"""Round 3: player props (2023-2026, The Odds API snapshots graded against nflverse results).

Per (game, snapshot, market, player) we build:
  - each book's main line: the two-way point priced closest to even
  - consensus point: median of books' main points
  - fair P(over) at the consensus point: average of books' de-vigged probabilities at that point
Then test market bias, line shopping / +EV vs a leave-one-out consensus, open->close movement, TD longshots,
and a naive rolling-median projection. Writes results/props.json.
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
EXCHANGES = {"novig", "prophetx", "kalshi", "polymarket", "betopenly", "betfair_ex_us", "sporttrade"}
CORE = ["player_reception_yds", "player_receptions", "player_rush_yds", "player_rush_attempts", "player_pass_yds",
        "player_pass_tds", "player_pass_completions", "player_pass_attempts", "player_pass_interceptions",
        "player_rush_reception_yds", "player_reception_longest", "player_rush_longest", "player_pass_longest_completion",
        "player_tackles_assists", "player_kicking_points", "player_field_goals", "player_pass_rush_yds", "player_sacks"]
R: dict = {}


def imp(o):
    o = np.asarray(o, float)
    return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))


def payout(o):
    o = np.asarray(o, float)
    return np.where(o < 0, 100 / -o, o / 100)


def binrec(win):
    win = pd.Series(win).dropna(); w = int(win.sum()); n = len(win)
    return {"n": n, "pct": round(w / n, 4) if n else None, "p": round(stats.binomtest(w, n, .5).pvalue, 4) if n else None}


def roi(win, odds):
    win = np.asarray(win, float); ok = ~np.isnan(win)
    pr = np.where(win[ok] == 1, payout(np.asarray(odds)[ok]), -1.0)
    return {"n": int(ok.sum()), "roi": round(float(pr.mean()), 4) if ok.sum() else None,
            "se": round(float(pr.std() / np.sqrt(ok.sum())), 4) if ok.sum() > 1 else None}


# ------------------------------------------------------------------ two-way lines, graded
tw = con.execute(f"""
    SELECT game_id, left(game_id, 4)::INT AS season, pass, bookmaker, market, player_key, player, point,
           over_price, under_price, actual, result, position,
           date_diff('second', last_update, snapshot_ts) / 60.0 AS quote_age_min
    FROM prop_results
    WHERE over_price IS NOT NULL AND under_price IS NOT NULL AND point IS NOT NULL AND actual IS NOT NULL
      AND market IN ({",".join("'" + m + "'" for m in CORE)})
""").df()
tw["p_over_raw"], tw["p_under_raw"] = imp(tw.over_price), imp(tw.under_price)
tw["hold"] = tw.p_over_raw + tw.p_under_raw - 1
tw = tw[(tw.hold > -0.02) & (tw.hold < 0.25)]  # drop broken quotes
tw["p_over_fair"] = tw.p_over_raw / (tw.p_over_raw + tw.p_under_raw)
tw["exch"] = tw.bookmaker.isin(EXCHANGES)
key = ["game_id", "pass", "market", "player_key"]

# each book's main line = the point whose fair prob is closest to 50%
tw["dist"] = (tw.p_over_fair - .5).abs()
main = tw.sort_values("dist").drop_duplicates(key + ["bookmaker"])
cons = (main[~main.exch].groupby(key).agg(cons_point=("point", "median"), n_books=("bookmaker", "nunique"),
                                            actual=("actual", "first"), season=("season", "first")).reset_index())
cons = cons[cons.n_books >= 3]
at = main.merge(cons[key + ["cons_point"]], on=key)
at_cons = at[at.point == at.cons_point]
fair = at_cons[~at_cons.exch].groupby(key).agg(fair_over=("p_over_fair", "mean"), best_over=("over_price", "max"),
                                                 best_under=("under_price", "max"), n_at=("bookmaker", "nunique")).reset_index()
P = cons.merge(fair, on=key)
P["over"] = np.where(P.actual > P.cons_point, 1.0, np.where(P.actual < P.cons_point, 0.0, np.nan))
R["coverage"] = P[P["pass"] == "close"].groupby("season").agg(props=("over", "size"), avg_books=("n_books", "mean")).round(1).reset_index().to_dict("records")

# ------------------------------------------------------------------ 1. market bias at close
C = P[P["pass"] == "close"]
bias = []
for m, d in C.groupby("market"):
    dk = at_cons[(at_cons["pass"] == "close") & (at_cons.market == m) & (at_cons.bookmaker == "draftkings")].merge(P[key + ["over"]], on=key)
    bias.append({"market": m, "props": len(d), "over_pct": binrec(d.over)["pct"], "fair_over": round(d.fair_over.mean(), 4),
                 "p": binrec(d.over)["p"],
                 "by_season": {int(s): binrec(dd.over)["pct"] for s, dd in d.groupby("season")},
                 "under_best_price_roi": roi(1 - d.over, d.best_under), "over_best_price_roi": roi(d.over, d.best_over),
                 "under_dk_roi": roi(1 - dk.over, dk.under_price)})
R["bias_close"] = sorted(bias, key=lambda r: -r["props"])
allc = C
R["bias_all_core"] = {"props": len(allc), "over_pct": binrec(allc.over), "fair_over": round(allc.fair_over.mean(), 4),
                      "under_best_roi": roi(1 - allc.over, allc.best_under), "over_best_roi": roi(allc.over, allc.best_over),
                      "by_season_under_best_roi": {int(s): roi(1 - d.over, d.best_under) for s, d in allc.groupby("season")}}

# calibration: fair P(over) buckets vs actual
C = C.assign(fb=pd.cut(C.fair_over, [0, .4, .45, .48, .52, .55, .6, 1]))
R["calibration_close"] = (C.groupby("fb", observed=True).agg(n=("over", "count"), fair=("fair_over", "mean"), actual=("over", "mean"))
                          .round(4).reset_index().assign(fb=lambda d: d.fb.astype(str)).to_dict("records"))

# ------------------------------------------------------------------ 2. line shopping / +EV vs leave-one-out consensus (same point)
cl = at_cons[at_cons["pass"] == "close"].merge(P[key + ["over"]], on=key)
g = cl.groupby(key)
cl["sum_fair"] = g.p_over_fair.transform("sum"); cl["cnt"] = g.p_over_fair.transform("count")
cl = cl[cl.cnt >= 4]
cl["ref_over"] = (cl.sum_fair - cl.p_over_fair) / (cl.cnt - 1)  # consensus excluding this book
cl["ev_over"] = cl.ref_over * (1 + payout(cl.over_price)) - 1
cl["ev_under"] = (1 - cl.ref_over) * (1 + payout(cl.under_price)) - 1
ev = {}
for th in (0.0, 0.02, 0.04, 0.06, 0.08):
    o = cl[cl.ev_over >= th]; u = cl[cl.ev_under >= th]
    bets_win = np.concatenate([o.over.values, 1 - u.over.values]); bets_odds = np.concatenate([o.over_price.values, u.under_price.values])
    bets_season = np.concatenate([o.season.values, u.season.values])
    ev[f"EV>={int(th*100)}%"] = {"all": roi(bets_win, bets_odds),
                                 "by_season": {int(s): roi(bets_win[bets_season == s], bets_odds[bets_season == s]) for s in np.unique(bets_season)},
                                 "avg_claimed_ev": round(float(np.concatenate([o.ev_over.values, u.ev_under.values]).mean()), 4) if len(bets_win) else None}
R["ev_same_point"] = ev
# which books are most often the +EV outlier
top = cl[(cl.ev_over >= .04) | (cl.ev_under >= .04)]
R["ev_by_book"] = []
for b, d in top.groupby("bookmaker"):
    w = np.where(d.ev_over >= .04, d.over, 1 - d.over); o = np.where(d.ev_over >= .04, d.over_price, d.under_price)
    R["ev_by_book"].append({"book": b, **roi(w, o)})
R["ev_by_book"].sort(key=lambda r: -r["n"])

# off-point lines: a book's main line differs from consensus
off = at[(at["pass"] == "close") & ~at.exch].merge(P[key + ["over"]], on=key)
off["gap"] = off.point - off.cons_point
off["rel_gap"] = off.gap / off.cons_point.clip(lower=1)
offr = {}
for m in ["player_reception_yds", "player_rush_yds", "player_pass_yds", "player_receptions"]:
    d = off[off.market == m]
    for lo, hi, lab in ((0.03, 0.08, "3-8% off"), (0.08, 1, "8%+ off")):
        lowline = d[(d.rel_gap <= -lo) & (d.rel_gap > -hi)]   # book line below consensus -> bet over there
        highline = d[(d.rel_gap >= lo) & (d.rel_gap < hi)]    # book line above consensus -> bet under there
        w = np.concatenate([(lowline.actual > lowline.point).astype(float).values, (highline.actual < highline.point).astype(float).values])
        o = np.concatenate([lowline.over_price.values, highline.under_price.values])
        offr[f"{m} {lab}"] = roi(w, o)
R["off_point"] = offr

# ------------------------------------------------------------------ 3. open -> close movement
mv = P[P["pass"].isin(["open", "day_before", "close"])].pivot_table(index=key[:1] + key[2:], columns="pass",
                                                                   values=["cons_point", "fair_over", "best_over", "best_under", "actual"], aggfunc="first")
mv.columns = [f"{a}_{b}" for a, b in mv.columns]
mv = mv.dropna(subset=["cons_point_open", "cons_point_close"]).reset_index()
mv["actual"] = mv.actual_close.fillna(mv.actual_open)
mv["move"] = mv.cons_point_close - mv.cons_point_open
mv["season"] = mv.game_id.str[:4].astype(int)
mv["over_open"] = np.where(mv.actual > mv.cons_point_open, 1.0, np.where(mv.actual < mv.cons_point_open, 0.0, np.nan))
mv["over_close"] = np.where(mv.actual > mv.cons_point_close, 1.0, np.where(mv.actual < mv.cons_point_close, 0.0, np.nan))
moved = mv[mv.move != 0]
up, down = moved[moved.move > 0], moved[moved.move < 0]
R["movement"] = {
    "props_with_open_and_close": len(mv), "share_line_moved": round(float((mv.move != 0).mean()), 4),
    "mae_open": round(float((mv.actual - mv.cons_point_open).abs().mean()), 3), "mae_close": round(float((mv.actual - mv.cons_point_close).abs().mean()), 3),
    "bet_open_in_move_direction (hindsight)": roi(np.concatenate([up.over_open, 1 - down.over_open]), np.concatenate([up.best_over_open, down.best_under_open])),
    "same bet graded at close line": roi(np.concatenate([up.over_close, 1 - down.over_close]), np.concatenate([up.best_over_close, down.best_under_close])),
    "under at open (best price)": roi(1 - mv.over_open, mv.best_under_open), "under at close (best price)": roi(1 - mv.over_close, mv.best_under_close),
}
by_mkt = []
for m, d in mv.groupby("market"):
    by_mkt.append({"market": m, "n": len(d), "moved": round(float((d.move != 0).mean()), 3), "mae_open": round(float((d.actual - d.cons_point_open).abs().mean()), 2),
                   "mae_close": round(float((d.actual - d.cons_point_close).abs().mean()), 2),
                   "under_open_roi": roi(1 - d.over_open, d.best_under_open)["roi"], "under_close_roi": roi(1 - d.over_close, d.best_under_close)["roi"]})
R["movement_by_market"] = sorted(by_mkt, key=lambda r: -r["n"])

# ------------------------------------------------------------------ 4. anytime TD (Yes prices) longshot bias
td = con.execute("""SELECT game_id, left(game_id,4)::INT season, pass, bookmaker, player_key, over_price, actual
                    FROM prop_results WHERE market='player_anytime_td' AND over_price IS NOT NULL AND actual IS NOT NULL AND pass='close'""").df()
td = td[~td.bookmaker.isin(EXCHANGES)]
td["scored"] = (td.actual >= 1).astype(float)
tdb = td.groupby(["game_id", "player_key"]).agg(best=("over_price", "max"), med=("over_price", "median"), scored=("scored", "first"), season=("season", "first")).reset_index()
tdb["impl_med"] = imp(tdb.med)
tdb["b"] = pd.cut(tdb.impl_med, [0, .1, .2, .3, .4, .5, .6, 1])
R["anytime_td"] = [{"implied": str(b), "n": len(d), "implied_avg": round(float(d.impl_med.mean()), 4), "scored": round(float(d.scored.mean()), 4),
                    "roi_best_price": roi(d.scored, d.best)["roi"], "roi_median_price": roi(d.scored, d.med)["roi"]}
                   for b, d in tdb.groupby("b", observed=True)]

# ------------------------------------------------------------------ 5. naive projection: rolling median of last 8 games vs line
hist = con.execute("""SELECT player_id, season, week, game_id, receiving_yards, receptions, rushing_yards, carries, passing_yards, completions, attempts
                      FROM stats_player_week WHERE season >= 2022 AND season_type IN ('REG','POST') ORDER BY player_id, season, week""").df()
stat_of = {"player_reception_yds": "receiving_yards", "player_receptions": "receptions", "player_rush_yds": "rushing_yards",
           "player_rush_attempts": "carries", "player_pass_yds": "passing_yards", "player_pass_completions": "completions", "player_pass_attempts": "attempts"}
for c in stat_of.values():
    hist[f"med8_{c}"] = hist.groupby("player_id")[c].transform(lambda s: s.shift().rolling(8, min_periods=4).median())
    hist[f"mean8_{c}"] = hist.groupby("player_id")[c].transform(lambda s: s.shift().rolling(8, min_periods=4).mean())
pid = con.execute("SELECT DISTINCT game_id, player_key, player_id FROM prop_results WHERE player_id IS NOT NULL").df()
proj_rows = []
Cc = P[(P["pass"] == "close") & P.market.isin(stat_of)].merge(pid, on=["game_id", "player_key"]).merge(hist.drop(columns=["season", "week"]), on=["player_id", "game_id"])
for m, c in stat_of.items():
    d = Cc[Cc.market == m].dropna(subset=[f"med8_{c}"]).copy()
    for kind in ("med8", "mean8"):
        d["edge"] = (d[f"{kind}_{c}"] - d.cons_point) / d.cons_point.clip(lower=1)
        for lab, msk in {"proj >= 15% above line": d.edge >= .15, "proj >= 15% below line": d.edge <= -.15}.items():
            dd = d[msk]; win = dd.over if "above" in lab else 1 - dd.over; odds = dd.best_over if "above" in lab else dd.best_under
            proj_rows.append({"market": m, "proj": kind, "rule": lab, **roi(win, odds),
                              "by_season": {int(s): roi(win[dd.season == s], odds[dd.season == s])["roi"] for s in sorted(dd.season.unique())}})
R["naive_projection"] = proj_rows

# ------------------------------------------------------------------ 6. where do Kalshi/Polymarket/exchanges sit?
R["exchange_coverage"] = con.execute("""SELECT bookmaker, left(game_id,4) season, count(DISTINCT game_id) games, count(DISTINCT market) markets
    FROM odds_raw WHERE bookmaker IN ('kalshi','polymarket','novig','prophetx','betopenly') AND market LIKE 'player_%' GROUP BY 1,2 ORDER BY 1,2""").df().to_dict("records")

(OUT / "props.json").write_text(json.dumps(R, indent=1, default=str))
print("done")

# ------------------------------------------------------------------ 7. robustness of +EV: regulated US books only; exchange as reference
OFFSHORE = {"betonlineag", "bovada", "lowvig", "mybookieag", "betus"}
reg = cl[~cl.bookmaker.isin(OFFSHORE | EXCHANGES)]
R["ev_regulated_only"] = {}
for th in (0.02, 0.04, 0.06):
    o = reg[reg.ev_over >= th]; u = reg[reg.ev_under >= th]
    w = np.concatenate([o.over.values, 1 - u.over.values]); od = np.concatenate([o.over_price.values, u.under_price.values])
    ss = np.concatenate([o.season.values, u.season.values])
    R["ev_regulated_only"][f"EV>={int(th*100)}%"] = {"all": roi(w, od), "by_season": {int(s): roi(w[ss == s], od[ss == s])["roi"] for s in np.unique(ss)}}
# exchange reference: de-vigged Novig/ProphetX price at the same point (2024+)
ex = at_cons[(at_cons["pass"] == "close") & at_cons.exch].groupby(key).p_over_fair.mean().rename("ex_over").reset_index()
bx = at_cons[(at_cons["pass"] == "close") & ~at_cons.exch & ~at_cons.bookmaker.isin(OFFSHORE)].merge(ex, on=key).merge(P[key + ["over"]], on=key)
bx["ev_over"] = bx.ex_over * (1 + payout(bx.over_price)) - 1
bx["ev_under"] = (1 - bx.ex_over) * (1 + payout(bx.under_price)) - 1
R["ev_vs_exchange"] = {}
for th in (0.0, 0.02, 0.04, 0.06):
    o = bx[bx.ev_over >= th]; u = bx[bx.ev_under >= th]
    w = np.concatenate([o.over.values, 1 - u.over.values]); od = np.concatenate([o.over_price.values, u.under_price.values])
    ss = np.concatenate([o.season.values, u.season.values])
    R["ev_vs_exchange"][f"EV>={int(th*100)}%"] = {"all": roi(w, od), "by_season": {int(s): roi(w[ss == s], od[ss == s])["roi"] for s in np.unique(ss)}}
# exchanges themselves: how does the exchange price do vs outcome (calibration of the "sharp" line)
exo = ex.merge(P[key + ["over", "fair_over"]], on=key)
R["exchange_calibration"] = {"n": len(exo), "exchange_fair_over": round(float(exo.ex_over.mean()), 4),
                             "book_fair_over": round(float(exo.fair_over.mean()), 4), "actual_over": round(float(exo.over.mean()), 4)}
(OUT / "props.json").write_text(json.dumps(R, indent=1, default=str))

# staleness: is the +EV quote old relative to the snapshot (i.e. possibly not bettable)?
bx["age"] = bx.quote_age_min
R["ev_vs_exchange_by_quote_age"] = {}
for lab, m in {"<2 min": bx.age < 2, "2-10 min": bx.age.between(2, 10), "10-60 min": bx.age.between(10, 60, inclusive="right"), ">60 min": bx.age > 60}.items():
    d = bx[m]; o = d[d.ev_over >= .02]; u = d[d.ev_under >= .02]
    R["ev_vs_exchange_by_quote_age"][lab] = roi(np.concatenate([o.over.values, 1 - u.over.values]), np.concatenate([o.over_price.values, u.under_price.values]))
R["ev_vs_exchange_by_book"] = []
for b, d in bx.groupby("bookmaker"):
    o = d[d.ev_over >= .02]; u = d[d.ev_under >= .02]
    R["ev_vs_exchange_by_book"].append({"book": b, **roi(np.concatenate([o.over.values, 1 - u.over.values]), np.concatenate([o.over_price.values, u.under_price.values]))})
R["ev_vs_exchange_by_market"] = []
for mk, d in bx.groupby("market"):
    o = d[d.ev_over >= .02]; u = d[d.ev_under >= .02]
    R["ev_vs_exchange_by_market"].append({"market": mk, "overs": len(o), "unders": len(u), **roi(np.concatenate([o.over.values, 1 - u.over.values]), np.concatenate([o.over_price.values, u.under_price.values]))})
R["ev_vs_exchange_side"] = {"over bets": roi(bx[bx.ev_over >= .02].over, bx[bx.ev_over >= .02].over_price),
                            "under bets": roi(1 - bx[bx.ev_under >= .02].over, bx[bx.ev_under >= .02].under_price)}
(OUT / "props.json").write_text(json.dumps(R, indent=1, default=str))
