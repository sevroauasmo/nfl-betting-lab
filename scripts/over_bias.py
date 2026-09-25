"""Is there a profitable over-bias in RB/WR props? (all games, no slog filter)

1. Decompose: actual under rate vs the no-vig (fair) under probability the books quote. The gap is the bias the
   market leaves on the table; the vig is what it costs to collect it. Per market, per book, per season.
2. Pre-registered "retail pressure" slices (where over money should be heaviest): stars, primetime, popular teams,
   after a big game, line above the player's recent median, line moved up during the week, early season, playoffs.
   Discovery 2023-24, confirmation 2025-26. Game-clustered bootstrap.
3. Kalshi yardage ladders (KXNFLRECYDS / KXNFLRSHYDS): pregame YES price vs how often the rung hit.
Writes results/over_bias.json.
"""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
EXCHANGES = ("novig", "prophetx")
MARKETS = ["player_reception_yds", "player_rush_yds", "player_receptions", "player_rush_attempts",
           "player_rush_reception_yds", "player_reception_longest", "player_rush_longest", "player_anytime_td"]
POPULAR = {"DAL", "KC", "PHI", "SF", "GB", "PIT", "BUF", "DET", "NE", "NYG", "CHI"}  # big retail followings, fixed a priori
rng = np.random.default_rng(11)
R: dict = {}


def imp(o):
    o = np.asarray(o, float)
    return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))


def dec(o):
    o = np.asarray(o, float)
    return 1 + np.where(o < 0, 100 / -o, o / 100)


# ------------------------------------------------------------------ one row per book quote, then consensus per bet
raw = con.execute(f"""
    SELECT p.game_id, left(p.game_id, 4)::INT AS season, p.pass, p.bookmaker, p.market, p.player_key, p.player_id,
           p.position, p.team, coalesce(p.point, 0.5) AS point, p.over_price, p.under_price, p.actual
    FROM prop_results p
    WHERE p.pass IN ('close', 'day_before') AND p.position IN ('RB', 'WR')
      AND p.market IN ({",".join("'" + m + "'" for m in MARKETS)})
      AND p.under_price IS NOT NULL AND p.over_price IS NOT NULL AND p.actual IS NOT NULL
""").df()
raw = raw[~((raw.market == "player_rush_yds") & (raw.position == "WR"))]
raw["exch"] = raw.bookmaker.isin(EXCHANGES)
raw["u_dec"] = dec(raw.under_price)
raw["fair_u"] = imp(raw.under_price) / (imp(raw.under_price) + imp(raw.over_price))
raw["hold"] = imp(raw.under_price) + imp(raw.over_price) - 1
bk = raw[~raw.exch]
key = ["game_id", "market", "player_key"]
mode_pt = (bk.groupby(["pass"] + key + ["point"]).size().rename("n").reset_index()
           .sort_values("n", ascending=False).drop_duplicates(["pass"] + key))
at = bk.merge(mode_pt.drop(columns="n"), on=["pass"] + key + ["point"])
bets = at.groupby(["pass"] + key + ["point"]).agg(
    season=("season", "first"), player_id=("player_id", "first"), position=("position", "first"), team=("team", "first"),
    actual=("actual", "first"), n_books=("bookmaker", "size"), fair_u=("fair_u", "median"), hold=("hold", "median"),
    u_med=("u_dec", "median"), u_best=("u_dec", "max")).reset_index()
exq = raw[raw.exch].merge(mode_pt.drop(columns="n"), on=["pass"] + key + ["point"]) \
    .groupby(["pass"] + key).agg(u_exch=("u_dec", "max"), fair_u_exch=("fair_u", "median")).reset_index()
bets = bets.merge(exq, on=["pass"] + key, how="left")
bets = bets[bets.n_books >= 2]
bets["win"] = np.where(bets.actual < bets.point, 1.0, np.where(bets.actual > bets.point, 0.0, np.nan))
bets = bets.dropna(subset=["win"])
for c in ["med", "best", "exch"]:
    bets[f"pnl_{c}"] = np.where(bets[f"u_{c}"].isna(), np.nan, np.where(bets.win == 1, bets[f"u_{c}"] - 1, -1.0))

# line movement: close point vs day-before point, same bet
mv = bets[bets.pass_ == "day_before"] if "pass_" in bets else bets[bets["pass"] == "day_before"]
mv = mv[key + ["point"]].rename(columns={"point": "point_db"})
b = bets[bets["pass"] == "close"].merge(mv, on=key, how="left")
b["move"] = b.point - b.point_db

# ------------------------------------------------------------------ context: game + player history (pre-game only)
g = con.execute("""SELECT game_id, week, weekday, gametime, gameday, home_team, away_team, game_type, total_line FROM games
                   WHERE season >= 2023""").df()
g["primetime"] = g.weekday.isin(["Thursday", "Monday"]) | ((g.weekday == "Sunday") & (g.gametime >= "19:00"))
b = b.merge(g, on="game_id", how="left")
b["popular"] = b.home_team.isin(POPULAR) | b.away_team.isin(POPULAR)
b["own_popular"] = b.team.isin(POPULAR)

h = con.execute("""
    SELECT s.player_id, s.game_id, g.gameday, s.receiving_yards, s.rushing_yards, s.receptions, s.carries,
           s.receiving_yards + s.rushing_yards AS rr_yds
    FROM stats_player_week s JOIN games g USING (game_id) WHERE s.season >= 2021
""").df().sort_values(["player_id", "gameday"])
stat_of = {"player_reception_yds": "receiving_yards", "player_rush_yds": "rushing_yards", "player_receptions": "receptions",
           "player_rush_attempts": "carries", "player_rush_reception_yds": "rr_yds"}
for s in set(stat_of.values()):
    grp = h.groupby("player_id")[s]
    h[f"{s}_last"] = grp.shift(1)
    h[f"{s}_med8"] = grp.transform(lambda x: x.shift(1).rolling(8, min_periods=4).median())
b = b.merge(h.drop(columns=["gameday"] + list(set(stat_of.values()))), on=["player_id", "game_id"], how="left")
b["last"] = np.nan; b["med8"] = np.nan
for m, s in stat_of.items():
    sel = b.market == m
    b.loc[sel, "last"] = b.loc[sel, f"{s}_last"]; b.loc[sel, "med8"] = b.loc[sel, f"{s}_med8"]
# league-wide rank of the line within (season, week, market): the names everyone is betting
b["wk_rank"] = b.groupby(["season", "week", "market"]).point.rank(ascending=False, method="first")
b["yds"] = b.market.isin(["player_reception_yds", "player_rush_yds"])
print(f"{len(b):,} close bets, {b.game_id.nunique()} games; RB/WR main yardage: {b.yds.sum():,}")


# ------------------------------------------------------------------ evaluation helpers
def evaluate(d, label, price="pnl_med", boot=1500):
    d = d.dropna(subset=[price, "fair_u"])
    if len(d) < 30:
        return None
    gs = d.groupby("game_id").agg(p=(price, "sum"), n=(price, "size"), w=("win", "sum"), f=("fair_u", "sum"))
    idx = rng.integers(0, len(gs), (boot, len(gs)))
    n_b = gs.n.values[idx].sum(1)
    rois = gs.p.values[idx].sum(1) / n_b
    bias = (gs.w.values[idx].sum(1) - gs.f.values[idx].sum(1)) / n_b
    return {"label": label, "games": len(gs), "bets": int(gs.n.sum()),
            "under": round(gs.w.sum() / gs.n.sum(), 4), "fair": round(gs.f.sum() / gs.n.sum(), 4),
            "bias": round((gs.w.sum() - gs.f.sum()) / gs.n.sum(), 4),
            "bias_ci90": [round(float(np.quantile(bias, q)), 4) for q in (.05, .95)],
            "roi": round(gs.p.sum() / gs.n.sum(), 4), "roi_ci90": [round(float(np.quantile(rois, q)), 4) for q in (.05, .95)],
            "p_roi_le0": round(float((rois <= 0).mean()), 3)}


def show(rows, title, cols=None):
    df = pd.DataFrame([r for r in rows if r])
    print(f"\n== {title}")
    if not df.empty:
        print(df[cols or df.columns].to_string(index=False))
    return df.to_dict("records")


# 1) decomposition by market
rows = []
for m in MARKETS:
    for pos in ["RB", "WR"]:
        d = b[(b.market == m) & (b.position == pos)]
        r = evaluate(d, f"{pos} {m.replace('player_', '')}")
        if r:
            r["hold"] = round(float(d.hold.median()), 4)
            r["roi_best"] = evaluate(d, "", "pnl_best")["roi"]
            e = evaluate(d, "", "pnl_exch"); r["roi_exch"] = e["roi"] if e else None; r["exch_bets"] = e["bets"] if e else 0
            rows.append(r)
R["by_market"] = show(rows, "bias = actual under rate - no-vig fair under prob (median book); ROI at median / best / exchange",
                      ["label", "bets", "under", "fair", "bias", "bias_ci90", "hold", "roi", "roi_best", "roi_exch", "exch_bets"])

Y = b[b.yds]  # the core bet the user wants: RB rush yds + RB/WR rec yds unders
R["core_by_season"] = show([evaluate(Y[Y.season == s], str(s)) for s in sorted(Y.season.unique())]
                           + [evaluate(Y[Y.season == s], f"{s} exch", "pnl_exch") for s in sorted(Y.season.unique())],
                           "core (RB/WR yardage) by season")

# 2) bias by book (who shades overs hardest?) on the core bet: fair under from each book vs outcome
bb = raw[(raw["pass"] == "close") & raw.market.isin(["player_reception_yds", "player_rush_yds"])].copy()
bb["win"] = np.where(bb.actual < bb.point, 1.0, np.where(bb.actual > bb.point, 0.0, np.nan))
bb = bb.dropna(subset=["win"])
bb["pnl"] = np.where(bb.win == 1, bb.u_dec - 1, -1.0)
byb = bb.groupby("bookmaker").agg(quotes=("win", "size"), under=("win", "mean"), fair=("fair_u", "mean"),
                                  hold=("hold", "median"), under_price=("under_price", "median"), roi=("pnl", "mean"))
byb["bias"] = byb.under - byb.fair
byb = byb[byb.quotes >= 2000].sort_values("roi", ascending=False).round(4)
print("\n== by book, core yardage unders at that book's own line and price\n", byb.to_string())
R["by_book"] = byb.reset_index().to_dict("records")

# 3) pre-registered retail-pressure slices, core bet
slices = {
    "star: top-24 line of week (league-wide)": Y.wk_rank <= 24,
    "not a star": Y.wk_rank > 24,
    "primetime (TNF/SNF/MNF)": Y.primetime,
    "Sunday day": ~Y.primetime,
    "player on a popular team": Y.own_popular,
    "big last game (last >= 1.5x line)": Y["last"] >= 1.5 * Y.point,
    "quiet last game (last <= 0.5x line)": Y["last"] <= 0.5 * Y.point,
    "line > trailing-8 median by 15%+": Y.point >= 1.15 * Y.med8,
    "line < trailing-8 median": Y.point < Y.med8,
    "line moved UP day-before->close": Y.move > 0,
    "line moved DOWN": Y.move < 0,
    "line unchanged": Y.move == 0,
    "weeks 1-4": Y.week <= 4,
    "weeks 5+": Y.week >= 5,
    "playoffs": Y.game_type != "REG",
    "under priced plus-money (median)": Y.u_med >= 2.0,
    "under juiced -125 or worse": Y.u_med <= 1.8,
    "high total (>=48)": Y.total_line >= 48,
}
rows = []
for k, m in slices.items():
    for per, sel in [("23-24", Y.season <= 2024), ("25-26", Y.season >= 2025)]:
        r = evaluate(Y[m & sel], f"{k} [{per}]")
        if r:
            e = evaluate(Y[m & sel], "", "pnl_exch"); r["roi_exch"] = e["roi"] if e else None
            rows.append(r)
R["slices"] = show(rows, "retail-pressure slices, core yardage unders (discovery 23-24 | confirm 25-26)",
                   ["label", "bets", "under", "fair", "bias", "bias_ci90", "roi", "roi_ci90", "roi_exch"])

# 4) exchange-only view: do exchange lines carry the same bias? (exchange fair prob vs outcome)
E = Y.dropna(subset=["fair_u_exch", "pnl_exch"]).copy()
E["fair_u"] = E.fair_u_exch
R["exchange_fair"] = show([evaluate(E, "exchange fair prob, all", "pnl_exch")]
                          + [evaluate(E[E.season == s], f"exchange {s}", "pnl_exch") for s in sorted(E.season.unique())],
                          "exchange (Novig/ProphetX) de-vigged fair under vs outcome")

# ------------------------------------------------------------------ 5) Kalshi yardage ladders
km = pd.read_parquet(ROOT / "data/kalshi/markets.parquet")
km = km[km.series.isin(["KXNFLRECYDS", "KXNFLRSHYDS"]) & km.result.isin(["yes", "no"])]
kick_of = pd.read_parquet(ROOT / "data/kalshi/game_kickoffs.parquet").set_index("ticker").kick.to_dict()
cdir = ROOT / "data/kalshi/candles"
rows = []
for t, occ, res, ser, strike in km[["ticker", "occurrence", "result", "series", "floor_strike"]].itertuples(index=False):
    f = cdir / f"{t}.json"
    if not f.exists():
        continue
    cs = json.loads(f.read_text())
    cs = cs.get("candlesticks", cs) if isinstance(cs, dict) else cs
    kick = kick_of.get(t)
    if kick is None:
        continue
    pre = [c for c in cs if c.get("end_period_ts", 0) <= kick - 15 * 60]  # last hourly candle ending 15m+ before kick
    if not pre:
        continue
    c = pre[-1]
    def px(side):
        v = (c.get(side) or {}).get("close_dollars") or (c.get(side) or {}).get("close")
        return float(v) / (1 if isinstance(v, str) or (v is not None and float(v) <= 1) else 100) if v is not None else None
    bid, ask = px("yes_bid"), px("yes_ask")
    if bid is None or ask is None or ask <= bid or ask - bid > .10:  # skip dead books
        continue
    rows.append({"ticker": t, "series": ser, "strike": strike, "yes": res == "yes", "bid": bid, "ask": ask,
                 "mid": (bid + ask) / 2, "season": pd.Timestamp(occ).year})
k = pd.DataFrame(rows)
if len(k):
    k["no_taker_roi"] = np.where(~k.yes, 1 / (1 - k.bid) - 1, -1.0)        # buy NO at 1 - yes_bid
    k["no_maker_roi"] = np.where(~k.yes, 1 / (1 - k.ask + .01) - 1, -1.0)  # rest a NO one tick inside (yes offer), if filled
    k["bucket"] = pd.cut(k.mid, [0, .1, .25, .4, .6, .75, .9, 1])
    kb = k.groupby("bucket", observed=True).agg(n=("yes", "size"), mid=("mid", "mean"), yes_rate=("yes", "mean"),
                                                 spread=("ask", lambda s: (s - k.loc[s.index, "bid"]).mean()),
                                                 no_taker_roi=("no_taker_roi", "mean")).round(4)
    print(f"\n== Kalshi yardage ladders: {len(k)} rungs with a pregame quote\n", kb.to_string())
    print("all rungs: yes_rate", round(k.yes.mean(), 4), "mid", round(k.mid.mean(), 4),
          "NO taker ROI", round(k.no_taker_roi.mean(), 4))
    R["kalshi"] = {"n": len(k), "yes_rate": round(float(k.yes.mean()), 4), "mid": round(float(k.mid.mean()), 4),
                   "no_taker_roi": round(float(k.no_taker_roi.mean()), 4),
                   "by_bucket": kb.reset_index().astype({"bucket": str}).to_dict("records")}

(ROOT / "results/over_bias.json").write_text(json.dumps(R, indent=1, default=str))

# ------------------------------------------------------------------ 6) what the bias is worth with zero vig
# ROI if every under traded at the exchange's own no-vig midpoint (a maker resting there, assuming fills were random).
# Real fills skew toward games where informed money wants the over, so this is a ceiling.
X = Y.dropna(subset=["fair_u_exch"]).copy()
X = X[(X.fair_u_exch > .3) & (X.fair_u_exch < .7)]
X["pnl_mid"] = np.where(X.win == 1, 1 / X.fair_u_exch - 1, -1.0)
X["fair_u"] = X.fair_u_exch
R["midpoint"] = show([evaluate(X, "all", "pnl_mid")] + [evaluate(X[X.season == s], str(s), "pnl_mid") for s in sorted(X.season.unique())]
                     + [evaluate(X[X.market == m], m, "pnl_mid") for m in ["player_reception_yds", "player_rush_yds"]],
                     "core unders at the exchange no-vig midpoint (zero-vig ceiling)")
(ROOT / "results/over_bias.json").write_text(json.dumps(R, indent=1, default=str))
