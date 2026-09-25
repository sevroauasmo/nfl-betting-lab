"""Does the prop market absorb gameday inactives (announced 90 min before kickoff) instantly?

Surprise inactive: rosters_weekly status INA for the game, but NOT listed Out/Doubtful on that week's final injury report.
Weighted by his prior usage (target share / carry share, last 3 games). For teammates' props we track the consensus line at
T-120 (before news), T-80 (10 min after), T-45 and T-10 (close), and grade bets placed at T-80 in the direction of the news.
Voids: props with no graded result (player inactive / no snaps) are excluded. Writes results/news_window.json.
"""
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
EXCH = ("novig", "prophetx", "kalshi", "polymarket", "betopenly")
PASSES = {"news_t120": "t120", "news_t80": "t80", "news_t45": "t45", "close": "t10"}
MK = {"player_reception_yds": "receiving", "player_receptions": "receiving", "player_rush_yds": "rushing", "player_rush_attempts": "rushing",
      "player_pass_yds": "passing", "player_pass_attempts": "passing"}
imp = lambda o: np.where(np.asarray(o, float) < 0, -np.asarray(o, float) / (-np.asarray(o, float) + 100), 100 / (np.asarray(o, float) + 100))  # noqa: E731
payout = lambda o: np.where(np.asarray(o, float) < 0, 100 / -np.asarray(o, float), np.asarray(o, float) / 100)  # noqa: E731
R: dict = {}

# ------------------------------------------------------------------ consensus line per snapshot
pl = con.execute(f"""
    SELECT r.game_id, r.pass, r.bookmaker, r.market, r.player_key, r.point, r.over_price, r.under_price, r.actual, pm.player_id, s.team
    FROM prop_results r LEFT JOIN prop_player_map pm USING (game_id, player_key)
    LEFT JOIN player_game_stats s ON s.game_id = r.game_id AND s.player_id = pm.player_id
    WHERE r.pass IN ({",".join("'" + p + "'" for p in PASSES)}) AND r.market IN ({",".join("'" + m + "'" for m in MK)})
      AND r.over_price IS NOT NULL AND r.under_price IS NOT NULL AND r.point IS NOT NULL AND r.actual IS NOT NULL
""").df()
pl["snap"] = pl["pass"].map(PASSES)
pl["po"], pl["pu"] = imp(pl.over_price), imp(pl.under_price)
pl = pl[(pl.po + pl.pu - 1).between(-0.02, 0.25)]
pl["fair"] = pl.po / (pl.po + pl.pu)
pl["dist"] = (pl.fair - .5).abs()
k = ["game_id", "snap", "market", "player_key"]
main = pl.sort_values("dist").drop_duplicates(k + ["bookmaker"])
books = main[~main.bookmaker.isin(EXCH)]
cons = books.groupby(k).agg(line=("point", "median"), nb=("bookmaker", "nunique"), actual=("actual", "first"),
                            player_id=("player_id", "first"), team=("team", "first")).reset_index()
cons = cons[cons.nb >= 3]
at = main.merge(cons[k + ["line"]], on=k)
at = at[at.point == at.line]
best = at[~at.bookmaker.isin(EXCH)].groupby(k).agg(fair=("fair", "mean"), best_over=("over_price", "max"), best_under=("under_price", "max")).reset_index()
exq = at[at.bookmaker.isin(EXCH)].groupby(k).agg(ex_over=("over_price", "max"), ex_under=("under_price", "max")).reset_index()
C = cons.merge(best, on=k).merge(exq, on=k, how="left")
W = C.pivot_table(index=["game_id", "market", "player_key", "player_id", "team", "actual"], columns="snap",
                  values=["line", "fair", "best_over", "best_under", "ex_over", "ex_under"], aggfunc="first")
W.columns = [f"{a}_{b}" for a, b in W.columns]
W = W.reset_index().dropna(subset=["line_t120", "line_t80", "line_t10"])
W["group"] = W.market.map(MK)
R["props_with_all_snapshots"] = len(W)

# ------------------------------------------------------------------ surprise inactives, weighted by prior usage
ina = con.execute("""
    SELECT g.game_id, w.team, w.gsis_id AS player_id, w.position, w.season, w.week
    FROM rosters_weekly w JOIN games g ON g.season = w.season AND g.week = w.week AND w.team IN (g.home_team, g.away_team)
    WHERE w.status = 'INA' AND w.season >= 2023 AND w.position IN ('WR', 'TE', 'RB', 'QB')""").df()
rep = con.execute("""SELECT CAST(season AS INT) AS season, CAST(week AS INT) AS week, gsis_id AS player_id, report_status
                     FROM injuries WHERE season >= 2023""").df().drop_duplicates(["season", "week", "player_id"])
ina = ina.merge(rep, on=["season", "week", "player_id"], how="left")
ina["surprise"] = ~ina.report_status.isin(["Out", "Doubtful"])
use = pd.read_parquet(ROOT / "data/player_games.parquet", columns=["player_id", "season", "week", "target_share_m3", "carry_share_m3"])
# usage going INTO this game: latest 3-game average from an earlier week (merge_asof, strictly before) -- no end-of-season lookahead
use["t"] = (use.season * 100 + use.week).astype("int64")
ina["t"] = (ina.season.astype(int) * 100 + ina.week.astype(int)).astype("int64")
ina = pd.merge_asof(ina.sort_values("t"), use.sort_values("t")[["player_id", "t", "target_share_m3", "carry_share_m3"]],
                    on="t", by="player_id", allow_exact_matches=False).fillna({"target_share_m3": 0, "carry_share_m3": 0})
qb_starts = con.execute("SELECT qb_id AS player_id, season, count(*) starts FROM team_games WHERE season >= 2023 GROUP BY ALL").df()
ina = ina.merge(qb_starts, on=["player_id", "season"], how="left").fillna({"starts": 0})
sur = ina[ina.surprise]
agg = sur.groupby(["game_id", "team"]).agg(sur_tgt=("target_share_m3", "sum"), sur_car=("carry_share_m3", "sum"),
                                            sur_qb=("starts", lambda s: float((s >= 2).any()))).reset_index()
R["surprise_inactives"] = {"all_inactive_skill_players": len(ina), "surprise": int(ina.surprise.sum()),
                           "surprise_with_target_share_ge_10pct": int(((ina.surprise) & (ina.target_share_m3 >= .10)).sum()),
                           "surprise_with_carry_share_ge_20pct": int(((ina.surprise) & (ina.carry_share_m3 >= .20)).sum()),
                           "surprise_starting_qb": int(((ina.surprise) & (ina.starts >= 2) & (ina.position == 'QB')).sum())}
W = W.merge(agg, on=["game_id", "team"], how="left").fillna({"sur_tgt": 0, "sur_car": 0, "sur_qb": 0})
W["news"] = np.select([(W.group == "receiving") & (W.sur_tgt >= .10), (W.group == "rushing") & (W.sur_car >= .20)], ["targets freed", "carries freed"], "none")

# ------------------------------------------------------------------ 1. how fast do lines move after the news?
W["move_total"] = (W.line_t10 - W.line_t120) / W.line_t120.clip(lower=1)
W["move_by_t80"] = (W.line_t80 - W.line_t120) / W.line_t120.clip(lower=1)
W["move_after_t80"] = (W.line_t10 - W.line_t80) / W.line_t120.clip(lower=1)
mv = {}
for (grp, nw), d in W.groupby(["group", "news"]):
    if len(d) < 30:
        continue
    mv[f"{grp} | {nw}"] = {"n": len(d), "avg_move_t120_to_close_pct": round(100 * d.move_total.mean(), 2),
                           "avg_move_by_t80_pct": round(100 * d.move_by_t80.mean(), 2), "avg_move_after_t80_pct": round(100 * d.move_after_t80.mean(), 2),
                           "share_lines_moved_by_t80": round(float((d.line_t80 != d.line_t120).mean()), 3),
                           "share_lines_moved_after_t80": round(float((d.line_t10 != d.line_t80).mean()), 3)}
R["line_movement"] = mv


# ------------------------------------------------------------------ 2. bet at T-80 in the direction of the news
def grade(d, snap, side, price_col):
    line = d[f"line_{snap}"]
    win = np.where(side == "over", d.actual > line, d.actual < line).astype(float)
    push = d.actual == line
    odds = d[price_col]
    ok = ~push & odds.notna()
    pr = np.where(win[ok] == 1, payout(odds[ok]), -1.0)
    return {"n": int(ok.sum()), "win_rate": round(float(win[ok].mean()), 4) if ok.sum() else None,
            "roi": round(float(pr.mean()), 4) if ok.sum() else None, "se": round(float(pr.std() / math.sqrt(ok.sum())), 4) if ok.sum() > 1 else None}


bets = {}
for nw, grp in (("targets freed", "receiving"), ("carries freed", "rushing")):
    d = W[(W.news == nw) & (W.group == grp)]
    for snap in ("t120", "t80", "t45", "t10"):
        bets[f"{nw}: OVER at {snap}, best book"] = grade(d, snap, "over", f"best_over_{snap}")
        bets[f"{nw}: OVER at {snap}, exchange"] = grade(d, snap, "over", f"ex_over_{snap}")
    bets[f"{nw}: closing-line value of T-80 line (close - T80, pts)"] = round(float((d.line_t10 - d.line_t80).mean()), 2)
    bets[f"{nw}: fair over at close vs actual over rate"] = [round(float(d.fair_t10.mean()), 4), round(float((d.actual > d.line_t10).mean()), 4)]
R["bets"] = bets

# ------------------------------------------------------------------ 3. any line: does movement between T-80 and close continue in the same direction?
W["m1"] = np.sign(W.line_t80 - W.line_t120); W["m2"] = np.sign(W.line_t10 - W.line_t80)
moved = W[W.m1 != 0]
R["momentum_after_news_window"] = {"props_that_moved_T120_T80": len(moved),
                                   "then_moved_same_direction": round(float((moved.m2 == moved.m1).mean()), 3),
                                   "then_reversed": round(float((moved.m2 == -moved.m1).mean()), 3),
                                   "bet_in_direction_of_early_move_at_T80_best_book": grade(
                                       moved.assign(side=np.where(moved.m1 > 0, "over", "under")), "t80",
                                       np.where(moved.m1 > 0, "over", "under"), "best_over_t80") if False else None}
d_up, d_dn = moved[moved.m1 > 0], moved[moved.m1 < 0]
R["momentum_after_news_window"]["over at T-80 after line rose T120->T80"] = grade(d_up, "t80", "over", "best_over_t80")
R["momentum_after_news_window"]["under at T-80 after line fell T120->T80"] = grade(d_dn, "t80", "under", "best_under_t80")

(ROOT / "results/news_window.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps(R, indent=1, default=str))

# ------------------------------------------------------------------ 4. vs baseline: over rate - fair at close, and UNDER ROI, by news group and season
W["season"] = W.game_id.str[:4].astype(int)
W["over_c"] = np.where(W.actual > W.line_t10, 1.0, np.where(W.actual < W.line_t10, 0.0, np.nan))
W["qb_out"] = W.sur_qb >= 1
base = {}
for (grp, nw), d in W.groupby(["group", "news"]):
    d = d.dropna(subset=["over_c"])
    u = np.where(d.over_c == 0, payout(d.best_under_t10), -1.0)
    base[f"{grp} | {nw}"] = {"n": len(d), "over": round(d.over_c.mean(), 4), "fair": round(d.fair_t10.mean(), 4),
                             "resid": round(d.over_c.mean() - d.fair_t10.mean(), 4), "under_roi_best_close": round(u.mean(), 4),
                             "by_season_resid": {int(s): round(x.over_c.mean() - x.fair_t10.mean(), 4) for s, x in d.groupby("season")}}
for grp in ("receiving", "passing"):
    d = W[(W.group == grp) & W.qb_out].dropna(subset=["over_c"])
    base[f"{grp} | surprise starting QB inactive"] = {"n": len(d), "over": round(d.over_c.mean(), 4), "fair": round(d.fair_t10.mean(), 4),
                                                      "resid": round(d.over_c.mean() - d.fair_t10.mean(), 4)}
R["vs_baseline"] = base
(ROOT / "results/news_window.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps(base, indent=1))

# ------------------------------------------------------------------ 5. "targets freed" robustness: clustered by team-game, by market, exchange prices, T-120 entry
def clustered(d, col):
    g_ = d.groupby(["game_id", "team"])[col].agg(["sum", "count"])
    m = g_["sum"].sum() / g_["count"].sum(); n = len(g_)
    r = g_["sum"] - m * g_["count"]
    return m, math.sqrt(n / (n - 1) * (r ** 2).sum()) / g_["count"].sum(), n


tf = W[(W.news == "targets freed")].dropna(subset=["over_c"]).copy()
bl = W[(W.group == "receiving") & (W.news == "none")].dropna(subset=["over_c"])
tf["resid_adj"] = (tf.over_c - tf.fair_t10) - (bl.over_c - bl.fair_t10).mean()
rob = {}
m, se, n = clustered(tf, "resid_adj")
rob["excess_resid_vs_no_news_clustered"] = {"estimate": round(m, 4), "se": round(se, 4), "t": round(m / se, 2), "team_games": n}
tf["u_best"] = np.where(tf.over_c == 0, payout(tf.best_under_t10), -1.0)
m, se, n = clustered(tf, "u_best"); rob["under_roi_best_close_clustered"] = {"roi": round(m, 4), "se": round(se, 4), "team_games": n}
for snap in ("t120", "t80"):
    x = tf.dropna(subset=[f"best_under_{snap}"]).copy()
    x["u"] = np.where(x.actual < x[f"line_{snap}"], payout(x[f"best_under_{snap}"]), np.where(x.actual > x[f"line_{snap}"], -1.0, 0.0))
    m, se, n = clustered(x, "u"); rob[f"under_roi_best_{snap}_clustered"] = {"roi": round(m, 4), "se": round(se, 4), "team_games": n}
ex = tf.dropna(subset=["ex_under_t10"]).copy()
ex["u_ex"] = np.where(ex.over_c == 0, payout(ex.ex_under_t10), -1.0)
m, se, n = clustered(ex, "u_ex"); rob["under_roi_exchange_close_clustered"] = {"n_props": len(ex), "roi": round(m, 4), "se": round(se, 4), "team_games": n}
for mk, x in tf.groupby("market"):
    rob[f"{mk}: over vs fair"] = [len(x), round(x.over_c.mean(), 4), round(x.fair_t10.mean(), 4)]
# dose-response: bigger vacated share -> bigger effect?
for lo, hi in ((.10, .18), (.18, .30), (.30, 2)):
    x = tf[(tf.sur_tgt >= lo) & (tf.sur_tgt < hi)]
    rob[f"vacated target share {lo:.2f}-{hi:.2f}"] = {"n": len(x), "resid": round((x.over_c - x.fair_t10).mean(), 4)}
# was the scratched player Questionable (market could half-anticipate) or not on the report at all?
R["targets_freed_robustness"] = rob
(ROOT / "results/news_window.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps(rob, indent=1))
