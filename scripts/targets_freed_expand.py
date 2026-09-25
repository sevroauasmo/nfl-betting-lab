"""How strong is the 'targets freed -> teammates' receiving unders' evidence, and does it generalize?

1. Bootstrap (resampling team-games) CI and P(ROI > 0) for the original rule; how many team-games to confirm +5%.
2. Variants of the same mechanism, each with 2023-24 vs 2025-26:
   - surprise vs KNOWN absences (listed Out/Doubtful, i.e. priced for days) vs both
   - target-share thresholds 5 / 10 / 15 / 20%
   - trigger position (WR, TE, RB)
   - affected markets: teammates' receptions / rec yds (original), rush+rec yds, longest reception; QB passing yds / completions
Graded at the closing consensus line; ROI at the best available book price; SEs clustered by team-game.
"""
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
payout = lambda o: np.where(np.asarray(o, float) < 0, 100 / -np.asarray(o, float), np.asarray(o, float) / 100)  # noqa: E731
imp = lambda o: np.where(np.asarray(o, float) < 0, -np.asarray(o, float) / (-np.asarray(o, float) + 100), 100 / (np.asarray(o, float) + 100))  # noqa: E731
R: dict = {}
MARKETS = ["player_receptions", "player_reception_yds", "player_rush_reception_yds", "player_reception_longest",
           "player_pass_yds", "player_pass_completions", "player_pass_attempts"]

# ------------------------------------------------------------------ closing consensus per prop (books only) + best under price
pl = con.execute(f"""
    SELECT r.game_id, r.bookmaker, r.market, r.player_key, r.point, r.over_price, r.under_price, r.actual, pm.player_id, s.team, s.position
    FROM prop_results r LEFT JOIN prop_player_map pm USING (game_id, player_key)
    LEFT JOIN player_game_stats s ON s.game_id = r.game_id AND s.player_id = pm.player_id
    WHERE r.pass = 'close' AND r.market IN ({",".join("'" + m + "'" for m in MARKETS)})
      AND r.over_price IS NOT NULL AND r.under_price IS NOT NULL AND r.point IS NOT NULL AND r.actual IS NOT NULL
      AND r.bookmaker NOT IN ('novig', 'prophetx', 'kalshi', 'polymarket', 'betopenly')""").df()
pl["po"], pl["pu"] = imp(pl.over_price), imp(pl.under_price)
pl = pl[(pl.po + pl.pu - 1).between(-0.02, 0.25)]
pl["fair"] = pl.po / (pl.po + pl.pu); pl["dist"] = (pl.fair - .5).abs()
k = ["game_id", "market", "player_key"]
main = pl.sort_values("dist").drop_duplicates(k + ["bookmaker"])
cons = main.groupby(k).agg(line=("point", "median"), nb=("bookmaker", "nunique"), actual=("actual", "first"), player_id=("player_id", "first"),
                           team=("team", "first"), position=("position", "first")).reset_index()
cons = cons[cons.nb >= 3]
at = main.merge(cons[k + ["line"]], on=k); at = at[at.point == at.line]
P = cons.merge(at.groupby(k).agg(fair=("fair", "mean"), best_under=("under_price", "max")).reset_index(), on=k)
P["under"] = np.where(P.actual < P.line, 1.0, np.where(P.actual > P.line, 0.0, np.nan))
P = P.dropna(subset=["under"])
P["season"] = P.game_id.str[:4].astype(int)
P["pnl"] = np.where(P.under == 1, payout(P.best_under), -1.0)
P["resid"] = (1 - P.under) - P.fair          # over-rate minus fair (negative = unders beat the price)

# ------------------------------------------------------------------ absences: inactive (INA) or listed Out/Doubtful and did not play
g = con.execute("SELECT game_id, season, week, home_team, away_team FROM games WHERE season >= 2023").df()
ro = con.execute("SELECT season, week, team, gsis_id AS player_id, position, status FROM rosters_weekly WHERE season >= 2023 AND position IN ('WR','TE','RB')").df()
inj = con.execute("SELECT CAST(season AS INT) AS season, CAST(week AS INT) AS week, gsis_id AS player_id, report_status FROM injuries WHERE season >= 2023").df().drop_duplicates(["season", "week", "player_id"])
use = pd.read_parquet(ROOT / "data/player_games.parquet", columns=["player_id", "season", "week", "target_share_m3"])
last = use.sort_values(["season", "week"])
ab = ro[ro.status.isin(["INA", "RES"])].merge(inj, on=["season", "week", "player_id"], how="left")
ab = ab.merge(g.melt(id_vars=["game_id", "season", "week"], value_vars=["home_team", "away_team"], value_name="team")[["game_id", "season", "week", "team"]],
              on=["season", "week", "team"])
# prior target share = most recent 3-game average from an EARLIER week
ab = pd.merge_asof(ab.assign(t=ab.season * 100 + ab.week).sort_values("t"),
                   last.assign(t=last.season * 100 + last.week).sort_values("t")[["player_id", "t", "target_share_m3"]],
                   on="t", by="player_id", allow_exact_matches=False)
ab["ts"] = ab.target_share_m3.fillna(0)
ab["kind"] = np.where(ab.report_status.isin(["Out", "Doubtful"]) | (ab.status == "RES"), "known", "surprise")
# only absences of players who were actually part of the offense recently (avoid long-term IR players whose share is stale)
ab = ab[ab.ts > 0]


def run(kind, thr, trig_pos, markets, label):
    a = ab[(ab.ts >= thr) & (ab.kind.isin(kind)) & ab.position.isin(trig_pos)]
    tg = a.groupby(["game_id", "team"]).ts.sum().rename("vac").reset_index()
    d = P[P.market.isin(markets)].merge(tg, on=["game_id", "team"])
    absent_same_game = set(zip(a.game_id, a.player_id))                      # only players absent in THIS game (their props are void anyway)
    d = d[[(gid, pid) not in absent_same_game for gid, pid in zip(d.game_id, d.player_id)]]
    if len(d) < 30:
        return None
    gsum = d.groupby(["game_id", "team"]).agg(pnl=("pnl", "sum"), n=("pnl", "size"), res=("resid", "sum"))
    m = gsum.pnl.sum() / gsum.n.sum(); r_ = gsum.pnl - m * gsum.n; ng = len(gsum)
    se = math.sqrt(ng / (ng - 1) * (r_ ** 2).sum()) / gsum.n.sum()
    out = {"variant": label, "team_games": ng, "bets": len(d), "under_rate": round(d.under.mean(), 3), "fair_under": round(1 - d.fair.mean(), 3),
           "roi": round(m, 4), "se": round(se, 4), "t": round(m / se, 2) if se else None}
    for lab, sd in (("2023-24", d[d.season <= 2024]), ("2025-26", d[d.season >= 2025])):
        out[f"roi_{lab}"] = round(sd.pnl.mean(), 4) if len(sd) else None
        out[f"n_{lab}"] = int(sd.groupby(["game_id", "team"]).ngroups)
    return out, d


REC = ["player_receptions", "player_reception_yds"]
rows = []
base, D0 = run(["surprise"], .10, ["WR", "TE", "RB"], REC, "ORIGINAL: surprise, >=10% share, rec + rec yds")
rows.append(base)
for kind, lab in ((["known"], "known absences (listed Out/Doubtful)"), (["surprise", "known"], "surprise + known")):
    rows.append(run(kind, .10, ["WR", "TE", "RB"], REC, f"{lab}, >=10%")[0])
for thr in (.05, .15, .20):
    for kind, lab in ((["surprise"], "surprise"), (["surprise", "known"], "surprise + known")):
        x = run(kind, thr, ["WR", "TE", "RB"], REC, f"{lab}, >={int(thr*100)}%")
        if x: rows.append(x[0])
for pos in (["WR"], ["TE"], ["RB"]):
    for kind, lab in ((["surprise"], "surprise"), (["surprise", "known"], "surprise + known")):
        x = run(kind, .10, pos, REC, f"{lab}, trigger is {pos[0]}")
        if x: rows.append(x[0])
for mk, lab in ((["player_rush_reception_yds"], "teammates' rush+rec yds"), (["player_reception_longest"], "teammates' longest reception"),
                (["player_pass_yds"], "QB pass yds"), (["player_pass_completions"], "QB completions"), (["player_pass_attempts"], "QB attempts")):
    for kind, klab in ((["surprise"], "surprise"), (["surprise", "known"], "surprise + known")):
        x = run(kind, .10, ["WR", "TE", "RB"], mk, f"{klab}, >=10% -> {lab}")
        if x: rows.append(x[0])
T = pd.DataFrame(rows)
R["variants"] = T.to_dict("records")

# ------------------------------------------------------------------ bootstrap for the original rule (resample team-games)
gs = D0.groupby(["game_id", "team"]).agg(pnl=("pnl", "sum"), n=("pnl", "size"))
rng = np.random.default_rng(11); boots = []
for _ in range(20000):
    s = gs.sample(len(gs), replace=True, random_state=rng.integers(1e9))
    boots.append(s.pnl.sum() / s.n.sum())
boots = np.array(boots)
sd_tg = (gs.pnl / gs.n).std()       # spread of per-team-game ROI
R["original_bootstrap"] = {"roi": round(float(gs.pnl.sum() / gs.n.sum()), 4), "ci90": [round(float(np.percentile(boots, 5)), 4), round(float(np.percentile(boots, 95)), 4)],
                           "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                           "p_roi_gt_0": round(float((boots > 0).mean()), 4), "p_roi_gt_5pct": round(float((boots > .05).mean()), 4),
                           "team_games": len(gs),
                           "team_games_to_confirm_5pct_at_2se": int(math.ceil((2 * sd_tg / .05) ** 2)),
                           "team_games_to_confirm_8pct_at_2se": int(math.ceil((2 * sd_tg / .08) ** 2))}
(ROOT / "results/targets_freed_expand.json").write_text(json.dumps(R, indent=1, default=str))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(json.dumps(R["original_bootstrap"], indent=1))
print(T.to_string(index=False))

# ------------------------------------------------------------------ baseline in the same framework: team-games with NO absence >= 5%
def roi_block(d, label):
    gsum = d.groupby(["game_id", "team"]).agg(pnl=("pnl", "sum"), n=("pnl", "size"))
    m = gsum.pnl.sum() / gsum.n.sum(); r_ = gsum.pnl - m * gsum.n; ng = len(gsum)
    se = math.sqrt(ng / (ng - 1) * (r_ ** 2).sum()) / gsum.n.sum()
    return {"variant": label, "team_games": ng, "bets": len(d), "under_rate": round(d.under.mean(), 3), "fair_under": round(1 - d.fair.mean(), 3),
            "roi": round(m, 4), "se": round(se, 4), "roi_2023-24": round(d[d.season <= 2024].pnl.mean(), 4), "roi_2025-26": round(d[d.season >= 2025].pnl.mean(), 4)}


rec = P[P.market.isin(REC)]
any5 = ab[ab.ts >= .05].groupby(["game_id", "team"]).size().rename("k").reset_index()
flag = rec.merge(any5, on=["game_id", "team"], how="left")
base_rows = [roi_block(rec, "ALL receiving props (rec + rec yds)"),
             roi_block(flag[flag.k.isna()], "team-games with NO absence >= 5% share"),
             roi_block(flag[flag.k.notna()], "team-games WITH an absence >= 5% share")]
# absence effect net of baseline, per season
nb = flag[flag.k.isna()]; wb = flag[flag.k.notna()]
base_rows.append({"variant": "difference (with - without), by season",
                  **{str(s): round(wb[wb.season == s].pnl.mean() - nb[nb.season == s].pnl.mean(), 4) for s in sorted(rec.season.unique())}})
R["baseline_check"] = base_rows
(ROOT / "results/targets_freed_expand.json").write_text(json.dumps(R, indent=1, default=str))
print(pd.DataFrame(base_rows).to_string(index=False))
