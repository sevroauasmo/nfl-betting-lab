"""First/second-half deep dive: do team half-splits (fast starters, halftime adjusters) beat the book's half lines?

Team tendencies are season-to-date and prior-season (shrunk), computed from game_periods before each game.
Graded against consensus closing 1H/2H spreads and totals (periods_close). Also the 1H share of the full-game line.
"""
import json, math
from pathlib import Path
import duckdb, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
gp = con.execute("SELECT * FROM game_periods WHERE season >= 2020").df()
gp["h1_h"], gp["h1_a"] = gp.q1_h + gp.q2_h, gp.q1_a + gp.q2_a
gp["h2_h"], gp["h2_a"] = gp.q3_h + gp.q4_h + gp.ot_h.fillna(0), gp.q3_a + gp.q4_a + gp.ot_a.fillna(0)
gd = con.execute("SELECT game_id, gameday FROM games").df()
gp = gp.merge(gd, on="game_id")
# team-game rows with half margins and half points, then pre-game rolling tendencies
rows = []
for side, opp in (("h", "a"), ("a", "h")):
    t = gp[["game_id", "season", "gameday"]].copy()
    t["team"] = gp.home_team if side == "h" else gp.away_team
    t["h1_margin"] = gp[f"h1_{side}"] - gp[f"h1_{opp}"]; t["h2_margin"] = gp[f"h2_{side}"] - gp[f"h2_{opp}"]
    t["h1_pts"] = gp[f"h1_{side}"] + gp[f"h1_{opp}"]; t["h2_pts"] = gp[f"h2_{side}"] + gp[f"h2_{opp}"]
    t["side"] = side
    rows.append(t)
T = pd.concat(rows).sort_values(["team", "gameday"])
g = T.groupby("team")
for c in ("h1_margin", "h2_margin", "h1_pts", "h2_pts"):
    T[f"{c}_r16"] = g[c].transform(lambda x: x.shift().rolling(16, min_periods=6).mean())   # ~last season of games
T["fast_start"] = T.h1_margin_r16 - T.h2_margin_r16          # better in 1H than 2H
T["h1_share"] = T.h1_pts_r16 / (T.h1_pts_r16 + T.h2_pts_r16)  # share of game points in 1H for this team's games
feat = T[["game_id", "side", "fast_start", "h1_margin_r16", "h2_margin_r16", "h1_share"]]
H = feat[feat.side == "h"].drop(columns="side").add_prefix("home_").rename(columns={"home_game_id": "game_id"}).merge(
    feat[feat.side == "a"].drop(columns="side").add_prefix("away_").rename(columns={"away_game_id": "game_id"}), on="game_id")

# consensus half lines at close (reuse quarters.py logic via the saved model frame is simpler to recompute here)
exec(open(ROOT / "scripts/quarters.py").read().split("# ------------------------------------------------------------------ 1. raw bias")[0])
X = L[(L["pass"] == "periods_close") & L.period.isin(["h1", "h2"])].dropna(subset=["a_win"]).merge(H, on="game_id")
X["resid"] = X.a_win - X.fair_a
out = {}
tests = {
    "spreads_h1": [("home fast-start minus away fast-start", X.home_fast_start - X.away_fast_start),
                   ("home 1H margin form minus away", X.home_h1_margin_r16 - X.away_h1_margin_r16)],
    "spreads_h2": [("home 2H margin form minus away (halftime adjusters)", X.home_h2_margin_r16 - X.away_h2_margin_r16),
                   ("home fast-start minus away (fade in 2H?)", X.home_fast_start - X.away_fast_start)],
    "totals_h1": [("avg 1H share of scoring, both teams", (X.home_h1_share + X.away_h1_share) / 2)],
    "totals_h2": [("avg 1H share of scoring, both teams", (X.home_h1_share + X.away_h1_share) / 2)],
}
for mk, lst in tests.items():
    d0 = X[(X.kind + "_" + X.period) == mk]
    for lab, f in lst:
        f = f.loc[d0.index]; d = d0[f.notna()]; f = f[f.notna()]
        q1, q5 = f.quantile([.25, .75]); lo, hi = d[f <= q1], d[f >= q5]
        diff = hi.resid.mean() - lo.resid.mean(); se = math.sqrt(hi.resid.var() / len(hi) + lo.resid.var() / len(lo))
        corr = float(np.corrcoef(f, d.resid)[0, 1])
        out[f"{mk}: {lab}"] = {"n": len(d), "top_minus_bottom_quartile": round(diff, 4), "z": round(diff / se, 2), "corr": round(corr, 4),
                               "top_q_a_rate_vs_fair": [round(hi.a_win.mean(), 4), round(hi.fair_a.mean(), 4)]}
# how books set 1H lines relative to the full game
fl = L[(L["pass"] == "periods_close") & (L.period == "h1")].copy()
out["book_1H_total_as_share_of_full_total"] = round(float((fl[fl.kind == "totals"].line / fl[fl.kind == "totals"].total_line).mean()), 4)
out["actual_1H_share_of_points_2023_26"] = round(float(((gp.h1_h + gp.h1_a).sum() / (gp.h1_h + gp.h1_a + gp.h2_h + gp.h2_a).sum())), 4)
out["book_1H_spread_as_share_of_full_spread"] = round(float((fl[fl.kind == "spreads"].line / fl[fl.kind == "spreads"].spread_line.replace(0, np.nan)).median()), 4)
(ROOT / "results/halves.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
