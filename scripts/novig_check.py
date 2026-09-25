"""Novig-only view: surprise-inactive receiver unders at Novig prices, Novig coverage, baseline, and Novig TD markets."""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
exec(open(ROOT / "scripts/news_window.py").read().split("# ------------------------------------------------------------------ 1. how fast")[0])
pay = lambda o: np.where(np.asarray(o, float) < 0, 100 / -np.asarray(o, float), np.asarray(o, float) / 100)  # noqa: E731
W["season"] = W.game_id.str[:4].astype(int)
T = W[(W.news == "targets freed") & (W.actual != W.line_t10)].copy()
nv = con.execute("""SELECT game_id, market, player_key, point, max(under_price) nv_under FROM prop_results
                    WHERE pass='close' AND bookmaker='novig' AND under_price IS NOT NULL GROUP BY ALL""").df()
X = T.merge(nv.rename(columns={"point": "line_t10"}), on=["game_id", "market", "player_key", "line_t10"], how="left")
X["win"] = (X.actual < X.line_t10).astype(float)


def clu(d, col):
    g = d.groupby(["game_id", "team"])[col].agg(["sum", "count"]); m = g["sum"].sum() / g["count"].sum(); r = g["sum"] - m * g["count"]; n = len(g)
    return round(m, 4), round(math.sqrt(n / (n - 1) * (r ** 2).sum()) / g["count"].sum(), 4) if n > 1 else None, n


out = {}
for s, d in [("2024", X[X.season == 2024]), ("2025", X[X.season == 2025]), ("2026", X[X.season == 2026]), ("2024-26", X[X.season >= 2024])]:
    have = d[d.nv_under.notna()].copy(); have["pnl"] = np.where(have.win == 1, pay(have.nv_under), -1.0)
    m, se, n = clu(have, "pnl") if len(have) > 2 else (None, None, 0)
    out[s] = {"qualifying_props": len(d), "novig_same_line": len(have), "coverage": round(len(have) / max(1, len(d)), 3),
              "under_win": round(have.win.mean(), 3) if len(have) else None, "median_price": int(have.nv_under.median()) if len(have) else None,
              "roi": m, "se": se, "team_games": n}
B = con.execute("""WITH c AS (SELECT game_id, market, player_key, point, under_price, actual FROM prop_results
                   WHERE pass='close' AND bookmaker='novig' AND market IN ('player_receptions','player_reception_yds') AND under_price IS NOT NULL AND actual IS NOT NULL)
                   SELECT left(game_id,4) season, count(*) n, avg((actual<point)::INT) FILTER (WHERE actual<>point) under_win,
                          avg(CASE WHEN actual<point THEN (CASE WHEN under_price<0 THEN 100.0/-under_price ELSE under_price/100.0 END) WHEN actual>point THEN -1 ELSE 0 END) roi
                   FROM c GROUP BY 1 ORDER BY 1""").df()
td = con.execute("""SELECT left(game_id,4) season, market, count(*) n, count(under_price) with_no_price, avg((actual>=1)::INT) scored,
                    median(over_price) med_yes, median(under_price) med_no,
                    avg(CASE WHEN under_price IS NULL THEN NULL WHEN actual<1 THEN (CASE WHEN under_price<0 THEN 100.0/-under_price ELSE under_price/100.0 END) ELSE -1 END) roi_no,
                    avg(CASE WHEN actual>=1 THEN (CASE WHEN over_price<0 THEN 100.0/-over_price ELSE over_price/100.0 END) ELSE -1 END) roi_yes
                    FROM prop_results WHERE pass='close' AND bookmaker='novig' AND market IN ('player_anytime_td','player_tds_over') AND actual IS NOT NULL
                    GROUP BY 1,2 ORDER BY 1,2""").df()
mk = con.execute("""SELECT left(game_id,4) season, count(DISTINCT game_id) games, count(DISTINCT market) markets FROM prop_results
                    WHERE pass='close' AND bookmaker='novig' GROUP BY 1 ORDER BY 1""").df()
res = {"strategy_at_novig": out, "baseline_all_novig_receiving_unders": B.to_dict("records"), "novig_td": td.to_dict("records"), "novig_coverage": mk.to_dict("records")}
(ROOT / "results/novig_check.json").write_text(json.dumps(res, indent=1, default=str))
print("SURPRISE-INACTIVE UNDERS AT NOVIG ONLY"); [print(" ", k, v) for k, v in out.items()]
print("BASELINE: every Novig receiving under"); print(B.to_string(index=False))
print("NOVIG TD MARKETS"); print(td.to_string(index=False))
print("NOVIG PROP COVERAGE"); print(mk.to_string(index=False))
