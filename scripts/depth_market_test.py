"""Do depth-chart changes predict prop results beyond the market price? Over rate vs de-vigged fair, by depth condition.
2025-26 = true point-in-time charts; 2023-24 = weekly charts (timing unverifiable)."""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
DF = pd.read_parquet(ROOT / "data/depth_features.parquet")
out = {}
for task in ("rec", "rush"):
    M = pd.read_parquet(ROOT / f"data/player_model_{task}_preds.parquet").merge(DF, on=["player_id", "game_id"], how="left", suffixes=("", "_d"))
    M["resid"] = M.over - M.fair_over
    conds = {"promoted (tier up)": M.promoted_d == 1 if "promoted_d" in M else M.promoted == 1,
             "demoted (tier down)": (M.demoted_d if "demoted_d" in M else M.demoted) == 1,
             "starter slot opened at position": M.starter_slots_opened >= 1,
             "starter (tier 1)": M.depth_tier == 1, "backup (tier 2)": M.depth_tier == 2, "not on chart / deep": M.depth_tier >= 3,
             "fine rank improved (2025+)": M.fine_rank_change > 0, "fine rank worse (2025+)": M.fine_rank_change < 0,
             "someone above removed (2025+)": M.above_removed >= 1}
    res = {}
    for era, m_era in (("2025-26 point-in-time", M.season >= 2025), ("2023-24 weekly", M.season <= 2024)):
        for lab, c in conds.items():
            d = M[m_era & c.fillna(False)]
            if len(d) < 30:
                continue
            res[f"{era} | {lab}"] = {"n": len(d), "over": round(d.over.mean(), 4), "fair": round(d.fair_over.mean(), 4),
                                     "resid": round(d.resid.mean(), 4), "t": round(d.resid.mean() / (d.resid.std() / math.sqrt(len(d))), 2)}
    out[task] = res
(ROOT / "results/depth_market_test.json").write_text(json.dumps(out, indent=1))
for t, r in out.items():
    print("#####", t)
    print(pd.DataFrame(r).T.to_string())
