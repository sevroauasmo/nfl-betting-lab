"""Deep prop study on data/props_features.parquet.

Everything is measured relative to the market: residual = over (1/0) - fair_over (de-vigged consensus at close).
A feature only matters if it moves the residual, i.e. the market under-reacts to it.
  1. single features: quintiles within market group, discovery 2023-24 vs holdout 2025-26
  2. pairs of feature extremes (top/bottom quintile x top/bottom quintile), BH-corrected, holdout-checked
  3. gradient boosting with the market price as an input: does it beat the market out of sample (log loss, ROI)?
Writes results/props_deep.json.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
F = pd.read_parquet(ROOT / "data/props_features.parquet")
F = F.dropna(subset=["over"]).copy()
F["resid"] = F.over - F.fair_over
F["period"] = np.where(F.season <= 2024, "disc", "hold")
GROUP = {"player_receptions": "receiving", "player_reception_yds": "receiving", "player_rush_yds": "rushing",
         "player_rush_attempts": "rushing", "player_rush_reception_yds": "rushing", "player_pass_yds": "passing",
         "player_pass_completions": "passing", "player_pass_attempts": "passing", "player_pass_tds": "passing",
         "player_pass_interceptions": "passing", "player_tackles_assists": "defense", "player_kicking_points": "kicking"}
F["group"] = F.market.map(GROUP)
FEATURES = ["fair_over", "cons_point", "n_books", "line_move_rel", "exch_gap", "l1_vs_line", "m3_vs_line", "m8_vs_line",
            "hit_rate_l8", "hot", "cv8", "season_gp", "target_share_m3", "air_yards_share_m3", "offense_pct_m3", "carry_share_m3",
            "prev_over", "prev_margin_rel", "over_rate_l5", "props_seen", "home", "team_line", "total_line", "implied_tt",
            "win_pct", "opp_win_pct", "opp_allow_rel", "opp_pass_epa_allowed", "opp_rush_epa_allowed", "opp_ypt_allowed",
            "questionable", "limited_or_dnp", "vacated_target_share", "vacated_carry_share", "dome", "wind_mph", "temp_f",
            "div_game", "primetime", "week"]
BINARY = {"home", "questionable", "limited_or_dnp", "dome", "div_game", "primetime", "prev_over"}
R: dict = {"n": len(F), "by_group": F.groupby("group").size().to_dict()}


def payout(o):
    o = np.asarray(o, float)
    return np.where(o < 0, 100 / -o, o / 100)


def side_roi(d, side):
    win = d.over.values if side == "over" else 1 - d.over.values
    odds = d.best_over.values if side == "over" else d.best_under.values
    return float(np.mean(np.where(win == 1, payout(odds), -1.0))) if len(d) else np.nan


def summarize(d):
    n = len(d)
    if n == 0:
        return None
    r = d.resid.values
    return {"n": n, "over": round(float(d.over.mean()), 4), "fair": round(float(d.fair_over.mean()), 4),
            "resid": round(float(r.mean()), 4), "t": round(float(r.mean() / (r.std() / math.sqrt(n))), 2) if n > 1 and r.std() > 0 else 0,
            "roi_over": round(side_roi(d, "over"), 4), "roi_under": round(side_roi(d, "under"), 4)}


def buckets(d, f):
    x = d[f]
    if f in BINARY or x.nunique() <= 2:
        return x.map(lambda v: f"{f}={int(v)}" if pd.notna(v) else None)
    try:
        q = pd.qcut(x, 5, labels=[f"{f} Q1 (low)", f"{f} Q2", f"{f} Q3", f"{f} Q4", f"{f} Q5 (high)"], duplicates="drop")
    except ValueError:
        return pd.Series([None] * len(d), index=d.index)
    return q.astype(object)


# ------------------------------------------------------------------ 1. single features
rows = []
for grp in ["all", "receiving", "rushing", "passing", "defense"]:
    d0 = F if grp == "all" else F[F.group == grp]
    for f in FEATURES:
        if d0[f].notna().sum() < 500:
            continue
        b = buckets(d0, f)
        for lab, d in d0.groupby(b):
            if len(d) < 150:
                continue
            a, dd, hh = summarize(d), summarize(d[d.period == "disc"]), summarize(d[d.period == "hold"])
            if not dd or not hh:
                continue
            rows.append({"group": grp, "feature": f, "bucket": lab, **a, "resid_disc": dd["resid"], "resid_hold": hh["resid"],
                         "n_hold": hh["n"], "roi_under_hold": hh["roi_under"], "roi_over_hold": hh["roi_over"]})
S = pd.DataFrame(rows)
S["p"] = [2 * (1 - stats.norm.cdf(abs(t))) for t in S.t]
S["q"] = multipletests(S.p, method="fdr_bh")[1]
S["consistent"] = (np.sign(S.resid_disc) == np.sign(S.resid_hold)) & (S.resid_disc.abs() >= .015) & (S.resid_hold.abs() >= .015)
S.to_csv(ROOT / "results/props_single_features.csv", index=False)
R["single_tests"] = len(S)
R["single_consistent_q10"] = S[S.consistent & (S.q < .10)].sort_values("p").head(40).to_dict("records")
R["single_top_any"] = S.sort_values("p").head(25).to_dict("records")

# ------------------------------------------------------------------ 2. pairs of extremes
pair_rows = []
for grp in ["all", "receiving", "rushing", "passing"]:
    d0 = F if grp == "all" else F[F.group == grp]
    ext = {}
    for f in FEATURES:
        if f in ("fair_over", "week") or d0[f].notna().sum() < 1000:
            continue
        if f in BINARY:
            ext[f"{f}=1"] = d0[f] == 1
        else:
            lo, hi = d0[f].quantile([.2, .8])
            if lo == hi:
                continue
            ext[f"{f} low"] = d0[f] <= lo
            ext[f"{f} high"] = d0[f] >= hi
    names = list(ext)
    for a, b in itertools.combinations(names, 2):
        if a.split(" ")[0].split("=")[0] == b.split(" ")[0].split("=")[0]:
            continue
        m = ext[a] & ext[b]
        if m.sum() < 200:
            continue
        d = d0[m]
        dd, hh = d[d.period == "disc"], d[d.period == "hold"]
        if len(dd) < 80 or len(hh) < 80:
            continue
        r = d.resid.values
        pair_rows.append({"group": grp, "setup": f"{a} + {b}", "n": len(d), "resid": round(float(r.mean()), 4),
                          "t": float(r.mean() / (r.std() / math.sqrt(len(r)))), "resid_disc": round(float(dd.resid.mean()), 4),
                          "resid_hold": round(float(hh.resid.mean()), 4), "n_hold": len(hh),
                          "roi_under_disc": round(side_roi(dd, "under"), 4), "roi_under_hold": round(side_roi(hh, "under"), 4),
                          "roi_over_disc": round(side_roi(dd, "over"), 4), "roi_over_hold": round(side_roi(hh, "over"), 4)})
Pp = pd.DataFrame(pair_rows)
Pp["p"] = [2 * (1 - stats.norm.cdf(abs(t))) for t in Pp.t]
Pp["q"] = multipletests(Pp.p, method="fdr_bh")[1]
# walk-forward honesty: rank by discovery residual only, then look at holdout
Pp["disc_abs"] = Pp.resid_disc.abs()
top = Pp.sort_values("disc_abs", ascending=False).head(100)
R["pairs_tested"] = len(Pp)
R["pairs_walkforward_top100"] = {"held_sign": int((np.sign(top.resid_disc) == np.sign(top.resid_hold)).sum()),
                                 "avg_disc_resid_abs": round(float(top.disc_abs.mean()), 4),
                                 "avg_hold_resid_same_dir": round(float((np.sign(top.resid_disc) * top.resid_hold).mean()), 4)}
Pp["best_hold_roi"] = np.where(Pp.resid_disc < 0, Pp.roi_under_hold, Pp.roi_over_hold)
Pp["best_disc_roi"] = np.where(Pp.resid_disc < 0, Pp.roi_under_disc, Pp.roi_over_disc)
R["pairs_both_periods_profitable"] = (Pp[(Pp.best_disc_roi > .03) & (Pp.best_hold_roi > .03) & (np.sign(Pp.resid_disc) == np.sign(Pp.resid_hold))]
                                      .sort_values("p").head(30).drop(columns=["disc_abs"]).to_dict("records"))
R["pairs_q_lt_10"] = int((Pp.q < .10).sum())
Pp.to_csv(ROOT / "results/props_pairs.csv", index=False)

# ------------------------------------------------------------------ 3. model vs market, walk-forward by season
X_cols = [f for f in FEATURES] + ["market_code"]
F["market_code"] = F.market.astype("category").cat.codes
F["logit_fair"] = np.log(F.fair_over / (1 - F.fair_over))
X_cols = X_cols + ["logit_fair"]
res = {}
preds = []
for test_season in (2024, 2025):
    tr = F[F.season < test_season]; te = F[F.season == test_season] if test_season < 2025 else F[F.season >= 2025]
    cols = [c for c in X_cols if tr[c].notna().sum() >= 200 and tr[c].nunique() > 1]  # e.g. no exchange prices before 2024
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=200,
                                         l2_regularization=1.0, categorical_features=[cols.index("market_code")], random_state=0)
    clf.fit(tr[cols], tr.over)
    p = np.clip(clf.predict_proba(te[cols])[:, 1], 1e-4, 1 - 1e-4)
    te = te.assign(p_model=p)
    preds.append(te)
    base = np.clip(te.fair_over.values, 1e-4, 1 - 1e-4)
    res[f"test {test_season}{'-26' if test_season == 2025 else ''}"] = {
        "n": len(te), "logloss_market": round(log_loss(te.over, base), 5), "logloss_model": round(log_loss(te.over, p), 5),
        "brier_market": round(float(np.mean((base - te.over) ** 2)), 5), "brier_model": round(float(np.mean((p - te.over) ** 2)), 5)}
PR = pd.concat(preds)
# bet when model's edge at the best available price clears a threshold
PR["ev_over"] = PR.p_model * (1 + payout(PR.best_over)) - 1
PR["ev_under"] = (1 - PR.p_model) * (1 + payout(PR.best_under)) - 1
bets = {}
for th in (0.0, 0.02, 0.04, 0.06, 0.08):
    o = PR[PR.ev_over >= th]; u = PR[PR.ev_under >= th]
    w = np.concatenate([o.over.values, 1 - u.over.values]); od = np.concatenate([o.best_over.values, u.best_under.values])
    ss = np.concatenate([o.season.values, u.season.values])
    prof = np.where(w == 1, payout(od), -1.0)
    bets[f"EV>={int(th*100)}%"] = {"n": len(w), "overs": len(o), "unders": len(u), "roi": round(float(prof.mean()), 4) if len(w) else None,
                                   "se": round(float(prof.std() / math.sqrt(len(w))), 4) if len(w) > 1 else None,
                                   "by_season": {int(s): round(float(prof[ss == s].mean()), 4) for s in np.unique(ss)}}
R["model"] = {"fit": res, "bets_at_best_price": bets}
# which features does the model lean on? permutation-free proxy: residual correlation of each feature in the holdout
R["holdout_feature_resid_corr"] = {f: round(float(PR[[f, "resid"]].dropna().corr().iloc[0, 1]), 4) for f in FEATURES if PR[f].notna().sum() > 500}

(ROOT / "results/props_deep.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps({k: R[k] for k in ("n", "by_group", "single_tests", "pairs_tested", "pairs_q_lt_10", "pairs_walkforward_top100", "model")}, indent=1, default=str))
