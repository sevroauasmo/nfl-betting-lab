"""Walk-forward prop model with feature selection inside the training window only.

For each market family (receiving / rushing / passing / other) and each test period:
  train = all earlier seasons; L1 logistic regression with the market's logit price as an input, penalty tuned by
  time-ordered CV on the training seasons only; score the test seasons. Nothing from the test period touches selection.
Reports log loss vs the market, and ROI when the model's edge clears thresholds at (a) Novig/ProphetX and (b) best book price.
Also prints the new-feature single effects (net of the market-wide under lean) for 2023-24 vs 2025-26.
Writes results/props_model.json.
"""
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
F = pd.read_parquet(ROOT / "data/props_features_v3.parquet").dropna(subset=["over"]).copy()
F["resid"] = F.over - F.fair_over
F["resid_adj"] = F.resid - F.groupby(["market", "season"]).resid.transform("mean")
F["mgroup"] = F.market.map(lambda m: "receiving" if "recep" in m else "rushing" if "rush" in m else "passing" if "pass" in m else "other")
F["logit_fair"] = np.log(F.fair_over / (1 - F.fair_over))
payout = lambda o: np.where(np.asarray(o, float) < 0, 100 / -np.asarray(o, float), np.asarray(o, float) / 100)  # noqa: E731
NUM = [c for c in F.columns if F[c].dtype.kind in "fi" and c not in {
    "over", "actual", "resid", "resid_adj", "season", "week", "best_over", "best_under", "fair_over", "logit_fair",
    "targets", "target_share", "air_yards_share", "carries", "receptions", "receiving_yards", "rushing_yards",  # same-game values
    "point_open", "point_day_before", "fair_open", "fair_day_before", "exch_over", "su_w_td", "gp_td", "opp_su_w_td", "opp_gp_td",
    "neutral", "playoff", "prev_line", "x_exp", "x_overperf", "proj_rec_yds", "proj_targets", "proj_carries", "line_range", "line_dispersion",
    "rest", "opp_rest", "tprr_man", "tprr_zone", "p_l1", "p_m3", "p_m8", "p_sd8", "n_books",
    "offense_pct", "carry_share", "season_gp_h"}]  # offense_pct / carry_share here are THIS game's values -> leak
# tripwire: a legitimate pre-game feature should barely correlate with the market residual; flag anything that does
leak = {c: round(float(F[[c, "resid"]].dropna().corr().iloc[0, 1]), 3) for c in NUM if F[c].notna().sum() > 1000}
suspects = {c: v for c, v in leak.items() if abs(v) > 0.08}
print("leak tripwire (|corr with residual| > 0.08):", suspects)
NUM = [c for c in NUM if c not in suspects]
R = {"features_considered": NUM}

# ------------------------------------------------------------------ new-feature single effects
NEW = ["x_exp_vs_line", "x_overperf_rel", "adot_m8", "yprr_m8", "routes_m3", "proj_vs_line", "returning_target_share", "own_ol_out",
       "qb_catchable_rate", "team_pa_rate", "opp_blitz_rate", "opp_box_vs_run", "line_wow", "rookie", "age", "altitude", "turf"]
rows = []
for grp, d0 in [("all", F)] + list(F.groupby("mgroup")):
    for f in NEW:
        d = d0.dropna(subset=[f])
        if len(d) < 1000 or d[f].nunique() < 2:
            continue
        if d[f].nunique() <= 2:
            lo, hi = d[d[f] == d[f].min()], d[d[f] == d[f].max()]
        else:
            q1, q5 = d[f].quantile([.2, .8]); lo, hi = d[d[f] <= q1], d[d[f] >= q5]
        if min(len(lo), len(hi)) < 150:
            continue
        diff = hi.resid_adj.mean() - lo.resid_adj.mean()
        se = math.sqrt(lo.resid_adj.var() / len(lo) + hi.resid_adj.var() / len(hi))
        e = lambda a, b: round(float(b.resid_adj.mean() - a.resid_adj.mean()), 4)  # noqa: E731
        rows.append({"group": grp, "feature": f, "high_minus_low": round(float(diff), 4), "z": round(diff / se, 2),
                     "2023-24": e(lo[lo.season <= 2024], hi[hi.season <= 2024]), "2025-26": e(lo[lo.season >= 2025], hi[hi.season >= 2025])})
A = pd.DataFrame(rows).sort_values("z", key=abs, ascending=False)
R["new_feature_effects"] = A.to_dict("records")

# ------------------------------------------------------------------ walk-forward L1 model
preds, fit = [], {}
for test in (2024, 2025):
    tr_all = F[F.season < test]; te_all = F[F.season == test] if test == 2024 else F[F.season >= 2025]
    fit[test] = {}
    for grp in ("receiving", "rushing", "passing", "other"):
        tr, te = tr_all[tr_all.mgroup == grp], te_all[te_all.mgroup == grp].copy()
        if len(tr) < 1000 or len(te) < 200:
            continue
        cols = [c for c in NUM if tr[c].notna().mean() > 0.3 and tr[c].nunique() > 1]
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        Z = lambda d: pd.concat([d[["logit_fair"]], ((d[cols] - mu) / sd).clip(-4, 4).fillna(0)], axis=1)  # noqa: E731
        # CV folds grouped by week so correlated props from the same slate stay together
        folds = GroupKFold(n_splits=4)
        clf = LogisticRegressionCV(Cs=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1], penalty="l1", solver="saga", scoring="neg_log_loss",
                                   cv=list(folds.split(Z(tr), tr.over, groups=tr.season * 100 + tr.week)), max_iter=3000, n_jobs=-1)
        clf.fit(Z(tr), tr.over)
        te["p"] = clf.predict_proba(Z(te))[:, 1]
        coefs = dict(zip(["logit_fair"] + cols, clf.coef_[0]))
        fit[test][grp] = {"n_train": len(tr), "n_test": len(te), "C": float(clf.C_[0]),
                          "ll_market": round(log_loss(te.over, te.fair_over), 5), "ll_model": round(log_loss(te.over, te.p), 5),
                          "selected": {k: round(v, 3) for k, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1])) if abs(v) > 1e-4}}
        preds.append(te)
R["fit"] = fit
PR = pd.concat(preds)
ex = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True).execute(
    """SELECT game_id, market, player_key, bookmaker, point, over_price, under_price FROM prop_results
       WHERE pass='close' AND bookmaker IN ('novig','prophetx') AND over_price IS NOT NULL AND under_price IS NOT NULL""").df()


def sim(df, op, up):
    out = {}
    for th in (0, .02, .04, .06, .08):
        o = df[df.p * (1 + payout(df[op])) - 1 >= th]; u = df[(1 - df.p) * (1 + payout(df[up])) - 1 >= th]
        w = np.r_[o.over, 1 - u.over]; od = np.r_[o[op], u[up]]; ss = np.r_[o.season, u.season]; mg = np.r_[o.mgroup, u.mgroup]
        pr = np.where(w == 1, payout(od), -1.0)
        out[f"EV>={int(th*100)}%"] = {"n": len(w), "unders": len(u), "roi": round(pr.mean(), 4) if len(w) else None,
                                      "se": round(pr.std() / math.sqrt(len(w)), 4) if len(w) > 1 else None,
                                      "by_season": {int(s): [int((ss == s).sum()), round(pr[ss == s].mean(), 4)] for s in np.unique(ss)},
                                      "by_group": {g_: [int((mg == g_).sum()), round(pr[mg == g_].mean(), 4)] for g_ in np.unique(mg)}}
    return out


R["roi_best_book"] = sim(PR, "best_over", "best_under")
X2 = PR.merge(ex, on=["game_id", "market", "player_key"]); X2 = X2[X2.point == X2.cons_point]
R["roi_exchange"] = sim(X2, "over_price", "under_price")
(ROOT / "results/props_model.json").write_text(json.dumps(R, indent=1, default=str))

pd.set_option("display.width", 200)
print(A.head(30).to_string(index=False))
for t, v in fit.items():
    for g_, x in v.items():
        print(t, g_, "C", x["C"], "ll market", x["ll_market"], "model", x["ll_model"], "| top:", list(x["selected"].items())[:8])
for lab in ("roi_exchange", "roi_best_book"):
    print(lab)
    for k, v in R[lab].items():
        print("  ", k, v["n"], v["roi"], v["se"], v["by_season"], v["by_group"])
