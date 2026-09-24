"""Yardage models for WR/TE receiving yards and RB rushing yards, compared with the prop market, plus SHAP.

1. LightGBM quantile models (10th..90th pct) + a mean model, walk-forward by season (train on all earlier seasons,
   which includes every player-game since 2021, not just games with props). Gives a full distribution -> P(over line).
2. Against the market on prop games (closing consensus line + de-vigged fair prob): median accuracy vs the line, Brier,
   and ROI when the model's edge clears thresholds (best book price and Novig/ProphetX).
3. A "market residual" model: target = actual - line. Does anything predict where the line is wrong, out of sample?
4. SHAP on both, rolled up by feature family.
5. Player persistence: is a player's market error in one season related to the next?
Writes results/player_model.json.
"""
import json
import math
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
import sys
VARIANT = sys.argv[1] if len(sys.argv) > 1 else ""          # "depth" -> add depth-chart features
D = pd.read_parquet(ROOT / "data/player_games.parquet")
FAM = json.loads((ROOT / "data/player_feature_families.json").read_text())
if VARIANT == "depth":
    DF = pd.read_parquet(ROOT / "data/depth_features.parquet")
    D = D.merge(DF, on=["player_id", "game_id"], how="left")
    FAM.update({c: "depth chart" for c in DF.columns if c not in ("player_id", "game_id")})
FEATS = list(FAM)
QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
payout = lambda o: np.where(np.asarray(o, float) < 0, 100 / -np.asarray(o, float), np.asarray(o, float) / 100)  # noqa: E731
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
TASKS = {"rec": {"target": "receiving_yards", "pos": ["WR", "TE"], "market": "player_reception_yds"},
         "rush": {"target": "rushing_yards", "pos": ["RB"], "market": "player_rush_yds"}}
PARAMS = dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=60, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, verbose=-1, seed=7)
R: dict = {}

# market lines (close consensus + fair prob + best prices) and exchange prices
PF = pd.read_parquet(ROOT / "data/props_features.parquet", columns=["game_id", "player_id", "market", "cons_point", "fair_over", "best_over", "best_under", "player_key"])
EX = con.execute("""SELECT game_id, market, player_key, point, max(over_price) ex_over, max(under_price) ex_under FROM prop_results
                    WHERE pass='close' AND bookmaker IN ('novig','prophetx') AND over_price IS NOT NULL AND under_price IS NOT NULL GROUP BY ALL""").df()


def p_over(qpred, line):
    """P(Y > line) from predicted quantiles by linear interpolation of the CDF; tails padded."""
    q = np.sort(qpred, axis=1)
    lo = np.minimum(q[:, :1] - (q[:, 1:2] - q[:, :1]) * 1.5, q[:, :1] - 1)  # ~0th pct
    hi = q[:, -1:] + (q[:, -1:] - q[:, -2:-1]) * 3 + 1                      # ~100th pct
    xs = np.hstack([lo, q, hi]); ps = np.array([0.0] + QS + [1.0])
    return np.array([1 - np.interp(l_, x, ps) for x, l_ in zip(xs, line)])


def sim(df, p, op, up):
    out = {}
    for th in (0, .03, .06, .09):
        eo = p * (1 + payout(df[op])) - 1; eu = (1 - p) * (1 + payout(df[up])) - 1
        o, u = df[eo >= th], df[eu >= th]
        w = np.r_[o.over, 1 - u.over]; od = np.r_[o[op], u[up]]; ss = np.r_[o.season, u.season]
        pr = np.where(w == 1, payout(od), -1.0)
        out[f"EV>={int(th*100)}%"] = {"n": len(pr), "unders": len(u), "roi": round(pr.mean(), 4) if len(pr) else None,
                                      "se": round(pr.std() / math.sqrt(len(pr)), 4) if len(pr) > 1 else None,
                                      "by_season": {int(s): [int((ss == s).sum()), round(pr[ss == s].mean(), 4)] for s in np.unique(ss)}}
    return out


for task, cfg in TASKS.items():
    d = D[D.position.isin(cfg["pos"])].copy()
    y = cfg["target"]
    d = d[d[f"{y}_m8"].notna() | (d.games_prior >= 1)]
    preds = []
    shap_rows = []
    for test in (2023, 2024, 2025, 2026):
        tr, te = d[d.season < test], d[d.season == test].copy()
        val = tr[tr.season == tr.season.max()]; trn = tr[tr.season < tr.season.max()]
        # mean model with early stopping on the last training season, then refit on all training seasons
        dm = lgb.train({**PARAMS, "objective": "regression"}, lgb.Dataset(trn[FEATS], trn[y]), 3000,
                       valid_sets=[lgb.Dataset(val[FEATS], val[y])], callbacks=[lgb.early_stopping(100, verbose=False)])
        n_iter = max(100, dm.best_iteration)
        mean_m = lgb.train({**PARAMS, "objective": "regression"}, lgb.Dataset(tr[FEATS], tr[y]), n_iter)
        te["pred_mean"] = mean_m.predict(te[FEATS])
        qp = np.column_stack([lgb.train({**PARAMS, "objective": "quantile", "alpha": a}, lgb.Dataset(tr[FEATS], tr[y]), n_iter).predict(te[FEATS]) for a in QS])
        for i, a in enumerate(QS):
            te[f"q{int(a*100)}"] = qp[:, i]
        te["test_season"] = test
        preds.append(te)
        if test in (2025,):
            ex = shap.TreeExplainer(mean_m)
            sv = ex.shap_values(te[FEATS])
            shap_rows.append(pd.DataFrame(np.abs(sv), columns=FEATS).mean())
    P = pd.concat(preds)
    qcols = [f"q{int(a*100)}" for a in QS]
    # pure forecasting accuracy on all test games
    R[f"{task}_forecast"] = {int(s): {"n": len(x), "mae_mean": round(float((x.pred_mean - x[y]).abs().mean()), 2),
                                      "mae_median": round(float((x.q50 - x[y]).abs().mean()), 2),
                                      "mae_naive_m8": round(float((x[f"{y}_m8"] - x[y]).abs().mean()), 2),
                                      "coverage_10_90": round(float(((x[y] >= x.q10) & (x[y] <= x.q90)).mean()), 3)}
                             for s, x in P.groupby("test_season")}
    # against the market
    M = P.merge(PF[PF.market == cfg["market"]], on=["game_id", "player_id"], how="inner")
    M = M[M.cons_point.notna()].copy()
    M["over"] = np.where(M[y] > M.cons_point, 1.0, np.where(M[y] < M.cons_point, 0.0, np.nan))
    M = M.dropna(subset=["over"])
    M["p_model"] = np.clip(p_over(M[qcols].values, M.cons_point.values), 0.02, 0.98)
    M["p_blend"] = 1 / (1 + np.exp(-(0.5 * np.log(M.fair_over / (1 - M.fair_over)) + 0.5 * np.log(M.p_model / (1 - M.p_model)))))
    R[f"{task}_vs_market"] = {int(s): {"n": len(x),
                                       "mae_line": round(float((x.cons_point - x[y]).abs().mean()), 2),
                                       "mae_model_median": round(float((x.q50 - x[y]).abs().mean()), 2),
                                       "model_closer_pct": round(float(((x.q50 - x[y]).abs() < (x.cons_point - x[y]).abs()).mean()), 4),
                                       "brier_market": round(float(((x.fair_over - x.over) ** 2).mean()), 5),
                                       "brier_model": round(float(((x.p_model - x.over) ** 2).mean()), 5),
                                       "brier_blend": round(float(((x.p_blend - x.over) ** 2).mean()), 5),
                                       "model_median_minus_line": round(float((x.q50 - x.cons_point).mean()), 2)}
                              for s, x in M.groupby("season")}
    R[f"{task}_roi_best_book_model"] = sim(M, M.p_model.values, "best_over", "best_under")
    R[f"{task}_roi_best_book_blend"] = sim(M, M.p_blend.values, "best_over", "best_under")
    MX = M.merge(EX[EX.market == cfg["market"]], on=["game_id", "market", "player_key"])
    MX = MX[MX.point == MX.cons_point]
    R[f"{task}_roi_exchange_model"] = sim(MX, MX.p_model.values, "ex_over", "ex_under")
    R[f"{task}_roi_exchange_blend"] = sim(MX, MX.p_blend.values, "ex_over", "ex_under")
    # SHAP of the yardage model, by family
    sv = shap_rows[0]
    R[f"{task}_shap_yards_family"] = sv.groupby(pd.Series(FAM)).sum().sort_values(ascending=False).round(3).to_dict()
    R[f"{task}_shap_yards_top"] = sv.sort_values(ascending=False).head(20).round(3).to_dict()

    # ---------------------------------------------------------------- market-residual model: what predicts where the line is wrong?
    M["mresid"] = M[y] - M.cons_point
    RF = FEATS + ["cons_point", "fair_over"]
    res_out, shap_res = {}, None
    for test in (2025,):
        tr, te = M[M.season < test], M[M.season >= test].copy()
        rm = lgb.train({**PARAMS, "objective": "huber", "alpha": 20.0, "num_leaves": 15, "min_data_in_leaf": 150},
                       lgb.Dataset(tr[RF], tr.mresid), 300)
        te["resid_pred"] = rm.predict(te[RF])
        c = float(np.corrcoef(te.resid_pred, te.mresid)[0, 1])
        # does the predicted residual sign predict over/under beyond the price?
        top = te[te.resid_pred >= te.resid_pred.quantile(0.8)]; bot = te[te.resid_pred <= te.resid_pred.quantile(0.2)]
        res_out = {"n_train": len(tr), "n_test": len(te), "corr_pred_vs_actual_resid": round(c, 4),
                   "top20_over_rate_vs_fair": [round(top.over.mean(), 4), round(top.fair_over.mean(), 4)],
                   "bottom20_over_rate_vs_fair": [round(bot.over.mean(), 4), round(bot.fair_over.mean(), 4)]}
        shap_res = pd.DataFrame(np.abs(shap.TreeExplainer(rm).shap_values(te[RF])), columns=RF).mean()
    fam_map = {**FAM, "cons_point": "market", "fair_over": "market"}
    R[f"{task}_residual_model"] = res_out
    R[f"{task}_shap_residual_family"] = shap_res.groupby(pd.Series(fam_map)).sum().sort_values(ascending=False).round(3).to_dict()
    R[f"{task}_shap_residual_top"] = shap_res.sort_values(ascending=False).head(15).round(3).to_dict()

    # ---------------------------------------------------------------- player persistence of market error
    M["err_rel"] = (M[y] - M.cons_point) / M.cons_point.clip(lower=5)
    pp_ = M.groupby(["player_id", "player", "season"]).agg(n=("err_rel", "size"), err=("err_rel", "mean"), over=("over", "mean"),
                                                           fair=("fair_over", "mean")).reset_index()
    pp_ = pp_[pp_.n >= 6]
    pairs = pp_.merge(pp_, on=["player_id", "player"], suffixes=("_a", "_b"))
    pairs = pairs[pairs.season_b == pairs.season_a + 1]
    R[f"{task}_player_persistence"] = {"player_season_pairs": len(pairs),
                                       "corr_err_next_season": round(float(pairs[["err_a", "err_b"]].corr().iloc[0, 1]), 4),
                                       "corr_over_minus_fair_next_season": round(float(np.corrcoef(pairs.over_a - pairs.fair_a, pairs.over_b - pairs.fair_b)[0, 1]), 4)}
    # model-level per-player: in which players did the model beat the line in BOTH of two consecutive seasons?
    M["model_better"] = ((M.q50 - M[y]).abs() < (M.cons_point - M[y]).abs()).astype(float)
    mb = M.groupby(["player", "season"]).agg(n=("model_better", "size"), better=("model_better", "mean")).reset_index()
    mb = mb[mb.n >= 8]
    mp = mb.merge(mb, on="player", suffixes=("_a", "_b")); mp = mp[mp.season_b == mp.season_a + 1]
    R[f"{task}_player_model_edge_persistence"] = {"pairs": len(mp), "corr": round(float(mp[["better_a", "better_b"]].corr().iloc[0, 1]), 4)}
    M.to_parquet(ROOT / f"data/player_model_{task}{'_' + VARIANT if VARIANT else ''}_preds.parquet")

(ROOT / f"results/player_model{'_' + VARIANT if VARIANT else ''}.json").write_text(json.dumps(R, indent=1, default=str))
for task in TASKS:
    print(f"\n######## {task}")
    print("forecast:", json.dumps(R[f"{task}_forecast"]))
    print("vs market:"); [print("  ", k, v) for k, v in R[f"{task}_vs_market"].items()]
    for lab in ("roi_exchange_model", "roi_exchange_blend", "roi_best_book_model", "roi_best_book_blend"):
        print(lab); [print("   ", k, v["n"], v["roi"], v["se"], v["by_season"]) for k, v in R[f"{task}_{lab}"].items()]
    print("SHAP yards by family:", R[f"{task}_shap_yards_family"])
    print("SHAP yards top:", list(R[f"{task}_shap_yards_top"].items())[:12])
    print("residual model:", R[f"{task}_residual_model"])
    print("SHAP residual by family:", R[f"{task}_shap_residual_family"])
    print("SHAP residual top:", list(R[f"{task}_shap_residual_top"].items())[:10])
    print("player persistence:", R[f"{task}_player_persistence"], R[f"{task}_player_model_edge_persistence"])
