"""Lean receiving-prop model (receptions + receiving yards) on the strongest, most explainable features.
Walk-forward (train on prior seasons only), evaluated at Novig/ProphetX prices and at the best book price."""
import json, math
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parent.parent
F = pd.read_parquet(ROOT / "data/props_features_v2.parquet").dropna(subset=["over"])
F = F[F.market.isin(["player_receptions", "player_reception_yds"])].copy()
F["resid"] = F.over - F.fair_over
payout = lambda o: np.where(np.asarray(o, float) < 0, 100 / -np.asarray(o, float), np.asarray(o, float) / 100)
R = {}
# robustness of the separation effect: by market x season, top vs bottom quintile of prior separation
d = F.dropna(subset=["ngs_avg_separation"])
q1, q5 = d.ngs_avg_separation.quantile([.2, .8])
R["separation_by_market_season"] = {f"{m} {s}": {"n_hi": int(((g.ngs_avg_separation >= q5)).sum()),
                                                "resid_hi": round(g[g.ngs_avg_separation >= q5].resid.mean(), 4),
                                                "resid_lo": round(g[g.ngs_avg_separation <= q1].resid.mean(), 4)}
                                    for (m, s), g in d.groupby(["market", "season"])}
X = ["home", "team_line", "implied_tt", "over_rate_l5", "cv8", "m8_vs_line", "tprr_m8", "route_share_m3", "cov_matchup",
     "opp_db_out", "opp_ypt_allowed", "team_pace", "wind_mph", "ngs_avg_separation", "backup_qb"]
F["logit_fair"] = np.log(F.fair_over / (1 - F.fair_over)); F["is_yds"] = (F.market == "player_reception_yds").astype(float)
preds, fit = [], {}
for test in (2024, 2025):
    tr = F[F.season < test]; te = (F[F.season == test] if test == 2024 else F[F.season >= 2025]).copy()
    mu, sd = tr[X].mean(), tr[X].std()
    Z = lambda x: pd.concat([x[["logit_fair", "is_yds"]], ((x[X] - mu) / sd).fillna(0)], axis=1)
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(Z(tr), tr.over)
    te["p"] = clf.predict_proba(Z(te))[:, 1]
    fit[test] = {"n": len(te), "ll_market": round(log_loss(te.over, te.fair_over), 5), "ll_model": round(log_loss(te.over, te.p), 5),
                 "coefs": dict(zip(["logit_fair", "is_yds"] + X, np.round(clf.coef_[0], 3)))}
    preds.append(te)
R["fit"] = fit
PR = pd.concat(preds)
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
ex = con.execute("""SELECT game_id, market, player_key, bookmaker, point, over_price, under_price FROM prop_results
                    WHERE pass='close' AND bookmaker IN ('novig','prophetx') AND over_price IS NOT NULL AND under_price IS NOT NULL""").df()


def sim(df, op, up, label):
    out = {}
    for th in (0, .02, .04, .06, .08):
        o = df[df.p * (1 + payout(df[op])) - 1 >= th]; u = df[(1 - df.p) * (1 + payout(df[up])) - 1 >= th]
        w = np.r_[o.over, 1 - u.over]; od = np.r_[o[op], u[up]]; ss = np.r_[o.season, u.season]
        pr = np.where(w == 1, payout(od), -1.0)
        out[f"EV>={int(th*100)}%"] = {"n": len(w), "unders": len(u), "roi": round(pr.mean(), 4) if len(w) else None,
                                      "se": round(pr.std() / math.sqrt(len(w)), 4) if len(w) > 1 else None,
                                      "by_season": {int(s): (int((ss == s).sum()), round(pr[ss == s].mean(), 4)) for s in np.unique(ss)}}
    R[label] = out


sim(PR, "best_over", "best_under", "best_book_price")
X2 = PR.merge(ex, on=["game_id", "market", "player_key"]); X2 = X2[X2.point == X2.cons_point]
sim(X2, "over_price", "under_price", "exchange_price")
(ROOT / "results/props_receiving.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps(R, indent=1, default=str))
