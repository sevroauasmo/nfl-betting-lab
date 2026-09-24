"""Follow-up to props_deep: remove the baseline under-lean, then (a) re-rank features, (b) fit a small regularized
logistic model on top of the market price, walk-forward, and simulate betting at the best available price."""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parent.parent
import sys
V2 = "v2" in sys.argv[1:]
F = pd.read_parquet(ROOT / ("data/props_features_v2.parquet" if V2 else "data/props_features.parquet")).dropna(subset=["over"]).copy()
F["resid"] = F.over - F.fair_over
F["resid_adj"] = F.resid - F.groupby(["market", "season"]).resid.transform("mean")   # net of the market-wide under lean
payout = lambda o: np.where(np.asarray(o, float) < 0, 100 / -np.asarray(o, float), np.asarray(o, float) / 100)
R = {"baseline_resid_by_season": F.groupby("season").resid.mean().round(4).to_dict()}

# (a) adjusted single-feature effects: bottom vs top quintile difference in adjusted residual, both periods
rows = []
FEATS = ["over_rate_l5", "hit_rate_l8", "prev_margin_rel", "m3_vs_line", "m8_vs_line", "l1_vs_line", "hot", "cv8", "home", "team_line",
         "implied_tt", "total_line", "wind_mph", "opp_allow_rel", "opp_pass_epa_allowed", "opp_rush_epa_allowed", "opp_ypt_allowed",
         "target_share_m3", "offense_pct_m3", "carry_share_m3", "vacated_target_share", "vacated_carry_share", "questionable",
         "line_move_rel", "exch_gap", "n_books", "win_pct", "season_gp", "primetime", "dome", "cons_point"]
NEW = ["route_share_m3", "tprr_m8", "cov_matchup", "man_zone_skew", "opp_man_rate", "opp_pressure_rate", "exp_plays", "team_pace",
       "team_proe", "opp_proe", "opp_db_out", "opp_front_out", "backup_qb", "total_move", "line_dispersion_rel", "fc_wind", "fc_precip",
       "rest_diff", "ngs_avg_separation", "ngs_avg_cushion", "ngs_avg_yac_above_expectation", "ngs_ryoe", "ngs_avg_time_to_throw",
       "ngs_aggressiveness"]
if V2:
    FEATS = FEATS + NEW
for grp, d0 in [("all", F)] + list(F.groupby(F.market.map(lambda m: "receiving" if "recep" in m else "rushing" if "rush" in m else "passing" if "pass" in m else "other"))):
    for f in FEATS:
        d = d0.dropna(subset=[f])
        if len(d) < 1000 or d[f].nunique() < 2:
            continue
        if d[f].nunique() <= 2:
            lo, hi = d[d[f] == d[f].min()], d[d[f] == d[f].max()]
        else:
            q1, q5 = d[f].quantile([.2, .8]); lo, hi = d[d[f] <= q1], d[d[f] >= q5]
        if len(lo) < 150 or len(hi) < 150:
            continue
        diff = lambda a, b: b.resid_adj.mean() - a.resid_adj.mean()
        se = math.sqrt(lo.resid_adj.var() / len(lo) + hi.resid_adj.var() / len(hi))
        e1 = diff(lo[lo.season <= 2024], hi[hi.season <= 2024]); e2 = diff(lo[lo.season >= 2025], hi[hi.season >= 2025])
        rows.append({"group": grp, "feature": f, "high_minus_low": round(diff(lo, hi), 4), "z": round(diff(lo, hi) / se, 2),
                     "disc": round(e1, 4), "hold": round(e2, 4), "n_low": len(lo), "n_high": len(hi)})
A = pd.DataFrame(rows)
A["consistent"] = np.sign(A.disc) == np.sign(A.hold)
R["adjusted_effects"] = A.sort_values("z", key=abs, ascending=False).head(40).to_dict("records")

# (b) small logistic model on top of the market, walk-forward
X = ["over_rate_l5", "hit_rate_l8", "prev_margin_rel", "m3_vs_line", "m8_vs_line", "cv8", "home", "team_line", "implied_tt",
     "wind_mph", "opp_allow_rel", "opp_pass_epa_allowed", "opp_rush_epa_allowed", "offense_pct_m3", "vacated_target_share"]
if V2:
    X = X + ["route_share_m3", "tprr_m8", "cov_matchup", "opp_man_rate", "opp_pressure_rate", "exp_plays", "team_proe", "opp_proe",
             "opp_db_out", "opp_front_out", "backup_qb", "total_move", "line_dispersion_rel", "fc_wind", "rest_diff",
             "ngs_avg_separation", "ngs_ryoe", "ngs_avg_time_to_throw"]
F["mgroup"] = F.market.map(lambda m: "receiving" if "recep" in m else "rushing" if "rush" in m else "passing" if "pass" in m else "other")
F["logit_fair"] = np.log(F.fair_over / (1 - F.fair_over))
res, preds = {}, []
for test in (2024, 2025):
    tr = F[F.season < test].copy(); te = (F[F.season == test] if test == 2024 else F[F.season >= 2025]).copy()
    mu, sd = tr[X].mean(), tr[X].std()
    Z = lambda d: pd.concat([d[["logit_fair"]], ((d[X] - mu) / sd).fillna(0)], axis=1)   # missing -> average
    p = np.zeros(len(te)); coefs = {}
    for grp in te.mgroup.unique():           # one model per market family: receiving / rushing / passing / other
        trg, idx = tr[tr.mgroup == grp], (te.mgroup == grp).values
        clf = LogisticRegression(C=0.05, max_iter=2000).fit(Z(trg), trg.over)
        p[idx] = clf.predict_proba(Z(te[idx]))[:, 1]
        coefs[grp] = {k: v for k, v in zip(["logit_fair"] + X, np.round(clf.coef_[0], 3)) if abs(v) >= 0.02}
    res[f"test {test}"] = {"n": len(te), "logloss_market": round(log_loss(te.over, te.fair_over), 5), "logloss_model": round(log_loss(te.over, p), 5),
                           "by_group": {g_: {"n": int((te.mgroup == g_).sum()), "ll_market": round(log_loss(te.over[te.mgroup == g_], te.fair_over[te.mgroup == g_]), 5),
                                             "ll_model": round(log_loss(te.over[te.mgroup == g_], p[(te.mgroup == g_).values]), 5)} for g_ in te.mgroup.unique()},
                           "coefs": coefs}
    preds.append(te.assign(p=p))
PR = pd.concat(preds)
PR["ev_o"] = PR.p * (1 + payout(PR.best_over)) - 1; PR["ev_u"] = (1 - PR.p) * (1 + payout(PR.best_under)) - 1
bets = {}
for th in (0, .02, .04, .06):
    o, u = PR[PR.ev_o >= th], PR[PR.ev_u >= th]
    w = np.r_[o.over, 1 - u.over]; od = np.r_[o.best_over, u.best_under]; ss = np.r_[o.season, u.season]
    pr = np.where(w == 1, payout(od), -1.0)
    bets[f"EV>={int(th*100)}%"] = {"n": len(w), "unders": len(u), "roi": round(pr.mean(), 4), "se": round(pr.std() / math.sqrt(len(w)), 4),
                                   "by_season": {int(s): round(pr[ss == s].mean(), 4) for s in np.unique(ss)}}
# benchmark: bet every under at best price (no model)
bu = np.where(PR.over == 0, payout(PR.best_under), -1.0)
bets["benchmark: all unders"] = {"n": len(PR), "roi": round(bu.mean(), 4)}
R["logit_model"] = {"fit": res, "bets": bets}
(ROOT / ("results/props_deep2_v2.json" if V2 else "results/props_deep2.json")).write_text(json.dumps(R, indent=1, default=str))
pd.set_option("display.width", 220)
print(R["baseline_resid_by_season"])
print(A.sort_values("z", key=abs, ascending=False).head(30).to_string(index=False))
print(json.dumps(R["logit_model"], indent=1, default=str))

# (c) the version you can actually bet: model picks at Novig / ProphetX prices (same line as consensus)
import duckdb
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
ex = con.execute("""SELECT game_id, market, player_key, bookmaker, point, over_price, under_price FROM prop_results
                    WHERE pass = 'close' AND bookmaker IN ('novig', 'prophetx') AND over_price IS NOT NULL AND under_price IS NOT NULL""").df()
X2 = PR.merge(ex, on=["game_id", "market", "player_key"])
X2 = X2[X2.point == X2.cons_point]
X2["ev_o"] = X2.p * (1 + payout(X2.over_price)) - 1; X2["ev_u"] = (1 - X2.p) * (1 + payout(X2.under_price)) - 1
exb = {}
for th in (0, .02, .04, .06):
    o, u = X2[X2.ev_o >= th], X2[X2.ev_u >= th]
    w = np.r_[o.over, 1 - u.over]; od = np.r_[o.over_price, u.under_price]; ss = np.r_[o.season, u.season]; bk = np.r_[o.bookmaker, u.bookmaker]
    mg = np.r_[o.mgroup, u.mgroup]
    pr = np.where(w == 1, payout(od), -1.0)
    exb[f"EV>={int(th*100)}%"] = {"n": len(w), "unders": len(u), "roi": round(pr.mean(), 4) if len(w) else None,
                                  "se": round(pr.std() / math.sqrt(len(w)), 4) if len(w) > 1 else None,
                                  "by_season": {int(s): round(pr[ss == s].mean(), 4) for s in np.unique(ss)},
                                  "by_venue": {b: round(pr[bk == b].mean(), 4) for b in np.unique(bk)},
                                  "by_group": {b: (int((mg == b).sum()), round(pr[mg == b].mean(), 4)) for b in np.unique(mg)}}
bu = np.where(X2.over == 0, payout(X2.under_price), -1.0)
exb["benchmark: all unders on exchange"] = {"n": len(X2), "roi": round(bu.mean(), 4)}
R["logit_model_exchange_prices"] = exb
(ROOT / ("results/props_deep2_v2.json" if V2 else "results/props_deep2.json")).write_text(json.dumps(R, indent=1, default=str))
print(json.dumps(exb, indent=1, default=str))
