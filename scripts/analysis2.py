"""Round 2: hypothesis-driven tests (not brute-force combos).

1. EPA power-rating model vs the closing spread / total (fit 2011-19, test 2020-25)
2. Luck regression: score margin vs EPA-implied margin, turnover luck, close-game record
3. Market overreaction: ATS margin autocorrelation (continuous, not buckets)
4. Implied team totals: favorite vs underdog scoring vs implied
5. League scoring-environment lag for totals
6. Backup QB starts
7. Moneyline vs spread for small dogs / buying half points
8. Dome & warm-weather teams in the cold; divisional rematches; early-season last-year anchoring

Writes results/round2.json. Run: uv run python scripts/analysis2.py
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
DISC_END, BE = 2019, 110 / 210
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
FR = {"SD": "LAC", "OAK": "LV", "STL": "LA"}  # pbp uses current franchise codes

tg = con.execute("SELECT * FROM team_games WHERE season BETWEEN 2009 AND 2025").df()
epa = con.execute("SELECT * FROM team_game_epa").df()
for df in (tg, epa):
    for c in df.columns:
        if pd.api.types.is_extension_array_dtype(df[c]) and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])):
            df[c] = df[c].astype("float64")
tg["team_pbp"] = tg.team.replace(FR); tg["opp_pbp"] = tg.opp.replace(FR)
e_off = epa.rename(columns={"team": "team_pbp"}).drop(columns="season")
e_def = epa[["game_id", "team", "off_epa", "off_epa_neutral", "off_sr", "giveaways", "plays", "off_epa_total"]].rename(
    columns={"team": "opp_pbp", "off_epa": "def_epa", "off_epa_neutral": "def_epa_neutral", "off_sr": "def_sr",
             "giveaways": "takeaways", "plays": "opp_plays", "off_epa_total": "def_epa_total"})
tg = tg.merge(e_off, on=["game_id", "team_pbp"], how="left").merge(e_def, on=["game_id", "opp_pbp"], how="left")
tg["gameday"] = pd.to_datetime(tg.gameday)
tg = tg.sort_values(["team", "gameday"]).reset_index(drop=True)
tg["net_epa"] = tg.off_epa - tg.def_epa
tg["net_epa_neutral"] = tg.off_epa_neutral - tg.def_epa_neutral
tg["epa_margin"] = tg.off_epa_total - tg.def_epa_total        # EPA is in points
tg["to_margin"] = tg.takeaways - tg.giveaways
tg["pace"] = tg.plays + tg.opp_plays
tg["pts_for"] = tg.pts; tg["pts_against"] = tg.opp_pts
R: dict = {}


def binrec(y):
    y = pd.Series(y).dropna(); w = int((y == 1).sum()); n = len(y)
    return {"w": w, "l": n - w, "pct": round(w / n, 4) if n else None,
            "p": round(stats.binomtest(w, n, .5).pvalue, 4) if n else None}


def eras(d, col="cover"):
    return {"all": binrec(d[col]), "2011-19": binrec(d[d.season <= DISC_END][col]), "2020-25": binrec(d[d.season > DISC_END][col])}


# ------------------------------------------------------------------ pregame ratings (franchise-level EWMA, carried across seasons)
ALPHA, CARRY = 0.12, 0.55
key = tg.team.replace(FR)
rate_cols = {"net_epa": "r_net", "net_epa_neutral": "r_net_n", "off_epa": "r_off", "def_epa": "r_def", "pace": "r_pace",
             "pts_for": "r_pf", "pts_against": "r_pa", "margin": "r_margin", "to_margin": "r_to", "epa_margin": "r_epam"}
means = {c: tg[c].mean() for c in rate_cols}
for c in rate_cols.values():
    tg[c] = np.nan
for team, idx in tg.groupby(key).groups.items():
    d = tg.loc[idx].sort_values("gameday")
    state = {c: means[c] for c in rate_cols}; last_season = None
    for i, row in d.iterrows():
        if last_season is not None and row.season != last_season:
            state = {c: means[c] + CARRY * (state[c] - means[c]) for c in rate_cols}
        for c, rc in rate_cols.items():
            tg.at[i, rc] = state[c]
        for c in rate_cols:
            if not np.isnan(row[c]):
                state[c] = (1 - ALPHA) * state[c] + ALPHA * row[c]
        last_season = row.season

opp_r = tg[["game_id", "team"] + list(rate_cols.values())].rename(columns={"team": "opp", **{v: "o_" + v for v in rate_cols.values()}})
tg = tg.merge(opp_r, on=["game_id", "opp"], how="left")
t = tg[tg.season.between(2011, 2025)].copy()
t["home_i"] = np.where(t.neutral.astype(bool), 0, np.where(t.side == "home", 1, -1))
t["d_net"] = t.r_net - t.o_r_net
t["d_net_n"] = t.r_net_n - t.o_r_net_n
t["d_margin"] = t.r_margin - t.o_r_margin
disc, hold = t[t.season <= DISC_END], t[t.season > DISC_END]

# 1a. spread model: predict margin, then test whether (model - line) predicts ATS margin out of sample
model_res = {}
for name, f in {"EPA/play (all)": "margin ~ d_net + home_i", "EPA/play (neutral WP)": "margin ~ d_net_n + home_i",
                "point differential": "margin ~ d_margin + home_i", "EPA + pt diff": "margin ~ d_net_n + d_margin + home_i"}.items():
    m = smf.ols(f, disc).fit()
    t["pred"] = m.predict(t)
    t["edge"] = t.pred - t.team_line
    h = t[t.season > DISC_END]; dd = t[t.season <= DISC_END]
    # does the model add information to the close?  ats_margin ~ edge  (slope > 0 = model knows something the line doesn't)
    slope_d = smf.ols("ats_margin ~ edge", dd).fit(); slope_h = smf.ols("ats_margin ~ edge", h).fit()
    sim = {}
    for th in (1, 2, 3, 4):
        pick = h[h.edge >= th]  # one row per game max, since edges are mirrored
        sim[f"edge>={th}"] = binrec(pick.cover)
    model_res[name] = {"mae_model": round(float((h.pred - h.margin).abs().mean()), 3),
                       "mae_line": round(float((h.team_line - h.margin).abs().mean()), 3),
                       "slope_disc": round(float(slope_d.params.edge), 3), "p_disc": round(float(slope_d.pvalues.edge), 4),
                       "slope_hold": round(float(slope_h.params.edge), 3), "p_hold": round(float(slope_h.pvalues.edge), 4),
                       "holdout_bets": sim}
R["spread_models"] = model_res

# 1b. line-informed: does the model predict where the line is *wrong* when combined? ats_margin ~ d_net_n + d_margin + home_i
m = smf.ols("ats_margin ~ d_net_n + d_margin + home_i + team_line", disc).fit()
mh = smf.ols("ats_margin ~ d_net_n + d_margin + home_i + team_line", hold).fit()
R["ats_residual_regression"] = {"disc": {k: [round(float(m.params[k]), 4), round(float(m.pvalues[k]), 4)] for k in m.params.index},
                                "hold": {k: [round(float(mh.params[k]), 4), round(float(mh.pvalues[k]), 4)] for k in mh.params.index}}

# 1c. totals model
gh = t[(t.side == "home")].copy()
gh["pred_inputs_ok"] = gh[["r_pf", "o_r_pf", "r_pa", "o_r_pa", "r_pace", "o_r_pace"]].notna().all(axis=1)
gh = gh[gh.pred_inputs_ok]
gh["exp_pts"] = (gh.r_pf + gh.o_r_pa) / 2 + (gh.o_r_pf + gh.r_pa) / 2
gh["exp_pace"] = (gh.r_pace + gh.o_r_pace) / 2
gh["off_sum"] = gh.r_off + gh.o_r_off; gh["def_sum"] = gh.r_def + gh.o_r_def
gh["dome"] = gh.roof.isin(["dome", "closed"]).astype(int)
gh["windy"] = ((gh.wind_mph.fillna(0) >= 10) & gh.roof.isin(["outdoors", "open"])).astype(int)
gh["over"] = gh.ou_result.map({"over": 1.0, "under": 0.0})
gdisc, ghold = gh[gh.season <= DISC_END], gh[gh.season > DISC_END]
tot_res = {}
for name, f in {"points ratings": "total ~ exp_pts", "EPA + pace": "total ~ off_sum + def_sum + exp_pace",
                "all + dome + wind": "total ~ exp_pts + off_sum + def_sum + exp_pace + dome + windy"}.items():
    mm = smf.ols(f, gdisc).fit()
    gh["tpred"] = mm.predict(gh); gh["tedge"] = gh.tpred - gh.total_line
    hh = gh[gh.season > DISC_END]
    s_h = smf.ols("ou_margin ~ tedge", hh).fit(); s_d = smf.ols("ou_margin ~ tedge", gh[gh.season <= DISC_END]).fit()
    sims = {}
    for th in (2, 3, 4):
        sims[f"over edge>={th}"] = binrec(hh[hh.tedge >= th].over)
        sims[f"under edge>={th}"] = binrec(1 - hh[hh.tedge <= -th].over)
    tot_res[name] = {"mae_model": round(float((hh.tpred - hh.total).abs().mean()), 3), "mae_line": round(float((hh.total_line - hh.total).abs().mean()), 3),
                     "slope_disc": round(float(s_d.params.tedge), 3), "p_disc": round(float(s_d.pvalues.tedge), 4),
                     "slope_hold": round(float(s_h.params.tedge), 3), "p_hold": round(float(s_h.pvalues.tedge), 4), "holdout_bets": sims}
R["total_models"] = tot_res
# wind within the totals residual (does wind survive controlling for ratings?)
mw = smf.ols("ou_margin ~ windy + dome + exp_pts - total_line", gh).fit() if False else smf.ols("ou_margin ~ windy + dome", gh).fit()
R["wind_residual"] = {k: [round(float(mw.params[k]), 3), round(float(mw.pvalues[k]), 4)] for k in mw.params.index}

# ------------------------------------------------------------------ 2. luck regression
tg["luck"] = tg.margin - tg.epa_margin  # scoreboard beat the EPA story by this much
g2 = tg.sort_values(["team", "gameday"])
g2["prev_luck"] = g2.groupby(["team", "season"]).luck.shift()
g2["prev_to"] = g2.groupby(["team", "season"]).to_margin.shift()
g2["luck_td"] = g2.groupby(["team", "season"]).luck.transform(lambda s: s.shift().expanding().mean())
g2["to_td"] = g2.groupby(["team", "season"]).to_margin.transform(lambda s: s.shift().expanding().mean())
g2["one_score"] = (g2.margin.abs() <= 8).astype(float)
g2["os_w"] = ((g2.margin.abs() <= 8) & (g2.margin > 0)).astype(float)
g2["os_wins_td"] = g2.groupby(["team", "season"]).os_w.transform(lambda s: s.shift().cumsum())
g2["os_games_td"] = g2.groupby(["team", "season"]).one_score.transform(lambda s: s.shift().cumsum())
g2["os_pct_td"] = g2.os_wins_td / g2.os_games_td.replace(0, np.nan)
L = g2[g2.season.between(2011, 2025) & g2.cover.notna()].copy()
luck = {}
for name, mask in {
    "last game: scoreboard beat EPA by 10+ (lucky)": L.prev_luck >= 10,
    "last game: scoreboard trailed EPA by 10+ (unlucky)": L.prev_luck <= -10,
    "last game: +3 or better turnover margin": L.prev_to >= 3,
    "last game: -3 or worse turnover margin": L.prev_to <= -3,
    "season-to-date luck >= +5/gm (4+ gp)": (L.luck_td >= 5) & (L.gp_td >= 4),
    "season-to-date luck <= -5/gm (4+ gp)": (L.luck_td <= -5) & (L.gp_td >= 4),
    "turnover margin >= +1.0/gm (4+ gp)": (L.to_td >= 1) & (L.gp_td >= 4),
    "turnover margin <= -1.0/gm (4+ gp)": (L.to_td <= -1) & (L.gp_td >= 4),
    "one-score record >= .75 (4+ such games)": (L.os_pct_td >= .75) & (L.os_games_td >= 4),
    "one-score record <= .25 (4+ such games)": (L.os_pct_td <= .25) & (L.os_games_td >= 4),
}.items():
    luck[name] = eras(L[mask])
R["luck"] = luck
R["luck_regression"] = {}
for col in ("prev_luck", "luck_td", "to_td", "prev_to"):
    d = L.dropna(subset=[col])
    fit = smf.ols(f"ats_margin ~ {col}", d).fit()
    fh = smf.ols(f"ats_margin ~ {col}", d[d.season > DISC_END]).fit()
    R["luck_regression"][col] = {"slope": round(float(fit.params[col]), 4), "p": round(float(fit.pvalues[col]), 4),
                                 "slope_2020_25": round(float(fh.params[col]), 4), "p_2020_25": round(float(fh.pvalues[col]), 4)}

# ------------------------------------------------------------------ 3. overreaction: ATS autocorrelation
g2["prev_ats"] = g2.groupby(["team", "season"]).ats_margin.shift()
g2["ats_last3"] = g2.groupby(["team", "season"]).ats_margin.transform(lambda s: s.shift().rolling(3).mean())
g2["ats_td"] = g2.groupby(["team", "season"]).ats_margin.transform(lambda s: s.shift().expanding().mean())
A = g2[g2.season.between(2011, 2025)]
R["ats_autocorr"] = {}
for col in ("prev_ats", "ats_last3", "ats_td"):
    d = A.dropna(subset=[col, "ats_margin"])
    fit = smf.ols(f"ats_margin ~ {col}", d).fit()
    R["ats_autocorr"][col] = {"slope": round(float(fit.params[col]), 4), "p": round(float(fit.pvalues[col]), 4), "n": int(len(d)),
                              "corr": round(float(d[col].corr(d.ats_margin)), 4)}
# market adjustment after a big result: does the next line move "too much"?
nx = g2.copy()
nx["next_line"] = nx.groupby(["team", "season"]).team_line.shift(-1)
nx["next_ats"] = nx.groupby(["team", "season"]).ats_margin.shift(-1)
nx = nx[nx.season.between(2011, 2025)].dropna(subset=["next_ats"])
R["after_big_ats_result"] = {
    "covered by 20+": {**binrec((nx[nx.ats_margin >= 20].next_ats > 0).where(nx[nx.ats_margin >= 20].next_ats != 0).astype(float)), "n": int((nx.ats_margin >= 20).sum())},
    "missed by 20+": {**binrec((nx[nx.ats_margin <= -20].next_ats > 0).where(nx[nx.ats_margin <= -20].next_ats != 0).astype(float)), "n": int((nx.ats_margin <= -20).sum())},
}

# ------------------------------------------------------------------ 4. implied team totals
tt = t.copy()
tt["implied_tt"] = (tt.total_line + tt.team_line) / 2
tt["tt_margin"] = tt.pts - tt.implied_tt
tt["tt_over"] = np.where(tt.tt_margin > 0, 1.0, np.where(tt.tt_margin < 0, 0.0, np.nan))
tt["role"] = np.where(tt.team_line > 0, "fav", np.where(tt.team_line < 0, "dog", "pk"))
tt["tt_b"] = pd.cut(tt.implied_tt, [0, 17, 20, 23, 26, 29, 60])
itt = []
for (role, b), d in tt[tt.role != "pk"].groupby(["role", "tt_b"], observed=True):
    itt.append({"role": role, "implied": str(b), "n": len(d), "avg_implied": round(d.implied_tt.mean(), 2), "avg_scored": round(d.pts.mean(), 2),
                "over_pct": binrec(d.tt_over)["pct"], "over_2011_19": binrec(d[d.season <= DISC_END].tt_over)["pct"],
                "over_2020_25": binrec(d[d.season > DISC_END].tt_over)["pct"], "p": binrec(d.tt_over)["p"]})
R["implied_team_totals"] = itt
R["implied_tt_by_side"] = {f"{r} {s}": eras(tt[(tt.role == r) & (tt.side == s) & ~tt.neutral.astype(bool)], "tt_over")
                           for r in ("fav", "dog") for s in ("home", "away")}

# ------------------------------------------------------------------ 5. league scoring-environment lag
G = con.execute("SELECT * FROM games WHERE season BETWEEN 2011 AND 2025 AND NOT playoff").df()
wk = G.groupby(["season", "week"]).agg(ou=("ou_margin", "mean"), n=("ou_margin", "size")).reset_index()
wk["prev2"] = wk.groupby("season").ou.transform(lambda s: s.shift().rolling(2).mean())
G = G.merge(wk[["season", "week", "prev2"]], on=["season", "week"])
G["over"] = G.ou_result.map({"over": 1.0, "under": 0.0})
fit = smf.ols("ou_margin ~ prev2", G.dropna(subset=["prev2"])).fit()
R["scoring_lag"] = {"slope": round(float(fit.params.prev2), 4), "p": round(float(fit.pvalues.prev2), 4),
                    "after 2 wks of overs by 3+ -> over%": eras(G[G.prev2 >= 3], "over"),
                    "after 2 wks of unders by 3+ -> over%": eras(G[G.prev2 <= -3], "over")}
R["over_by_week_block"] = {b: eras(G[m], "over") for b, m in {"wk 1": G.week == 1, "wk 2-4": G.week.between(2, 4),
                                                               "wk 5-9": G.week.between(5, 9), "wk 10-14": G.week.between(10, 14), "wk 15+": G.week >= 15}.items()}

# ------------------------------------------------------------------ 6. backup QBs
q = tg.sort_values(["team", "gameday"]).copy()
def primary_starter(g):
    out, counts = [], {}
    for qb in g.qb_id:
        out.append(max(counts, key=counts.get) if counts else None)
        if qb is not None and qb == qb: counts[qb] = counts.get(qb, 0) + 1
    return pd.Series(out, index=g.index)
q["primary_qb"] = q.groupby(["team", "season"], group_keys=False).apply(primary_starter)
q["backup_start"] = (q.gp_td >= 3) & q.primary_qb.notna() & (q.qb_id != q.primary_qb)
bq = q[["game_id", "team", "backup_start"]]
q = q.merge(bq.rename(columns={"team": "opp", "backup_start": "opp_backup"}), on=["game_id", "opp"], how="left")
Q = q[q.season.between(2011, 2025)]
Q = Q.assign(over=Q.ou_result.map({"over": 1.0, "under": 0.0}))
R["backup_qb"] = {
    "team starting non-primary QB (ATS)": eras(Q[Q.backup_start]),
    "... as dog": eras(Q[Q.backup_start & Q.dog]), "... as fav": eras(Q[Q.backup_start & Q.fav]),
    "opp starting non-primary QB (ATS)": eras(Q[Q.opp_backup == True]),
    "game with any backup QB (over%)": eras(Q[(Q.side == "home") & (Q.backup_start | (Q.opp_backup == True))], "over"),
    "n_backup_starts": int(Q.backup_start.sum()),
}

# ------------------------------------------------------------------ 7. ML vs spread for small dogs; half-point value
ml = t[t.dog & t.ml.notna()].copy()
ml["ml_profit"] = np.where(ml.su_win == 1, ml.ml_payout, np.where(ml.su_win == 0, -1.0, 0.0))
ml["sp_profit"] = np.where(ml.cover == 1, ml.spread_payout.fillna(100 / 110), np.where(ml.cover == 0, -1.0, 0.0))
mlr = []
for line in (-1, -1.5, -2, -2.5, -3, -3.5, -4, -6.5, -7, -7.5, -10):
    d = ml[ml.team_line == line]
    if len(d) >= 40:
        mlr.append({"dog_line": f"+{-line:g}", "n": len(d), "su_win": round(d.su_win.mean(), 3), "avg_ml": round(d.ml.median()),
                    "ml_roi": round(d.ml_profit.mean(), 4), "spread_cover": binrec(d.cover)["pct"], "spread_roi": round(d.sp_profit.mean(), 4)})
R["dog_ml_vs_spread"] = mlr
home_g = G[G.location.isna()] if "location" in G else G
res = G.result.abs()
R["half_point_value"] = {f"land exactly on {k}": round(float((res == k).mean()), 4) for k in (1, 2, 3, 4, 6, 7, 10, 14)}
R["half_point_value"]["closing line = 3: margin exactly 3"] = round(float((G[G.spread_line.abs() == 3].result.abs() == 3).mean()), 4)
R["half_point_value"]["closing line = 2.5/3.5: margin exactly 3"] = round(float((G[G.spread_line.abs().isin([2.5, 3.5])].result.abs() == 3).mean()), 4)
R["half_point_value"]["closing line = 7: margin exactly 7"] = round(float((G[G.spread_line.abs() == 7].result.abs() == 7).mean()), 4)

# ------------------------------------------------------------------ 8. misc structural angles
home_roof = (con.execute("SELECT season, home_team team, mode(roof) roof FROM games GROUP BY 1,2").df())
home_roof["team_env"] = np.where(home_roof.roof.isin(["dome", "closed"]), "dome", "open")
warm = {"MIA", "TB", "JAX", "LAC", "SD", "LA", "ARI", "SF", "LV", "OAK", "NO", "HOU", "ATL", "DAL", "CAR"}
t2 = t.merge(home_roof[["season", "team", "team_env"]], on=["season", "team"], how="left")
cold_out = t2.roof.isin(["outdoors", "open"]) & (t2.temp_f <= 35)
t2["over"] = t2.ou_result.map({"over": 1.0, "under": 0.0})
R["cold_weather"] = {
    "dome team, road, outdoor <=35F": eras(t2[cold_out & (t2.team_env == "dome") & (t2.side == "away")]),
    "warm-weather team, road, outdoor <=35F": eras(t2[cold_out & t2.team.isin(warm) & (t2.side == "away")]),
    "home team hosting dome/warm team in <=35F": eras(t2[cold_out & (t2.side == "home") & (t2.opp.isin(warm) | t2.opp.map(dict(zip(t2.team, t2.team_env))).eq("dome"))]),
}
# divisional rematch: lost first meeting
dv = t[t.div_game.astype(bool) & ~t.playoff.astype(bool)].sort_values("gameday").copy()
dv["meeting"] = dv.groupby(["season", "team", "opp"]).cumcount() + 1
first = dv[dv.meeting == 1][["season", "team", "opp", "margin", "cover"]].rename(columns={"margin": "m1", "cover": "c1"})
sec = dv[dv.meeting == 2].merge(first, on=["season", "team", "opp"])
R["div_rematch"] = {"lost 1st meeting": eras(sec[sec.m1 < 0]), "lost 1st meeting by 14+": eras(sec[sec.m1 <= -14]),
                    "won 1st meeting by 14+": eras(sec[sec.m1 >= 14]), "failed to cover 1st meeting": eras(sec[sec.c1 == 0])}
# early-season anchoring on last season: regression of ATS margin on last-year win% in weeks 1-4 vs later
early = t[t.week <= 4].dropna(subset=["prev_season_win_pct"])
late = t[(t.week >= 10) & ~t.playoff.astype(bool)].dropna(subset=["prev_season_win_pct"])
fe = smf.ols("ats_margin ~ prev_season_win_pct", early).fit(); fl = smf.ols("ats_margin ~ prev_season_win_pct", late).fit()
R["last_year_anchor"] = {"weeks 1-4 slope": round(float(fe.params.prev_season_win_pct), 3), "weeks 1-4 p": round(float(fe.pvalues.prev_season_win_pct), 4),
                         "weeks 10+ slope": round(float(fl.params.prev_season_win_pct), 3), "weeks 10+ p": round(float(fl.pvalues.prev_season_win_pct), 4),
                         "wk1-4, last yr <=.35 (ATS)": eras(early[early.prev_season_win_pct <= .35]),
                         "wk1-4, last yr >=.70 (ATS)": eras(early[early.prev_season_win_pct >= .70])}
# EPA-vs-record gap: teams whose record outruns their efficiency
t["win_pct_td"] = t.su_w_td / t.gp_td.replace(0, np.nan)
gap = t[(t.gp_td >= 5)].copy()
gap["rank_gap"] = gap.groupby(["season", "week"]).win_pct_td.rank(pct=True) - gap.groupby(["season", "week"]).r_net_n.rank(pct=True)
R["record_vs_epa"] = {"record much better than EPA (top gap 15%)": eras(gap[gap.rank_gap >= .35]),
                      "record much worse than EPA": eras(gap[gap.rank_gap <= -.35])}

(OUT / "round2.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps(R, indent=1, default=str))

# ------------------------------------------------------------------ follow-up: does the totals model add anything beyond wind?
mm = smf.ols("total ~ exp_pts + off_sum + def_sum + exp_pace + dome + windy", gdisc).fit()
gh["tpred"] = mm.predict(gh); gh["tedge"] = gh.tpred - gh.total_line
hh = gh[gh.season > DISC_END]
R["totals_model_ex_wind"] = {}
for lbl, d in {"windy": hh[hh.windy == 1], "calm/dome": hh[hh.windy == 0]}.items():
    R["totals_model_ex_wind"][lbl] = {f"under edge>={th}": binrec(1 - d[d.tedge <= -th].over) for th in (2, 3, 4)}
    R["totals_model_ex_wind"][lbl].update({f"over edge>={th}": binrec(d[d.tedge >= th].over) for th in (2, 3)})
(OUT / "round2.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps(R["totals_model_ex_wind"], indent=1))
