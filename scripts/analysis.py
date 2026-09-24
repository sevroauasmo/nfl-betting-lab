"""Betting-trend scan over nflverse closing lines.

Writes CSVs to results/ and results/summary.json. Run: uv run python scripts/analysis.py

Method
- ATS scan is from one team's perspective (team_games). Filters are "atoms" in groups;
  combos take at most one atom per group (singles, pairs, triples), n >= MIN_N.
- Each combo is scored on the full 2011-2025 window, then honestly re-checked:
  discovery = 2011-2019, holdout = 2020-2025. A trend only "survives" if it is on the same
  side of 52.4% (the -110 break-even) in the holdout.
- p-values are two-sided binomial vs 50%; q-values are Benjamini-Hochberg across all combos tested.
"""
import itertools
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
FIRST, LAST = 2011, 2025
DISC_END = 2019  # discovery 2011-2019, holdout 2020-2025
BE = 110 / 210  # -110 break-even, 52.38%
MIN_N = 100

con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
tg = con.execute(f"SELECT * FROM team_games WHERE season BETWEEN {FIRST} AND {LAST}").df()
g = con.execute(f"""
    SELECT g.*, w.rain, w.snow
    FROM games g LEFT JOIN (
        SELECT game_id,
               bool_or((weather ILIKE '%rain%' OR weather ILIKE '%shower%')
                       AND weather NOT ILIKE '%no rain%' AND weather NOT ILIKE '%chance%') AS rain,
               bool_or(weather ILIKE '%snow%' AND weather NOT ILIKE '%no snow%'
                       AND weather NOT ILIKE '%chance%') AS snow
        FROM pbp WHERE season BETWEEN {FIRST} AND {LAST} GROUP BY 1) w USING (game_id)
    WHERE season BETWEEN {FIRST} AND {LAST}""").df()
summary: dict = {}


def to_numpy_dtypes(df):
    """DuckDB returns nullable pandas dtypes; use plain numpy (NaN for missing) so masks/where behave."""
    for c in df.columns:
        if pd.api.types.is_extension_array_dtype(df[c]) or df[c].dtype == object:
            if pd.api.types.is_bool_dtype(df[c]) and not df[c].isna().any():
                df[c] = df[c].astype(bool)
            elif pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
                df[c] = df[c].astype("float64")
    return df


tg, g = to_numpy_dtypes(tg), to_numpy_dtypes(g)


def rec(cover: pd.Series, profit: pd.Series | None = None) -> dict:
    """cover: 1/0/NaN(push)."""
    w = int((cover == 1).sum()); l = int((cover == 0).sum()); p = int(cover.isna().sum())
    n = w + l
    out = {"w": w, "l": l, "p": p, "n": n, "pct": w / n if n else np.nan}
    out["pval"] = stats.binomtest(w, n, 0.5).pvalue if n else np.nan
    out["roi_110"] = (w * (100 / 110) - l) / (n + p) if n else np.nan
    if profit is not None:
        out["roi_actual"] = profit.mean()
    return out


def fmt_rec(r):
    return f"{r['w']}-{r['l']}-{r['p']}"


# ---------------------------------------------------------------- baselines
tg["neutral"] = tg["neutral"].astype(bool)
tg["profit"] = np.where(tg["cover"] == 1, tg["spread_payout"].fillna(100 / 110), np.where(tg["cover"] == 0, -1.0, 0.0))
tg["ml_profit"] = np.where(tg["su_win"] == 1, tg["ml_payout"], np.where(tg["su_win"] == 0, -1.0, 0.0))
home = tg[(tg.side == "home") & ~tg.neutral]

base_rows = []
for s, d in home.groupby("season"):
    r = rec(d.cover)
    base_rows.append({"season": s, "home_cover_pct": r["pct"],
                      "fav_cover_pct": rec(tg[(tg.season == s) & tg.fav].cover)["pct"],
                      "over_pct": rec(g[g.season == s].ou_result.map({"over": 1, "under": 0}))["pct"],
                      "avg_home_line": d.team_line.mean(), "avg_home_margin": d.margin.mean(),
                      "home_su_pct": d.su_win.mean()})
by_season = pd.DataFrame(base_rows)
by_season.to_csv(OUT / "baseline_by_season.csv", index=False)
summary["baseline_by_season"] = by_season.round(4).to_dict("records")
summary["overall"] = {
    "games": int(len(g)),
    "home": rec(home.cover), "fav": rec(tg[tg.fav].cover),
    "home_fav": rec(tg[tg.fav & (tg.side == "home") & ~tg.neutral].cover),
    "home_dog": rec(tg[tg.dog & (tg.side == "home") & ~tg.neutral].cover),
    "over": rec(g.ou_result.map({"over": 1, "under": 0})),
    "ats_push_rate": float((g.ats_winner == "push").mean()),
    "ou_push_rate": float((g.ou_result == "push").mean()),
    "mean_abs_ats_error": float(g.home_ats_margin.abs().mean()),
    "sd_ats_error": float(g.home_ats_margin.std()),
    "mean_abs_total_error": float(g.ou_margin.abs().mean()),
    "sd_total_error": float(g.ou_margin.std()),
}

# ---------------------------------------------------------------- spread bucket x side (user's example)
def line_bucket(x):
    a = abs(x)
    if a == 0: return "PK"
    if a < 3: return "0.5-2.5"
    if a == 3: return "3"
    if a < 7: return "3.5-6.5"
    if a == 7: return "7"
    if a < 10: return "7.5-9.5"
    if a < 14: return "10-13.5"
    return "14+"

order = ["PK", "0.5-2.5", "3", "3.5-6.5", "7", "7.5-9.5", "10-13.5", "14+"]
tg["lb"] = tg.team_line.map(line_bucket)
rows = []
for (role, side), d in tg[~tg.neutral & (tg.team_line != 0)].groupby(
        [np.where(tg.fav, "fav", "dog")[~tg.neutral & (tg.team_line != 0)], "side"]):
    for b in order[1:]:
        dd = d[d.lb == b]
        if len(dd):
            r = rec(dd.cover, dd.profit)
            r1 = rec(dd[dd.season <= DISC_END].cover); r2 = rec(dd[dd.season > DISC_END].cover)
            rows.append({"role": f"{side} {role}", "line": b, **r, "pct_2011_19": r1["pct"], "pct_2020_25": r2["pct"],
                         "su_pct": dd.su_win.mean(), "avg_ats_margin": dd.ats_margin.mean()})
spread_tbl = pd.DataFrame(rows)
spread_tbl.to_csv(OUT / "spread_bucket_by_side.csv", index=False)
summary["spread_bucket_by_side"] = spread_tbl.round(4).to_dict("records")

# ---------------------------------------------------------------- moneyline by price
tg["impl"] = np.where(tg.ml < 0, -tg.ml / (-tg.ml + 100), 100 / (tg.ml + 100))
bins = [0, .15, .25, .35, .45, .55, .65, .75, .85, 1]
tg["impl_b"] = pd.cut(tg.impl, bins)
ml = (tg.dropna(subset=["ml"]).groupby(["impl_b"], observed=True)
      .agg(n=("su_win", "size"), win_pct=("su_win", "mean"), implied=("impl", "mean"), roi=("ml_profit", "mean"))
      .reset_index())
ml["impl_b"] = ml.impl_b.astype(str)
ml.to_csv(OUT / "moneyline_by_price.csv", index=False)
summary["moneyline_by_price"] = ml.round(4).to_dict("records")
ml_side = (tg.dropna(subset=["ml"]).assign(role=lambda d: np.where(d.side.eq("home") & ~d.neutral, "home", "away/neutral") + np.where(d.fav, " fav", " dog"))
           .groupby("role").agg(n=("su_win", "size"), win_pct=("su_win", "mean"), implied=("impl", "mean"), roi=("ml_profit", "mean")).reset_index())
summary["moneyline_by_role"] = ml_side.round(4).to_dict("records")

# ---------------------------------------------------------------- key numbers + teasers
home_g = g[~g.neutral.astype(bool)]
mfreq = g.result.abs().value_counts(normalize=True).sort_index()
summary["margin_freq"] = {int(k): round(float(v), 4) for k, v in mfreq.head(21).items()}
tot_freq = g.total.value_counts(normalize=True).sort_values(ascending=False).head(12)
summary["total_freq"] = {int(k): round(float(v), 4) for k, v in tot_freq.items()}

# 6-pt teaser legs: team gets +6
tg["teased_cover"] = np.where(tg.ats_margin + 6 > 0, 1.0, np.where(tg.ats_margin + 6 < 0, 0.0, np.nan))
teas = []
for label, m in {
    "Wong: fav -7.5 to -8.5 (-> -1.5/-2.5)": tg.team_line.between(7.5, 8.5),
    "Wong: dog +1.5 to +2.5 (-> +7.5/+8.5)": tg.team_line.between(-2.5, -1.5),
    "Wong fav, home": tg.team_line.between(7.5, 8.5) & (tg.side == "home"),
    "Wong fav, road": tg.team_line.between(7.5, 8.5) & (tg.side == "away"),
    "Wong dog, home": tg.team_line.between(-2.5, -1.5) & (tg.side == "home"),
    "Wong dog, road": tg.team_line.between(-2.5, -1.5) & (tg.side == "away"),
    "Wong dog, total <= 44": tg.team_line.between(-2.5, -1.5) & (tg.total_line <= 44),
    "Wong fav, total <= 44": tg.team_line.between(7.5, 8.5) & (tg.total_line <= 44),
    "All favorites": tg.fav, "All dogs": tg.dog,
    "Fav -3 (-> +3)": tg.team_line == 3, "Dog +3 (-> +9)": tg.team_line == -3,
    "Fav -6.5 to -7 (-> +0.5/PK)": tg.team_line.between(6.5, 7),
    "Dog +7 (-> +13)": tg.team_line == -7,
}.items():
    d = tg[m]
    for era, dd in [("all", d), ("2011-19", d[d.season <= DISC_END]), ("2020-25", d[d.season > DISC_END])]:
        r = rec(dd.teased_cover)
        teas.append({"leg": label, "era": era, "n": r["n"], "win_pct": r["pct"], "pushes": r["p"]})
teas = pd.DataFrame(teas)
teas.to_csv(OUT / "teasers.csv", index=False)
summary["teasers"] = teas.round(4).to_dict("records")

# ---------------------------------------------------------------- ATS atom scan
t = tg
early_season = t.week <= 4
home_nn = (t.side == "home") & ~t.neutral
away_nn = (t.side == "away") & ~t.neutral
wpct = t.su_w_td / t.gp_td.replace(0, np.nan)
owpct = t.opp_su_w_td / t.opp_gp_td.replace(0, np.nan)
atspct = t.ats_w_td / t.ats_n_td.replace(0, np.nan)
opp_atspct = t.opp_ats_w_td / t.opp_ats_n_td.replace(0, np.nan)
outdoor = t.roof.isin(["outdoors", "open"])
L = t.team_line

ATOMS = {
    "side": {"home": home_nn, "away": away_nn},
    "line": {
        "fav": L > 0, "dog": L < 0,
        "fav 0.5-2.5": L.between(0.5, 2.5), "dog 0.5-2.5": L.between(-2.5, -0.5),
        "fav 3": L == 3, "dog 3": L == -3,
        "fav 3.5-6.5": L.between(3.5, 6.5), "dog 3.5-6.5": L.between(-6.5, -3.5),
        "fav 7": L == 7, "dog 7": L == -7,
        "fav 7.5-9.5": L.between(7.5, 9.5), "dog 7.5-9.5": L.between(-9.5, -7.5),
        "fav 10+": L >= 10, "dog 10+": L <= -10,
        "fav 7+": L >= 7, "dog 7+": L <= -7,
        "fav 9+": L >= 9, "dog 9+": L <= -9,
    },
    "rest": {
        "off bye": t.rest >= 13, "opp off bye": t.opp_rest >= 13,
        "rest adv 3+ days": (t.rest - t.opp_rest) >= 3, "rest disadv 3+ days": (t.rest - t.opp_rest) <= -3,
        "short week (<=5d)": t.rest <= 5, "opp short week": t.opp_rest <= 5,
    },
    "prev_su": {
        "off SU loss": t.prev_su_win == 0, "off SU win": t.prev_su_win == 1,
        "off blowout loss (17+)": t.prev_margin <= -17, "off blowout win (17+)": t.prev_margin >= 17,
        "off upset loss (lost as fav)": (t.prev_su_win == 0) & (t.prev_fav == True),
        "off upset win (won as dog)": (t.prev_su_win == 1) & (t.prev_fav == False),
        "off 2 straight SU losses": (t.prev_su_win == 0) & (t.prev2_su_win == 0),
        "off 2 straight SU wins": (t.prev_su_win == 1) & (t.prev2_su_win == 1),
        "off 1-score loss": t.prev_margin.between(-8, -1),
    },
    "prev_ats": {
        "off ATS loss": t.prev_cover == 0, "off ATS win": t.prev_cover == 1,
        "off ATS miss by 14+": t.prev_ats_margin <= -14, "off ATS cover by 14+": t.prev_ats_margin >= 14,
        "off 2 straight ATS losses": (t.prev_cover == 0) & (t.prev2_cover == 0),
        "off 2 straight ATS wins": (t.prev_cover == 1) & (t.prev2_cover == 1),
    },
    "opp_prev": {
        "opp off SU loss": t.opp_prev_su_win == 0, "opp off SU win": t.opp_prev_su_win == 1,
        "opp off blowout win (17+)": t.opp_prev_margin >= 17, "opp off blowout loss (17+)": t.opp_prev_margin <= -17,
        "opp off ATS cover by 14+": t.opp_prev_ats_margin >= 14, "opp off ATS miss by 14+": t.opp_prev_ats_margin <= -14,
    },
    "record": {
        "winless (2+ gp)": (t.gp_td >= 2) & (t.su_w_td == 0),
        "win pct <= .333 (4+ gp)": (t.gp_td >= 4) & (wpct <= 1 / 3),
        "win pct >= .667 (4+ gp)": (t.gp_td >= 4) & (wpct >= 2 / 3),
        "worse record than opp by .3+": (t.gp_td >= 4) & ((wpct - owpct) <= -0.3),
        "better record than opp by .3+": (t.gp_td >= 4) & ((wpct - owpct) >= 0.3),
        "ATS <= .35 (6+ gp)": (t.ats_n_td >= 6) & (atspct <= 0.35),
        "ATS >= .65 (6+ gp)": (t.ats_n_td >= 6) & (atspct >= 0.65),
        "opp ATS >= .65 (6+ gp)": (t.opp_ats_n_td >= 6) & (opp_atspct >= 0.65),
        "opp ATS <= .35 (6+ gp)": (t.opp_ats_n_td >= 6) & (opp_atspct <= 0.35),
        "pt diff/gm <= -7 (4+ gp)": (t.gp_td >= 4) & (t.margin_td / t.gp_td <= -7),
        "pt diff/gm >= +7 (4+ gp)": (t.gp_td >= 4) & (t.margin_td / t.gp_td >= 7),
    },
    "last_season": {
        "made playoffs last yr": t.prev_season_playoffs == True,
        "win pct <= .35 last yr": t.prev_season_win_pct <= 0.35,
        "opp made playoffs last yr": t.opp_prev_season_win_pct >= 0.65,
    },
    "timing": {
        "week 1": t.week == 1, "weeks 1-4": early_season, "weeks 5-12": t.week.between(5, 12),
        "weeks 13+ (reg)": (t.week >= 13) & ~t.playoff, "final 2 reg weeks": ~t.playoff & (t.week >= t.groupby("season").week.transform(lambda w: w[~t.loc[w.index, "playoff"]].max()) - 1),
        "playoffs": t.playoff,
    },
    "slot": {
        "primetime": t.primetime, "1pm ET window": t.early_window, "late afternoon": ~t.primetime & ~t.early_window,
        "Thursday": t.weekday == "Thursday", "Monday": t.weekday == "Monday", "Saturday": t.weekday == "Saturday",
    },
    "matchup": {"division game": t.div_game.astype(bool), "non-division": ~t.div_game.astype(bool)},
    "env": {
        "outdoor <=35F": outdoor & (t.temp_f <= 35), "outdoor >=80F": outdoor & (t.temp_f >= 80),
        "wind >=15mph": outdoor & (t.wind_mph >= 15), "dome/closed": t.roof.isin(["dome", "closed"]),
        "total <=41": t.total_line <= 41, "total >=50": t.total_line >= 50,
    },
    "travel": {
        "West team, 1pm ET road game": away_nn & (t.team_tz == "P") & (t.home_tz == "E") & t.early_window,
        "East team at West coast": away_nn & (t.team_tz == "E") & (t.home_tz == "P"),
        "cross-country road trip": away_nn & (((t.team_tz == "E") & (t.home_tz == "P")) | ((t.team_tz == "P") & (t.home_tz == "E"))),
        "opp is West team, 1pm ET at us": home_nn & (t.opp.map(dict(zip(t.team, t.team_tz))) == "P") & (t.home_tz == "E") & t.early_window,
    },
    "qb": {"new starting QB vs last game": t.qb_changed.astype(bool), "opp new starting QB": t.opp_qb_changed.astype(bool)},
}
# travel map above via dict(zip) uses last-seen tz per team, fine since tz is static per franchise code.

names, groups, masks = [], [], []
for grp, d in ATOMS.items():
    for k, m in d.items():
        names.append(k); groups.append(grp); masks.append(m.fillna(False).to_numpy(bool))
M = np.vstack(masks)
cover = t.cover.to_numpy(float)
is_w = cover == 1; is_l = cover == 0; is_p = np.isnan(cover)
disc = (t.season <= DISC_END).to_numpy(); hold = ~disc
profit = t.profit.to_numpy()


def score(mask):
    w = int((mask & is_w).sum()); l = int((mask & is_l).sum()); n = w + l
    if n < MIN_N: return None
    wd = int((mask & is_w & disc).sum()); ld = int((mask & is_l & disc).sum())
    wh = int((mask & is_w & hold).sum()); lh = int((mask & is_l & hold).sum())
    return w, l, int((mask & is_p).sum()), wd, ld, wh, lh, float(np.nanmean(profit[mask]))


res = []
idx = range(len(names))
for k in (1, 2, 3):
    for combo in itertools.combinations(idx, k):
        if len({groups[i] for i in combo}) < k:
            continue
        m = M[list(combo)].all(axis=0)
        s = score(m)
        if s:
            res.append((" + ".join(names[i] for i in combo), k, *s))

ats = pd.DataFrame(res, columns=["setup", "k", "w", "l", "p", "w_disc", "l_disc", "w_hold", "l_hold", "roi_actual"])
ats["n"] = ats.w + ats.l
ats["pct"] = ats.w / ats.n
ats["pct_disc"] = ats.w_disc / (ats.w_disc + ats.l_disc)
ats["n_hold"] = ats.w_hold + ats.l_hold
ats["pct_hold"] = ats.w_hold / ats.n_hold
ats["pval"] = [stats.binomtest(w, n, .5).pvalue for w, n in zip(ats.w, ats.n)]
ats["pval_disc"] = [stats.binomtest(w, n, .5).pvalue for w, n in zip(ats.w_disc, ats.w_disc + ats.l_disc)]
ats["qval"] = multipletests(ats.pval, method="fdr_bh")[1]
ats["edge"] = (ats.pct - .5).abs()
ats["bet"] = np.where(ats.pct >= .5, "ON", "FADE")
ats["disc_side_ok"] = np.where(ats.pct_disc >= .5, ats.pct_disc >= BE, ats.pct_disc <= 1 - BE)
ats["hold_same_side"] = np.sign(ats.pct_disc - .5) == np.sign(ats.pct_hold - .5)
ats["hold_beats_vig"] = np.where(ats.pct_disc >= .5, ats.pct_hold >= BE, ats.pct_hold <= 1 - BE)
ats = ats.sort_values("pval")
ats.to_csv(OUT / "ats_scan.csv", index=False)

n_tests = len(ats)
summary["ats_scan"] = {
    "atoms": len(names), "combos_tested": n_tests,
    "by_k": ats.groupby("k").size().to_dict(),
    "p_lt_05": int((ats.pval < .05).sum()), "expected_by_chance_05": round(n_tests * .05),
    "p_lt_01": int((ats.pval < .01).sum()), "expected_by_chance_01": round(n_tests * .01),
    "q_lt_10": int((ats.qval < .10).sum()),
    "min_q": float(ats.qval.min()),
}

# Honest walk-forward: take top discovery-period setups, see what the holdout says.
wf = ats[(ats.w_disc + ats.l_disc >= 80) & (ats.n_hold >= 40)].sort_values("pval_disc").head(100).copy()
summary["ats_walkforward"] = {
    "top_n": len(wf),
    "held_same_side": int(wf.hold_same_side.sum()),
    "beat_vig_in_holdout": int(wf.hold_beats_vig.sum()),
    "avg_disc_edge": float((wf.pct_disc - .5).abs().mean()),
    "avg_hold_edge_same_dir": float((np.sign(wf.pct_disc - .5) * (wf.pct_hold - .5)).mean()),
}
wf.to_csv(OUT / "ats_walkforward_top100.csv", index=False)

cols = ["setup", "bet", "w", "l", "p", "pct", "pct_disc", "pct_hold", "n_hold", "pval", "qval", "roi_actual"]
summary["ats_singles"] = ats[ats.k == 1].sort_values("edge", ascending=False)[cols].round(4).to_dict("records")
summary["ats_top_full"] = ats.head(40)[cols].round(4).to_dict("records")
robust = ats[ats.disc_side_ok & ats.hold_beats_vig & (ats.n_hold >= 50)].sort_values("pval")
robust.to_csv(OUT / "ats_robust.csv", index=False)
summary["ats_robust"] = robust.head(40)[cols].round(4).to_dict("records")
summary["ats_robust_count"] = len(robust)

# Null check: shuffle covers within season to see how many "robust" setups pure noise produces.
rng = np.random.default_rng(7)
null_counts = []
seasons = t.season.to_numpy()
base_k = [(list(c)) for k in (1, 2) for c in itertools.combinations(idx, k) if len({groups[i] for i in c}) == k]
for rep in range(5):
    perm = cover.copy()
    for s_ in np.unique(seasons):
        ix = np.where(seasons == s_)[0]
        perm[ix] = rng.permutation(perm[ix])
    pw = perm == 1; pl = perm == 0
    cnt = 0
    for c in base_k:
        m = M[c].all(axis=0)
        wd = (m & pw & disc).sum(); ld = (m & pl & disc).sum(); wh = (m & pw & hold).sum(); lh = (m & pl & hold).sum()
        if wd + ld + wh + lh < MIN_N or wh + lh < 50: continue
        pd_, ph = wd / (wd + ld), wh / (wh + lh)
        if (pd_ >= BE and ph >= BE) or (pd_ <= 1 - BE and ph <= 1 - BE): cnt += 1
    null_counts.append(int(cnt))
real_k12 = int(((ats.k <= 2) & ats.disc_side_ok & ats.hold_beats_vig & (ats.n_hold >= 50)).sum())
summary["ats_null"] = {"real_robust_k1_2": real_k12, "shuffled_robust_k1_2": null_counts}

# ---------------------------------------------------------------- totals scan (game level)
gg = g.copy()
gg["over"] = gg.ou_result.map({"over": 1.0, "under": 0.0})
gdisc = (gg.season <= DISC_END).to_numpy()
out_ = gg.roof.isin(["outdoors", "open"])
tgh = tg[tg.side == "home"].set_index("game_id"); tga = tg[tg.side == "away"].set_index("game_id")
hp = tgh.reindex(gg.game_id); ap = tga.reindex(gg.game_id)
TL = gg.total_line
TATOMS = {
    "total": {"total <=38": TL <= 38, "total 38.5-41.5": TL.between(38.5, 41.5), "total 42-44.5": TL.between(42, 44.5),
              "total 45-47.5": TL.between(45, 47.5), "total 48-50.5": TL.between(48, 50.5), "total 51+": TL >= 51},
    "spread": {"spread <=3": gg.spread_line.abs() <= 3, "spread 3.5-7": gg.spread_line.abs().between(3.5, 7),
               "spread 7.5+": gg.spread_line.abs() >= 7.5, "spread 10+": gg.spread_line.abs() >= 10,
               "home fav": gg.spread_line > 0, "home dog": gg.spread_line < 0},
    "env": {"dome/closed": gg.roof.isin(["dome", "closed"]), "outdoor": out_,
            "wind >=10": out_ & (gg.wind_mph >= 10), "wind >=15": out_ & (gg.wind_mph >= 15), "wind >=20": out_ & (gg.wind_mph >= 20),
            "temp <=32": out_ & (gg.temp_f <= 32), "temp 33-45": out_ & gg.temp_f.between(33, 45), "temp >=80": out_ & (gg.temp_f >= 80),
            "rain": out_ & (gg.rain == True), "snow": out_ & (gg.snow == True)},
    "surface": {"grass": gg.surface.str.contains("grass", case=False, na=False), "turf": ~gg.surface.str.contains("grass", case=False, na=True)},
    "slot": {"primetime": gg.primetime, "Thursday": gg.weekday == "Thursday", "Monday": gg.weekday == "Monday",
             "SNF (Sun night)": (gg.weekday == "Sunday") & gg.primetime, "1pm ET": gg.early_window, "Saturday": gg.weekday == "Saturday"},
    "timing": {"week 1": gg.week == 1, "weeks 1-4": gg.week <= 4, "weeks 5-12": gg.week.between(5, 12),
               "weeks 13+ reg": (gg.week >= 13) & ~gg.playoff, "playoffs": gg.playoff},
    "matchup": {"division": gg.div_game.astype(bool), "non-division": ~gg.div_game.astype(bool)},
    "form": {"both off over": (hp.prev_ou.values == "over") & (ap.prev_ou.values == "over"),
             "both off under": (hp.prev_ou.values == "under") & (ap.prev_ou.values == "under"),
             "home off bye": hp.rest.values >= 13, "away off bye": ap.rest.values >= 13,
             "either team new QB": hp.qb_changed.fillna(False).values.astype(bool) | ap.qb_changed.fillna(False).values.astype(bool)},
    "era": {},
}
tn, tgp, tm = [], [], []
for grp, d in TATOMS.items():
    for k, m in d.items():
        tn.append(k); tgp.append(grp); tm.append(pd.Series(m).fillna(False).to_numpy(bool))
TM = np.vstack(tm)
ov = gg.over.to_numpy(float); ow = ov == 1; ol = ov == 0
oprofit = np.where(ov == 1, np.where(gg.over_odds < 0, 100 / -gg.over_odds, gg.over_odds / 100), np.where(ov == 0, -1.0, 0.0))
tres = []
for k in (1, 2, 3):
    for combo in itertools.combinations(range(len(tn)), k):
        if len({tgp[i] for i in combo}) < k: continue
        m = TM[list(combo)].all(axis=0)
        w = int((m & ow).sum()); l = int((m & ol).sum())
        if w + l < 80: continue
        wd = int((m & ow & gdisc).sum()); ld = int((m & ol & gdisc).sum())
        tres.append((" + ".join(tn[i] for i in combo), k, w, l, wd, ld, w - wd, l - ld, float(np.nanmean(oprofit[m]))))
tot = pd.DataFrame(tres, columns=["setup", "k", "over", "under", "o_disc", "u_disc", "o_hold", "u_hold", "roi_over_actual"])
tot["n"] = tot.over + tot.under
tot["over_pct"] = tot.over / tot.n
tot["pct_disc"] = tot.o_disc / (tot.o_disc + tot.u_disc)
tot["n_hold"] = tot.o_hold + tot.u_hold
tot["pct_hold"] = tot.o_hold / tot.n_hold
tot["pval"] = [stats.binomtest(w, n, .5).pvalue for w, n in zip(tot.over, tot.n)]
tot["pval_disc"] = [stats.binomtest(w, n, .5).pvalue for w, n in zip(tot.o_disc, tot.o_disc + tot.u_disc)]
tot["qval"] = multipletests(tot.pval, method="fdr_bh")[1]
tot["bet"] = np.where(tot.over_pct >= .5, "OVER", "UNDER")
tot["disc_side_ok"] = np.where(tot.pct_disc >= .5, tot.pct_disc >= BE, tot.pct_disc <= 1 - BE)
tot["hold_beats_vig"] = np.where(tot.pct_disc >= .5, tot.pct_hold >= BE, tot.pct_hold <= 1 - BE)
tot = tot.sort_values("pval")
tot.to_csv(OUT / "totals_scan.csv", index=False)
tcols = ["setup", "bet", "over", "under", "over_pct", "pct_disc", "pct_hold", "n_hold", "pval", "qval", "roi_over_actual"]
summary["totals_scan"] = {"combos_tested": len(tot), "p_lt_05": int((tot.pval < .05).sum()),
                          "expected_by_chance_05": round(len(tot) * .05), "q_lt_10": int((tot.qval < .10).sum())}
summary["totals_singles"] = tot[tot.k == 1].sort_values("pval")[tcols].round(4).to_dict("records")
summary["totals_top"] = tot.head(40)[tcols].round(4).to_dict("records")
trob = tot[tot.disc_side_ok & tot.hold_beats_vig & (tot.n_hold >= 40)].sort_values("pval")
trob.to_csv(OUT / "totals_robust.csv", index=False)
summary["totals_robust"] = trob.head(30)[tcols].round(4).to_dict("records")
summary["totals_robust_count"] = len(trob)
twf = tot[(tot.o_disc + tot.u_disc >= 60) & (tot.n_hold >= 30)].sort_values("pval_disc").head(100)
summary["totals_walkforward"] = {"top_n": len(twf), "held_same_side": int((np.sign(twf.pct_disc - .5) == np.sign(twf.pct_hold - .5)).sum()),
                                 "beat_vig_in_holdout": int(twf.hold_beats_vig.sum())}

# ---------------------------------------------------------------- referees (totals) & coaches (ATS)
ref = (gg.groupby("referee").agg(n=("over", "count"), over_pct=("over", "mean"), avg_ou_margin=("ou_margin", "mean"),
                                 first=("season", "min"), last=("season", "max")).query("n >= 80").reset_index())
ref["pval"] = [stats.binomtest(int(round(p * n)), n, .5).pvalue for p, n in zip(ref.over_pct, ref.n)]
ref = ref.sort_values("over_pct")
ref.to_csv(OUT / "referees_totals.csv", index=False)
summary["referees"] = ref.round(4).to_dict("records")
summary["referees_qmin"] = float(multipletests(ref.pval, method="fdr_bh")[1].min())

coach = (tg.groupby("coach").agg(n=("cover", "count"), cover_pct=("cover", "mean"), dog_n=("dog", "sum"),
                                 su_pct=("su_win", "mean"), roi=("profit", "mean")).query("n >= 80").reset_index())
coach_dog = tg[tg.dog].groupby("coach").agg(dog_ats_n=("cover", "count"), dog_cover_pct=("cover", "mean"))
coach = coach.merge(coach_dog, on="coach", how="left")
coach["pval"] = [stats.binomtest(int(round(p * n)), n, .5).pvalue for p, n in zip(coach.cover_pct, coach.n)]
coach["qval"] = multipletests(coach.pval, method="fdr_bh")[1]
coach = coach.sort_values("cover_pct", ascending=False)
coach.to_csv(OUT / "coaches_ats.csv", index=False)
summary["coaches"] = coach.round(4).to_dict("records")

team = (tg.groupby("team").agg(n=("cover", "count"), cover_pct=("cover", "mean"), roi=("profit", "mean")).reset_index())
team["pval"] = [stats.binomtest(int(round(p * n)), n, .5).pvalue for p, n in zip(team.cover_pct, team.n)]
summary["teams"] = team.sort_values("cover_pct", ascending=False).round(4).to_dict("records")

# ---------------------------------------------------------------- calibration: is the spread unbiased?
cal = (home_g.assign(b=pd.cut(home_g.spread_line, [-30, -10, -7, -4, -1, 1, 4, 7, 10, 30]))
       .groupby("b", observed=True).agg(n=("result", "size"), avg_line=("spread_line", "mean"), avg_result=("result", "mean"),
                                         home_cover=("ats_winner", lambda s: (s == "home").sum() / (s != "push").sum())).reset_index())
cal["b"] = cal.b.astype(str)
summary["calibration"] = cal.round(3).to_dict("records")
tcal = (gg.assign(b=pd.cut(gg.total_line, [0, 38, 41.5, 44.5, 47.5, 50.5, 70]))
        .groupby("b", observed=True).agg(n=("total", "size"), avg_line=("total_line", "mean"), avg_total=("total", "mean"),
                                          over_pct=("over", "mean")).reset_index())
tcal["b"] = tcal.b.astype(str)
summary["total_calibration"] = tcal.round(3).to_dict("records")

(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(json.dumps({k: summary[k] for k in ["overall", "ats_scan", "ats_walkforward", "ats_null", "ats_robust_count",
                                          "totals_scan", "totals_walkforward", "totals_robust_count"]}, indent=1, default=str))
