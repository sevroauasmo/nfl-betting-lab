"""Render results/ into a single self-contained HTML report: results/report.html."""
import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
s = json.loads((R / "summary.json").read_text())
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)

wind = con.execute("""
    SELECT CASE WHEN wind_mph<5 THEN '0–4' WHEN wind_mph<10 THEN '5–9' WHEN wind_mph<15 THEN '10–14'
                WHEN wind_mph<20 THEN '15–19' ELSE '20+' END AS bucket,
           count(*) n, avg(total_line) line, avg(total) pts,
           sum((ou_result='under')::int)/sum((ou_result<>'push')::int) under_pct,
           sum((ou_result='under' AND season<=2019)::int)/sum((ou_result<>'push' AND season<=2019)::int) u1,
           sum((ou_result='under' AND season>=2020)::int)/sum((ou_result<>'push' AND season>=2020)::int) u2
    FROM games WHERE season BETWEEN 2011 AND 2025 AND roof IN ('outdoors','open') AND wind_mph IS NOT NULL
    GROUP BY 1 ORDER BY min(wind_mph)""").df()
wind_seasons = con.execute("""
    SELECT season, count(*) n, avg((ou_result='under')::int) u FROM games
    WHERE season BETWEEN 2011 AND 2025 AND roof IN ('outdoors','open') AND wind_mph BETWEEN 10 AND 19 AND ou_result<>'push'
    GROUP BY 1 ORDER BY 1""").df()

ats = pd.read_csv(R / "ats_scan.csv")
tot = pd.read_csv(R / "totals_scan.csv")
keep_a = ats[(ats.k <= 2) | (ats.pval < 0.03)]
keep_t = tot[(tot.k <= 2) | (tot.pval < 0.10)]


def rows(df, win, loss, pct):
    out = []
    for r in df.itertuples():
        out.append([r.setup, int(r.k), getattr(r, win), getattr(r, loss), round(getattr(r, pct), 4),
                    round(r.pct_disc, 4), round(r.pct_hold, 4), int(r.n_hold), float(f"{r.pval:.3g}"),
                    float(f"{r.qval:.3g}"), bool(r.disc_side_ok and r.hold_beats_vig)])
    return out


data = {
    "overall": s["overall"], "seasons": s["baseline_by_season"],
    "spread": [r for r in s["spread_bucket_by_side"] if r["role"] in ("home fav", "away fav")],
    "wind": wind.round(4).to_dict("records"), "wind_seasons": wind_seasons.round(4).to_dict("records"),
    "teasers": s["teasers"], "ml": s["moneyline_by_price"],
    "ats_scan": s["ats_scan"], "ats_wf": s["ats_walkforward"], "ats_null": s["ats_null"],
    "tot_scan": s["totals_scan"], "tot_wf": s["totals_walkforward"],
    "ats_rows": rows(keep_a, "w", "l", "pct"), "tot_rows": rows(keep_t, "over", "under", "over_pct"),
    "refs_qmin": s["referees_qmin"], "coaches": s["coaches"][:6] + s["coaches"][-4:],
}
R2 = json.loads((R / "round2.json").read_text())


def e(d):  # eras dict -> {era: [w, l, pct]}
    return {k: [v["w"], v["l"], v["pct"]] for k, v in d.items()}


sm = R2["spread_models"]["EPA + pt diff"]
sn = R2["spread_models"]["EPA/play (neutral WP)"]
tm = R2["total_models"]["all + dome + wind"]
hp = R2["half_point_value"]
data["r2"] = [
    {"g": "Spread models", "t": "EPA + point-differential power ratings vs closing spread",
     "d": f"The model misses the final margin by {sm['mae_model']} pts on average vs {sm['mae_line']} for the line (2020–25). Betting 3+ pt disagreements: {sm['holdout_bets']['edge>=3']['w']}–{sm['holdout_bets']['edge>=3']['l']}.", "v": "noise"},
    {"g": "Spread models", "t": "EPA/play only (neutral game states)",
     "d": f"Betting 3+ pt edges in 2020–25: {sn['holdout_bets']['edge>=3']['w']}–{sn['holdout_bets']['edge>=3']['l']}. The model adds nothing to the close (slope p = {sn['p_hold']}).", "v": "noise"},
    {"g": "Totals models", "t": "Scoring/EPA/pace ratings + dome + wind vs closing total",
     "d": f"Big model unders (4+ pts) went {tm['holdout_bets']['under edge>=4']['w']}–{tm['holdout_bets']['under edge>=4']['l']} in 2020–25, but nearly all of that is the wind term. In calm or dome games the model has no edge.", "v": "lean"},
    {"g": "Luck", "t": "Last game: scoreboard trailed EPA by 10+ (unlucky)", "e": e(R2["luck"]["last game: scoreboard trailed EPA by 10+ (unlucky)"]),
     "d": "About 53% ATS in both eras, which doesn't clear −110. The 'lucky' side flipped between eras.", "v": "noise"},
    {"g": "Luck", "t": "Turnover margin ≥ +1.0/gm, season to date", "e": e(R2["luck"]["turnover margin >= +1.0/gm (4+ gp)"]),
     "d": "The market already regresses turnover luck.", "v": "noise"},
    {"g": "Luck", "t": "One-score record ≤ .25 (4+ such games)", "e": e(R2["luck"]["one-score record <= .25 (4+ such games)"]),
     "d": "Priced in both directions.", "v": "noise"},
    {"g": "Overreaction", "t": "ATS margin autocorrelation",
     "d": f"Correlation between one game's ATS margin and the next: {R2['ats_autocorr']['prev_ats']['corr']} (n = {R2['ats_autocorr']['prev_ats']['n']}). No momentum and no mean reversion.", "v": "noise"},
    {"g": "Overreaction", "t": "Record much better than EPA says (fade)", "e": e(R2["record_vs_epa"]["record much better than EPA (top gap 15%)"]),
     "d": "Teams whose record outruns their efficiency cover at a coin-flip rate.", "v": "noise"},
    {"g": "Team totals", "t": "Favorites with implied team total ≤ 20",
     "d": "In very low-total games, favorites scored under their implied team total 68% of the time (n = 106), in both eras (73% / 64%). Implied TT = (total + spread) / 2, so check actual team-total lines.", "v": "lean"},
    {"g": "Totals", "t": "League scoring trend (after 2 weeks of overs by 3+)", "e": e(R2["scoring_lag"]["after 2 wks of overs by 3+ -> over%"]),
     "d": "Totals adjust to league-wide scoring swings within a week.", "v": "noise"},
    {"g": "QB", "t": "Team starting a non-primary QB", "e": e(R2["backup_qb"]["team starting non-primary QB (ATS)"]),
     "d": f"{R2['backup_qb']['n_backup_starts']} starts. Backups underperformed in 2011–19 but not since. Lines now adjust correctly.", "v": "noise"},
    {"g": "Prices", "t": "Dog +10: spread vs moneyline",
     "d": "Dogs of exactly +10 covered 60% (n = 119) but won outright only 17.6% (ML ROI −16%). Take the points, not the price.", "v": "lean"},
    {"g": "Prices", "t": "Key number 3",
     "d": f"{hp['land exactly on 3']*100:.1f}% of all games end on exactly 3, and {hp['closing line = 3: margin exactly 3']*100:.1f}% when the line is 3. Going from +2.5 to +3 turns the 7.9% of games the favorite wins by exactly 3 from losses into pushes. At historical cover rates, +3 is worth about −125, so a half-point costing more than about 15 cents isn't worth it. Laying −3 instead of −3.5 saves the 9.2% of games decided by exactly 3.", "v": "info"},
    {"g": "Situational", "t": "Warm-weather team on the road in ≤ 35°F", "e": e(R2["cold_weather"]["warm-weather team, road, outdoor <=35F"]),
     "d": "Same direction in both eras, but only 90 games.", "v": "noise"},
    {"g": "Situational", "t": "Divisional rematch after losing the first meeting", "e": e(R2["div_rematch"]["lost 1st meeting"]),
     "d": "Similar in both eras but doesn't clear −110.", "v": "noise"},
    {"g": "Situational", "t": "Weeks 1–4, team won ≥ 70% last season", "e": e(R2["last_year_anchor"]["wk1-4, last yr >=.70 (ATS)"]),
     "d": f"The market doesn't over-anchor on last year (regression p = {R2['last_year_anchor']['weeks 1-4 p']}).", "v": "noise"},
]

PR = json.loads((R / "props.json").read_text())
data["props"] = {
    "coverage": PR["coverage"],
    "bias": [{"market": r["market"].replace("player_", "").replace("_", " "), "props": r["props"], "over": r["over_pct"],
              "fair": r["fair_over"], "seasons": r["by_season"], "under_best": r["under_best_price_roi"]["roi"],
              "over_best": r["over_best_price_roi"]["roi"]} for r in PR["bias_close"]],
    "all": {"over": PR["bias_all_core"]["over_pct"]["pct"], "fair": PR["bias_all_core"]["fair_over"], "n": PR["bias_all_core"]["props"],
            "under_by_season": {k: v["roi"] for k, v in PR["bias_all_core"]["by_season_under_best_roi"].items()}},
    "ev_consensus": {k: {**v["all"], "seasons": {s: x["roi"] for s, x in v["by_season"].items()}} for k, v in PR["ev_same_point"].items()},
    "ev_exchange": {k: {**v["all"], "seasons": v["by_season"]} for k, v in PR["ev_vs_exchange"].items()},
    "ev_side": PR["ev_vs_exchange_side"], "ev_age": PR["ev_vs_exchange_by_quote_age"],
    "ev_market": sorted([r for r in PR["ev_vs_exchange_by_market"] if r["n"] >= 75], key=lambda r: -r["n"]),
    "movement": PR["movement"], "td": PR["anytime_td"],
    "proj": [r for r in PR["naive_projection"] if r["proj"] == "mean8"],
    "exch_cov": PR["exchange_coverage"], "exch_cal": PR["exchange_calibration"],
}

html = (ROOT / "scripts/report_template.html").read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")))
(R / "report.html").write_text(html)
print(f"wrote {R/'report.html'} ({len(html)/1e3:.0f} KB)")
