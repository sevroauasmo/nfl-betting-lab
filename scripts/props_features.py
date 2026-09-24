"""Build a feature table for every graded prop (consensus close line), using only pre-game information.

Output: data/props_features.parquet — one row per (game, market, player) at the close, with:
  market:   cons_point, fair_over (de-vigged consensus), best prices, n_books, open->close line move, exchange gap
  player:   last1/3/8 averages vs line, L8 hit rate vs today's line, volatility, season-to-date, snap %, target/carry share,
            previous prop results in this market, own injury designation + practice status
  team:     spread, total, implied team total, fav/dog, home, record, vacated target/carry share from teammates ruled out
  opponent: stat allowed to this position (season-to-date, vs league), pass/rush EPA allowed, yards/target allowed (PFR)
  game:     division, primetime, roof, temp, wind, week
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
EXCH = {"novig", "prophetx", "kalshi", "polymarket", "betopenly"}
STAT = {"player_receptions": "receptions", "player_reception_yds": "receiving_yards", "player_rush_yds": "rushing_yards",
        "player_rush_attempts": "carries", "player_pass_yds": "passing_yards", "player_pass_completions": "completions",
        "player_pass_attempts": "attempts", "player_pass_tds": "passing_tds", "player_pass_interceptions": "passing_interceptions",
        "player_rush_reception_yds": "rush_rec_yds", "player_tackles_assists": "tackles_combined", "player_kicking_points": "kicking_points"}


def imp(o):
    o = np.asarray(o, float)
    return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))


# ------------------------------------------------------------------ consensus lines per pass
tw = con.execute(f"""
    SELECT r.game_id, r.pass, r.bookmaker, r.market, r.player_key, r.point, r.over_price, r.under_price, r.actual,
           pm.player_id
    FROM prop_results r LEFT JOIN prop_player_map pm USING (game_id, player_key)
    WHERE r.over_price IS NOT NULL AND r.under_price IS NOT NULL AND r.point IS NOT NULL
      AND r.market IN ({",".join("'" + m + "'" for m in STAT)}) AND r.pass IN ('open', 'day_before', 'close')
""").df()
tw["po"], tw["pu"] = imp(tw.over_price), imp(tw.under_price)
tw = tw[(tw.po + tw.pu - 1).between(-0.02, 0.25)]
tw["fair"] = tw.po / (tw.po + tw.pu)
tw["exch"] = tw.bookmaker.isin(EXCH)
key = ["game_id", "pass", "market", "player_key"]
tw["dist"] = (tw.fair - .5).abs()
main = tw.sort_values("dist").drop_duplicates(key + ["bookmaker"])
cons = main[~main.exch].groupby(key).agg(cons_point=("point", "median"), n_books=("bookmaker", "nunique"),
                                         actual=("actual", "first"), player_id=("player_id", "first")).reset_index()
cons = cons[cons.n_books >= 3]
at = main.merge(cons[key + ["cons_point"]], on=key)
at = at[at.point == at.cons_point]
fair = at[~at.exch].groupby(key).agg(fair_over=("fair", "mean"), best_over=("over_price", "max"), best_under=("under_price", "max")).reset_index()
exf = at[at.exch].groupby(key).fair.mean().rename("exch_over").reset_index()
P = cons.merge(fair, on=key).merge(exf, on=key, how="left")
close = P[P["pass"] == "close"].drop(columns="pass")
for ps in ("open", "day_before"):
    o = P[P["pass"] == ps][["game_id", "market", "player_key", "cons_point", "fair_over"]].rename(
        columns={"cons_point": f"point_{ps}", "fair_over": f"fair_{ps}"})
    close = close.merge(o, on=["game_id", "market", "player_key"], how="left")
F = close.dropna(subset=["actual", "player_id"]).copy()
F["over"] = np.where(F.actual > F.cons_point, 1.0, np.where(F.actual < F.cons_point, 0.0, np.nan))
F["line_move"] = F.cons_point - F.point_open
F["line_move_rel"] = F.line_move / F.point_open.clip(lower=1)
F["exch_gap"] = F.exch_over - F.fair_over
print("props:", len(F))

# ------------------------------------------------------------------ player weekly history (2021+)
h = con.execute("""
    SELECT s.player_id, s.season, s.week, s.game_id, s.team, s.opponent_team, s.position, s.position_group,
           s.receptions, s.receiving_yards, s.rushing_yards, s.carries, s.passing_yards, s.completions, s.attempts,
           s.passing_tds, s.passing_interceptions, s.targets, s.target_share, s.air_yards_share,
           s.rushing_yards + s.receiving_yards AS rush_rec_yds,
           coalesce(p.def_tackles_combined, s.def_tackles_solo + s.def_tackles_with_assist + s.def_tackle_assists) AS tackles_combined,
           3 * coalesce(s.fg_made, 0) + coalesce(s.pat_made, 0) AS kicking_points,
           sc.offense_pct
    FROM stats_player_week s
    LEFT JOIN pfr_adv_week_def p ON p.game_id = s.game_id AND lower(p.pfr_player_name) = lower(s.player_display_name)
    LEFT JOIN players pl ON pl.gsis_id = s.player_id
    LEFT JOIN snap_counts sc ON sc.game_id = s.game_id AND sc.pfr_player_id = pl.pfr_id
    WHERE s.season >= 2021 AND s.season_type IN ('REG', 'POST')
""").df()
h = h.sort_values(["player_id", "season", "week"])
team_carries = h.groupby("game_id").carries.transform("sum").where(lambda x: x > 0)
h["carry_share"] = h.carries / h.groupby(["game_id", "team"]).carries.transform("sum").replace(0, np.nan)
g = h.groupby("player_id")
for c in set(STAT.values()):
    s = g[c].shift()
    h[f"{c}_l1"] = s
    h[f"{c}_m3"] = g[c].transform(lambda x: x.shift().rolling(3, min_periods=2).mean())
    h[f"{c}_m8"] = g[c].transform(lambda x: x.shift().rolling(8, min_periods=3).mean())
    h[f"{c}_sd8"] = g[c].transform(lambda x: x.shift().rolling(8, min_periods=4).std())
    for k in range(1, 9):
        h[f"{c}_lag{k}"] = g[c].shift(k)
for c in ("target_share", "air_yards_share", "offense_pct", "carry_share"):
    h[f"{c}_m3"] = g[c].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
    h[f"{c}_l1"] = g[c].shift()
h["season_gp"] = h.groupby(["player_id", "season"]).cumcount()

F = F.merge(h, on=["player_id", "game_id"], how="left", suffixes=("", "_h"))
F = F.dropna(subset=["season"])
F["stat"] = F.market.map(STAT)
pick = lambda suffix: F.apply(lambda r: r.get(f"{r.stat}{suffix}"), axis=1)  # noqa: E731
for sfx in ("_l1", "_m3", "_m8", "_sd8"):
    F[f"p{sfx}"] = pick(sfx).astype(float)
lags = np.column_stack([pick(f"_lag{k}").astype(float).values for k in range(1, 9)])
F["hit_rate_l8"] = np.nansum(lags > F.cons_point.values[:, None], axis=1) / np.maximum((~np.isnan(lags)).sum(axis=1), 1)
F.loc[(~np.isnan(lags)).sum(axis=1) < 4, "hit_rate_l8"] = np.nan
F["l1_vs_line"] = F.p_l1 / F.cons_point.clip(lower=0.5)
F["m3_vs_line"] = F.p_m3 / F.cons_point.clip(lower=0.5)
F["m8_vs_line"] = F.p_m8 / F.cons_point.clip(lower=0.5)
F["hot"] = F.p_m3 / F.p_m8.replace(0, np.nan)          # recent vs longer form
F["cv8"] = F.p_sd8 / F.p_m8.replace(0, np.nan)         # volatility
F = F.drop(columns=[c for c in F.columns if any(c.endswith(f"_lag{k}") for k in range(1, 9))])

# previous prop results for this player+market
F = F.sort_values(["player_key", "market", "season", "week"])
gp = F.groupby(["player_id", "market"])
F["prev_over"] = gp.over.shift()
F["prev_margin_rel"] = gp.apply(lambda d: ((d.actual - d.cons_point) / d.cons_point.clip(lower=0.5)).shift()).reset_index(level=[0, 1], drop=True)
F["over_rate_l5"] = gp.over.transform(lambda x: x.shift().rolling(5, min_periods=3).mean())
F["props_seen"] = gp.cumcount()

# ------------------------------------------------------------------ game / team context
tg = con.execute("""SELECT game_id, team, opp, side, neutral, team_line, total_line, div_game, primetime, weekday, roof,
                           temp_f, wind_mph, su_w_td, gp_td, opp_su_w_td, opp_gp_td, playoff FROM team_games""").df()
F = F.merge(tg, on=["game_id", "team"], how="left")
F["home"] = (F.side == "home").astype(float)
F["implied_tt"] = (F.total_line + F.team_line) / 2
F["win_pct"] = F.su_w_td / F.gp_td.replace(0, np.nan)
F["opp_win_pct"] = F.opp_su_w_td / F.opp_gp_td.replace(0, np.nan)
F["dome"] = F.roof.isin(["dome", "closed"]).astype(float)

# opponent: stat allowed to this position group, season-to-date per game, relative to league
allowed = (h.groupby(["season", "week", "game_id", "opponent_team", "position_group"])[list(set(STAT.values()))].sum().reset_index()
           .sort_values(["season", "week"]))
ga = allowed.groupby(["season", "opponent_team", "position_group"])
for c in set(STAT.values()):
    allowed[f"allow_{c}"] = ga[c].transform(lambda x: x.shift().expanding().mean())
    lg = allowed.groupby(["season", "week", "position_group"])[f"allow_{c}"].transform("mean")
    allowed[f"allow_rel_{c}"] = allowed[f"allow_{c}"] / lg.replace(0, np.nan)
allowed = allowed.rename(columns={"opponent_team": "opp"})[["game_id", "opp", "position_group"] + [f"allow_rel_{c}" for c in set(STAT.values())]]
F = F.merge(allowed, on=["game_id", "opp", "position_group"], how="left")
F["opp_allow_rel"] = F.apply(lambda r: r.get(f"allow_rel_{r.stat}"), axis=1).astype(float)
F = F.drop(columns=[c for c in F.columns if c.startswith("allow_rel_")])

# opponent EPA allowed (pass / rush), season-to-date
ep = con.execute("""SELECT e.game_id, e.season, g.week, e.team AS offense, CASE WHEN e.team = g.home_team THEN g.away_team ELSE g.home_team END AS defense,
                           e.pass_epa, e.rush_epa FROM team_game_epa e JOIN games g USING (game_id) WHERE e.season >= 2021""").df()
fr = {"SD": "LAC", "OAK": "LV", "STL": "LA"}
ep = ep.sort_values(["season", "week"])
gd = ep.groupby(["season", "defense"])
ep["opp_pass_epa_allowed"] = gd.pass_epa.transform(lambda x: x.shift().expanding(min_periods=2).mean())
ep["opp_rush_epa_allowed"] = gd.rush_epa.transform(lambda x: x.shift().expanding(min_periods=2).mean())
F = F.merge(ep[["game_id", "defense", "opp_pass_epa_allowed", "opp_rush_epa_allowed"]].rename(columns={"defense": "opp"}),
            on=["game_id", "opp"], how="left")
# coverage proxy from PFR per-defender charting: yards per target allowed, season-to-date
cov = con.execute("""SELECT season, week, team AS opp, sum(def_yards_allowed) y, sum(def_targets) t
                     FROM pfr_adv_week_def WHERE season >= 2021 GROUP BY ALL ORDER BY season, week""").df()
gc = cov.groupby(["season", "opp"])
cov["opp_ypt_allowed"] = gc.y.transform(lambda x: x.shift().expanding(min_periods=2).sum()) / gc.t.transform(lambda x: x.shift().expanding(min_periods=2).sum())
F = F.merge(cov[["season", "week", "opp", "opp_ypt_allowed"]], on=["season", "week", "opp"], how="left")

# ------------------------------------------------------------------ injuries: own status + vacated usage from teammates ruled out
inj = con.execute("""SELECT CAST(season AS INT) AS season, CAST(week AS INT) AS week, gsis_id AS player_id, team, report_status, practice_status
                     FROM injuries WHERE season >= 2023""").df()
F = F.merge(inj[["season", "week", "player_id", "report_status", "practice_status"]], on=["season", "week", "player_id"], how="left")
F["questionable"] = (F.report_status == "Questionable").astype(float)
F["limited_or_dnp"] = F.practice_status.fillna("").str.contains("Limited|Did Not").astype(float)
out = inj[inj.report_status.isin(["Out", "Doubtful"])][["season", "week", "player_id", "team"]]
share = h[["player_id", "season", "week", "target_share_m3", "carry_share_m3"]]
vac = out.merge(share, on=["player_id", "season", "week"], how="left").groupby(["season", "week", "team"]).agg(
    vacated_target_share=("target_share_m3", "sum"), vacated_carry_share=("carry_share_m3", "sum")).reset_index()
F = F.merge(vac, on=["season", "week", "team"], how="left")
F[["vacated_target_share", "vacated_carry_share"]] = F[["vacated_target_share", "vacated_carry_share"]].fillna(0)

F["season"] = F.season.astype(int)
F = F.drop(columns=[c for c in F.columns if c.endswith("_h")])
keep = [c for c in F.columns if not (c in set(STAT.values()) or any(c.startswith(v + "_") for v in set(STAT.values())))]
F = F[keep]
F.to_parquet(ROOT / "data/props_features.parquet")
print(F.shape)
print(F.isna().mean().round(2).sort_values().to_string())
