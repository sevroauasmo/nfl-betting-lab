"""Feature table for every WR/TE/RB game since 2021 (not just games with props), pre-game information only.

Target columns: receiving_yards, rushing_yards (same-game, never used as features).
Every feature is either rolled from earlier games (shift / merge_asof with allow_exact_matches=False) or known before
kickoff (game line, schedule, injury report). FAMILY maps each feature to a group for SHAP roll-ups.
Output: data/player_games.parquet, data/player_feature_families.json
"""
import json
from pathlib import Path
from zoneinfo import ZoneInfo  # noqa: F401  (kept for parity with other scripts)

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
FAMILY: dict = {}


def fam(name, cols):
    for c in cols:
        FAMILY[c] = name


def asof(left, right, cols, by):
    right = right[[*([by] if isinstance(by, str) else by), "t"] + cols].sort_values("t")
    return pd.merge_asof(left.sort_values("t"), right, on="t", by=by, allow_exact_matches=False)


def T(df):
    return (df.season.astype(int) * 100 + df.week.astype(int)).astype("int64")


# ------------------------------------------------------------------ base rows + targets
B = con.execute("""
    SELECT s.player_id, s.player_display_name AS player, s.position, s.season, s.week, s.game_id, s.team, s.opponent_team AS opp,
           s.receiving_yards, s.rushing_yards, s.receptions, s.targets, s.carries, s.target_share, s.air_yards_share, s.wopr,
           s.receiving_air_yards, s.receiving_yards_after_catch, s.receiving_epa, s.rushing_epa, s.receiving_first_downs, s.rushing_first_downs
    FROM stats_player_week s
    WHERE s.season >= 2021 AND s.season_type IN ('REG', 'POST') AND s.position IN ('WR', 'TE', 'RB')
""").df()
B["t"] = T(B)
B = B.sort_values(["player_id", "t"]).reset_index(drop=True)
print("base rows", len(B))

# ------------------------------------------------------------------ form: rolling production (prior games only)
g = B.groupby("player_id")
FORM = []
for c in ("receiving_yards", "rushing_yards", "receptions", "targets", "carries", "receiving_air_yards", "receiving_yards_after_catch",
          "receiving_epa", "rushing_epa", "receiving_first_downs", "rushing_first_downs"):
    for w_ in (1, 3, 8):
        n = f"{c}_m{w_}"
        B[n] = g[c].transform(lambda x: x.shift().rolling(w_, min_periods=1 if w_ == 1 else 2).mean()); FORM.append(n)
    B[f"{c}_ewm"] = g[c].transform(lambda x: x.shift().ewm(halflife=4, min_periods=2).mean()); FORM.append(f"{c}_ewm")
for c in ("receiving_yards", "rushing_yards"):
    B[f"{c}_sd8"] = g[c].transform(lambda x: x.shift().rolling(8, min_periods=4).std())
    B[f"{c}_max8"] = g[c].transform(lambda x: x.shift().rolling(8, min_periods=3).max())
    B[f"{c}_med8"] = g[c].transform(lambda x: x.shift().rolling(8, min_periods=3).median())
    B[f"{c}_m24"] = g[c].transform(lambda x: x.shift().rolling(24, min_periods=6).mean())
    FORM += [f"{c}_sd8", f"{c}_max8", f"{c}_med8", f"{c}_m24"]
B["games_prior"] = g.cumcount()
B["season_gp"] = B.groupby(["player_id", "season"]).cumcount()
fam("form", FORM + ["games_prior", "season_gp"])

# efficiency
B["ypt_m8"] = g.receiving_yards.transform(lambda x: x.shift().rolling(8, min_periods=3).sum()) / g.targets.transform(lambda x: x.shift().rolling(8, min_periods=3).sum()).replace(0, np.nan)
B["ypc_m8"] = g.rushing_yards.transform(lambda x: x.shift().rolling(8, min_periods=3).sum()) / g.carries.transform(lambda x: x.shift().rolling(8, min_periods=3).sum()).replace(0, np.nan)
B["catch_rate_m8"] = g.receptions.transform(lambda x: x.shift().rolling(8, min_periods=3).sum()) / g.targets.transform(lambda x: x.shift().rolling(8, min_periods=3).sum()).replace(0, np.nan)
fam("efficiency", ["ypt_m8", "ypc_m8", "catch_rate_m8"])

# usage shares
for c in ("target_share", "air_yards_share", "wopr"):
    B[f"{c}_m3"] = g[c].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
    B[f"{c}_m8"] = g[c].transform(lambda x: x.shift().rolling(8, min_periods=2).mean())
team_carries = B.groupby(["game_id", "team"]).carries.transform("sum").replace(0, np.nan)
B["carry_share"] = B.carries / team_carries
B["carry_share_m3"] = g.carry_share.transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
B["carry_share_m8"] = g.carry_share.transform(lambda x: x.shift().rolling(8, min_periods=2).mean())
USAGE = [f"{c}_{w_}" for c in ("target_share", "air_yards_share", "wopr") for w_ in ("m3", "m8")] + ["carry_share_m3", "carry_share_m8"]

snaps = con.execute("""SELECT sc.season, sc.week, pl.gsis_id AS player_id, sc.offense_pct FROM snap_counts sc
                       JOIN players pl ON pl.pfr_id = sc.pfr_player_id WHERE sc.season >= 2021""").df()
snaps["t"] = T(snaps); snaps = snaps.sort_values(["player_id", "t"])
snaps["snap_pct_m3"] = snaps.groupby("player_id").offense_pct.transform(lambda x: x.rolling(3, min_periods=1).mean())
snaps["snap_pct_l1"] = snaps.offense_pct
B = asof(B, snaps, ["snap_pct_m3", "snap_pct_l1"], "player_id")
USAGE += ["snap_pct_m3", "snap_pct_l1"]

# routes / coverage splits from participation (2021-2025)
pp = con.execute("""
    SELECT p.game_id, p.season, p.week, p.posteam AS team, p.defteam AS opp, p.play_id, p.receiver_player_id, p.receiving_yards,
           pa.offense_players, pa.defense_man_zone_type AS mz, pa.was_pressure
    FROM pbp p JOIN pbp_participation pa ON pa.nflverse_game_id = p.game_id AND pa.play_id = p.play_id
    WHERE p.season >= 2021 AND p.qb_dropback = 1 AND p.play_type = 'pass'""").df()
pp["man"] = np.where(pp.mz == "MAN_COVERAGE", 1.0, np.where(pp.mz == "ZONE_COVERAGE", 0.0, np.nan))
on = pp.assign(pid=pp.offense_players.str.split(";")).explode("pid").reset_index(drop=True)
on = on[on.pid.notna() & (on.pid != "")]
on["tgt"] = (on.pid == on.receiver_player_id).astype(float)
on["yds"] = np.where(on.tgt == 1, on.receiving_yards.fillna(0), 0.0)
on["man_r"] = (on.man == 1).astype(float); on["zone_r"] = (on.man == 0).astype(float)
on["man_t"] = on.tgt * on.man_r; on["zone_t"] = on.tgt * on.zone_r
drop = pp.groupby(["game_id", "team"]).size().rename("dropbacks")
rg = on.groupby(["season", "week", "game_id", "team", "pid"]).agg(routes=("tgt", "size"), rt=("tgt", "sum"), ry=("yds", "sum"),
                                                                     man_r=("man_r", "sum"), zone_r=("zone_r", "sum"), man_t=("man_t", "sum"), zone_t=("zone_t", "sum")).reset_index()
rg = rg.join(drop, on=["game_id", "team"]).rename(columns={"pid": "player_id"})
rg["route_share"] = rg.routes / rg.dropbacks
rg["t"] = T(rg); rg = rg.sort_values(["player_id", "t"])
gr = rg.groupby("player_id")
roll = lambda c, w_, mp=2: gr[c].transform(lambda x: x.rolling(w_, min_periods=mp).sum())  # noqa: E731  (through current week; asof shifts)
rg["route_share_m3"] = gr.route_share.transform(lambda x: x.rolling(3, min_periods=1).mean())
rg["routes_m3"] = gr.routes.transform(lambda x: x.rolling(3, min_periods=1).mean())
rg["tprr_m8"] = roll("rt", 8) / roll("routes", 8).replace(0, np.nan)
rg["yprr_m8"] = roll("ry", 8) / roll("routes", 8).replace(0, np.nan)
rg["tprr_man"] = roll("man_t", 17, 3) / roll("man_r", 17, 3).replace(0, np.nan)
rg["tprr_zone"] = roll("zone_t", 17, 3) / roll("zone_r", 17, 3).replace(0, np.nan)
B = asof(B, rg, ["route_share_m3", "routes_m3", "tprr_m8", "yprr_m8", "tprr_man", "tprr_zone"], "player_id")
USAGE += ["route_share_m3", "routes_m3", "tprr_m8"]
fam("usage", USAGE)
fam("efficiency", ["yprr_m8"])

# ------------------------------------------------------------------ expected opportunity (ffopportunity) + NGS
ep = con.execute("""SELECT CAST(season AS INT) AS season, CAST(week AS INT) AS week, player_id, receptions_exp, rec_yards_gained_exp,
                           rush_yards_gained_exp, rec_yards_gained_diff, rush_yards_gained_diff, rec_air_yards, rec_attempt, rush_attempt
                    FROM read_parquet('data/raw/ffopportunity/*.parquet', union_by_name = true)""").df()
ep["t"] = T(ep); ep = ep.sort_values(["player_id", "t"]); ge = ep.groupby("player_id")
XC = []
for c in ("receptions_exp", "rec_yards_gained_exp", "rush_yards_gained_exp", "rec_yards_gained_diff", "rush_yards_gained_diff"):
    for w_ in (4, 8):
        ep[f"x_{c}_m{w_}"] = ge[c].transform(lambda x: x.rolling(w_, min_periods=2).mean()); XC.append(f"x_{c}_m{w_}")
ep["adot_m8"] = ge.rec_air_yards.transform(lambda x: x.rolling(8, min_periods=3).sum()) / ge.rec_attempt.transform(lambda x: x.rolling(8, min_periods=3).sum()).replace(0, np.nan)
B = asof(B, ep, XC + ["adot_m8"], "player_id")
fam("expected", XC); fam("usage", ["adot_m8"])


def ngs(table, cols, prefix="ngs_"):
    d = con.execute(f"SELECT season, week, player_gsis_id AS player_id, {', '.join(cols)} FROM {table} WHERE week > 0 AND season >= 2020").df()
    d["t"] = T(d); d = d.sort_values(["player_id", "t"])
    for c in cols:
        d[f"{prefix}{c}"] = d.groupby("player_id")[c].transform(lambda x: x.rolling(8, min_periods=2).mean())
    return d, [f"{prefix}{c}" for c in cols]


d, c1 = ngs("ngs_receiving", ["avg_separation", "avg_cushion", "avg_yac_above_expectation", "avg_intended_air_yards", "catch_percentage"])
B = asof(B, d, c1, "player_id")
d, c2 = ngs("ngs_rushing", ["rush_yards_over_expected_per_att", "efficiency", "percent_attempts_gte_eight_defenders", "avg_time_to_los"])
B = asof(B, d, c2, "player_id")
fam("tracking (NGS)", c1 + c2)

# ------------------------------------------------------------------ game / team context (known pre-game)
tg = con.execute("""SELECT game_id, team, side, team_line, total_line, div_game, primetime, roof, temp_f, wind_mph, rest, opp_rest,
                           su_w_td, gp_td, opp_su_w_td, opp_gp_td, qb_id, season, gameday FROM team_games WHERE season >= 2021""").df()
B = B.merge(tg.drop(columns=["season"]), on=["game_id", "team"], how="left")
B["home"] = (B.side == "home").astype(float)
B["implied_tt"] = (B.total_line + B.team_line) / 2
B["dome"] = B.roof.isin(["dome", "closed"]).astype(float)
B["win_pct"] = B.su_w_td / B.gp_td.replace(0, np.nan); B["opp_win_pct"] = B.opp_su_w_td / B.opp_gp_td.replace(0, np.nan)
B["rest_diff"] = B.rest - B.opp_rest
fam("game script", ["team_line", "total_line", "implied_tt", "home", "div_game", "primetime", "win_pct", "opp_win_pct", "rest_diff", "rest"])
wf = con.execute("SELECT game_id, wind_d1 AS fc_wind, gust_d1 AS fc_gust, precip_d1 AS fc_precip FROM game_wind_forecast").df()
B = B.merge(wf, on="game_id", how="left")
fam("weather", ["dome", "temp_f", "wind_mph", "fc_wind", "fc_gust", "fc_precip"])

# team pace / pass rate over expected / QB situation (season-to-date prior)
pace = con.execute("""SELECT season, week, game_id, posteam AS team, defteam AS opp, count(*) plays,
                             count(*) FILTER (WHERE qb_dropback = 1) dropbacks, count(*) FILTER (WHERE play_type = 'run') runs,
                             avg(pass_oe) FILTER (WHERE wp BETWEEN 0.2 AND 0.8 AND down IN (1, 2) AND qtr <= 3) AS neutral_proe
                      FROM pbp WHERE season >= 2021 AND play_type IN ('pass', 'run') AND posteam IS NOT NULL GROUP BY ALL""").df()
pace["t"] = T(pace); pace = pace.sort_values("t")
for c in ("plays", "neutral_proe", "dropbacks", "runs"):
    pace[f"team_{c}_std"] = pace.groupby(["season", "team"])[c].transform(lambda x: x.expanding().mean())
B = asof(B, pace, ["team_plays_std", "team_neutral_proe_std", "team_dropbacks_std", "team_runs_std"], "team")
dp = pace.rename(columns={"team": "offense", "opp": "defense"})
for c in ("plays", "dropbacks", "runs"):
    dp[f"opp_{c}_allowed"] = dp.groupby(["season", "defense"])[c].transform(lambda x: x.expanding().mean())
B = asof(B, dp.rename(columns={"defense": "opp"}), ["opp_plays_allowed", "opp_dropbacks_allowed", "opp_runs_allowed"], "opp")
fam("pace / tendency", ["team_plays_std", "team_neutral_proe_std", "team_dropbacks_std", "team_runs_std",
                        "opp_plays_allowed", "opp_dropbacks_allowed", "opp_runs_allowed"])
# projected team volume from the game line (fit on prior seasons)
tv = pace[["season", "week", "t", "game_id", "team", "dropbacks", "runs"]].merge(tg[["game_id", "team", "team_line", "total_line"]], on=["game_id", "team"])
tv = asof(tv, pace, ["team_neutral_proe_std", "team_plays_std"], "team")
for target in ("dropbacks", "runs"):
    B[f"proj_{target}"] = np.nan
    cols = ["team_line", "total_line", "team_neutral_proe_std", "team_plays_std"]
    for s in range(2022, 2027):
        tr = tv[tv.season < s].dropna(subset=cols + [target])
        lr = LinearRegression().fit(tr[cols], tr[target])
        m = B.season == s
        B.loc[m, f"proj_{target}"] = lr.predict(B.loc[m, cols].fillna(tr[cols].mean()))
B["proj_targets"] = B.proj_dropbacks * B.route_share_m3 * B.tprr_m8
B["proj_rec_yds"] = B.proj_dropbacks * B.route_share_m3 * B.yprr_m8
B["proj_carries"] = B.proj_runs * B.carry_share_m3
B["proj_rush_yds"] = B.proj_carries * B.ypc_m8
fam("projection", ["proj_dropbacks", "proj_runs", "proj_targets", "proj_rec_yds", "proj_carries", "proj_rush_yds"])

# QB: backup starting; QB accuracy / play-action (FTN)
q = tg.sort_values(["team", "season", "gameday"]); prim = []
for (t_, s_), d_ in q.groupby(["team", "season"]):
    cnt = {}
    for r in d_.itertuples():
        prim.append((r.game_id, t_, max(cnt, key=cnt.get) if cnt else None)); cnt[r.qb_id] = cnt.get(r.qb_id, 0) + 1
prim = pd.DataFrame(prim, columns=["game_id", "team", "primary_qb"])
B = B.merge(prim, on=["game_id", "team"], how="left")
B["backup_qb"] = (B.primary_qb.notna() & (B.qb_id != B.primary_qb)).astype(float)
ftn = con.execute("""
    SELECT f.season, f.week, p.posteam AS team, p.defteam AS defense,
           avg(f.is_catchable_ball::INT) FILTER (WHERE p.pass_attempt = 1 AND p.sack = 0) catchable,
           avg(f.is_play_action::INT) FILTER (WHERE p.qb_dropback = 1) play_action,
           avg((f.n_blitzers > 0)::INT) FILTER (WHERE p.qb_dropback = 1) blitz,
           avg(f.n_defense_box) FILTER (WHERE p.play_type = 'run') box_vs_run,
           avg(f.is_screen_pass::INT) FILTER (WHERE p.pass_attempt = 1) screen,
           avg(f.is_motion::INT) motion
    FROM ftn_charting f JOIN pbp p ON p.game_id = f.nflverse_game_id AND p.play_id = f.nflverse_play_id GROUP BY ALL""").df()
ftn["t"] = T(ftn); ftn = ftn.sort_values("t")
for c in ("catchable", "play_action", "screen", "motion"):
    ftn[f"team_{c}_std"] = ftn.groupby(["season", "team"])[c].transform(lambda x: x.expanding().mean())
for c in ("blitz", "box_vs_run"):
    ftn[f"opp_{c}_std"] = ftn.groupby(["season", "defense"])[c].transform(lambda x: x.expanding().mean())
B = asof(B, ftn, ["team_catchable_std", "team_play_action_std", "team_screen_std", "team_motion_std"], "team")
B = asof(B, ftn.rename(columns={"defense": "opp"}), ["opp_blitz_std", "opp_box_vs_run_std"], "opp")
fam("QB / scheme", ["backup_qb", "team_catchable_std", "team_play_action_std", "team_screen_std", "team_motion_std"])

# ------------------------------------------------------------------ opponent defense
allowed = B.groupby(["season", "week", "t", "game_id", "opp", "position"])[["receiving_yards", "rushing_yards", "targets", "carries"]].sum().reset_index()
allowed = allowed.sort_values("t")
OPP = []
for c in ("receiving_yards", "rushing_yards", "targets", "carries"):
    allowed[f"opp_allow_{c}"] = allowed.groupby(["season", "opp", "position"])[c].transform(lambda x: x.expanding().mean())
    lg = allowed.groupby(["t", "position"])[f"opp_allow_{c}"].transform("mean")
    allowed[f"opp_allow_{c}_rel"] = allowed[f"opp_allow_{c}"] / lg.replace(0, np.nan)
    OPP.append(f"opp_allow_{c}_rel")
B = asof(B, allowed, OPP, ["opp", "position"])
epa = con.execute("""SELECT e.season, g.week, CASE WHEN e.team = g.home_team THEN g.away_team ELSE g.home_team END AS opp,
                            e.pass_epa, e.rush_epa, e.off_sr FROM team_game_epa e JOIN games g USING (game_id) WHERE e.season >= 2021""").df()
epa["t"] = T(epa); epa = epa.sort_values("t")
for c in ("pass_epa", "rush_epa", "off_sr"):
    epa[f"opp_{c}_allowed"] = epa.groupby(["season", "opp"])[c].transform(lambda x: x.expanding().mean())
B = asof(B, epa, ["opp_pass_epa_allowed", "opp_rush_epa_allowed", "opp_off_sr_allowed"], "opp")
dcov = pp.groupby(["season", "week", "opp"]).agg(man=("man", "mean"), pressure=("was_pressure", "mean")).reset_index()
dcov["t"] = T(dcov); dcov = dcov.sort_values("t")
for c in ("man", "pressure"):
    dcov[f"opp_{c}_rate"] = dcov.groupby(["season", "opp"])[c].transform(lambda x: x.expanding().mean())
B = asof(B, dcov, ["opp_man_rate", "opp_pressure_rate"], "opp")
cov = con.execute("""SELECT season, week, team AS opp, sum(def_yards_allowed) y, sum(def_targets) tt, sum(def_missed_tackles) mt
                     FROM pfr_adv_week_def WHERE season >= 2021 GROUP BY ALL""").df()
cov["t"] = T(cov); cov = cov.sort_values("t"); gcv = cov.groupby(["season", "opp"])
cov["opp_ypt_allowed"] = gcv.y.transform(lambda x: x.expanding().sum()) / gcv.tt.transform(lambda x: x.expanding().sum()).replace(0, np.nan)
cov["opp_missed_tackles_pg"] = gcv.mt.transform(lambda x: x.expanding().mean())
B = asof(B, cov, ["opp_ypt_allowed", "opp_missed_tackles_pg"], "opp")
B["cov_matchup"] = (B.opp_man_rate * B.tprr_man + (1 - B.opp_man_rate) * B.tprr_zone) / B.tprr_m8.replace(0, np.nan)
fam("opponent", OPP + ["opp_pass_epa_allowed", "opp_rush_epa_allowed", "opp_off_sr_allowed", "opp_man_rate", "opp_pressure_rate",
                       "opp_ypt_allowed", "opp_missed_tackles_pg", "opp_blitz_std", "opp_box_vs_run_std", "cov_matchup"])

# ------------------------------------------------------------------ injuries / roster (this week's report is pre-game)
inj = con.execute("""SELECT CAST(season AS INT) AS season, CAST(week AS INT) AS week, gsis_id AS player_id, team, position, report_status, practice_status
                     FROM injuries WHERE season >= 2021""").df()
B = B.merge(inj[["season", "week", "player_id", "report_status", "practice_status"]].drop_duplicates(["season", "week", "player_id"]),
            on=["season", "week", "player_id"], how="left")
B["questionable"] = (B.report_status == "Questionable").astype(float)
B["practice_limited"] = B.practice_status.fillna("").str.contains("Limited").astype(float)
B["practice_dnp"] = B.practice_status.fillna("").str.contains("Did Not").astype(float)
out = inj[inj.report_status.isin(["Out", "Doubtful"])]
shares = B[["player_id", "season", "week", "target_share_m3", "carry_share_m3"]]
vac = out.merge(shares, on=["player_id", "season", "week"], how="left").groupby(["season", "week", "team"]).agg(
    vacated_target_share=("target_share_m3", "sum"), vacated_carry_share=("carry_share_m3", "sum")).reset_index()
B = B.merge(vac, on=["season", "week", "team"], how="left")
prev_out = out.assign(week=out.week + 1)
now_out = set(zip(out.season, out.week, out.player_id))
ret = prev_out[[(a, b, c) not in now_out for a, b, c in zip(prev_out.season, prev_out.week, prev_out.player_id)]]
last_share = B.sort_values("t").groupby(["player_id", "season"]).target_share_m3.last().reset_index()
rv = ret.merge(last_share, on=["player_id", "season"], how="left").groupby(["season", "week", "team"]).target_share_m3.sum().rename("returning_target_share").reset_index()
B = B.merge(rv, on=["season", "week", "team"], how="left")
dsn = con.execute("""SELECT sc.season, sc.week, pl.gsis_id AS player_id, sc.team, sc.position, sc.defense_pct, sc.offense_pct FROM snap_counts sc
                     JOIN players pl ON pl.pfr_id = sc.pfr_player_id WHERE sc.season >= 2021""").df().sort_values(["player_id", "season", "week"])
dsn["d4"] = dsn.groupby("player_id").defense_pct.transform(lambda x: x.rolling(4, min_periods=1).mean())
dsn["o4"] = dsn.groupby("player_id").offense_pct.transform(lambda x: x.rolling(4, min_periods=1).mean())
lastsn = dsn.groupby(["player_id", "season"])[["d4", "o4"]].last().reset_index()
oi = out.merge(lastsn, on=["player_id", "season"], how="left").fillna({"d4": 0, "o4": 0})
agg = lambda pos, col, name: oi[oi.position.isin(pos)].groupby(["season", "week", "team"])[col].sum().rename(name).reset_index()  # noqa: E731
B = B.merge(agg(["CB", "S", "SS", "FS", "DB"], "d4", "opp_db_out").rename(columns={"team": "opp"}), on=["season", "week", "opp"], how="left")
B = B.merge(agg(["DE", "DT", "OLB", "EDGE", "DL", "LB", "ILB", "MLB"], "d4", "opp_front_out").rename(columns={"team": "opp"}), on=["season", "week", "opp"], how="left")
B = B.merge(agg(["T", "G", "C", "OL", "OT", "OG"], "o4", "own_ol_out"), on=["season", "week", "team"], how="left")
for c in ("vacated_target_share", "vacated_carry_share", "returning_target_share", "opp_db_out", "opp_front_out", "own_ol_out"):
    B[c] = B[c].fillna(0)
fam("injury / roster", ["questionable", "practice_limited", "practice_dnp", "vacated_target_share", "vacated_carry_share",
                        "returning_target_share", "opp_db_out", "opp_front_out", "own_ol_out"])

# ------------------------------------------------------------------ player
pl = con.execute("SELECT gsis_id AS player_id, rookie_season, birth_date, height, weight, draft_pick AS draft_number FROM players").df()
B = B.merge(pl, on="player_id", how="left")
B["age"] = B.season - pd.to_datetime(B.birth_date, errors="coerce").dt.year
B["rookie"] = (B.rookie_season == B.season).astype(float)
B["experience"] = B.season - B.rookie_season
B["is_te"] = (B.position == "TE").astype(float); B["is_rb"] = (B.position == "RB").astype(float)
fam("player", ["age", "rookie", "experience", "height", "weight", "draft_number", "is_te", "is_rb", "week"])

feats = [c for c in FAMILY if c in B.columns]
B.to_parquet(ROOT / "data/player_games.parquet")
(ROOT / "data/player_feature_families.json").write_text(json.dumps({c: FAMILY[c] for c in feats}, indent=1))
print(B.shape, "features:", len(feats))
cov_ = B[feats].notna().mean().sort_values()
print(cov_.head(15).round(2).to_string())
# leak tripwire: a pre-game feature should not track this game's yards better than an 8-game average does
for tgt in ("receiving_yards", "rushing_yards"):
    base = abs(B[[f"{tgt}_m8", tgt]].corr().iloc[0, 1])
    sus = {c: round(abs(B[[c, tgt]].corr().iloc[0, 1]), 3) for c in feats if B[c].notna().sum() > 1000 and abs(B[[c, tgt]].corr().iloc[0, 1]) > base + 0.05}
    print(f"tripwire {tgt}: m8 corr {base:.3f}; features clearly above it: {sus}")
