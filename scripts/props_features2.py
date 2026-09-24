"""Second feature layer for props (adds to data/props_features.parquet -> data/props_features_v2.parquet).

All features use only games before the one being bet (season-to-date or rolling, shifted).
  usage:     pass-play snap share (proxy for routes run), targets per pass-play snap
  coverage:  opponent man-coverage rate + pressure rate; player's target rate vs man vs zone -> expected rate vs this opponent
  pace/PROE: team + opponent plays per game, neutral-situation pass rate over expected
  injuries:  opponent DBs / pass rushers ruled out, weighted by their snap share
  QB:        team starting a non-primary QB
  NGS:       receiver separation / cushion / YAC over expected, rusher RYOE, QB time to throw
  market:    game total + spread move open->close, dispersion of books' prop lines
  weather:   forecast wind/precip 1 day out (2024+), rest days
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
F = pd.read_parquet(ROOT / "data/props_features.parquet")
n0 = len(F)


def roll_prior(df, by, cols, order=("season", "week"), within_season=True, min_periods=1):
    """Season-to-date mean of cols, excluding the current row."""
    df = df.sort_values(list(order))
    grp = by + (["season"] if within_season else [])
    g = df.groupby(grp)
    for c in cols:
        df[f"{c}_std"] = g[c].transform(lambda x: x.shift().expanding(min_periods=min_periods).mean())
    return df


# ------------------------------------------------------------------ play-level pass plays with participation (2018+)
pp = con.execute("""
    SELECT p.game_id, p.season, p.week, p.posteam AS team, p.defteam AS opp, p.play_id, p.receiver_player_id, p.passer_player_id,
           p.pass_attempt, p.complete_pass, p.yards_gained, p.qb_dropback,
           pa.offense_players, pa.defense_man_zone_type AS mz, pa.was_pressure, pa.time_to_throw
    FROM pbp p JOIN pbp_participation pa ON pa.nflverse_game_id = p.game_id AND pa.play_id = p.play_id
    WHERE p.season >= 2021 AND p.qb_dropback = 1 AND p.play_type = 'pass'
""").df()
pp["man"] = (pp.mz == "MAN_COVERAGE").astype(float).where(pp.mz.isin(["MAN_COVERAGE", "ZONE_COVERAGE"]))
# defense tendencies per game -> season-to-date
dg = pp.groupby(["season", "week", "game_id", "opp"]).agg(man_rate=("man", "mean"), pressure_rate=("was_pressure", "mean")).reset_index()
dg = roll_prior(dg, ["opp"], ["man_rate", "pressure_rate"])
F = F.merge(dg[["game_id", "opp", "man_rate_std", "pressure_rate_std"]].rename(columns={"man_rate_std": "opp_man_rate", "pressure_rate_std": "opp_pressure_rate"}),
            on=["game_id", "opp"], how="left")

# player on-field pass plays (route proxy) and targets split by coverage
on = pp[["game_id", "season", "week", "team", "play_id", "offense_players", "receiver_player_id", "man"]].copy()
on["pid"] = on.offense_players.str.split(";")
on = on.explode("pid").drop(columns="offense_players")
on = on[on.pid.notna() & (on.pid != "")]
on = on.reset_index(drop=True)
on["tgt"] = (on.pid == on.receiver_player_id).astype(float)
on["man_snaps"] = (on.man == 1).astype(float); on["zone_snaps"] = (on.man == 0).astype(float)
on["man_tgts"] = on.tgt * on.man_snaps; on["zone_tgts"] = on.tgt * on.zone_snaps
team_db = pp.groupby(["game_id", "team"]).size().rename("team_dropbacks").reset_index()
pg = on.groupby(["game_id", "season", "week", "team", "pid"]).agg(
    pass_snaps=("tgt", "size"), tgts=("tgt", "sum"), man_snaps=("man_snaps", "sum"), zone_snaps=("zone_snaps", "sum"),
    man_tgts=("man_tgts", "sum"), zone_tgts=("zone_tgts", "sum"),
).reset_index().merge(team_db, on=["game_id", "team"])
pg["route_share"] = pg.pass_snaps / pg.team_dropbacks
pg = pg.sort_values(["pid", "season", "week"])
g = pg.groupby("pid")
pg["route_share_m3"] = g.route_share.transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
pg["tprr_m8"] = g.tgts.transform(lambda x: x.shift().rolling(8, min_periods=2).sum()) / g.pass_snaps.transform(lambda x: x.shift().rolling(8, min_periods=2).sum())
for c in ("man_snaps", "zone_snaps", "man_tgts", "zone_tgts"):  # last ~season of games, prior only
    pg[f"{c}_r"] = g[c].transform(lambda x: x.shift().rolling(17, min_periods=3).sum())
pg["tprr_man"] = pg.man_tgts_r / pg.man_snaps_r.replace(0, np.nan)
pg["tprr_zone"] = pg.zone_tgts_r / pg.zone_snaps_r.replace(0, np.nan)
F = F.merge(pg[["game_id", "pid", "route_share_m3", "tprr_m8", "tprr_man", "tprr_zone"]].rename(columns={"pid": "player_id"}),
            on=["game_id", "player_id"], how="left")
# expected target rate vs this defense's coverage mix, relative to the player's overall rate
F["cov_matchup"] = (F.opp_man_rate * F.tprr_man + (1 - F.opp_man_rate) * F.tprr_zone) / F.tprr_m8.replace(0, np.nan)
F["man_zone_skew"] = F.tprr_man / F.tprr_zone.replace(0, np.nan)

# ------------------------------------------------------------------ pace + pass rate over expected
pace = con.execute("""
    SELECT season, week, game_id, posteam AS team, defteam AS opp, count(*) AS plays,
           avg(pass_oe) FILTER (WHERE wp BETWEEN 0.2 AND 0.8 AND down IN (1, 2) AND qtr <= 3) AS neutral_proe
    FROM pbp WHERE season >= 2021 AND play_type IN ('pass', 'run') AND posteam IS NOT NULL
    GROUP BY ALL""").df()
pace = roll_prior(pace, ["team"], ["plays", "neutral_proe"])
F = F.merge(pace[["game_id", "team", "plays_std", "neutral_proe_std"]].rename(columns={"plays_std": "team_pace", "neutral_proe_std": "team_proe"}),
            on=["game_id", "team"], how="left")
F = F.merge(pace[["game_id", "team", "plays_std", "neutral_proe_std"]].rename(columns={"team": "opp", "plays_std": "opp_pace", "neutral_proe_std": "opp_proe"}),
            on=["game_id", "opp"], how="left")
# plays the opponent's defense allows per game
dpace = pace.rename(columns={"opp": "defense"})
dpace = roll_prior(dpace[["season", "week", "game_id", "defense", "plays"]].rename(columns={"plays": "plays_allowed"}), ["defense"], ["plays_allowed"])
F = F.merge(dpace[["game_id", "defense", "plays_allowed_std"]].rename(columns={"defense": "opp", "plays_allowed_std": "opp_plays_allowed"}),
            on=["game_id", "opp"], how="left")
F["exp_plays"] = (F.team_pace + F.opp_plays_allowed) / 2

# ------------------------------------------------------------------ opponent injuries weighted by snap share
snaps = con.execute("""SELECT sc.game_id, sc.season, sc.week, sc.team, pl.gsis_id AS player_id, sc.position, sc.defense_pct, sc.offense_pct
                       FROM snap_counts sc JOIN players pl ON pl.pfr_id = sc.pfr_player_id WHERE sc.season >= 2022""").df()
snaps = snaps.sort_values(["player_id", "season", "week"])
snaps["def_share_m4"] = snaps.groupby("player_id").defense_pct.transform(lambda x: x.rolling(4, min_periods=1).mean())
last_share = snaps.groupby(["player_id", "season"]).agg(last_week=("week", "max")).reset_index()
inj = con.execute("""SELECT CAST(season AS INT) AS season, CAST(week AS INT) AS week, gsis_id AS player_id, team, position, report_status
                     FROM injuries WHERE season >= 2023 AND report_status IN ('Out', 'Doubtful')""").df()
# most recent defensive snap share before this week
sh = snaps[["player_id", "season", "week", "def_share_m4"]].rename(columns={"week": "w_snap"})
ij = inj.merge(sh, on=["player_id", "season"], how="left")
ij = ij[ij.w_snap < ij.week].sort_values("w_snap").groupby(["season", "week", "player_id"]).tail(1)
ij = inj.merge(ij[["season", "week", "player_id", "def_share_m4"]], on=["season", "week", "player_id"], how="left")
ij["def_share_m4"] = ij.def_share_m4.fillna(0)
db = ij[ij.position.isin(["CB", "S", "SS", "FS", "DB"])].groupby(["season", "week", "team"]).def_share_m4.sum().rename("opp_db_out")
pr = ij[ij.position.isin(["DE", "DT", "OLB", "EDGE", "DL", "LB"])].groupby(["season", "week", "team"]).def_share_m4.sum().rename("opp_front_out")
opp_inj = pd.concat([db, pr], axis=1).reset_index().rename(columns={"team": "opp"})
F = F.merge(opp_inj, on=["season", "week", "opp"], how="left")
F[["opp_db_out", "opp_front_out"]] = F[["opp_db_out", "opp_front_out"]].fillna(0)

# ------------------------------------------------------------------ QB situation: team starting a non-primary QB
q = con.execute("SELECT game_id, season, gameday, team, qb_id FROM team_games WHERE season >= 2022 ORDER BY team, season, gameday").df()
prim = []
for (t, s), d in q.groupby(["team", "season"]):
    counts = {}
    for r in d.itertuples():
        prim.append((r.game_id, t, max(counts, key=counts.get) if counts else None))
        counts[r.qb_id] = counts.get(r.qb_id, 0) + 1
prim = pd.DataFrame(prim, columns=["game_id", "team", "primary_qb"])
q = q.merge(prim, on=["game_id", "team"])
q["backup_qb"] = (q.primary_qb.notna() & (q.qb_id != q.primary_qb)).astype(float)
F = F.merge(q[["game_id", "team", "backup_qb"]], on=["game_id", "team"], how="left")

# ------------------------------------------------------------------ Next Gen Stats (as-of prior weeks only)
# NGS only publishes a weekly row when the player *qualified in that game* (enough targets/carries/attempts), so joining
# on the current week leaks this game's volume. Instead: rolling mean through each week, then take the latest value
# from a strictly earlier week (merge_asof, allow_exact_matches=False).
F["t"] = (F.season * 100 + F.week).astype("int64")


def ngs_asof(F, table, cols, prefix="ngs_"):
    d = con.execute(f"SELECT season, week, player_gsis_id AS player_id, {', '.join(cols)} FROM {table} WHERE week > 0 AND season >= 2021").df()
    d = d.sort_values(["player_id", "season", "week"])
    for c in cols:
        d[f"{prefix}{c}"] = d.groupby("player_id")[c].transform(lambda x: x.rolling(8, min_periods=2).mean())
    d["t"] = (d.season * 100 + d.week).astype("int64")
    d = d[["player_id", "t"] + [f"{prefix}{c}" for c in cols]].sort_values("t")
    out = pd.merge_asof(F.sort_values("t"), d, on="t", by="player_id", allow_exact_matches=False)
    return out


F = ngs_asof(F, "ngs_receiving", ["avg_separation", "avg_cushion", "avg_yac_above_expectation", "avg_intended_air_yards"])
F = ngs_asof(F, "ngs_rushing", ["rush_yards_over_expected_per_att"])
F = F.rename(columns={"ngs_rush_yards_over_expected_per_att": "ngs_ryoe"})
F = ngs_asof(F, "ngs_passing", ["avg_time_to_throw", "aggressiveness", "avg_air_yards_differential"])
F = F.drop(columns="t")

# ------------------------------------------------------------------ market: game line moves + prop line dispersion
gl = con.execute("""
    WITH t AS (SELECT game_id, pass, market, bookmaker, outcome, point FROM odds_raw
               WHERE market IN ('totals', 'spreads') AND pass IN ('open', 'close')
                 AND bookmaker NOT IN ('kalshi', 'polymarket', 'novig', 'prophetx', 'betopenly'))
    SELECT game_id,
      median(point) FILTER (WHERE market = 'totals' AND pass = 'close' AND outcome = 'Over') - median(point) FILTER (WHERE market = 'totals' AND pass = 'open' AND outcome = 'Over') AS total_move
    FROM t GROUP BY 1""").df()
F = F.merge(gl, on="game_id", how="left")
F["total_move"] = F.total_move.astype(float)
disp = con.execute("""
    WITH m AS (SELECT game_id, market, player_key, bookmaker, point,
                      abs(0.5 - (CASE WHEN over_price < 0 THEN -over_price / (-over_price + 100.0) ELSE 100.0 / (over_price + 100) END)) d,
                      row_number() OVER (PARTITION BY game_id, market, player_key, bookmaker ORDER BY
                        abs(0.5 - (CASE WHEN over_price < 0 THEN -over_price / (-over_price + 100.0) ELSE 100.0 / (over_price + 100) END))) rn
               FROM prop_results WHERE pass = 'close' AND over_price IS NOT NULL AND under_price IS NOT NULL AND point IS NOT NULL
                 AND bookmaker NOT IN ('kalshi', 'polymarket', 'novig', 'prophetx', 'betopenly'))
    SELECT game_id, market, player_key, stddev_samp(point) AS line_dispersion, max(point) - min(point) AS line_range
    FROM m WHERE rn = 1 GROUP BY ALL""").df()
F = F.merge(disp, on=["game_id", "market", "player_key"], how="left")
F["line_dispersion_rel"] = F.line_dispersion / F.cons_point.clip(lower=0.5)

# ------------------------------------------------------------------ forecast weather (2024+) and rest
wf = con.execute("SELECT game_id, wind_d1 AS fc_wind, gust_d1 AS fc_gust, precip_d1 AS fc_precip FROM game_wind_forecast").df()
F = F.merge(wf, on="game_id", how="left")
rest = con.execute("SELECT game_id, team, rest, opp_rest FROM team_games").df()
F = F.merge(rest, on=["game_id", "team"], how="left")
F["rest_diff"] = F.rest - F.opp_rest

assert len(F) == n0, (len(F), n0)
F.to_parquet(ROOT / "data/props_features_v2.parquet")
new = [c for c in F.columns if c not in pd.read_parquet(ROOT / "data/props_features.parquet").columns]
print(len(F), "rows;", len(new), "new features")
print(F[new].notna().mean().round(2).sort_values().to_string())
