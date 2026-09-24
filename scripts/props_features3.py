"""Third feature layer (v2 -> data/props_features_v3.parquet). Prior-game information only.

  expected opportunity (ffverse ffopportunity xgboost model): rolling expected receptions / rec yards / rush yards / pass yards,
      expected-vs-line, and over-performance vs expectation (luck that the line may have absorbed)
  efficiency: yards per route run (YPRR), aDOT, air-yards per route
  volume projection: predicted team dropbacks/carries from spread + total + PROE + pace (model fit on prior seasons),
      x player share x efficiency -> projected stat vs line
  roster: teammates returning from injury (were Out last week, not listed this week) weighted by pre-absence share;
      own offensive-line starters out (snap-weighted)
  charting (FTN, 2022+): QB catchable-ball rate, team play-action rate, opponent blitz rate, opponent box count
  market: this player's line vs his line last week in the same market
  player: rookie, altitude (Denver), turf
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
F = pd.read_parquet(ROOT / "data/props_features_v2.parquet")
n0 = len(F)
F["t"] = (F.season * 100 + F.week).astype("int64")


def asof(F, d, cols, by="player_id"):
    """Attach d[cols] (already rolled *through* each week) from the latest strictly-earlier week."""
    d = d[[by, "t"] + cols].dropna(subset=[by]).sort_values("t")
    return pd.merge_asof(F.sort_values("t"), d, on="t", by=by, allow_exact_matches=False)


# ------------------------------------------------------------------ expected opportunity (ffopportunity)
ep = con.execute("""SELECT CAST(season AS INT) AS season, CAST(week AS INT) AS week, player_id, posteam,
                           receptions_exp, rec_yards_gained_exp, rush_yards_gained_exp, pass_yards_gained_exp, pass_completions_exp,
                           rec_yards_gained_diff, rush_yards_gained_diff, pass_yards_gained_diff, receptions_diff,
                           rec_air_yards, rec_attempt, rush_attempt, pass_attempt
                    FROM read_parquet('data/raw/ffopportunity/*.parquet', union_by_name = true)""").df()
ep["t"] = (ep.season * 100 + ep.week).astype("int64")
ep = ep.sort_values(["player_id", "t"])
g = ep.groupby("player_id")
roll = {}
for c in ("receptions_exp", "rec_yards_gained_exp", "rush_yards_gained_exp", "pass_yards_gained_exp", "pass_completions_exp",
          "rec_yards_gained_diff", "rush_yards_gained_diff", "pass_yards_gained_diff", "receptions_diff"):
    ep[f"x_{c}_m4"] = g[c].transform(lambda x: x.rolling(4, min_periods=2).mean())
ep["adot_m8"] = g.rec_air_yards.transform(lambda x: x.rolling(8, min_periods=3).sum()) / g.rec_attempt.transform(lambda x: x.rolling(8, min_periods=3).sum()).replace(0, np.nan)
xcols = [c for c in ep.columns if c.startswith("x_")] + ["adot_m8"]
F = asof(F, ep, xcols)
XMAP = {"player_receptions": "x_receptions_exp_m4", "player_reception_yds": "x_rec_yards_gained_exp_m4",
        "player_rush_yds": "x_rush_yards_gained_exp_m4", "player_pass_yds": "x_pass_yards_gained_exp_m4",
        "player_pass_completions": "x_pass_completions_exp_m4",
        "player_rush_reception_yds": None}
DMAP = {"player_receptions": "x_receptions_diff_m4", "player_reception_yds": "x_rec_yards_gained_diff_m4",
        "player_rush_yds": "x_rush_yards_gained_diff_m4", "player_pass_yds": "x_pass_yards_gained_diff_m4"}
F["x_exp"] = [r[XMAP[m]] if XMAP.get(m) else (r["x_rec_yards_gained_exp_m4"] + r["x_rush_yards_gained_exp_m4"] if m == "player_rush_reception_yds" else np.nan)
              for m, (_, r) in zip(F.market, F.iterrows())]
F["x_exp_vs_line"] = F.x_exp / F.cons_point.clip(lower=0.5)
F["x_overperf"] = [r[DMAP[m]] if m in DMAP else np.nan for m, (_, r) in zip(F.market, F.iterrows())]
F["x_overperf_rel"] = F.x_overperf / F.cons_point.clip(lower=0.5)

# ------------------------------------------------------------------ YPRR (receiving yards per pass-play snap on field)
pp = con.execute("""
    SELECT p.game_id, p.season, p.week, p.posteam AS team, p.play_id, p.receiver_player_id, p.receiving_yards, pa.offense_players
    FROM pbp p JOIN pbp_participation pa ON pa.nflverse_game_id = p.game_id AND pa.play_id = p.play_id
    WHERE p.season >= 2021 AND p.qb_dropback = 1 AND p.play_type = 'pass'""").df()
on = pp.assign(pid=pp.offense_players.str.split(";")).explode("pid").reset_index(drop=True)
on = on[on.pid.notna() & (on.pid != "")]
on["yds"] = np.where(on.pid == on.receiver_player_id, on.receiving_yards.fillna(0), 0.0)
yp = on.groupby(["season", "week", "pid"]).agg(routes=("yds", "size"), ryds=("yds", "sum")).reset_index().rename(columns={"pid": "player_id"})
yp["t"] = (yp.season * 100 + yp.week).astype("int64")
yp = yp.sort_values(["player_id", "t"])
gy = yp.groupby("player_id")
yp["yprr_m8"] = gy.ryds.transform(lambda x: x.rolling(8, min_periods=3).sum()) / gy.routes.transform(lambda x: x.rolling(8, min_periods=3).sum())
yp["routes_m3"] = gy.routes.transform(lambda x: x.rolling(3, min_periods=1).mean())
F = asof(F, yp, ["yprr_m8", "routes_m3"])

# ------------------------------------------------------------------ volume projection from the game line
tv = con.execute("""
    SELECT t.game_id, t.season, t.week, t.team, t.team_line, t.total_line,
           count(*) FILTER (WHERE p.qb_dropback = 1) AS dropbacks, count(*) FILTER (WHERE p.play_type = 'run') AS carries
    FROM team_games t JOIN pbp p ON p.game_id = t.game_id AND p.posteam = t.team
    WHERE t.season >= 2021 AND p.play_type IN ('pass', 'run') GROUP BY ALL""").df()
tv = tv.merge(F[["game_id", "team", "team_proe", "team_pace"]].drop_duplicates(["game_id", "team"]), on=["game_id", "team"], how="left")
tv[["team_proe", "team_pace"]] = tv[["team_proe", "team_pace"]].fillna(tv[["team_proe", "team_pace"]].mean())
cols = ["team_line", "total_line", "team_proe", "team_pace"]
for target in ("dropbacks", "carries"):
    F[f"proj_{target}"] = np.nan
    for s in sorted(F.season.unique()):  # fit on prior seasons only
        tr = tv[tv.season < s].dropna(subset=cols + [target])
        if len(tr) < 200:
            continue
        lr = LinearRegression().fit(tr[cols], tr[target])
        m = F.season == s
        F.loc[m, f"proj_{target}"] = lr.predict(F.loc[m, cols].fillna(tv[cols].mean()))
# projected stat = projected team volume x player's share x efficiency (all prior)
F["proj_targets"] = F.proj_dropbacks * F.route_share_m3 * F.tprr_m8
F["proj_rec_yds"] = F.proj_dropbacks * F.route_share_m3 * F.yprr_m8
F["proj_carries"] = F.proj_carries * F.carry_share_m3
proj = np.select([F.market == "player_reception_yds", F.market == "player_receptions", F.market == "player_rush_attempts"],
                 [F.proj_rec_yds, F.proj_targets * 0.65, F.proj_carries], np.nan)
F["proj_vs_line"] = proj / F.cons_point.clip(lower=0.5)

# ------------------------------------------------------------------ roster: teammates returning, own OL out
inj = con.execute("""SELECT CAST(season AS INT) AS season, CAST(week AS INT) AS week, gsis_id AS player_id, team, position, report_status
                     FROM injuries WHERE season >= 2023""").df()
out = inj[inj.report_status.isin(["Out", "Doubtful"])][["season", "week", "player_id", "team", "position"]]
prev = out.assign(week=out.week + 1)  # was Out last week
this_out = set(zip(out.season, out.week, out.player_id))
ret = prev[[(s, w, p) not in this_out for s, w, p in zip(prev.season, prev.week, prev.player_id)]]
share = pd.read_parquet(ROOT / "data/props_features.parquet", columns=["player_id", "season", "target_share_m3"]).groupby(["player_id", "season"]).target_share_m3.max()
h = con.execute("SELECT player_id, CAST(season AS INT) AS season, week, target_share, carries, team FROM stats_player_week WHERE season >= 2023").df()
h = h.sort_values(["player_id", "season", "week"])
h["ts_m4"] = h.groupby("player_id").target_share.transform(lambda x: x.rolling(4, min_periods=1).mean())
last = h.groupby(["player_id", "season"]).agg(ts=("ts_m4", "last")).reset_index()
ret = ret.merge(last, on=["player_id", "season"], how="left")
rv = ret.groupby(["season", "week", "team"]).ts.sum().rename("returning_target_share").reset_index()
F = F.merge(rv, on=["season", "week", "team"], how="left")
F["returning_target_share"] = F.returning_target_share.fillna(0)
snaps = con.execute("""SELECT sc.season, sc.week, pl.gsis_id AS player_id, sc.offense_pct FROM snap_counts sc
                       JOIN players pl ON pl.pfr_id = sc.pfr_player_id WHERE sc.season >= 2023 AND sc.position IN ('T', 'G', 'C', 'OL')""").df()
snaps = snaps.sort_values(["player_id", "season", "week"])
snaps["os_m4"] = snaps.groupby("player_id").offense_pct.transform(lambda x: x.rolling(4, min_periods=1).mean())
ol = out[out.position.isin(["T", "G", "C", "OL", "OT", "OG"])].merge(
    snaps.groupby(["player_id", "season"]).os_m4.last().reset_index(), on=["player_id", "season"], how="left")
olv = ol.groupby(["season", "week", "team"]).os_m4.sum().rename("own_ol_out").reset_index()
F = F.merge(olv, on=["season", "week", "team"], how="left")
F["own_ol_out"] = F.own_ol_out.fillna(0)

# ------------------------------------------------------------------ FTN charting tendencies (season-to-date, prior weeks)
ftn = con.execute("""
    SELECT f.season, f.week, p.posteam AS team, p.defteam AS defense,
           avg(f.is_catchable_ball::INT) FILTER (WHERE p.pass_attempt = 1 AND p.sack = 0) AS catchable,
           avg(f.is_play_action::INT) FILTER (WHERE p.qb_dropback = 1) AS play_action,
           avg((f.n_blitzers > 0)::INT) FILTER (WHERE p.qb_dropback = 1) AS blitz,
           avg(f.n_defense_box) FILTER (WHERE p.play_type = 'run') AS box_vs_run
    FROM ftn_charting f JOIN pbp p ON p.game_id = f.nflverse_game_id AND p.play_id = f.nflverse_play_id
    GROUP BY ALL""").df()
ftn["t"] = (ftn.season * 100 + ftn.week).astype("int64")
ftn = ftn.sort_values("t")
for c, key in (("catchable", "team"), ("play_action", "team"), ("blitz", "defense"), ("box_vs_run", "defense")):
    ftn[f"{c}_std"] = ftn.groupby(["season", key])[c].transform(lambda x: x.expanding().mean())
F = asof(F, ftn.rename(columns={"catchable_std": "qb_catchable_rate", "play_action_std": "team_pa_rate"}), ["qb_catchable_rate", "team_pa_rate"], by="team")
d2 = ftn.rename(columns={"defense": "opp", "blitz_std": "opp_blitz_rate", "box_vs_run_std": "opp_box_vs_run"})
F = asof(F, d2, ["opp_blitz_rate", "opp_box_vs_run"], by="opp")

# ------------------------------------------------------------------ market: line vs this player's line last time out
F = F.sort_values(["player_id", "market", "t"])
F["prev_line"] = F.groupby(["player_id", "market"]).cons_point.shift()
F["line_wow"] = (F.cons_point - F.prev_line) / F.prev_line.clip(lower=0.5)

# ------------------------------------------------------------------ player / venue
pl = con.execute("SELECT gsis_id AS player_id, rookie_season, birth_date FROM players").df()
F = F.merge(pl, on="player_id", how="left")
F["rookie"] = (F.rookie_season == F.season).astype(float)
F["age"] = F.season - pd.to_datetime(F.birth_date, errors="coerce").dt.year
g2 = con.execute("SELECT game_id, home_team, surface FROM games").df()
F = F.merge(g2, on="game_id", how="left")
F["altitude"] = (F.home_team == "DEN").astype(float)
F["turf"] = (~F.surface.fillna("").str.contains("grass")).astype(float)
F = F.drop(columns=["t", "rookie_season", "birth_date", "home_team", "surface"])

assert len(F) == n0, (len(F), n0)
F.to_parquet(ROOT / "data/props_features_v3.parquet")
new = [c for c in F.columns if c not in pd.read_parquet(ROOT / "data/props_features_v2.parquet").columns]
print(len(F), "rows;", len(new), "new:")
print(F[new].notna().mean().round(2).sort_values().to_string())
