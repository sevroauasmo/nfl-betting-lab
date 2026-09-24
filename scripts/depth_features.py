"""Depth-chart features per player-game (WR/TE/RB), point-in-time where the data allows.

2025+ : ESPN daily snapshots with timestamps -> use the latest snapshot strictly before kickoff (true point-in-time).
2021-24: one chart per team per game week, no timestamp -> assumed to be the pre-game chart (cannot be verified).
Common features (both eras): depth tier (1 starter, 2 backup, 3 reserve, 4 not listed), tier change vs the team's previous
game, promoted / demoted flags, and "starter slots opened" at the player's position (last game's starters no longer starters).
2025+ only: fine rank within position, rank change, and players listed above him last game who are gone or now below him.
Output: data/depth_features.parquet keyed by (player_id, game_id).
"""
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
POS = {"WR": "WR", "TE": "TE", "RB": "RB", "HB": "RB"}

# games per team with kickoff (UTC) and the team's previous game
sch = con.execute("""SELECT game_id, season, week, gameday, gametime, home_team, away_team FROM schedules
                     WHERE season >= 2021 AND gametime IS NOT NULL""").df()
et = ZoneInfo("America/New_York")
sch["kick"] = [datetime.fromisoformat(f"{d} {t}").replace(tzinfo=et).astimezone(timezone.utc) for d, t in zip(sch.gameday, sch.gametime)]
tg = pd.concat([sch.assign(team=sch.home_team), sch.assign(team=sch.away_team)])[["game_id", "season", "week", "team", "kick"]]
tg = tg.sort_values(["team", "kick"])
tg["prev_game_id"] = tg.groupby("team").game_id.shift()

# ------------------------------------------------------------------ old format (2021-24): weekly, coarse tiers
old = con.execute("""SELECT season, week, club_code AS team, gsis_id AS player_id, position, depth_position, depth_team
                     FROM depth_charts WHERE season BETWEEN 2021 AND 2024 AND gsis_id IS NOT NULL""").df()
old["pos"] = old.depth_position.str.strip().map(POS)
old = old[old.pos.notna() & old.position.isin(["WR", "TE", "RB"])]
old["tier"] = pd.to_numeric(old.depth_team, errors="coerce").clip(upper=3)
old = old.groupby(["season", "week", "team", "player_id", "pos"]).tier.min().reset_index()
old = old.merge(tg[["season", "week", "team", "game_id"]], on=["season", "week", "team"])

# ------------------------------------------------------------------ new format (2025+): latest snapshot before kickoff
new = con.execute("""SELECT dt, team, gsis_id AS player_id, pos_abb AS pos, pos_rank FROM depth_charts
                     WHERE dt IS NOT NULL AND pos_abb IN ('WR', 'TE', 'RB') AND gsis_id IS NOT NULL""").df()
new["dt"] = pd.to_datetime(new.dt, utc=True)
snaps = new[["team", "dt"]].drop_duplicates().sort_values("dt")
g25 = tg[tg.season >= 2025].sort_values("kick")
pick = pd.merge_asof(g25, snaps.rename(columns={"dt": "snap_dt"}), left_on="kick", right_on="snap_dt", by="team",
                     allow_exact_matches=False, direction="backward")
pick = pick[pick.snap_dt.notna() & (pick.kick - pick.snap_dt < pd.Timedelta(days=4))]   # stale charts are not "this week"
nw = pick[["game_id", "season", "week", "team", "snap_dt"]].merge(new.rename(columns={"dt": "snap_dt"}), on=["team", "snap_dt"])
cut = {"WR": [3, 6], "TE": [1, 2], "RB": [1, 2]}   # fine rank -> tier: WR1-3 starters, WR4-6 backups, etc.
nw["tier"] = [1 if r <= cut[p][0] else 2 if r <= cut[p][1] else 3 for p, r in zip(nw.pos, nw.pos_rank)]
nw = nw.rename(columns={"pos_rank": "fine_rank"})

# ------------------------------------------------------------------ combine + change features
dc = pd.concat([old[["game_id", "team", "player_id", "pos", "tier"]], nw[["game_id", "team", "player_id", "pos", "tier", "fine_rank", "snap_dt"]]])
dc = dc.sort_values("tier").drop_duplicates(["game_id", "player_id"])
base = pd.read_parquet(ROOT / "data/player_games.parquet", columns=["player_id", "game_id", "team", "position"]).rename(columns={"position": "pos"})
base = base.merge(tg[["game_id", "team", "prev_game_id"]], on=["game_id", "team"], how="left")
F = base.merge(dc[["game_id", "player_id", "tier", "fine_rank"]], on=["game_id", "player_id"], how="left")
charted_games = set(dc.game_id)
F["on_chart"] = F.tier.notna().astype(float)
F.loc[~F.game_id.isin(charted_games), "on_chart"] = np.nan          # no chart at all for that game -> unknown
F["depth_tier"] = F.tier.fillna(4).where(F.on_chart.notna())
prev = dc[["game_id", "player_id", "tier", "fine_rank"]].rename(columns={"game_id": "prev_game_id", "tier": "prev_tier", "fine_rank": "prev_fine_rank"})
F = F.merge(prev, on=["prev_game_id", "player_id"], how="left")
F["prev_depth_tier"] = F.prev_tier.fillna(4).where(F.prev_game_id.isin(charted_games))
F["depth_tier_change"] = F.prev_depth_tier - F.depth_tier                 # + = promoted
F["promoted"] = (F.depth_tier_change > 0).astype(float).where(F.depth_tier_change.notna())
F["demoted"] = (F.depth_tier_change < 0).astype(float).where(F.depth_tier_change.notna())
F["fine_rank_change"] = F.prev_fine_rank - F.fine_rank
# starter slots opened at the position: last game's tier-1 players at this team+pos who are not tier-1 now
st_now = dc[dc.tier == 1].groupby(["game_id", "team", "pos"]).player_id.apply(set).rename("now_st")
st_prev = dc[dc.tier == 1].groupby(["game_id", "team", "pos"]).player_id.apply(set).rename("prev_st").reset_index().rename(columns={"game_id": "prev_game_id"})
slots = tg[["game_id", "team", "prev_game_id"]].merge(st_now.reset_index(), on=["game_id", "team"]).merge(st_prev, on=["prev_game_id", "team", "pos"])
slots["starter_slots_opened"] = [len(p - n) for p, n in zip(slots.prev_st, slots.now_st)]
F = F.merge(slots[["game_id", "team", "pos", "starter_slots_opened"]], on=["game_id", "team", "pos"], how="left")
# 2025+: players listed above him last game who are gone from the chart or now below him
above = []
nwk = nw.set_index(["game_id", "team", "pos"])
for r in F[F.fine_rank.notna() & F.prev_fine_rank.notna()].itertuples():
    try:
        pv = nwk.loc[(r.prev_game_id, r.team, r.pos)]; cur = nwk.loc[(r.game_id, r.team, r.pos)]
    except KeyError:
        continue
    above_prev = set(pv[pv.fine_rank < r.prev_fine_rank].player_id)
    cur_rank = dict(zip(cur.player_id, cur.fine_rank))
    above.append((r.player_id, r.game_id, sum(1 for p in above_prev if p not in cur_rank or cur_rank[p] > r.fine_rank)))
F = F.merge(pd.DataFrame(above, columns=["player_id", "game_id", "above_removed"]), on=["player_id", "game_id"], how="left")
F = F[["player_id", "game_id", "on_chart", "depth_tier", "prev_depth_tier", "depth_tier_change", "promoted", "demoted",
       "starter_slots_opened", "fine_rank", "fine_rank_change", "above_removed"]]
F.to_parquet(ROOT / "data/depth_features.parquet")
print(F.shape)
print(F.notna().mean().round(3).to_string())
print(F.depth_tier.value_counts(dropna=False).to_string())
