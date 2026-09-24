#!/usr/bin/env bash
# Build data/nfl.duckdb from data/raw parquet. One table per file pattern (yearly files unioned).
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f data/nfl.duckdb
{
t() { echo "CREATE TABLE $1 AS SELECT * FROM read_parquet('data/raw/$2', union_by_name=true);"; }
t schedules            'schedules/games.parquet'
t pbp                  'pbp/play_by_play_*.parquet'
t pbp_participation    'pbp_participation/pbp_participation_[0-9]*.parquet'
t stats_player_week    'stats_player/stats_player_week_*.parquet'
t stats_player_reg     'stats_player/stats_player_reg_[0-9]*.parquet'
t stats_player_post    'stats_player/stats_player_post_*.parquet'
t stats_team_week      'stats_team/stats_team_week_*.parquet'
t stats_team_reg       'stats_team/stats_team_reg_[0-9]*.parquet'
t stats_team_post      'stats_team/stats_team_post_*.parquet'
t player_stats_def     'player_stats/player_stats_def_[0-9]*.parquet'
t player_stats_kicking 'player_stats/player_stats_kicking_[0-9]*.parquet'
t snap_counts          'snap_counts/*.parquet'
t injuries             'injuries/*.parquet'
t depth_charts         'depth_charts/*.parquet'
t rosters              'rosters/*.parquet'
t rosters_weekly       'weekly_rosters/*.parquet'
t ftn_charting         'ftn_charting/*.parquet'
t ngs_passing          'nextgen_stats/ngs_passing.parquet'
t ngs_rushing          'nextgen_stats/ngs_rushing.parquet'
t ngs_receiving        'nextgen_stats/ngs_receiving.parquet'
for k in def pass rec rush; do
t pfr_adv_week_$k      "pfr_advstats/advstats_week_${k}_*.parquet"
t pfr_adv_season_$k    "pfr_advstats/advstats_season_${k}.parquet"
done
t qbr_week             'espn_data/qbr_week_level.parquet'
t qbr_season           'espn_data/qbr_season_level.parquet'
t players              'players/players.parquet'
t otc_players          'players_components/otc_players.parquet'
t pfr_rosters          'misc/pfr_rosters.parquet'
t contracts            'contracts/historical_contracts.parquet'
t draft_picks          'draft_picks/draft_picks.parquet'
t combine              'combine/combine.parquet'
t officials            'officials/officials.parquet'
t teams                'teams/teams_colors_logos.parquet'
t trades               'trades/trades.parquet'
} | duckdb data/nfl.duckdb
duckdb data/nfl.duckdb -c "SELECT table_name, estimated_size AS rows, column_count AS cols FROM duckdb_tables() ORDER BY 1"
ls -lh data/nfl.duckdb
