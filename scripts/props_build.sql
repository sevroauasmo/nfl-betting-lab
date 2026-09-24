-- Turn odds_raw prop rows into graded, analysis-ready tables.
--   prop_lines   : one row per (game, pass, book, market, player, point) with over/under (or yes/no) prices paired
--   prop_results : prop_lines joined to the player's actual stat line from nflverse, graded over/under/push
-- Player matching: same game_id + normalized name (lowercase, no punctuation, no Jr/Sr/II/III/IV suffix).

CREATE OR REPLACE MACRO norm_name(s) AS
  trim(regexp_replace(regexp_replace(regexp_replace(lower(s), '\([^)]*\)', '', 'g'), '[^a-z ]', '', 'g'), '\s+(jr|sr|ii|iii|iv|v)$', ''));
CREATE OR REPLACE MACRO last_tok(k) AS regexp_extract(k, '(\S+)$', 1);

CREATE OR REPLACE TABLE prop_lines AS
SELECT game_id, event_id, pass, snapshot_ts, commence_time, bookmaker, market,
       player, norm_name(player) AS player_key, point,
       max(price) FILTER (WHERE outcome IN ('Over', 'Yes')) AS over_price,
       max(price) FILTER (WHERE outcome IN ('Under', 'No')) AS under_price,
       max(last_update) AS last_update
FROM odds_raw
WHERE market LIKE 'player_%' AND player IS NOT NULL
GROUP BY ALL;

CREATE OR REPLACE TABLE player_game_stats AS
SELECT s.game_id, s.player_id, s.player_display_name, norm_name(s.player_display_name) AS player_key, s.position,
       s.team, s.opponent_team,
       s.completions, s.attempts, s.passing_yards, s.passing_tds, s.passing_interceptions,
       s.carries, s.rushing_yards, s.rushing_tds,
       s.receptions, s.targets, s.receiving_yards, s.receiving_tds,
       s.fg_made, s.pat_made, s.def_sacks, s.def_tackles_solo, s.def_tackle_assists, s.def_interceptions, s.special_teams_tds,
       -- books settle on official combined tackles (solo + assisted); prefer PFR's number, else rebuild it from nflverse parts
       coalesce(p.def_tackles_combined, s.def_tackles_solo + s.def_tackles_with_assist + s.def_tackle_assists) AS tackles_combined
FROM stats_player_week s
LEFT JOIN pfr_adv_week_def p ON p.game_id = s.game_id AND norm_name(p.pfr_player_name) = norm_name(s.player_display_name)
WHERE s.season >= 2023;

-- Longest plays come from play-by-play.
CREATE OR REPLACE TABLE player_game_longest AS
SELECT game_id, player_id, max(longest_rec) AS longest_rec, max(longest_rush) AS longest_rush, max(longest_comp) AS longest_comp
FROM (
  SELECT game_id, receiver_player_id AS player_id, CASE WHEN complete_pass = 1 THEN yards_gained END AS longest_rec, NULL::DOUBLE AS longest_rush, NULL::DOUBLE AS longest_comp
  FROM pbp WHERE season >= 2023 AND receiver_player_id IS NOT NULL
  UNION ALL
  SELECT game_id, rusher_player_id, NULL, yards_gained, NULL FROM pbp WHERE season >= 2023 AND rusher_player_id IS NOT NULL AND play_type = 'run'
  UNION ALL
  SELECT game_id, passer_player_id, NULL, NULL, CASE WHEN complete_pass = 1 THEN yards_gained END FROM pbp WHERE season >= 2023 AND passer_player_id IS NOT NULL
) GROUP BY ALL;

-- Name match: exact normalized name within the game, else a unique last-name match within the game
-- (handles Gabriel/Gabe Davis, Chigoziem/Chig Okonkwo, Nathaniel/Tank Dell, Joshua/Josh Palmer).
CREATE OR REPLACE TABLE prop_player_map AS
WITH names AS (SELECT DISTINCT game_id, player_key FROM prop_lines),
exact AS (
  SELECT n.game_id, n.player_key, s.player_id FROM names n JOIN player_game_stats s USING (game_id, player_key)
),
initial AS (  -- same last name + first initial, unique in game (Gabriel/Gabe Davis next to Ray Davis)
  SELECT n.game_id, n.player_key, any_value(s.player_id) AS player_id
  FROM names n JOIN player_game_stats s ON s.game_id = n.game_id AND last_tok(s.player_key) = last_tok(n.player_key)
                                        AND left(s.player_key, 1) = left(n.player_key, 1)
  WHERE NOT EXISTS (SELECT 1 FROM exact e WHERE e.game_id = n.game_id AND e.player_key = n.player_key)
  GROUP BY 1, 2 HAVING count(*) = 1
),
fallback AS (  -- same last name, unique in game (Nathaniel/Tank Dell)
  SELECT n.game_id, n.player_key, any_value(s.player_id) AS player_id
  FROM names n JOIN player_game_stats s ON s.game_id = n.game_id AND last_tok(s.player_key) = last_tok(n.player_key)
  WHERE NOT EXISTS (SELECT 1 FROM exact e WHERE e.game_id = n.game_id AND e.player_key = n.player_key)
    AND NOT EXISTS (SELECT 1 FROM initial i WHERE i.game_id = n.game_id AND i.player_key = n.player_key)
  GROUP BY 1, 2 HAVING count(*) = 1
)
SELECT * FROM exact UNION ALL SELECT * FROM initial UNION ALL SELECT * FROM fallback;

CREATE OR REPLACE TABLE prop_results AS
WITH j AS (
  SELECT l.*, s.player_id, s.position, s.team,
    CASE l.market
      WHEN 'player_pass_yds' THEN s.passing_yards        WHEN 'player_pass_yds_alternate' THEN s.passing_yards
      WHEN 'player_pass_tds' THEN s.passing_tds          WHEN 'player_pass_tds_alternate' THEN s.passing_tds
      WHEN 'player_pass_attempts' THEN s.attempts        WHEN 'player_pass_attempts_alternate' THEN s.attempts
      WHEN 'player_pass_completions' THEN s.completions  WHEN 'player_pass_completions_alternate' THEN s.completions
      WHEN 'player_pass_interceptions' THEN s.passing_interceptions WHEN 'player_pass_interceptions_alternate' THEN s.passing_interceptions
      WHEN 'player_rush_yds' THEN s.rushing_yards        WHEN 'player_rush_yds_alternate' THEN s.rushing_yards
      WHEN 'player_rush_attempts' THEN s.carries         WHEN 'player_rush_attempts_alternate' THEN s.carries
      WHEN 'player_rush_tds' THEN s.rushing_tds          WHEN 'player_rush_tds_alternate' THEN s.rushing_tds
      WHEN 'player_receptions' THEN s.receptions         WHEN 'player_receptions_alternate' THEN s.receptions
      WHEN 'player_reception_yds' THEN s.receiving_yards WHEN 'player_reception_yds_alternate' THEN s.receiving_yards
      WHEN 'player_reception_tds' THEN s.receiving_tds   WHEN 'player_reception_tds_alternate' THEN s.receiving_tds
      WHEN 'player_rush_reception_yds' THEN s.rushing_yards + s.receiving_yards
      WHEN 'player_rush_reception_yds_alternate' THEN s.rushing_yards + s.receiving_yards
      WHEN 'player_pass_rush_yds' THEN s.passing_yards + s.rushing_yards
      WHEN 'player_pass_rush_yds_alternate' THEN s.passing_yards + s.rushing_yards
      WHEN 'player_pass_rush_reception_yds' THEN s.passing_yards + s.rushing_yards + s.receiving_yards
      WHEN 'player_rush_reception_tds' THEN s.rushing_tds + s.receiving_tds
      WHEN 'player_anytime_td' THEN s.rushing_tds + s.receiving_tds + coalesce(s.special_teams_tds, 0)
      WHEN 'player_tds_over' THEN s.rushing_tds + s.receiving_tds + coalesce(s.special_teams_tds, 0)
      WHEN 'player_field_goals' THEN s.fg_made           WHEN 'player_field_goals_alternate' THEN s.fg_made
      WHEN 'player_kicking_points' THEN 3 * s.fg_made + s.pat_made
      WHEN 'player_kicking_points_alternate' THEN 3 * s.fg_made + s.pat_made
      WHEN 'player_pats' THEN s.pat_made
      WHEN 'player_sacks' THEN s.def_sacks               WHEN 'player_sacks_alternate' THEN s.def_sacks
      WHEN 'player_solo_tackles' THEN s.def_tackles_solo
      WHEN 'player_tackles_assists' THEN s.tackles_combined
      WHEN 'player_tackles_assists_alternate' THEN s.tackles_combined
      WHEN 'player_defensive_interceptions' THEN s.def_interceptions
      WHEN 'player_reception_longest' THEN CASE WHEN s.player_id IS NOT NULL THEN coalesce(lg.longest_rec, 0) END  -- played, no catch = 0
      WHEN 'player_rush_longest' THEN CASE WHEN s.player_id IS NOT NULL THEN coalesce(lg.longest_rush, 0) END
      WHEN 'player_pass_longest_completion' THEN CASE WHEN s.player_id IS NOT NULL THEN coalesce(lg.longest_comp, 0) END
    END AS actual
  FROM prop_lines l
  LEFT JOIN prop_player_map pm ON pm.game_id = l.game_id AND pm.player_key = l.player_key
  LEFT JOIN player_game_stats s ON s.game_id = l.game_id AND s.player_id = pm.player_id
  LEFT JOIN player_game_longest lg ON lg.game_id = l.game_id AND lg.player_id = s.player_id
)
SELECT *,
  -- Yes/No markets (anytime TD, 1st/last TD) have no point: treat as over 0.5
  coalesce(point, 0.5) AS line,
  CASE WHEN actual IS NULL THEN NULL
       WHEN actual > coalesce(point, 0.5) THEN 'over'
       WHEN actual < coalesce(point, 0.5) THEN 'under' ELSE 'push' END AS result
FROM j
WHERE market NOT IN ('player_1st_td', 'player_last_td', 'player_pass_yds_q1');  -- need play-level ordering; graded separately if ever needed
