-- One row per team per game with offensive efficiency from play-by-play (pass + run plays only).
CREATE OR REPLACE TABLE team_game_epa AS
SELECT game_id, season, posteam AS team,
  count(*) AS plays,
  avg(epa) AS off_epa,
  avg(epa) FILTER (WHERE wp BETWEEN 0.1 AND 0.9) AS off_epa_neutral,
  avg(success) AS off_sr,
  avg(epa) FILTER (WHERE play_type = 'pass') AS pass_epa,
  avg(epa) FILTER (WHERE play_type = 'run') AS rush_epa,
  avg((play_type = 'pass')::int) FILTER (WHERE wp BETWEEN 0.2 AND 0.8 AND down IN (1,2)) AS early_down_pass_rate,
  sum(interception) + sum(fumble_lost) AS giveaways,
  sum(fumble) AS fumbles, sum(fumble_lost) AS fumbles_lost,
  sum(epa) AS off_epa_total
FROM pbp
WHERE play_type IN ('pass','run') AND epa IS NOT NULL AND posteam IS NOT NULL AND season >= 2009
GROUP BY ALL;
