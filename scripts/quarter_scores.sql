-- Points by quarter for every game (1999+), from cumulative scores in play-by-play. qtr 5 = overtime.
CREATE OR REPLACE TABLE quarter_scores AS
WITH endq AS (
  SELECT game_id, qtr, max(total_home_score) AS h, max(total_away_score) AS a
  FROM pbp WHERE qtr BETWEEN 1 AND 5 GROUP BY 1, 2
), filled AS (   -- carry the score forward through quarters with no plays logged (rare)
  SELECT g.game_id, q.qtr,
         max(e.h) OVER (PARTITION BY g.game_id ORDER BY q.qtr ROWS UNBOUNDED PRECEDING) AS h_cum,
         max(e.a) OVER (PARTITION BY g.game_id ORDER BY q.qtr ROWS UNBOUNDED PRECEDING) AS a_cum
  FROM (SELECT DISTINCT game_id FROM endq) g CROSS JOIN range(1, 6) q(qtr)
  LEFT JOIN endq e ON e.game_id = g.game_id AND e.qtr = q.qtr
)
SELECT game_id, qtr,
       coalesce(h_cum, 0) - coalesce(lag(h_cum) OVER w, 0) AS home_pts,
       coalesce(a_cum, 0) - coalesce(lag(a_cum) OVER w, 0) AS away_pts
FROM filled WINDOW w AS (PARTITION BY game_id ORDER BY qtr);

CREATE OR REPLACE TABLE game_periods AS
SELECT g.game_id, g.season, g.week, g.home_team, g.away_team, g.spread_line, g.total_line, g.result, g.total, g.roof, g.wind_mph,
       sum(home_pts) FILTER (WHERE qtr = 1) q1_h, sum(away_pts) FILTER (WHERE qtr = 1) q1_a,
       sum(home_pts) FILTER (WHERE qtr = 2) q2_h, sum(away_pts) FILTER (WHERE qtr = 2) q2_a,
       sum(home_pts) FILTER (WHERE qtr = 3) q3_h, sum(away_pts) FILTER (WHERE qtr = 3) q3_a,
       sum(home_pts) FILTER (WHERE qtr = 4) q4_h, sum(away_pts) FILTER (WHERE qtr = 4) q4_a,
       sum(home_pts) FILTER (WHERE qtr = 5) ot_h, sum(away_pts) FILTER (WHERE qtr = 5) ot_a
FROM games g JOIN quarter_scores q USING (game_id) GROUP BY ALL;
