-- Derived betting tables built on top of `schedules`.
--   games      : one row per game, betting outcomes + game context
--   team_games : one row per team per game (two per game), side-specific ATS/ML outcomes
--                plus situational features computed only from info known before kickoff.
-- Conventions (nflverse): spread_line > 0 means HOME favored by that many; result = home - away.

CREATE OR REPLACE TABLE games AS
WITH tz AS (
  SELECT * FROM (VALUES
    ('SEA','P'),('SF','P'),('OAK','P'),('LV','P'),('LA','P'),('LAC','P'),('SD','P'),
    ('ARI','M'),('DEN','M'),
    ('CHI','C'),('GB','C'),('MIN','C'),('DAL','C'),('HOU','C'),('KC','C'),('NO','C'),('TEN','C'),('STL','C')
  ) t(team, tz)
)
SELECT
  s.game_id, s.season, s.game_type, s.game_type <> 'REG' AS playoff, s.week,
  CAST(s.gameday AS DATE) AS gameday, s.weekday, s.gametime,
  s.home_team, s.away_team, s.home_score, s.away_score,
  s.location = 'Neutral' AS neutral,
  s.result, s.total, s.overtime,
  s.spread_line, s.total_line, s.home_moneyline, s.away_moneyline,
  s.home_spread_odds, s.away_spread_odds, s.over_odds, s.under_odds,
  s.div_game = 1 AS div_game, s.roof, s.surface, s."temp" AS temp_f, s.wind AS wind_mph,
  s.home_rest, s.away_rest, s.home_qb_id, s.away_qb_id, s.home_qb_name, s.away_qb_name,
  s.home_coach, s.away_coach, s.referee, s.stadium,
  COALESCE(th.tz, 'E') AS home_tz, COALESCE(ta.tz, 'E') AS away_tz,
  s.gametime >= '19:00' AS primetime,
  s.gametime < '14:00' AS early_window,
  -- outcomes
  CASE WHEN s.result > s.spread_line THEN 'home' WHEN s.result < s.spread_line THEN 'away' ELSE 'push' END AS ats_winner,
  CASE WHEN s.total > s.total_line THEN 'over' WHEN s.total < s.total_line THEN 'under' ELSE 'push' END AS ou_result,
  s.result - s.spread_line AS home_ats_margin,
  s.total - s.total_line AS ou_margin
FROM schedules s
LEFT JOIN tz th ON th.team = s.home_team
LEFT JOIN tz ta ON ta.team = s.away_team
WHERE s.result IS NOT NULL AND s.spread_line IS NOT NULL;

CREATE OR REPLACE TABLE team_games AS
WITH sides AS (
  SELECT g.*, 'home' AS side, home_team AS team, away_team AS opp,
         home_score AS pts, away_score AS opp_pts,
         spread_line AS team_line,            -- expected team margin (positive = team favored)
         home_spread_odds AS spread_odds, home_moneyline AS ml, away_moneyline AS opp_ml,
         home_rest AS rest, away_rest AS opp_rest, home_qb_id AS qb_id, home_coach AS coach,
         home_tz AS team_tz
  FROM games g
  UNION ALL
  SELECT g.*, 'away', away_team, home_team, away_score, home_score,
         -spread_line, away_spread_odds, away_moneyline, home_moneyline,
         away_rest, home_rest, away_qb_id, away_coach, away_tz
  FROM games g
),
base AS (
  SELECT *,
    pts - opp_pts AS margin,
    (pts - opp_pts) - team_line AS ats_margin,
    CASE WHEN (pts - opp_pts) > team_line THEN 1 WHEN (pts - opp_pts) < team_line THEN 0 END AS cover,  -- NULL = push
    CASE WHEN pts > opp_pts THEN 1 WHEN pts < opp_pts THEN 0 END AS su_win,
    team_line > 0 AS fav, team_line < 0 AS dog
  FROM sides
),
lagged AS (
  SELECT *,
    ROW_NUMBER() OVER w AS game_num,
    LAG(margin)     OVER w AS prev_margin,
    LAG(ats_margin) OVER w AS prev_ats_margin,
    LAG(cover)      OVER w AS prev_cover,
    LAG(su_win)     OVER w AS prev_su_win,
    LAG(fav)        OVER w AS prev_fav,
    LAG(side)       OVER w AS prev_side,
    LAG(primetime)  OVER w AS prev_primetime,
    LAG(total)      OVER w AS prev_total,
    LAG(ou_result)  OVER w AS prev_ou,
    LAG(qb_id)      OVER w AS prev_qb_id,
    LAG(cover, 2)   OVER w AS prev2_cover,
    LAG(su_win, 2)  OVER w AS prev2_su_win,
    SUM(cover)  OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS ats_w_td,
    COUNT(cover) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS ats_n_td,
    SUM(su_win)  OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS su_w_td,
    COUNT(*)     OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS gp_td,
    SUM(ats_margin) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS ats_margin_td,
    SUM(margin)     OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS margin_td
  FROM base
  WINDOW w AS (PARTITION BY team, season ORDER BY gameday)
),
prev_season AS (
  SELECT team, season + 1 AS season,
         AVG(su_win) FILTER (WHERE NOT playoff) AS prev_season_win_pct,
         BOOL_OR(playoff) AS prev_season_playoffs,
         AVG(cover) FILTER (WHERE NOT playoff) AS prev_season_ats_pct
  FROM base GROUP BY 1, 2
)
SELECT l.*,
  p.prev_season_win_pct, p.prev_season_playoffs, p.prev_season_ats_pct,
  o.su_w_td AS opp_su_w_td, o.gp_td AS opp_gp_td, o.ats_w_td AS opp_ats_w_td, o.ats_n_td AS opp_ats_n_td,
  o.prev_su_win AS opp_prev_su_win, o.prev_cover AS opp_prev_cover, o.prev_margin AS opp_prev_margin,
  o.prev_ats_margin AS opp_prev_ats_margin, o.margin_td AS opp_margin_td,
  o.prev_season_win_pct AS opp_prev_season_win_pct,
  (l.qb_id IS DISTINCT FROM l.prev_qb_id AND l.prev_qb_id IS NOT NULL) AS qb_changed,
  (o.qb_id IS DISTINCT FROM o.prev_qb_id AND o.prev_qb_id IS NOT NULL) AS opp_qb_changed,
  -- American odds -> profit per 1 unit risked
  CASE WHEN l.spread_odds < 0 THEN 100.0 / -l.spread_odds ELSE l.spread_odds / 100.0 END AS spread_payout,
  CASE WHEN l.ml < 0 THEN 100.0 / -l.ml ELSE l.ml / 100.0 END AS ml_payout
FROM lagged l
LEFT JOIN prev_season p USING (team, season)
LEFT JOIN (
  SELECT lg.*, ps.prev_season_win_pct
  FROM lagged lg LEFT JOIN prev_season ps USING (team, season)
) o ON o.game_id = l.game_id AND o.team = l.opp;
