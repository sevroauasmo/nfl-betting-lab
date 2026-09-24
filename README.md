# NFL betting lab

Research into whether public NFL data finds an edge in betting markets, plus a live paper-trading test of the one
strategy that survived: **resting orders on Kalshi touchdown-scorer markets**.

- Full write-up: `results/report.html` (published as a private Claude artifact).
- Live paper-trading results: [`paper/SUMMARY.md`](paper/SUMMARY.md) (Kalshi market-maker) and
  [`paper/TARGETS_FREED.md`](paper/TARGETS_FREED.md) (surprise-inactive receiver unders), updated automatically on game days.

## What's automated

`.github/workflows/paper-maker.yml` runs on GitHub's servers on NFL game days (roughly every 20 minutes around kickoff
windows, every few hours on off days). Nothing runs on a local machine. Each run:

1. **place**: for games kicking off within 6h (and again within 1h), snapshot every Kalshi market in the TD-scorer,
   first-TD, spread, total, team-total, game-winner and yardage/reception series. Record paper resting orders on both
   sides (a NO bid, i.e. selling YES at the ask, and a YES bid), at the current best price (`join`, back of the queue)
   and one tick inside (`improve`, front of the queue).
2. **fill**: after kickoff, replay Kalshi's real trade tape from placement to kickoff. An order fills only after
   the size resting ahead of it at placement has traded through. Cancellations ahead of us are ignored, so fills are pessimistic.
3. **settle**: once Kalshi finalizes a market, grade fills and subtract an estimated maker fee.
4. Rewrite `paper/SUMMARY.md` and commit `paper/state/orders.csv` back to the repo.

A second strategy, `paper/targets_freed.py`, runs in the same job. At about T-85 (just after inactives are announced)
it snapshots Kalshi receiving-ladder order books, sportsbook and Novig/ProphetX odds (The Odds API, ~4 credits per game,
`ODDS_API_KEY` repo secret) and ESPN's injury feed. After the game it identifies surprise inactives (INA but not Out/Doubtful,
≥10% target share) and grades paper unders on their teammates' receptions and receiving yards: Kalshi taking the ask
(all / only when the spread ≤ 4¢), Kalshi resting a NO bid at the mid (filled from the trade tape), best sportsbook, and
best exchange.

No real orders are ever sent. The Kalshi public API needs no key. Trigger a run manually from the Actions tab
(`workflow_dispatch`).

## Research pipeline (local)

| Step | Script |
|---|---|
| Download all nflverse data (~700 MB parquet, no key) | `scripts/download_nflverse.sh` |
| Build `data/nfl.duckdb` (39 tables) | `scripts/build_db.sh` |
| Betting tables (`games`, `team_games`) | `scripts/betting_tables.sql` |
| Game-line trend scan (2011-2025) | `scripts/analysis.py` |
| Power ratings, luck, overreaction | `scripts/analysis2.py` |
| Historical odds + props (The Odds API, needs `ODDS_API_KEY` in `.env`) | `scripts/pull_odds.py` |
| Prop grading | `scripts/props_build.sql` |
| Props: bias, line shopping, exchange-referenced EV | `scripts/props_analysis.py` |
| Props: features (3 layers) + walk-forward models | `scripts/props_features*.py`, `scripts/props_deep*.py`, `scripts/props_model.py` |
| Exchanges vs book consensus | `scripts/exchange_ev.py` |
| Forecast-wind unders | `scripts/wind_forecast.py` |
| Kalshi history (public API) + longshot / maker analysis | `scripts/kalshi_fetch.py`, `scripts/kalshi_analysis.py` |
| Report | `scripts/build_report.py` → `results/report.html` |

`data/` is git-ignored (about 2 GB). Rebuild it with the scripts above.
