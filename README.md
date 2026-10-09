# Surprise-inactive receiver unders

One NFL betting edge that survived a season-length search through game lines, totals, quarter and half markets,
player-prop models and prediction-market liquidity. The full search, including everything that didn't work, is
in [`docs/RESEARCH.md`](docs/RESEARCH.md).

## The rule

About **90 minutes before kickoff**, NFL teams publish their inactive lists.

1. **Trigger:** a WR, TE or RB is **inactive** but was **not listed Out or Doubtful** on that week's final injury
   report, and he had **≥ 10% of his team's targets** over his previous 3 games.
2. **Bet:** the **under** on each remaining teammate's **receptions** and **receiving yards**.
3. **Where:**
   - **Novig** first. Take the listed under at the consensus line.
   - **Kalshi** if Novig doesn't list the prop. Buy NO on the ladder contract that matches the line ("5+
     receptions" for under 4.5), **only when the spread is ≤ 4¢**. Otherwise rest a NO bid at the mid until kickoff.
     Never take a wide Kalshi book: the spread eats the whole edge.
4. **Stake:** flat and small. About 1–2 triggers a week, 15–20 bets.

## Why it works

When a role player is a surprise scratch, books and bettors push his teammates' receiving lines up to absorb the
freed targets, and they push them too far. The freed volume spreads across more players and a less efficient
offense than the lines assume. The market prices absences it has known about for days correctly (no edge there).
The overreaction is specific to **late, surprise** scratches, and it grows with the size of the vacated role.

## Evidence

### Backtest, 2023–2026
All triggers use only information available before kickoff: the scratched player's usage from earlier games,
Friday's report and game-day inactives. Graded at the closing line, with SEs clustered by team-game.

| | Result |
|---|---|
| Qualifying team-games | 217 |
| Teammates' receiving overs vs. de-vigged fair price | 43.9% vs 49.7% |
| Edge beyond the normal receiving under lean | −4.0 pts, t = −2.6 |
| ROI at the best sportsbook price | **+5.4% to +7.5%** (95% CI on the broad prop set: +0.2% to +10.6%) |
| ROI at Novig/ProphetX prices | **+10.2%** (584 bets) |
| ROI at Novig only, 2025 (Novig listed 87.5% of qualifying props) | **+11.5%** (70 team-games); Novig baseline for all receiving unders: +0.9% |
| By season, best book | 2023 +0.7%, 2024 +8.5%, 2025 +13.1% |
| Dose-response (vacated target share 10–18% / 18–30% / 30%+) | −3.3 / −9.6 / −7.9 pts vs. fair |
| Kalshi (2025–26, priced 1 min after inactives) | taking any ask −1.1%; spread ≤ 4¢ **+13.1%** (105 contracts); at the mid +11.6% |
| Probability the true ROI > 0 (bootstrap) | ~98% |

### Live paper test, 2026 (weeks 2–4)
| Venue | Bets | Win rate | ROI |
|---|---|---|---|
| Novig | 41 | 56% | +10.2% |
| Best exchange (Novig/ProphetX) | 65 | 62% | +16.4% |
| Best sportsbook (reference only) | 71 | 59% | +10.5% |
| Kalshi, spread ≤ 4¢ | 64 | 66% | +19.0% |
| Kalshi, resting at the mid | 37 filled of 42 | 62% | +20.5% |

Six triggered team-games (Puka Nacua, RJ Harvey, Adonai Mitchell, Jalen Coker, Keenan Allen, Terry McLaurin).
Weeks 2–3 overlap the backtest window, so only week 4 is fully out of sample. Running totals are in
[`paper/TARGETS_FREED.md`](paper/TARGETS_FREED.md).

### Projection
Backtest winners shrink going forward, so plan on **about +5% on Novig** (90% range roughly −2% to +12%).
About 50 more triggered team-games (roughly the rest of this season) confirms or rules out +5% at 2 standard errors.

## Caveats
- Small sample: about 55 qualifying team-games a season, and the bets inside one game move together.
- 2023 was roughly flat. Most of the profit is 2024–25.
- Exchange liquidity is probably a few hundred dollars per prop.
- You have about 90 minutes between the inactives announcement and kickoff to act.
- Clear any real trading with your employer's compliance rules first.

## Automation (paper only)

`.github/workflows/paper-maker.yml` runs on GitHub Actions on game days and calls
[`paper/targets_freed.py`](paper/targets_freed.py). No real orders are sent.

- **snapshot** (after inactives, before kickoff): Kalshi receiving-ladder order books, and sportsbook and
  Novig/ProphetX odds via The Odds API (`ODDS_API_KEY` repo secret, ~4 credits a game).
- **backfill**: GitHub's scheduler drops many runs. Any game the live run missed is rebuilt afterward from
  Kalshi's 1-minute candles at T−85 (free) and The Odds API's history (paid plan only).
- **grade** (after nflverse publishes game-day rosters): finds triggers and grades every venue: Novig, best
  exchange, best sportsbook, Kalshi taking (all / tight spread), and Kalshi resting at the mid (filled from the real
  trade tape behind the queue). Rules match book settlement: active with snaps but no stats counts as 0, and an
  inactive player or one with no snaps is void.
- **summarize**: writes [`paper/TARGETS_FREED.md`](paper/TARGETS_FREED.md).

Run it by hand: Actions tab → *kalshi-paper-maker* → *Run workflow*, or locally:

```bash
uv run --no-project --python 3.12 --with requests --with pandas --with pyarrow python paper/targets_freed.py
```

### Rebuilding the backtest
`scripts/news_window.py` (trigger and line-movement analysis), `scripts/targets_freed_pnl.py` (P&L by season and
venue), `scripts/targets_freed_expand.py` (bootstrap, variants vs. baseline), `scripts/novig_check.py`
(Novig-only) and `scripts/kalshi_targets_freed.py` (Kalshi order-book pricing). These need the local research
database. See [`docs/RESEARCH.md`](docs/RESEARCH.md).
