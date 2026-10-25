# Paper test: surprise inactive -> teammates' receiving unders

_Updated 2026-10-25 22:00 UTC. Prices snapshotted ~T-85 (after inactives); nothing is actually bet._

Games snapshotted: 105 (62 rebuilt after the fact from 1-minute Kalshi candles / odds history) · graded: 10 with a trigger, 85 without · pending: 7

| strategy | bets | fills | win rate | staked (units) | P&L (units) | ROI |
|---|---|---|---|---|---|---|
| book_best | 120 | 120 | 62.5% | 120.0 | +20.70 | +17.2% |
| exchange_best | 112 | 112 | 63.4% | 112.0 | +23.72 | +21.2% |
| kalshi_rest_mid | 91 | 82 | 63.4% | 43.2 | +8.78 | +20.3% |
| kalshi_take_all | 113 | 113 | 65.5% | 62.7 | +11.32 | +18.1% |
| kalshi_take_tight | 112 | 112 | 65.2% | 62.2 | +10.80 | +17.4% |
| novig | 88 | 88 | 61.4% | 88.0 | +16.88 | +19.2% |

Kalshi bets stake the contract cost, so ROI is per dollar risked. Book/exchange bets stake 1 unit at American odds.
