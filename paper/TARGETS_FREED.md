# Paper test: surprise inactive -> teammates' receiving unders

_Updated 2026-10-29 22:35 UTC. Prices snapshotted ~T-85 (after inactives); nothing is actually bet._

Games snapshotted: 107 (62 rebuilt after the fact from 1-minute Kalshi candles / odds history) · graded: 12 with a trigger, 95 without · pending: 0

| strategy | bets | fills | win rate | staked (units) | P&L (units) | ROI |
|---|---|---|---|---|---|---|
| book_best | 120 | 120 | 62.5% | 120.0 | +20.70 | +17.2% |
| exchange_best | 112 | 112 | 63.4% | 112.0 | +23.72 | +21.2% |
| kalshi_rest_mid | 111 | 98 | 60.2% | 51.5 | +7.40 | +14.4% |
| kalshi_take_all | 133 | 133 | 61.7% | 73.5 | +8.54 | +11.6% |
| kalshi_take_tight | 132 | 132 | 61.4% | 73.0 | +8.02 | +11.0% |
| novig | 88 | 88 | 61.4% | 88.0 | +16.88 | +19.2% |

Kalshi bets stake the contract cost, so ROI is per dollar risked. Book/exchange bets stake 1 unit at American odds.
