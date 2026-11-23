# Paper test: surprise inactive -> teammates' receiving unders

_Updated 2026-11-23 23:15 UTC. Prices snapshotted ~T-85 (after inactives); nothing is actually bet._

Games snapshotted: 162 (62 rebuilt after the fact from 1-minute Kalshi candles / odds history) · graded: 24 with a trigger, 138 without · pending: 0

| strategy | bets | fills | win rate | staked (units) | P&L (units) | ROI |
|---|---|---|---|---|---|---|
| book_best | 280 | 280 | 59.6% | 280.0 | +34.75 | +12.4% |
| exchange_best | 269 | 269 | 60.2% | 269.0 | +41.74 | +15.5% |
| kalshi_rest_mid | 265 | 212 | 61.3% | 112.0 | +17.22 | +15.4% |
| kalshi_take_all | 287 | 287 | 59.6% | 158.7 | +12.28 | +7.7% |
| kalshi_take_tight | 285 | 285 | 59.6% | 157.5 | +12.48 | +7.9% |
| novig | 243 | 243 | 59.7% | 243.0 | +35.91 | +14.8% |

Kalshi bets stake the contract cost, so ROI is per dollar risked. Book/exchange bets stake 1 unit at American odds.
