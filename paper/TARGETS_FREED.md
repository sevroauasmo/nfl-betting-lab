# Paper test: surprise inactive -> teammates' receiving unders

_Updated 2026-11-26 18:20 UTC. Prices snapshotted ~T-85 (after inactives); nothing is actually bet._

Games snapshotted: 165 (62 rebuilt after the fact from 1-minute Kalshi candles / odds history) · graded: 25 with a trigger, 139 without · pending: 0

| strategy | bets | fills | win rate | staked (units) | P&L (units) | ROI |
|---|---|---|---|---|---|---|
| book_best | 292 | 292 | 60.6% | 292.0 | +42.86 | +14.7% |
| exchange_best | 281 | 281 | 61.2% | 281.0 | +50.38 | +17.9% |
| kalshi_rest_mid | 277 | 218 | 61.9% | 115.2 | +19.10 | +16.6% |
| kalshi_take_all | 299 | 299 | 60.5% | 165.1 | +15.88 | +9.6% |
| kalshi_take_tight | 297 | 297 | 60.6% | 163.9 | +16.08 | +9.8% |
| novig | 255 | 255 | 60.8% | 255.0 | +44.48 | +17.4% |

Kalshi bets stake the contract cost, so ROI is per dollar risked. Book/exchange bets stake 1 unit at American odds.
