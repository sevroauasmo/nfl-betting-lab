# Paper test: surprise inactive -> teammates' receiving unders

_Updated 2026-10-18 19:20 UTC. Prices snapshotted ~T-85 (after inactives); nothing is actually bet._

Games snapshotted: 91 (62 rebuilt after the fact from 1-minute Kalshi candles / odds history) · graded: 7 with a trigger, 74 without · pending: 0

| strategy | bets | fills | win rate | staked (units) | P&L (units) | ROI |
|---|---|---|---|---|---|---|
| book_best | 95 | 95 | 63.2% | 95.0 | +18.35 | +19.3% |
| exchange_best | 89 | 89 | 65.2% | 89.0 | +22.38 | +25.2% |
| kalshi_rest_mid | 63 | 56 | 66.1% | 29.3 | +7.71 | +26.3% |
| kalshi_take_all | 85 | 85 | 67.1% | 46.9 | +10.09 | +21.5% |
| kalshi_take_tight | 85 | 85 | 67.1% | 46.9 | +10.09 | +21.5% |
| novig | 65 | 65 | 63.1% | 65.0 | +15.89 | +24.4% |

Kalshi bets stake the contract cost, so ROI is per dollar risked. Book/exchange bets stake 1 unit at American odds.
