# Paper test: surprise inactive -> teammates' receiving unders

_Updated 2026-10-12 19:49 UTC. Prices snapshotted ~T-85 (after inactives); nothing is actually bet._

Games snapshotted: 78 (62 rebuilt after the fact from 1-minute Kalshi candles / odds history) · graded: 6 with a trigger, 72 without · pending: 0

| strategy | bets | fills | win rate | staked (units) | P&L (units) | ROI |
|---|---|---|---|---|---|---|
| book_best | 81 | 81 | 60.5% | 81.0 | +11.10 | +13.7% |
| exchange_best | 75 | 75 | 62.7% | 75.0 | +14.70 | +19.6% |
| kalshi_rest_mid | 50 | 43 | 62.8% | 22.4 | +4.58 | +20.4% |
| kalshi_take_all | 72 | 72 | 65.3% | 39.7 | +7.29 | +18.4% |
| kalshi_take_tight | 72 | 72 | 65.3% | 39.7 | +7.29 | +18.4% |
| novig | 51 | 51 | 58.8% | 51.0 | +8.25 | +16.2% |

Kalshi bets stake the contract cost, so ROI is per dollar risked. Book/exchange bets stake 1 unit at American odds.
