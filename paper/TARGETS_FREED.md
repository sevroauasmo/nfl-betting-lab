# Paper test: surprise inactive -> teammates' receiving unders

_Updated 2026-11-12 22:30 UTC. Prices snapshotted ~T-85 (after inactives); nothing is actually bet._

Games snapshotted: 136 (62 rebuilt after the fact from 1-minute Kalshi candles / odds history) · graded: 17 with a trigger, 119 without · pending: 0

| strategy | bets | fills | win rate | staked (units) | P&L (units) | ROI |
|---|---|---|---|---|---|---|
| book_best | 198 | 198 | 61.1% | 198.0 | +29.40 | +14.8% |
| exchange_best | 189 | 189 | 61.9% | 189.0 | +33.95 | +18.0% |
| kalshi_rest_mid | 189 | 158 | 61.4% | 83.1 | +13.18 | +15.9% |
| kalshi_take_all | 211 | 211 | 60.2% | 116.9 | +10.13 | +8.7% |
| kalshi_take_tight | 210 | 210 | 60.0% | 116.4 | +9.61 | +8.3% |
| novig | 165 | 165 | 60.6% | 165.0 | +26.61 | +16.1% |

Kalshi bets stake the contract cost, so ROI is per dollar risked. Book/exchange bets stake 1 unit at American odds.
