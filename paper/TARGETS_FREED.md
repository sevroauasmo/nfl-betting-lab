# Paper test: surprise inactive -> teammates' receiving unders

_Updated 2026-11-16 23:10 UTC. Prices snapshotted ~T-85 (after inactives); nothing is actually bet._

Games snapshotted: 149 (62 rebuilt after the fact from 1-minute Kalshi candles / odds history) · graded: 21 with a trigger, 128 without · pending: 0

| strategy | bets | fills | win rate | staked (units) | P&L (units) | ROI |
|---|---|---|---|---|---|---|
| book_best | 246 | 246 | 59.8% | 246.0 | +30.77 | +12.5% |
| exchange_best | 235 | 235 | 60.4% | 235.0 | +37.09 | +15.8% |
| kalshi_rest_mid | 237 | 193 | 61.1% | 102.2 | +15.10 | +14.8% |
| kalshi_take_all | 259 | 259 | 59.5% | 143.4 | +10.61 | +7.4% |
| kalshi_take_tight | 257 | 257 | 59.5% | 142.2 | +10.81 | +7.6% |
| novig | 209 | 209 | 59.8% | 209.0 | +31.55 | +15.1% |

Kalshi bets stake the contract cost, so ROI is per dollar risked. Book/exchange bets stake 1 unit at American odds.
