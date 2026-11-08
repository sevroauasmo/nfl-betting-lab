# Paper test: surprise inactive -> teammates' receiving unders

_Updated 2026-11-08 13:25 UTC. Prices snapshotted ~T-85 (after inactives); nothing is actually bet._

Games snapshotted: 123 (62 rebuilt after the fact from 1-minute Kalshi candles / odds history) · graded: 15 with a trigger, 107 without · pending: 0

| strategy | bets | fills | win rate | staked (units) | P&L (units) | ROI |
|---|---|---|---|---|---|---|
| book_best | 167 | 167 | 62.3% | 167.0 | +27.81 | +16.7% |
| exchange_best | 158 | 158 | 63.3% | 158.0 | +31.37 | +19.9% |
| kalshi_rest_mid | 158 | 135 | 61.5% | 71.2 | +11.64 | +16.3% |
| kalshi_take_all | 180 | 180 | 61.1% | 99.4 | +10.56 | +10.6% |
| kalshi_take_tight | 179 | 179 | 60.9% | 99.0 | +10.04 | +10.1% |
| novig | 134 | 134 | 61.9% | 134.0 | +24.07 | +18.0% |

Kalshi bets stake the contract cost, so ROI is per dollar risked. Book/exchange bets stake 1 unit at American odds.
