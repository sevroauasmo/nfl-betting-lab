# Paper test: surprise inactive -> teammates' receiving unders

_Updated 2026-10-09 16:59 UTC. Prices snapshotted ~T-85 (after inactives); nothing is actually bet._

Games snapshotted: 65 (62 rebuilt after the fact from 1-minute Kalshi candles / odds history) · graded: 5 with a trigger, 60 without · pending: 0

| strategy | bets | fills | win rate | staked (units) | P&L (units) | ROI |
|---|---|---|---|---|---|---|
| book_best | 71 | 71 | 59.2% | 71.0 | +7.44 | +10.5% |
| exchange_best | 65 | 65 | 61.5% | 65.0 | +10.64 | +16.4% |
| kalshi_rest_mid | 42 | 37 | 62.2% | 19.1 | +3.91 | +20.5% |
| kalshi_take_all | 64 | 64 | 65.6% | 35.3 | +6.72 | +19.0% |
| kalshi_take_tight | 64 | 64 | 65.6% | 35.3 | +6.72 | +19.0% |
| novig | 41 | 41 | 56.1% | 41.0 | +4.19 | +10.2% |

Kalshi bets stake the contract cost, so ROI is per dollar risked. Book/exchange bets stake 1 unit at American odds.
