"""Ideas #3 and #4 on Kalshi NFL markets.

#3 longshot bias:
  - futures: snapshot every still-open contract on the 1st/15th of each month (daily candles), price vs outcome,
    and ROI of buying NO on longshots at the ask (taker) or at the bid (resting order, if filled) after fees.
  - game markets: same idea at 1h / 24h before kickoff (hourly candles).
#4 maker vs taker: from sampled pre-game trade tapes, P&L of the taker vs the resting (maker) side, by price.
Writes results/kalshi.json.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
K = ROOT / "data/kalshi"
sys.path.insert(0, str(ROOT / "scripts"))
R: dict = {}


def fee(p):  # Kalshi taker fee per $1 contract, rounded up to the cent
    return np.ceil(0.07 * p * (1 - p) * 100) / 100


def val(c, side, k="close"):
    d = c.get(side) or {}
    v = d.get(f"{k}_dollars", d.get(k))
    return float(v) if v not in (None, "") else np.nan


def load_candles(tickers):
    rows = []
    for t in tickers:
        p = K / "candles" / f"{t}.json"
        if not p.exists():
            continue
        for c in json.loads(p.read_text()):
            rows.append((t, c["end_period_ts"], val(c, "yes_bid"), val(c, "yes_ask"), val(c, "price"),
                         float(c.get("volume_fp", c.get("volume", 0)) or 0)))
    return pd.DataFrame(rows, columns=["ticker", "ts", "bid", "ask", "last", "vol"])


def stats_block(d, p_col="mid"):
    """Calibration + ROI for buying YES/NO at ask (taker) and at bid (maker, if filled)."""
    d = d.dropna(subset=["bid", "ask"])
    d = d[(d.ask > d.bid) & (d.bid > 0) & (d.ask < 1)]
    y = d.yes.values
    out = {"n": len(d), "avg_price": round(float(d[p_col].mean()), 4), "yes_rate": round(float(y.mean()), 4)}
    # buy YES at ask / NO at (1 - bid) as taker, with fee
    cy = d.ask.values + fee(d.ask.values); cn = (1 - d.bid.values) + fee(1 - d.bid.values)
    out["roi_buy_yes_taker"] = round(float(np.mean(np.where(y == 1, 1 / cy - 1, -1))), 4)
    out["roi_buy_no_taker"] = round(float(np.mean(np.where(y == 0, 1 / cn - 1, -1))), 4)
    # resting orders filled at your own price (bid for YES buyer; 1-ask for NO buyer), no taker fee
    my = d.bid.values; mn = 1 - d.ask.values
    out["roi_buy_yes_maker"] = round(float(np.mean(np.where(y == 1, 1 / my - 1, -1))), 4)
    out["roi_buy_no_maker"] = round(float(np.mean(np.where(y == 0, 1 / mn - 1, -1))), 4)
    prof = np.where(y == 0, 1 / cn - 1, -1)
    out["se_no_taker"] = round(float(prof.std() / math.sqrt(len(d))), 4) if len(d) > 1 else None
    return out


m = pd.read_parquet(K / "markets.parquet")
m["yes"] = (m.result == "yes").astype(float)
BUCKETS = [0, .03, .06, .10, .20, .35, .50, .65, .80, .90, .97, 1]

# ------------------------------------------------------------------ #3a futures
fut = m[(m.kind == "future") & (m.volume > 0)]
fc = load_candles(fut.ticker)
if len(fc):
    fc = fc.merge(fut[["ticker", "series", "event_ticker", "yes", "open_time", "close_time"]], on="ticker")
    fc["date"] = pd.to_datetime(fc.ts, unit="s", utc=True)
    snaps = []
    for (tk), g in fc.groupby("ticker"):
        g = g.sort_values("date")
        close = pd.Timestamp(g.close_time.iloc[0])
        anchors = pd.date_range(g.date.min().normalize(), close, freq="SMS", tz="UTC")  # 1st and 15th
        for a in anchors:
            prior = g[g.date <= a]
            if len(prior) and a < close - pd.Timedelta(days=1):
                r = prior.iloc[-1]
                snaps.append((tk, a, r.bid, r.ask, r.last, r.yes, r.series, r.event_ticker))
    fs = pd.DataFrame(snaps, columns=["ticker", "anchor", "bid", "ask", "last", "yes", "series", "event_ticker"])
    fs["mid"] = (fs.bid + fs.ask) / 2
    fs["b"] = pd.cut(fs.mid, BUCKETS)
    R["futures_n_markets"] = int(fs.ticker.nunique()); R["futures_n_snapshots"] = len(fs)
    R["futures_by_price"] = [{"bucket": str(b), **stats_block(g)} for b, g in fs.groupby("b", observed=True)]
    fs["family"] = np.select([fs.series.str.startswith("KXNFLWINS"), fs.series.isin(["KXSB", "KXNFLAFCCHAMP", "KXNFLNFCCHAMP", "KXAFC", "KXNFC"]),
                              fs.series.str.contains("EAST|WEST|NORTH|SOUTH"), fs.series.str.contains("PLAYOFF|SEED|STAGE")],
                             ["win totals", "SB / conference", "division", "playoffs / seeding"], "awards / other")
    R["futures_longshots_by_family"] = {f: stats_block(g[g.mid <= .10]) for f, g in fs.groupby("family")}
    R["futures_favorites_by_family"] = {f: stats_block(g[g.mid >= .80]) for f, g in fs.groupby("family")}
    # one snapshot per market (earliest) to avoid over-counting long-lived contracts
    first = fs.sort_values("anchor").drop_duplicates("ticker")
    R["futures_first_snapshot_longshots"] = stats_block(first[first.mid <= .10])
    R["futures_first_snapshot_by_price"] = [{"bucket": str(b), **stats_block(g)} for b, g in first.groupby("b", observed=True)]
    fs.drop(columns="b").to_parquet(K / "futures_snapshots.parquet")

ONLY_FUTURES = "futures" in sys.argv[1:]
if ONLY_FUTURES:
    print(json.dumps(R, indent=1, default=str)); sys.exit()

# ------------------------------------------------------------------ #3b game markets at 1h / 24h before kickoff
from kalshi_fetch import kickoffs  # noqa: E402
look = kickoffs()
g = m[(m.kind == "game") & (m.volume > 0)].copy()
g["key"] = g.event_ticker.str.split("-").str[1]
g["kick"] = g.key.map(lambda k: look.get(k, (None, None))[1])
g = g.dropna(subset=["kick"])
have = {f[:-5] for f in os.listdir(K / "candles")}
g = g[g.ticker.isin(have)]
R["game_markets_with_candles"] = int(len(g))
if len(g):
    gc = load_candles(g.ticker).merge(g[["ticker", "series", "yes", "kick"]], on="ticker")
    res = {}
    for h in (1, 24):
        s = gc[gc.ts <= gc.kick - h * 3600].sort_values("ts").groupby("ticker").tail(1)
        s = s[s.ts >= s.kick - (h + 6) * 3600]  # quote must be reasonably fresh
        s["mid"] = (s.bid + s.ask) / 2
        s["b"] = pd.cut(s.mid, BUCKETS)
        res[f"{h}h"] = {"by_price": [{"bucket": str(b), **stats_block(x)} for b, x in s.groupby("b", observed=True)],
                        "longshots_by_series": {ser: stats_block(x[x.mid <= .15]) for ser, x in s.groupby("series") if (x.mid <= .15).sum() >= 30},
                        "spread_cents_median": round(float((s.ask - s.bid).median()), 3)}
    R["game_markets"] = res

# ------------------------------------------------------------------ #4 maker vs taker from trade tapes
tp = K / "trades"
if tp.exists():
    rows = []
    res_map = dict(zip(m.ticker, m.yes)); ser_map = dict(zip(m.ticker, m.series))
    for f in os.listdir(tp):
        t = f[:-5]
        for x in json.loads((tp / f).read_text()):
            rows.append((t, ser_map.get(t), res_map.get(t), x["taker_side"], x["yes_price"], x["count"], x["window"]))
    tr = pd.DataFrame(rows, columns=["ticker", "series", "yes", "taker_side", "yes_price", "count", "window"])
    if len(tr):
        # taker P&L per contract before fees; maker = negative of that
        tp_ = np.where(tr.taker_side == "yes", tr.yes - tr.yes_price, (1 - tr.yes) - (1 - tr.yes_price))
        paid = np.where(tr.taker_side == "yes", tr.yes_price, 1 - tr.yes_price)
        tr["taker_pnl"] = tp_; tr["taker_paid"] = paid; tr["taker_fee"] = fee(paid)
        tr["w"] = tr["count"]

        def agg(d):
            w = d.w.values; tot = w.sum()
            return {"trades": len(d), "contracts": round(float(tot)),
                    "maker_pnl_per_contract_c": round(float(-(d.taker_pnl * w).sum() / tot * 100), 3),
                    "taker_fee_per_contract_c": round(float((d.taker_fee * w).sum() / tot * 100), 3),
                    "maker_return_on_capital": round(float(-(d.taker_pnl * w).sum() / ((1 - d.taker_paid) * w).sum()), 4),
                    "markets": int(d.ticker.nunique())}
        R["maker_taker_all"] = agg(tr)
        R["maker_taker_by_series"] = {s: agg(d) for s, d in tr.groupby("series")}
        R["maker_taker_by_window_h"] = {int(k): agg(d) for k, d in tr.groupby("window")}
        tr["pb"] = pd.cut(tr.taker_paid, [0, .1, .25, .5, .75, .9, 1])
        R["maker_taker_by_taker_price"] = {str(k): agg(d) for k, d in tr.groupby("pb", observed=True)}
        R["maker_taker_by_taker_side"] = {k: agg(d) for k, d in tr.groupby("taker_side")}

(ROOT / "results/kalshi.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps({k: v for k, v in R.items() if k not in ("game_markets",)}, indent=1, default=str)[:6000])
