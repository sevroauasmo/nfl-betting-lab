"""Pull settled Kalshi NFL markets, their price history, and sampled pre-game trades (public API, no key).

  uv run python scripts/kalshi_fetch.py markets   # settled markets for FUTURES + GAME series -> data/kalshi/markets.parquet
  uv run python scripts/kalshi_fetch.py candles   # daily candles (futures), hourly candles (game markets) -> data/kalshi/candles/
  uv run python scripts/kalshi_fetch.py trades    # pre-game trade samples for game markets -> data/kalshi/trades/
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
K = ROOT / "data/kalshi"
B = "https://api.elections.kalshi.com/trade-api/v2"
FUTURES = ["KXSB", "KXNFLAFCCHAMP", "KXNFLNFCCHAMP", "KXAFC", "KXNFC",
           "KXNFLAFCEAST", "KXNFLAFCNORTH", "KXNFLAFCSOUTH", "KXNFLAFCWEST", "KXNFLNFCEAST", "KXNFLNFCNORTH", "KXNFLNFCSOUTH", "KXNFLNFCWEST",
           "KXNFLPLAYOFF", "KXNFL1SEED", "KXNFLSEED", "KXNFLSTAGEOFELIM", "KXNFLMVP", "KXNFLOPOTY", "KXNFLDPOTY", "KXNFLOROTY", "KXNFLDROTY",
           "KXNFLCOTY", "KXNFLCPOTY", "KXNFLOPOY", "KXNFLDPOY", "KXNFLOROY", "KXNFLDROY",
           "KXLEADERNFLPYDS", "KXLEADERNFLPTDS", "KXLEADERNFLRUSHYDS", "KXLEADERNFLRUSHTDS", "KXLEADERNFLRYDS", "KXLEADERNFLRTDS",
           "KXLEADERNFLSACKS", "KXLEADERNFLINT", "KXNFLCOACHOUT", "KXCOACHOUTNFL", "KXNFLWINS"] + \
          [f"KXNFLWINS-{t}" for t in "ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAC KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS".split()]
GAME = ["KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL", "KXNFLTEAMTOTAL", "KXNFLANYTD", "KXNFL2TD", "KXNFLFIRSTTD",
        "KXNFLPASSYDS", "KXNFLRECYDS", "KXNFLRSHYDS", "KXNFLREC", "KXNFLPASSTDS", "KXNFL1H", "KXNFL1HWINNER", "KXNFL1HTOTAL", "KXNFL1HSPREAD"]
S = requests.Session()
S.mount("https://", requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32))
RATE_LIMITED = [0]


def get(path, **params):
    for attempt in range(8):
        try:
            r = S.get(B + path, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt); continue
        if r.status_code == 429 or r.status_code >= 500:
            RATE_LIMITED[0] += 1
            time.sleep(min(30, 1.5 ** attempt)); continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"failed {path} {params}")


def markets():
    rows = []
    for kind, series in (("future", FUTURES), ("game", GAME)):
        for s in series:
          n = 0
          # markets settled before Kalshi's historical cutoff (~late Jul 2026) live under /historical
          for src, path, extra in (("live", "/markets", {"status": "settled"}), ("hist", "/historical/markets", {})):
            cur = None
            while True:
                p = {"series_ticker": s, "limit": 1000, **extra}
                if cur: p["cursor"] = cur
                d = get(path, **p)
                for m in d.get("markets", []):
                    if m.get("result") not in ("yes", "no"):
                        continue
                    rows.append({"kind": kind, "src": src, "series": s, "ticker": m["ticker"], "event_ticker": m["event_ticker"],
                                 "title": m.get("title"), "yes_sub_title": m.get("yes_sub_title"), "result": m.get("result"),
                                 "settle_value": float(m.get("settlement_value_dollars") or "nan"),
                                 "open_time": m.get("open_time"), "close_time": m.get("close_time"),
                                 "occurrence": m.get("occurrence_datetime"), "volume": float(m.get("volume_fp") or 0),
                                 "floor_strike": m.get("floor_strike"), "strike_type": m.get("strike_type")})
                    n += 1
                cur = d.get("cursor")
                if not cur or not d.get("markets"): break
          print(f"{kind:6s} {s}: {n} settled markets", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df.to_parquet(K / "markets.parquet")
    print(df.groupby("kind").agg(markets=("ticker", "size"), volume=("volume", "sum")))


def ts(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def candles():
    m = pd.read_parquet(K / "markets.parquet")
    m = m[m.volume > 0]
    out = K / "candles"; out.mkdir(exist_ok=True)

    def one(r):
        path = out / f"{r.ticker}.json"
        if path.exists():
            return
        end = ts(r.close_time)
        if r.kind == "future":
            start, period = ts(r.open_time), 1440
        else:
            start, period = max(ts(r.open_time), end - 7 * 86400), 60
        cs = []
        while start < end:  # API caps candles per request; walk forward in chunks
            chunk_end = min(end, start + (5000 * period * 60))
            path_ = (f"/historical/markets/{r.ticker}/candlesticks" if r.src == "hist"
                     else f"/series/{r.series}/markets/{r.ticker}/candlesticks")
            d = get(path_, start_ts=start, end_ts=chunk_end, period_interval=period)
            cs += d.get("candlesticks", [])
            start = chunk_end
        path.write_text(json.dumps(cs))

    # public API allows ~4 req/s here: all futures + all game winners, 700 random markets from each other series
    m = pd.concat([d if (k == "future" or s == "KXNFLGAME") else d.sample(min(len(d), 700), random_state=7)
                   for (k, s), d in m.groupby(["kind", "series"])])
    order = {"future": 0, "KXNFLGAME": 1, "KXNFLSPREAD": 2, "KXNFLTOTAL": 2}
    m = m.assign(prio=[order.get(k, order.get(s, 3)) for k, s in zip(m.kind, m.series)]).sort_values("prio")
    todo = [r for r in m.itertuples() if not (out / f"{r.ticker}.json").exists()]
    print(f"candles: {len(todo)} markets to fetch", flush=True)
    with ThreadPoolExecutor(6) as ex:
        for i, f in enumerate(as_completed([ex.submit(one, r) for r in todo])):
            f.result()
            if i % 500 == 0: print(f"  {i}/{len(todo)} (429s so far: {RATE_LIMITED[0]})", flush=True)


def kickoffs():
    """Map Kalshi game events (e.g. KXNFLGAME-26SEP21NYGLAR) to nflverse kickoff times (UTC)."""
    con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
    g = con.execute("SELECT game_id, gameday, gametime, home_team, away_team FROM schedules WHERE season >= 2024").df()
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    alias = {"LA": ["LA", "LAR"], "JAX": ["JAX", "JAC"], "WAS": ["WAS", "WSH"], "LV": ["LV", "LVR"]}
    look = {}
    for r in g.itertuples():
        k = datetime.fromisoformat(f"{r.gameday} {r.gametime}").replace(tzinfo=et).astimezone(timezone.utc)
        dcode = datetime.fromisoformat(str(r.gameday)).strftime("%y%b%d").upper()
        for a in alias.get(r.away_team, [r.away_team]):
            for h in alias.get(r.home_team, [r.home_team]):
                look[f"{dcode}{a}{h}"] = (r.game_id, int(k.timestamp()))
    return look


def trades():
    m = pd.read_parquet(K / "markets.parquet")
    m = m[(m.kind == "game") & (m.volume > 0)].copy()
    look = kickoffs()
    m["key"] = m.event_ticker.str.split("-").str[1]
    m["game_id"] = m.key.map(lambda k: look.get(k, (None, None))[0])
    m["kick"] = m.key.map(lambda k: look.get(k, (None, None))[1])
    print(f"game markets: {len(m)}, matched to schedule: {m.kick.notna().sum()}", flush=True)
    m[["ticker", "event_ticker", "series", "src", "game_id", "kick"]].to_parquet(K / "game_kickoffs.parquet")
    m = m.dropna(subset=["kick"])
    # sample: every game-winner market, 400 random markets from each other series
    m = pd.concat([g if s == "KXNFLGAME" else g.sample(min(len(g), 150), random_state=7) for s, g in m.groupby("series")])
    out = K / "trades"; out.mkdir(exist_ok=True)
    WINDOWS = [(3600, 0), (6 * 3600, 5 * 3600), (24 * 3600, 23 * 3600), (72 * 3600, 71 * 3600)]  # (from, to) seconds before kickoff

    def one(r):
        path = out / f"{r.ticker}.json"
        if path.exists():
            return
        rows = []
        for lo, hi in WINDOWS:
            cur, pages = None, 0
            while pages < 3:
                p = {"ticker": r.ticker, "limit": 1000, "min_ts": int(r.kick) - lo, "max_ts": int(r.kick) - hi}
                if cur: p["cursor"] = cur
                d = get("/historical/trades" if r.src == "hist" else "/markets/trades", **p)
                for t in d.get("trades", []):
                    rows.append({"ts": t["created_time"], "taker_side": t["taker_side"], "yes_price": float(t["yes_price_dollars"]),
                                 "count": float(t["count_fp"]), "window": lo // 3600})
                pages += 1; cur = d.get("cursor")
                if not cur or not d.get("trades"): break
        path.write_text(json.dumps(rows))

    todo = [r for r in m.itertuples() if not (out / f"{r.ticker}.json").exists()]
    print(f"trades: {len(todo)} markets to sample", flush=True)
    with ThreadPoolExecutor(6) as ex:
        for i, f in enumerate(as_completed([ex.submit(one, r) for r in todo])):
            f.result()
            if i % 500 == 0: print(f"  {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    K.mkdir(parents=True, exist_ok=True)
    {"markets": markets, "candles": candles, "trades": trades}[sys.argv[1]]()
