"""Paper market-maker for Kalshi NFL game markets. No real orders are ever sent.

Each run (GitHub Actions, game days):
  place   - for games kicking off within 6h (round "T-6h") or 1h (round "T-1h"), snapshot every market's order book and
            record paper resting orders: a NO bid (= selling YES at the ask) and a YES bid, each at two prices:
              join    = at the current best price, behind all size already resting there
              improve = one tick better (front of a new level), when the spread allows
  fill    - after kickoff, pull the real trade tape from placement to kickoff; a paper order fills only once the
            queue ahead of it (recorded at placement, cancellations ignored -> pessimistic) has traded through
  settle  - once Kalshi finalizes a market, grade filled orders; subtract an estimated maker fee
State lives in paper/state/*.csv (committed back to the repo by the workflow). Public API, no key needed.
"""
import csv
import io
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state"
STATE.mkdir(exist_ok=True)
ORDERS = STATE / "orders.csv"
B = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = ["KXNFLTD", "KXNFLFIRSTTD", "KXNFLSPREAD", "KXNFLTOTAL", "KXNFLTEAMTOTAL", "KXNFLGAME",
          "KXNFLRECYDS", "KXNFLREC", "KXNFLRSHYDS", "KXNFLPASSYDS"]
ROUNDS = {"T-6h": 6 * 3600, "T-1h": 3600}
SIZE = 100                      # paper contracts per order
MAKER_FEE = 0.0175              # conservative estimate: fee = ceil(0.0175 * size * p * (1 - p)) dollars
FIELDS = ["order_id", "round", "placed_ts", "kickoff_ts", "game", "series", "ticker", "title", "side", "variant",
          "price_yes", "queue_ahead", "best_yes_bid", "best_yes_ask", "status", "filled", "result", "pnl", "fee", "graded_ts"]
S = requests.Session()


def get(path, **params):
    for attempt in range(8):
        try:
            r = S.get(B + path, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** min(attempt, 5)); continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(20, 1.5 ** attempt)); continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"failed {path}")


def load():
    if not ORDERS.exists():
        return []
    with ORDERS.open() as f:
        return list(csv.DictReader(f))


def save(rows):
    with ORDERS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)


def kickoffs():
    """Kalshi event code (e.g. 26SEP24ATLGB) -> kickoff unix ts, from nflverse's schedule."""
    txt = S.get("https://github.com/nflverse/nfldata/raw/master/data/games.csv", timeout=60).text
    et = ZoneInfo("America/New_York")
    alias = {"LA": ["LA", "LAR"], "JAX": ["JAX", "JAC"], "WAS": ["WAS", "WSH"], "LV": ["LV", "LVR"]}
    out = {}
    for r in csv.DictReader(io.StringIO(txt)):
        if int(r["season"]) < 2026 or not r["gametime"]:
            continue
        k = datetime.fromisoformat(f"{r['gameday']} {r['gametime']}").replace(tzinfo=et)
        code = k.strftime("%y%b%d").upper()
        for a in alias.get(r["away_team"], [r["away_team"]]):
            for h in alias.get(r["home_team"], [r["home_team"]]):
                out[f"{code}{a}{h}"] = int(k.timestamp())
    return out


def book(ticker):
    """Best YES bid/ask and the size resting at each (orderbook lists bids for YES and NO; YES ask = 1 - best NO bid)."""
    d = get(f"/markets/{ticker}/orderbook") or {}
    ob = d.get("orderbook_fp") or d.get("orderbook") or {}
    yes = [(float(p), float(q)) for p, q in (ob.get("yes_dollars") or [])]
    no = [(float(p), float(q)) for p, q in (ob.get("no_dollars") or [])]
    if not yes or not no:
        return None
    yb, yq = max(yes); nb, nq = max(no)
    return {"yes_bid": yb, "yes_bid_q": yq, "yes_ask": round(1 - nb, 4), "yes_ask_q": nq}


def place(orders):
    now = int(time.time())
    ko = kickoffs()
    have = {(o["round"], o["ticker"]) for o in orders}
    new = 0
    for s in SERIES:
        cur = None
        while True:
            p = {"series_ticker": s, "status": "open", "limit": 1000}
            if cur: p["cursor"] = cur
            d = get("/markets", **p) or {}
            for m in d.get("markets", []):
                game = m["event_ticker"].split("-")[1]
                k = ko.get(game)
                if not k or k <= now:
                    continue
                for rnd, secs in ROUNDS.items():
                    # place each round once, on the first run inside its window (T-6h round stops at T-1h)
                    lo = k - secs; hi = k - (ROUNDS["T-1h"] if rnd == "T-6h" else 0)
                    if not (lo <= now < hi) or (rnd, m["ticker"]) in have:
                        continue
                    b = book(m["ticker"])
                    if not b or b["yes_ask"] - b["yes_bid"] <= 0:
                        continue
                    spread = round(b["yes_ask"] - b["yes_bid"], 2)
                    quotes = [("no_bid", "join", b["yes_ask"], b["yes_ask_q"]), ("yes_bid", "join", b["yes_bid"], b["yes_bid_q"])]
                    if spread >= 0.02:  # room to step inside the spread
                        quotes += [("no_bid", "improve", round(b["yes_ask"] - 0.01, 2), 0.0), ("yes_bid", "improve", round(b["yes_bid"] + 0.01, 2), 0.0)]
                    for side, var, px, q in quotes:
                        orders.append({"order_id": f"{rnd}|{m['ticker']}|{side}|{var}", "round": rnd, "placed_ts": now, "kickoff_ts": k,
                                       "game": game, "series": s, "ticker": m["ticker"], "title": m.get("yes_sub_title") or m.get("title"),
                                       "side": side, "variant": var, "price_yes": px, "queue_ahead": q,
                                       "best_yes_bid": b["yes_bid"], "best_yes_ask": b["yes_ask"], "status": "open",
                                       "filled": 0, "result": "", "pnl": "", "fee": "", "graded_ts": ""})
                        new += 1
                    have.add((rnd, m["ticker"]))
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
    print(f"place: {new} new paper orders")


def fill(orders):
    """After kickoff: walk the real trade tape from placement to kickoff and fill orders through the recorded queue."""
    now = int(time.time())
    todo = {}
    for o in orders:
        if o["status"] == "open" and int(o["kickoff_ts"]) <= now:
            todo.setdefault(o["ticker"], []).append(o)
    for t, os_ in todo.items():
        start = min(int(o["placed_ts"]) for o in os_); end = int(os_[0]["kickoff_ts"])
        trades, cur = [], None
        while True:
            p = {"ticker": t, "limit": 1000, "min_ts": start, "max_ts": end}
            if cur: p["cursor"] = cur
            d = get("/markets/trades", **p) or {}
            trades += d.get("trades", [])
            cur = d.get("cursor")
            if not cur or not d.get("trades"):
                break
        for o in os_:
            placed, px, ahead = int(o["placed_ts"]), float(o["price_yes"]), float(o["queue_ahead"])
            vol_at, through = 0.0, False
            for tr in trades:
                ts = datetime.fromisoformat(tr["created_time"].replace("Z", "+00:00")).timestamp()
                if ts < placed:
                    continue
                yp, c = float(tr["yes_price_dollars"]), float(tr["count_fp"])
                if o["side"] == "no_bid" and tr["taker_side"] == "yes":      # takers buying YES hit resting NO bids
                    if abs(yp - px) < 1e-9: vol_at += c
                    elif yp > px + 1e-9: through = True                     # traded past our level -> level exhausted
                elif o["side"] == "yes_bid" and tr["taker_side"] == "no":    # takers selling YES hit resting YES bids
                    if abs(yp - px) < 1e-9: vol_at += c
                    elif yp < px - 1e-9: through = True
            filled = SIZE if through else max(0.0, min(SIZE, vol_at - ahead))
            o["filled"] = round(filled, 2)
            o["status"] = "filled" if filled > 0 else "unfilled"
    print(f"fill: evaluated {sum(len(v) for v in todo.values())} orders on {len(todo)} markets")


def settle(orders):
    done = 0
    for t in {o["ticker"] for o in orders if o["status"] == "filled"}:
        m = (get(f"/markets/{t}") or {}).get("market") or {}
        if m.get("result") not in ("yes", "no"):
            continue
        y = 1.0 if m["result"] == "yes" else 0.0
        for o in orders:
            if o["ticker"] != t or o["status"] != "filled":
                continue
            q, px = float(o["filled"]), float(o["price_yes"])
            pnl = q * ((px - y) if o["side"] == "no_bid" else (y - px))     # sold YES at px / bought YES at px
            fee = math.ceil(MAKER_FEE * q * px * (1 - px) * 100) / 100
            o.update(status="settled", result=m["result"], pnl=round(pnl, 2), fee=fee, graded_ts=int(time.time()))
            done += 1
    print(f"settle: graded {done} orders")


def summarize(orders):
    rows = [o for o in orders if o["status"] == "settled"]
    lines = ["# Kalshi paper market-maker", "",
             f"_Updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Paper orders only; fills from the real trade tape behind the recorded queue (pessimistic)._", ""]
    placed = len(orders); filled = sum(1 for o in orders if o["status"] in ("filled", "settled"))
    lines += [f"Orders placed: **{placed}** · filled: **{filled}** · settled: **{len(rows)}**", ""]
    if rows:
        def agg(keyf):
            g = {}
            for o in rows:
                k = keyf(o); a = g.setdefault(k, {"n": 0, "q": 0.0, "pnl": 0.0, "fee": 0.0, "cap": 0.0, "games": set()})
                q, px = float(o["filled"]), float(o["price_yes"])
                a["n"] += 1; a["q"] += q; a["pnl"] += float(o["pnl"]); a["fee"] += float(o["fee"]); a["games"].add(o["game"])
                a["cap"] += q * ((1 - px) if o["side"] == "no_bid" else px)          # capital at risk
            return g
        for title, keyf in (("By series × side", lambda o: (o["series"], o["side"])),
                            ("By variant × round", lambda o: (o["variant"], o["round"])),
                            ("By series × side × variant", lambda o: (o["series"], o["side"], o["variant"]))):
            lines += [f"## {title}", "", "| group | fills | games | contracts | P&L $ | fees $ | net return on capital |", "|---|---|---|---|---|---|---|"]
            for k, a in sorted(agg(keyf).items(), key=lambda kv: -kv[1]["q"]):
                ret = (a["pnl"] - a["fee"]) / a["cap"] if a["cap"] else 0
                lines.append(f"| {' / '.join(k)} | {a['n']} | {len(a['games'])} | {a['q']:.0f} | {a['pnl']:.2f} | {a['fee']:.2f} | {ret:+.1%} |")
            lines.append("")
    (ROOT / "SUMMARY.md").write_text("\n".join(lines))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    steps = sys.argv[1:] or ["place", "fill", "settle", "summarize"]
    orders = load()
    for st in steps:
        {"place": place, "fill": fill, "settle": settle, "summarize": summarize}[st](orders)
        save(orders)
