"""Live trader: 'surprise inactive -> teammates' receiving unders' on Kalshi.

Each run (cron-job.org dispatches the workflow every 5 min in pregame windows):
  1. For games kicking off in the next ~110 min, build candidates: WR/TE/RB with >=10% target share over their last
     3 games who are NOT listed Out/Doubtful on the week's injury report.
  2. Check each candidate's ESPN player-news feed (RotoWire blurbs) for "X is inactive" (posted ~T-75..T-90).
  3. On a trigger, buy NO on each active teammate's receptions + receiving-yards ladder at the strike priced nearest
     50/50, sized ~$BET each. Spread <= 4c: take the ask. Wider: rest a post-only NO bid at the mid. Every order expires
     at kickoff. Decisions use the production order book (public); orders go to KALSHI_ENV (demo | live).
Cap: $PER_WEEK per NFL week (beyond that, the account balance is the limit). Each ticker is ordered at most once.
State: paper/state/live/orders.csv (every order attempt), paper/state/live/triggers.json.
"""
import base64
import io
import json
import math
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from targets_freed import ALIAS, K, NV, SERIES, fee, get, norm, schedule  # noqa: E402

ROOT = Path(__file__).resolve().parent
LIVE = ROOT / "state" / "live"
ORDERS = LIVE / "orders.csv"
TRIG = LIVE / "triggers.json"
ENV = os.environ.get("KALSHI_ENV", "demo")                       # demo | live | off
TRADE = {"demo": "https://external-api.demo.kalshi.co/trade-api/v2", "live": "https://external-api.kalshi.com/trade-api/v2"}
BET, PER_WEEK = 5.0, 250.0
TIGHT = 0.04
WINDOW = (110 * 60, 10 * 60)                                      # act from T-110 to T-10
INACTIVE = r"^{last}(?: \([^)]*\))?,?(?: [^.]{{0,60}}?,)? (?:is|was|has been|will be) (?:listed as |ruled |declared )?inactive\b"   # subject must be the player
HEDGE = re.compile(r"\b(expected|likely|could|may|might|questionable|game-time|trending|probable|considered|doubtful)\b", re.I)
UA = {"User-Agent": "Mozilla/5.0"}


# ------------------------------------------------------------------ Kalshi auth
def _key():
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    return load_pem_private_key(os.environ["KALSHI_PRIVATE_KEY"].encode(), None)


def signed(method, path, body=None):
    key, ts = _key(), str(int(time.time() * 1000))
    full = "/trade-api/v2" + path
    msg = (ts + method + full.split("?")[0]).encode()
    try:                                                          # Ed25519 keys sign raw; RSA keys use PSS
        sig = key.sign(msg)
    except TypeError:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        sig = key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
    h = {"KALSHI-ACCESS-KEY": os.environ["KALSHI_KEY_ID"], "KALSHI-ACCESS-TIMESTAMP": ts,
         "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(), "Content-Type": "application/json"}
    r = requests.request(method, TRADE[ENV] + path, headers=h, data=json.dumps(body) if body else None, timeout=30)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"raw": r.text[:300]}


# ------------------------------------------------------------------ candidates + trigger detection
_nv = {}


def nflverse(season):
    if season not in _nv:
        rd = lambda url: pd.read_parquet(io.BytesIO(requests.get(url, timeout=120).content))  # noqa: E731
        _nv[season] = {"stats": rd(f"{NV}/stats_player/stats_player_week_{season}.parquet"),
                       "injuries": rd(f"{NV}/injuries/injuries_{season}.parquet"),
                       "players": rd(f"{NV}/players/players.parquet")[["gsis_id", "espn_id", "display_name", "position"]]}
    return _nv[season]


def espn_injuries(g):
    """(name -> status) from ESPN's game summary; used as a fallback when nflverse hasn't published the week's report."""
    sb = get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard") or {}
    homes = ALIAS.get(g["home"], [g["home"]])
    eid = next((e["id"] for e in sb.get("events", []) if abs(datetime.fromisoformat(e["date"].replace("Z", "+00:00")).timestamp() - g["kick"]) < 600
                and any(c["team"]["abbreviation"] in homes for c in e["competitions"][0]["competitors"])), None)
    out = {}
    for t in ((get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary", event=eid) or {}).get("injuries") or []) if eid else []:
        for x in t.get("injuries", []):
            when = datetime.fromisoformat(x.get("date", "1970-01-01T00:00Z").replace("Z", "+00:00")).timestamp()
            if when < g["kick"] - 12 * 3600:                   # Friday/Saturday designations only, not game-day updates
                out[norm(x.get("athlete", {}).get("displayName", ""))] = x.get("status")
    return out


def candidates(g):
    nv = nflverse(g["season"])
    st = nv["stats"]
    prior = st[(st.week < g["week"]) & (st.season_type == "REG") & st.team.isin([g["home"], g["away"]])].sort_values("week")
    last3 = prior.groupby("player_id").tail(3)
    ts = last3.groupby("player_id").target_share.mean()
    team = last3.groupby("player_id").team.last()             # current team = team in most recent game
    pos = last3.groupby("player_id").position.last()
    c = pd.DataFrame({"ts": ts, "team": team, "position": pos}).reset_index().rename(columns={"player_id": "gsis_id"})
    c = c[(c.ts >= 0.10) & c.position.isin(["WR", "TE", "RB"])].merge(nv["players"][["gsis_id", "espn_id", "display_name"]], on="gsis_id")
    inj = nv["injuries"]; inj = inj[(inj.week == g["week"]) & inj.team.isin([g["home"], g["away"]])]
    if inj.empty:                                              # nflverse not updated yet -> ESPN's pre-game-day designations
        e = espn_injuries(g)
        listed = {n for n, s in e.items() if s in ("Out", "Doubtful")}
        c = c[~c.display_name.map(norm).isin(listed)]
    else:
        c = c[~c.gsis_id.isin(set(inj[inj.report_status.isin(["Out", "Doubtful"])].gsis_id))]
    return c, inj


def inactive_news(espn_id, name, g):
    last = re.escape(re.sub(r"\s+(Jr\.?|Sr\.?|II|III|IV|V)$", "", name).split()[-1])
    feed = (get("https://site.api.espn.com/apis/fantasy/v2/games/ffl/news/players", playerId=int(espn_id), limit=30) or {}).get("feed", [])
    for a in feed:
        t = datetime.fromisoformat(a["published"].replace("Z", "+00:00")).timestamp()
        h = a.get("headline", "")
        if g["kick"] - 30 * 3600 <= t <= g["kick"] and re.search(INACTIVE.format(last=last), h, re.I) and not HEDGE.search(h):
            return {"published": a["published"], "headline": h}
    return None


# ------------------------------------------------------------------ orders
def team_of(ticker, g):
    """KXNFLREC-26OCT15SEADEN-SEATHORTON15-4 -> SEA (longest matching alias of either team)."""
    part = ticker.split("-")[2]
    opts = sorted(((a, t) for t in (g["home"], g["away"]) for a in ALIAS.get(t, [t])), key=lambda x: -len(x[0]))
    return next((t for a, t in opts if part.startswith(a)), None)


def ladders(g, team, exclude):
    """Per (player, stat): the strike priced nearest 50/50, with current production best bids."""
    picks = []
    for series, stat in SERIES.items():
        d = get(f"{K}/markets", series_ticker=series, status="open", limit=1000) or {}
        rows = []
        for m in d.get("markets", []):
            if m["event_ticker"].split("-")[1] not in g["codes"] or team_of(m["ticker"], g) != team:
                continue
            sub = m.get("yes_sub_title") or ""
            pn, strike = norm(sub.split(":")[0]), re.search(r":\s*([\d.]+)\+", sub)
            if pn in exclude or not strike:
                continue
            ob = (get(f"{K}/markets/{m['ticker']}/orderbook") or {}).get("orderbook_fp") or {}
            yes = [float(p) for p, q in ob.get("yes_dollars") or []]
            no = [float(p) for p, q in ob.get("no_dollars") or []]
            if yes and no:
                yb, nb = max(yes), max(no)
                rows.append({"player": pn, "stat": stat, "ticker": m["ticker"], "strike": float(strike.group(1)),
                             "yes_bid": yb, "no_bid": nb, "mid": (yb + 1 - nb) / 2})
        if rows:
            df = pd.DataFrame(rows)
            picks += [grp.loc[(grp.mid - 0.5).abs().idxmin()].to_dict() for _, grp in df.groupby("player")]
    return picks


def place(r, g):
    spread = round((1 - r["no_bid"]) - r["yes_bid"], 2)
    if spread <= TIGHT:
        mode, no_px, post_only = "take", round(1 - r["yes_bid"], 2), False
    else:
        mode, no_px, post_only = "rest", round(1 - round(r["mid"], 2), 2), True
    count = int(BET // (no_px + (fee(no_px) if mode == "take" else 0)))
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "env": ENV, "ticker": r["ticker"], "player": r["player"],
           "stat": r["stat"], "strike": r["strike"], "mode": mode, "spread": spread, "no_price": no_px, "count": count,
           "cost": round(count * no_px, 2), "status": "", "order_id": "", "error": ""}
    if count < 1:
        return {**row, "status": "skipped", "error": "price too high for bet size"}
    body = {"ticker": r["ticker"], "client_order_id": str(uuid.uuid4()), "side": "ask",          # ask == buy NO
            "count": f"{count}.00", "price": f"{1 - no_px:.4f}",                               # V2 prices are on the YES scale
            "time_in_force": "good_till_canceled", "expiration_time": int(g["kick"]),
            "self_trade_prevention_type": "taker_at_cross", "post_only": post_only}
    code, resp = signed("POST", "/portfolio/events/orders", body)
    if code in (200, 201):
        o = resp.get("order", resp)
        return {**row, "status": o.get("status", "placed"), "order_id": o.get("order_id", "")}
    return {**row, "status": f"rejected {code}", "error": json.dumps(resp.get("error", resp))[:200]}


# ------------------------------------------------------------------ main
def run(games, now=None):
    now = now or int(time.time())
    if ENV == "off":
        print("KALSHI_ENV=off: not trading"); return
    LIVE.mkdir(parents=True, exist_ok=True)
    trig = json.loads(TRIG.read_text()) if TRIG.exists() else {}
    orders = pd.read_csv(ORDERS) if ORDERS.exists() else pd.DataFrame(columns=["game_id", "team", "week", "ticker", "cost", "status"])
    new = []

    def spent(pred):
        live_ = [o for o in orders.to_dict("records") + new if not str(o["status"]).startswith(("rejected", "skipped"))]
        return sum(float(o["cost"]) for o in live_ if pred(o))

    for g in games:
        if not (g["kick"] - WINDOW[0] <= now <= g["kick"] - WINDOW[1]):
            continue
        cand, inj = candidates(g)
        out_names = set(inj[inj.report_status.isin(["Out", "Doubtful"])].full_name.map(norm)) if not inj.empty else \
            {n for n, st_ in espn_injuries(g).items() if st_ in ("Out", "Doubtful")}
        print(f"{g['game_id']} T-{(g['kick'] - now) // 60}m: watching {', '.join(f'{n} ({t:.0%})' for n, t in zip(cand.display_name, cand.ts)) or 'nobody'}")
        hits = {}
        for _, c in cand.iterrows():
            if pd.isna(c.espn_id):
                continue
            n = inactive_news(c.espn_id, c.display_name, g)
            if n:
                hits.setdefault(c.team, []).append({"player": c.display_name, "ts": round(float(c.ts), 3), **n})
        for team, who in hits.items():
            key = f"{g['game_id']}|{team}"
            trig.setdefault(key, {"first_seen": datetime.now(timezone.utc).isoformat(timespec="seconds"), "players": who})
            print(f"  TRIGGER {team}: " + "; ".join(f"{w['player']} ({w['ts']:.0%}) - {w['headline']}" for w in who))
            exclude = {norm(w["player"]) for w in who} | out_names        # scratched players + anyone already ruled out
            spent_week = spent(lambda o: o["week"] == g["week"])
            for r in ladders(g, team, exclude):
                if r["ticker"] in set(orders.ticker) | {o["ticker"] for o in new}:
                    continue
                if spent_week + BET > PER_WEEK:
                    print(f"  weekly cap reached (${spent_week:.0f})"); break
                o = {"game_id": g["game_id"], "team": team, "week": g["week"], **place(r, g)}
                new.append(o)
                if not o["status"].startswith(("rejected", "skipped")):
                    spent_week += o["cost"]
                print(f"  {o['mode']:4s} NO {o['ticker']} x{o['count']} @ {o['no_price']:.2f} (spread {o['spread']:.2f}) -> {o['status']} {o['error']}")
    TRIG.write_text(json.dumps(trig, indent=1))
    if new:
        pd.concat([orders, pd.DataFrame(new)]).to_csv(ORDERS, index=False)


if __name__ == "__main__":
    run(schedule())
