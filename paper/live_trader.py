"""Live trader: 'surprise inactive -> teammates' receiving unders' on Kalshi.

Each run (cron-job.org dispatches the workflow every 5 min in pregame windows):
  1. For games kicking off in the next ~110 min, build candidates: WR/TE/RB with >=5% target share over their last
     3 games who are NOT listed Out/Doubtful on the week's injury report. A team triggers when its surprise inactives'
     target shares add up to >= 15% (the backtest's 10-15% bucket was break-even: +1% best book, -5% exchange, 89 games).
  2. Check each candidate's ESPN player-news feed (RotoWire blurbs) for "X is inactive" (posted ~T-75..T-90).
  3. On a trigger, buy NO on each active teammate's receptions + receiving-yards ladder at the strike priced nearest
     50/50, sized ~$BET each. Spread <= 4c: take the ask. Wider: rest a post-only NO bid at the mid. Every order expires
     at kickoff. Decisions use the production order book (public); orders go to KALSHI_ENV (demo | live).
Cap: $PER_WEEK per NFL week (beyond that, the account balance is the limit). Each ticker is ordered at most once.
State: paper/state/live/orders.csv (every order attempt), triggers.json, fills.json (per-order fills + settlement).
Report: paper/LIVE.md (real fills and P&L from the exchange, next to the paper test for the same games).
Alerts: triggers, orders, failures and settled results go to $ALERTS_FILE (JSON lines); paper/notify.py posts them
as GitHub issues/comments.
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
DEPTH = LIVE / "depth.csv"                                       # depth of every ordered contract, every run until kickoff
FILLS = LIVE / "fills.json"
REPORT = ROOT / "LIVE.md"
ALERTS = []                                                      # (key, title, body) for paper/notify.py
ENV = os.environ.get("KALSHI_ENV", "demo")                       # demo | live | off
TRADE = {"demo": "https://external-api.demo.kalshi.co/trade-api/v2", "live": "https://external-api.kalshi.com/trade-api/v2"}
BET, PER_WEEK = 10.0, 250.0
WATCH_MIN = 0.05                                                  # watch anyone with >= 5% target share ...
MIN_VACATED = 0.15                                                # ... and trigger when a team's scratched share sums to >= 15%
TIGHT = 0.04
MAX_REST_SPREAD = 0.20                                            # wider books are bench players with no real market
WINDOW = (110 * 60, 10 * 60)                                      # act from T-110 to T-10
MAX_MID_DIST = 0.20                                               # chosen strike must price 30-70c; else it's not the bet we tested
EXPIRE_BEFORE_KICK = 120                                          # orders expire 2 min before scheduled kickoff
SWEEP_AFTER_KICK = 6 * 3600                                       # cancel anything still resting from T-2m to T+6h
INACTIVE = r"^(?:{first} )?{last}(?: \([^)]*\))?,?(?: [^.]{{0,80}}?,)? (?:is|was|has been|will be) (?:officially )?(?:listed as |ruled |declared )?inactive\b"
HEDGE = re.compile(r"\b(expected|likely|could|may|might|trending|considered|projected|anticipated)\b", re.I)   # within the matched phrase
TRANSIENT = ("rejected 429", "rejected 5", "error")              # order attempts worth retrying on the next run
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
    c = c[(c.ts >= WATCH_MIN) & c.position.isin(["WR", "TE", "RB"])].merge(nv["players"][["gsis_id", "espn_id", "display_name"]], on="gsis_id")
    inj = nv["injuries"]; inj = inj[(inj.week == g["week"]) & inj.team.isin([g["home"], g["away"]])]
    out = inj[inj.report_status.isin(["Out", "Doubtful"])]
    out_names = set(out.full_name.map(norm)) | {n for n, s in espn_injuries(g).items() if s in ("Out", "Doubtful")}   # either source
    c = c[~c.gsis_id.isin(set(out.gsis_id)) & ~c.display_name.map(norm).isin(out_names)]
    teams = dict(zip(last3.player_display_name.map(norm), last3.team))   # name -> team, for assigning Kalshi ladders
    return c, out_names, teams


def inactive_news(espn_id, name, g):
    first, _, last = re.sub(r"\s+(Jr\.?|Sr\.?|II|III|IV|V)$", "", name).partition(" ")   # "Amon-Ra St. Brown" -> "St. Brown"
    pat = INACTIVE.format(first=re.escape(first), last=re.escape(last or first))
    feed = (get("https://site.api.espn.com/apis/fantasy/v2/games/ffl/news/players", playerId=int(espn_id), limit=30) or {}).get("feed", [])
    for a in feed:
        t = datetime.fromisoformat(a["published"].replace("Z", "+00:00")).timestamp()
        h = a.get("headline", "")
        m = re.search(pat, h, re.I)
        if g["kick"] - 30 * 3600 <= t <= g["kick"] and m and not HEDGE.search(m.group(0)):
            return {"published": a["published"], "headline": h}
    return None


# ------------------------------------------------------------------ orders
def team_of(ticker, g):
    """KXNFLREC-26OCT15SEADEN-SEATHORTON15-4 -> SEA (longest matching alias of either team)."""
    part = ticker.split("-")[2]
    opts = sorted(((a, t) for t in (g["home"], g["away"]) for a in ALIAS.get(t, [t])), key=lambda x: -len(x[0]))
    return next((t for a, t in opts if part.startswith(a)), None)


def ladders(g, team, exclude, teams):
    """Per (player, stat): the strike priced nearest 50/50, with current production best bids."""
    picks = []
    for series, stat in SERIES.items():
        rows = []
        for m in [m for code in g["codes"] for m in (get(f"{K}/markets", event_ticker=f"{series}-{code}", status="open", limit=1000) or {}).get("markets", [])]:
            sub = m.get("yes_sub_title") or ""
            pn, strike = norm(sub.split(":")[0]), re.search(r":\s*([\d.]+)\+", sub)
            if (teams.get(pn) or team_of(m["ticker"], g)) != team or pn in exclude or not strike:
                continue
            ob = (get(f"{K}/markets/{m['ticker']}/orderbook") or {}).get("orderbook_fp") or {}
            yes = [(float(p), float(q)) for p, q in ob.get("yes_dollars") or []]
            no = [(float(p), float(q)) for p, q in ob.get("no_dollars") or []]
            if yes and no:
                yb, nb = max(yes)[0], max(no)[0]
                depth = lambda c: round(sum(q * (1 - p) for p, q in yes if p >= yb - c - 1e-9), 2)   # $ of NO takeable within c of best  # noqa: E731
                rows.append({"player": pn, "stat": stat, "ticker": m["ticker"], "strike": float(strike.group(1)),
                             "yes_bid": yb, "no_bid": nb, "mid": (yb + 1 - nb) / 2,
                             "depth_best": depth(0), "depth_1c": depth(0.01), "depth_3c": depth(0.03)})
        if rows:
            df = pd.DataFrame(rows)
            picks += [grp.loc[(grp.mid - 0.5).abs().idxmin()].to_dict() for _, grp in df.groupby("player")]
    return picks


def prop(ticker):
    """KXNFLREC-26OCT18BALCLE-CLEHFANNIN44-4 -> KXNFLREC-26OCT18BALCLE-CLEHFANNIN44: one bet per player+stat, whatever the strike."""
    return ticker.rsplit("-", 1)[0]


def ordered_on_exchange(g):
    """Props in this game that already have an order of ours on the exchange (any status). The CSV can lag a failed push."""
    out = set()
    for series in SERIES:
        for code in g["codes"]:
            code_, resp = signed("GET", f"/portfolio/orders?event_ticker={series}-{code}&limit=1000")
            if code_ == 404:                                      # no such event on this exchange (alias spelling, demo gaps)
                continue
            if code_ != 200:
                raise RuntimeError(f"can't read existing orders ({code_}): {resp}")   # never trade blind
            out |= {prop(o["ticker"]) for o in resp.get("orders", [])}
    return out


def place(r, g):
    spread = round((1 - r["no_bid"]) - r["yes_bid"], 2)
    if spread <= TIGHT:
        mode, no_px, post_only = "take", round(1 - r["yes_bid"], 2), False
    elif spread <= MAX_REST_SPREAD:
        mode, no_px, post_only = "rest", round(1 - round(r["mid"], 2), 2), True
    else:
        mode, no_px, post_only = "none", round(1 - round(r["mid"], 2), 2), True
    if abs(r["mid"] - 0.5) > MAX_MID_DIST or not 0.05 <= no_px <= 0.95:
        mode = "none"
    count = int(BET // (no_px + (fee(no_px) if mode == "take" else 0)))
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "env": ENV, "ticker": r["ticker"], "player": r["player"],
           "stat": r["stat"], "strike": r["strike"], "mode": mode, "spread": spread, "no_price": no_px, "count": count,
           "cost": round(count * no_px, 2), "status": "", "order_id": "", "error": "",
           "depth_best": r.get("depth_best"), "depth_1c": r.get("depth_1c"), "depth_3c": r.get("depth_3c")}   # $ of NO takeable at order time
    if mode == "none":
        return {**row, "status": "skipped", "error": f"spread {spread:.2f} / mid {r['mid']:.2f} outside limits"}
    if count < 1:
        return {**row, "status": "skipped", "error": "price too high for bet size"}
    body = {"ticker": r["ticker"], "client_order_id": str(uuid.uuid4()), "side": "ask",          # ask == buy NO
            "count": f"{count}.00", "price": f"{1 - no_px:.4f}",                               # V2 prices are on the YES scale
            "time_in_force": "good_till_canceled", "expiration_time": int(g["kick"]) - EXPIRE_BEFORE_KICK,
            "self_trade_prevention_type": "taker_at_cross", "post_only": post_only}
    try:
        code, resp = signed("POST", "/portfolio/events/orders", body)
    except requests.RequestException as e:                        # outcome unknown: next run checks the exchange first
        return {**row, "status": "error", "error": str(e)[:200]}
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
    all_orders = pd.read_csv(ORDERS) if ORDERS.exists() else pd.DataFrame(columns=["game_id", "team", "week", "ticker", "cost", "status", "env"])
    orders = all_orders[all_orders.env == ENV] if "env" in all_orders else all_orders     # demo history never counts against live
    new = []

    def spent(pred):
        live_ = [o for o in orders.to_dict("records") + new if not str(o["status"]).startswith(("rejected", "skipped", "error"))]
        return sum(float(o["cost"]) for o in live_ if pred(o))

    try:
        try:
            cancel_stale(games, now)
        except Exception as e:
            print(f"sweep failed: {e}")
            ALERTS.append(("errors", "[trade] errors", f"Post-kickoff cancel sweep failed: `{type(e).__name__}: {e}`"))
        for g in games:
            if g["kick"] - WINDOW[0] <= now <= g["kick"] - WINDOW[1]:
                try:
                    trade_game(g, now, trig, orders, new, spent)
                except Exception as e:                            # one game's failure must not stop the others
                    print(f"{g['game_id']}: FAILED {type(e).__name__}: {e}")
                    ALERTS.append(("errors", "[trade] errors", f"`{g['game_id']}` failed at T-{(g['kick'] - now) // 60}m: `{type(e).__name__}: {e}`"))
        try:
            log_depth(games, pd.concat([orders, pd.DataFrame(new)]) if new else orders, now)
        except Exception as e:                                    # measurement only; never let it break trading
            print(f"depth log failed: {e}")
    finally:
        TRIG.write_text(json.dumps(trig, indent=1))
        if new:
            pd.concat([all_orders, pd.DataFrame(new)]).to_csv(ORDERS, index=False)


def cancel_stale(games, now):
    """Belt and braces for expiry: cancel any order of ours still resting once a game is within 2 min of kickoff
    (late kickoffs, schedule drift). Never leave a pregame bid exposed to in-game flow."""
    for g in games:
        if not (g["kick"] - EXPIRE_BEFORE_KICK <= now <= g["kick"] + SWEEP_AFTER_KICK):
            continue
        for series in SERIES:
            for code in g["codes"]:
                c, r = signed("GET", f"/portfolio/orders?event_ticker={series}-{code}&status=resting&limit=1000")
                for o in (r.get("orders", []) if c == 200 else []):
                    c2, r2 = signed("DELETE", f"/portfolio/events/orders/{o['order_id']}?market_ticker={o['ticker']}")
                    print(f"  swept {o['ticker']} ({g['game_id']}): {c2}")
                    if c2 not in (200, 204, 404):
                        ALERTS.append(("errors", "[trade] errors", f"Couldn't cancel resting order on `{o['ticker']}` after kickoff ({c2}: `{json.dumps(r2)[:150]}`)."))


def log_depth(games, orders, now):
    """How the book we traded into evolves to kickoff: best-level and within-1c/3c NO depth, per ordered ticker, per run."""
    kicks = {g["game_id"]: g["kick"] for g in games if g["kick"] - WINDOW[0] <= now <= g["kick"]}
    if orders.empty or not kicks:
        return
    rows = []
    for o in orders[orders.game_id.isin(kicks) & ~orders.status.astype(str).str.startswith(("rejected", "skipped", "error"))].to_dict("records"):
        ob = (get(f"{K}/markets/{o['ticker']}/orderbook") or {}).get("orderbook_fp") or {}
        yes = [(float(p), float(q)) for p, q in ob.get("yes_dollars") or []]
        if not yes:
            continue
        yb = max(yes)[0]
        d = lambda c: round(sum(q * (1 - p) for p, q in yes if p >= yb - c - 1e-9), 2)  # noqa: E731
        rows.append({"ts": now, "t_minus": (kicks[o["game_id"]] - now) // 60, "game_id": o["game_id"], "ticker": o["ticker"],
                     "no_ask": round(1 - yb, 2), "depth_best": d(0), "depth_1c": d(0.01), "depth_3c": d(0.03)})
    if rows:
        pd.DataFrame(rows).to_csv(DEPTH, mode="a", header=not DEPTH.exists(), index=False)


def trade_game(g, now, trig, orders, new, spent):
    cand, out_names, teams = candidates(g)
    print(f"{g['game_id']} T-{(g['kick'] - now) // 60}m: watching {', '.join(f'{n} ({t:.0%})' for n, t in zip(cand.display_name, cand.ts)) or 'nobody'}")
    hits = {}
    for _, c in cand.iterrows():
        if pd.isna(c.espn_id):
            continue
        n = inactive_news(c.espn_id, c.display_name, g)
        if n:
            hits.setdefault(c.team, []).append({"player": c.display_name, "ts": round(float(c.ts), 3), **n})
    for team, who in hits.items():
        vacated = sum(w["ts"] for w in who)
        if vacated < MIN_VACATED:
            print(f"  below threshold {team}: " + "; ".join(f"{w['player']} ({w['ts']:.0%})" for w in who) + f" = {vacated:.0%} < {MIN_VACATED:.0%}")
            continue
        key = f"{g['game_id']}|{team}"
        is_new = key not in trig
        n_before = len(new)
        trig.setdefault(key, {"first_seen": datetime.now(timezone.utc).isoformat(timespec="seconds")})["players"] = who
        print(f"  TRIGGER {team}: " + "; ".join(f"{w['player']} ({w['ts']:.0%}) - {w['headline']}" for w in who))
        exclude = {norm(w["player"]) for w in who} | out_names        # scratched players + anyone already ruled out
        spent_week = spent(lambda o: o["week"] == g["week"])
        done = ordered_on_exchange(g) | {prop(o["ticker"]) for o in orders.to_dict("records") + new if not str(o["status"]).startswith(TRANSIENT)}
        for r in ladders(g, team, exclude, teams):
            if prop(r["ticker"]) in done:
                continue
            if spent_week + BET > PER_WEEK:
                print(f"  weekly cap reached (${spent_week:.0f})"); break
            o = {"game_id": g["game_id"], "team": team, "week": g["week"], **place(r, g)}
            new.append(o)
            done.add(prop(r["ticker"]))
            if not o["status"].startswith(("rejected", "skipped", "error")):
                spent_week += o["cost"]
            print(f"  {o['mode']:4s} NO {o['ticker']} x{o['count']} @ {o['no_price']:.2f} (spread {o['spread']:.2f}) -> {o['status']} {o['error']}")
        mine = [o for o in new[n_before:] if not o["status"].startswith(("rejected", "skipped"))]   # alert on real orders only
        if is_new or mine:
            lines = [f"**{w['player']}** ({w['ts']:.0%} target share): {w['headline']}" for w in who] if is_new else []
            lines += ["", "| order | stat | strike | mode | NO price | contracts | cost | status |", "|---|---|---|---|---|---|---|---|"] if mine else []
            lines += [f"| {o['player']} | {o['stat']} | {o['strike']:g}+ | {o['mode']} | {o['no_price']:.2f} | {o['count']} | ${o['cost']:.2f} | {o['status']} {o['error']} |" for o in mine]
            ALERTS.append((key, f"[trade] {g['away']} @ {g['home']}: {team} trigger ({', '.join(w['player'] for w in who)}) [{ENV}]",
                           f"T-{(g['kick'] - now) // 60}m\n\n" + "\n".join(lines)))


# ------------------------------------------------------------------ fills, settlement, report
def report(games):
    """Reconcile every order with its exchange fills and the market result; write paper/LIVE.md."""
    if not ORDERS.exists():
        return
    orders = pd.read_csv(ORDERS)
    orders = orders[(orders.env == ENV) & orders.order_id.notna() & (orders.order_id.astype(str) != "")]
    fills = json.loads(FILLS.read_text()) if FILLS.exists() else {}
    for o in orders.to_dict("records"):
        f = fills.get(o["order_id"])
        if f and f.get("result"):
            continue
        try:
            code, resp = signed("GET", f"/portfolio/fills?order_id={o['order_id']}&limit=1000")
            if code != 200:
                print(f"fills {o['order_id']}: {code} {resp}"); continue
            fs = resp.get("fills", [])
            n = sum(float(x["count_fp"]) for x in fs)
            cost = sum(float(x["count_fp"]) * float(x["no_price_dollars"]) for x in fs)
            fee_ = sum(float(x.get("fee_cost") or 0) for x in fs)
            m = (get(f"{TRADE[ENV]}/markets/{o['ticker']}") or {}).get("market") or {}
        except Exception as e:                                    # one bad lookup must not sink the report
            print(f"fills {o['order_id']}: {type(e).__name__}: {e}"); continue
        res = m.get("result") if m.get("result") in ("yes", "no") else None
        if res is None and n == 0 and m.get("status") in ("settled", "finalized", "determined"):
            res = "unfilled"                                      # settled with no fill: nothing to track
        elif res is None and m.get("status") in ("settled", "finalized") :
            res = "void"                                          # settled without yes/no (voided): refunded at cost
        pnl = {"no": n - cost - fee_, "yes": -cost - fee_, "void": 0.0, "unfilled": 0.0}.get(res)
        fills[o["order_id"]] = {"filled": n, "cost": round(cost, 4), "fee": round(fee_, 4), "result": res,
                                "pnl": round(pnl, 4) if pnl is not None else None}
    FILLS.write_text(json.dumps(fills, indent=1))
    if orders.empty:
        return
    d = orders.assign(**{k: orders.order_id.map(lambda i, k=k: (fills.get(i) or {}).get(k)) for k in ("filled", "cost", "fee", "result", "pnl")}, lim_cost=orders.cost)
    d["cost"] = d.cost.fillna(0) + d.fee.fillna(0)
    paper = pd.read_csv(ROOT / "state" / "tf_bets.csv") if (ROOT / "state" / "tf_bets.csv").exists() else pd.DataFrame()
    paper = paper[paper.strategy == "kalshi_take_tight"] if not paper.empty else paper
    trig = json.loads(TRIG.read_text()) if TRIG.exists() else {}
    notified = set(trig.get("_settled_notified", []))
    lines = ["# Live trades: surprise-inactive receiver unders", "",
             f"_Updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · exchange: **{ENV}** · real fills and settlements from Kalshi._", ""]
    f_ = d[d.filled.fillna(0) > 0]; s_ = f_[f_.result.isin(["yes", "no"])]
    tot_pnl, tot_cost = s_.pnl.sum(), s_.cost.sum()
    lines += [f"**Settled:** {len(s_)} bets, {(s_.result == 'no').sum()}-{(s_.result == 'yes').sum()}, "
              f"P&L **${tot_pnl:+.2f}** on ${tot_cost:.2f} risked ({tot_pnl / tot_cost if tot_cost else 0:+.1%}). "
              f"**Open:** {f_.result.isna().sum()} filled bets, ${f_[f_.result.isna()].cost.sum():.2f} at risk. "
              f"**Unfilled:** {(d.filled.fillna(0) == 0).sum()} of {len(d)} orders.", "",
              "| game | team | trigger | orders | filled | W-L | risked | P&L | ROI | paper (Kalshi tight) ROI |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (gid, team), x in d.groupby(["game_id", "team"], sort=False):
        fx = x[x.filled.fillna(0) > 0]; sx = fx[fx.result.isin(["yes", "no"])]
        who = ", ".join(w["player"] for w in trig.get(f"{gid}|{team}", {}).get("players", []))
        pp = paper[(paper.game_id == gid) & (paper.team == team)] if not paper.empty else paper
        p_roi = f"{pp.pnl.sum() / pp.stake.sum():+.1%}" if len(pp) and pp.stake.sum() else "–"
        roi = f"{sx.pnl.sum() / sx.cost.sum():+.1%}" if len(sx) and sx.cost.sum() else "–"
        lines.append(f"| {gid} | {team} | {who} | {len(x)} | {len(fx)} | {(sx.result == 'no').sum()}-{(sx.result == 'yes').sum()} | "
                     f"${fx.cost.sum():.2f} | ${sx.pnl.sum():+.2f} | {roi} | {p_roi} |")
        key = f"{gid}|{team}"
        if len(fx) and fx.result.notna().all() and key not in notified:           # every filled bet in this trigger has settled
            ALERTS.append((key, None, f"**Settled:** {(sx.result == 'no').sum()}-{(sx.result == 'yes').sum()}, "
                                      f"P&L ${sx.pnl.sum():+.2f} on ${sx.cost.sum():.2f} ({roi}). Paper test, same game: {p_roi}."))
            notified.add(key)
    trig["_settled_notified"] = sorted(notified)
    TRIG.write_text(json.dumps(trig, indent=1))
    lines += ["", "Paper column: the paper test's Kalshi strategy for the same trigger (take when spread ≤ 4¢), priced at its T-85 snapshot.", ""]
    REPORT.write_text("\n".join(lines))


def write_alerts():
    if ALERTS and os.environ.get("ALERTS_FILE"):
        with open(os.environ["ALERTS_FILE"], "a") as f:
            for key, title, body in ALERTS:
                f.write(json.dumps({"key": key, "title": title, "body": body}) + "\n")


def preflight():
    """Every run: prove the key works and show the balance, so a bad key surfaces now, not at the next trigger."""
    try:
        code, r = signed("GET", "/portfolio/balance")
    except Exception as e:
        code, r = None, {"error": str(e)}
    if code == 200:
        print(f"[{ENV}] balance ${float(r.get('balance_dollars') or r.get('balance', 0) / 100):,.2f}")
        return True
    print(f"[{ENV}] PREFLIGHT FAILED: {code} {r}")
    ALERTS.append(("errors", "[trade] errors", f"Preflight failed on **{ENV}**: can't read the balance ({code}: `{json.dumps(r)[:200]}`). Check the API key secrets."))
    return False


if __name__ == "__main__":
    games = schedule()
    try:
        if ENV != "off" and not preflight():
            sys.exit(1)
        run(games)
        if ENV != "off":
            report(games)
    finally:
        write_alerts()
