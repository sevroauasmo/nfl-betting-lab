"""Paper test: 'surprise inactive -> teammates' receiving unders', on Kalshi and (for comparison) sportsbooks/exchanges.

snapshot  (T-90..T-35, once per game, after inactives are announced at T-90):
            Kalshi order books for every KXNFLREC / KXNFLRECYDS market in the game; sportsbook + Novig/ProphetX odds for
            receptions / receiving yards from The Odds API (if ODDS_API_KEY is set; ~4 credits per game); ESPN injury feed.
grade     (after the game, once nflverse has published game-day rosters): identify surprise inactives (status INA but not
            Out/Doubtful on the week's injury report) with >=10% target share over their last 3 games. For each remaining
            teammate WR/TE/RB, record paper bets:
              kalshi_take_tight - buy NO at the ask, only if the spread <= 4c           (taker fee included)
              kalshi_take_all   - buy NO at the ask, any spread                         (benchmark)
              kalshi_rest_mid   - rest a NO bid at the mid; filled from the real trade tape behind the queue
              book_best         - under at the best US sportsbook price at the consensus line
              exchange_best     - under at the best Novig/ProphetX price at that line
          Kalshi bets settle on Kalshi's result; book/exchange bets on nflverse stats (active + snaps + no stats = 0; else void).
summarize -> paper/TARGETS_FREED.md
"""
import csv
import gzip
import io
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
SNAP = ROOT / "state" / "tf"
SNAP.mkdir(parents=True, exist_ok=True)
BETS = ROOT / "state" / "tf_bets.csv"
DONE = ROOT / "state" / "tf_graded.json"
K = "https://api.elections.kalshi.com/trade-api/v2"
ODDS = "https://api.the-odds-api.com/v4"
NV = "https://github.com/nflverse/nflverse-data/releases/download"
SERIES = {"KXNFLREC": "receptions", "KXNFLRECYDS": "receiving_yards"}
ODDS_MKT = {"player_receptions": "receptions", "player_reception_yds": "receiving_yards"}
EXCH = {"novig", "prophetx"}
ALIAS = {"LA": ["LA", "LAR"], "JAX": ["JAX", "JAC"], "WAS": ["WAS", "WSH"], "LV": ["LV", "LVR"]}
NAMES = {"ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills", "CAR": "Carolina Panthers",
         "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys", "DEN": "Denver Broncos",
         "DET": "Detroit Lions", "GB": "Green Bay Packers", "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
         "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers", "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
         "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants", "NYJ": "New York Jets",
         "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers",
         "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders"}
S = requests.Session()


def get(url, **params):
    for a in range(6):
        try:
            r = S.get(url, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** a); continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(20, 1.5 ** a)); continue
        return r.json() if r.status_code == 200 else None
    return None


def norm(s):
    s = re.sub(r"\([^)]*\)", "", str(s).lower())
    return re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", re.sub(r"[^a-z ]", "", s)).strip()


def schedule():
    txt = S.get("https://github.com/nflverse/nfldata/raw/master/data/games.csv", timeout=60).text
    et = ZoneInfo("America/New_York")
    rows = []
    for r in csv.DictReader(io.StringIO(txt)):
        if int(r["season"]) < 2026 or not r["gametime"]:
            continue
        k = datetime.fromisoformat(f"{r['gameday']} {r['gametime']}").replace(tzinfo=et)
        codes = [f"{k.strftime('%y%b%d').upper()}{a}{h}" for a in ALIAS.get(r["away_team"], [r["away_team"]]) for h in ALIAS.get(r["home_team"], [r["home_team"]])]
        rows.append({"game_id": r["game_id"], "season": int(r["season"]), "week": int(r["week"]), "home": r["home_team"], "away": r["away_team"],
                     "kick": int(k.timestamp()), "codes": codes})
    return rows


# ------------------------------------------------------------------ snapshot
def kalshi_books(g):
    out = []
    for series in SERIES:
        d = get(f"{K}/markets", series_ticker=series, status="open", limit=1000) or {}
        for m in d.get("markets", []):
            if m["event_ticker"].split("-")[1] not in g["codes"]:
                continue
            ob = (get(f"{K}/markets/{m['ticker']}/orderbook") or {}).get("orderbook_fp") or {}
            yes = [(float(p), float(q)) for p, q in ob.get("yes_dollars") or []]
            no = [(float(p), float(q)) for p, q in ob.get("no_dollars") or []]
            sub = m.get("yes_sub_title") or ""
            strike = re.search(r":\s*([\d.]+)\+", sub)
            out.append({"ticker": m["ticker"], "series": series, "player": sub.split(":")[0], "strike": float(strike.group(1)) if strike else None,
                        "yes_bid": max(yes)[0] if yes else None, "yes_bid_q": max(yes)[1] if yes else None,
                        "no_bid": max(no)[0] if no else None, "no_bid_q": max(no)[1] if no else None,
                        "team_code": m["ticker"].split("-")[2][:3]})
    return out


def odds_snapshot(g, key):
    if not key:
        return None
    evs = get(f"{ODDS}/sports/americanfootball_nfl/events", apiKey=key) or []   # free endpoint
    ev = next((e for e in evs if e["home_team"] == NAMES.get(g["home"]) and e["away_team"] == NAMES.get(g["away"])), None)
    if not ev:
        return None
    d = get(f"{ODDS}/sports/americanfootball_nfl/events/{ev['id']}/odds", apiKey=key, regions="us,us_ex",
            markets=",".join(ODDS_MKT), oddsFormat="american")
    return {"event": ev, "odds": d}


def snapshot(games, key):
    now = int(time.time())
    for g in games:
        path = SNAP / f"{g['game_id']}.json.gz"
        if path.exists() or not (g["kick"] - 90 * 60 <= now <= g["kick"] - 35 * 60):
            continue
        espn = None
        sb = get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard") or {}
        eid = next((e["id"] for e in sb.get("events", []) if abs(datetime.fromisoformat(e["date"].replace("Z", "+00:00")).timestamp() - g["kick"]) < 600
                    and g["home"] in [c["team"]["abbreviation"] for c in e["competitions"][0]["competitors"]]), None)
        if eid:
            espn = (get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary", event=eid) or {}).get("injuries")
        snap = {"game": g, "ts": now, "kalshi": kalshi_books(g), "odds": odds_snapshot(g, key), "espn_injuries": espn}
        with gzip.open(path, "wt") as f:
            json.dump(snap, f)
        print(f"snapshot {g['game_id']}: {len(snap['kalshi'])} kalshi books, odds={'yes' if snap['odds'] else 'no'}, T-{(g['kick'] - now) // 60}m")


# ------------------------------------------------------------------ grading
_nv = {}


def nflverse(season):
    if season not in _nv:
        rd = lambda url: pd.read_parquet(io.BytesIO(S.get(url, timeout=120).content))  # noqa: E731
        _nv[season] = {"rosters": rd(f"{NV}/weekly_rosters/roster_weekly_{season}.parquet"),
                       "injuries": rd(f"{NV}/injuries/injuries_{season}.parquet"),
                       "stats": rd(f"{NV}/stats_player/stats_player_week_{season}.parquet"),
                       "snaps": rd(f"{NV}/snap_counts/snap_counts_{season}.parquet"),
                       "players": rd(f"{NV}/players/players.parquet")[["gsis_id", "pfr_id", "display_name"]]}
    return _nv[season]


def fee(p):
    return math.ceil(0.07 * p * (1 - p) * 100) / 100


def payout(o):
    return 100 / -o if o < 0 else o / 100


def kalshi_result(ticker):
    m = (get(f"{K}/markets/{ticker}") or {}).get("market") or {}
    return m.get("result") if m.get("result") in ("yes", "no") else None


def rest_fill(ticker, yes_px, queue_ahead, start, end, size=100):
    trades, cur = [], None
    while True:
        p = {"ticker": ticker, "limit": 1000, "min_ts": start, "max_ts": end}
        if cur: p["cursor"] = cur
        d = get(f"{K}/markets/trades", **p) or {}
        trades += d.get("trades", []); cur = d.get("cursor")
        if not cur or not d.get("trades"):
            break
    at, through = 0.0, False
    for t in trades:
        if t["taker_side"] != "yes":
            continue
        yp = float(t["yes_price_dollars"])
        if abs(yp - yes_px) < 1e-9: at += float(t["count_fp"])
        elif yp > yes_px + 1e-9: through = True
    return size if through else max(0.0, min(size, at - queue_ahead))


def grade_game(snap, rows):
    g = snap["game"]; nv = nflverse(g["season"])
    ro = nv["rosters"]; ro = ro[(ro.week == g["week"]) & ro.team.isin([g["home"], g["away"]])]
    if ro.empty or not (ro.status == "INA").any():
        return "waiting"                                  # game-day rosters not published yet
    inj = nv["injuries"]; inj = inj[inj.week == g["week"]]
    st = nv["stats"]; prior = st[(st.week < g["week"]) & (st.season_type == "REG")].sort_values("week")
    ts = prior.groupby("player_id").tail(3).groupby("player_id").target_share.mean()
    ina = ro[(ro.status == "INA") & ro.position.isin(["WR", "TE", "RB"])]
    listed = set(inj[inj.report_status.isin(["Out", "Doubtful"])].gsis_id)
    sur = ina[~ina.gsis_id.isin(listed)].assign(ts=lambda d: d.gsis_id.map(ts).fillna(0))
    sur = sur[sur.ts >= 0.10]
    if sur.empty:
        return "no_trigger"
    game_stats = st[st.week == g["week"]].set_index("player_id")
    sn = nv["snaps"].merge(nv["players"], left_on="pfr_player_id", right_on="pfr_id")
    played = set(sn[(sn.week == g["week"]) & (sn.offense_snaps > 0)].gsis_id)
    kick, ts_snap = g["kick"], snap["ts"]
    for team, tsur in sur.groupby("team"):
        mates = ro[(ro.team == team) & (ro.status == "ACT") & ro.position.isin(["WR", "TE", "RB"])]
        names = {norm(n): gid for n, gid in zip(mates.full_name, mates.gsis_id)}
        trig = "; ".join(f"{n} ({s:.0%})" for n, s in zip(tsur.full_name, tsur.ts))
        # sportsbook quotes from the snapshot: (player, stat) -> [(book, side, point, price)]
        quotes = {}
        if snap.get("odds") and snap["odds"].get("odds"):
            for b in snap["odds"]["odds"].get("bookmakers", []):
                for m in b["markets"]:
                    for o in m["outcomes"]:
                        quotes.setdefault((norm(o.get("description")), ODDS_MKT[m["key"]]), []).append((b["key"], o["name"], o.get("point"), o["price"]))
        cons = {}
        for (pn, stat), qs in quotes.items():
            if pn not in names:
                continue
            gid = names[pn]
            books = [q for q in qs if q[0] not in EXCH]
            pts = sorted(q[2] for q in books if q[1] == "Under" and q[2] is not None)
            if not pts:
                continue
            L = pts[len(pts) // 2]; cons[(pn, stat)] = L
            actual = float(game_stats.loc[gid, stat]) if gid in game_stats.index else (0.0 if gid in played else None)   # None -> void
            for venue, pool in (("book_best", [q for q in books if q[1] == "Under" and q[2] == L]),
                                ("exchange_best", [q for q in qs if q[0] in EXCH and q[1] == "Under" and q[2] == L])):
                if not pool or actual is None:
                    continue
                bk, _, _, price = max(pool, key=lambda q: q[3])
                res = "push" if actual == L else ("win" if actual < L else "loss")
                pnl = 0.0 if res == "push" else (payout(price) if res == "win" else -1.0)
                rows.append({"game_id": g["game_id"], "team": team, "trigger": trig, "player": pn, "stat": stat, "strategy": venue, "venue": bk,
                             "line_or_strike": L, "price": price, "stake": 1.0, "filled": 1.0, "result": res, "pnl": round(pnl, 4)})
        # Kalshi ladders: strike = floor(line)+1 when a sportsbook line exists, else the strike priced nearest 50/50
        kb = pd.DataFrame([x for x in snap["kalshi"] if norm(x["player"]) in names and x["strike"] is not None
                           and x["yes_bid"] is not None and x["no_bid"] is not None])
        if kb.empty:
            continue
        kb["pn"] = kb.player.map(norm); kb["stat"] = kb.series.map(SERIES)
        kb["mid"] = (kb.yes_bid + (1 - kb.no_bid)) / 2
        for (pn, stat), grp in kb.groupby(["pn", "stat"]):
            L = cons.get((pn, stat))
            r = grp.loc[(grp.strike - (math.floor(L) + 1)).abs().idxmin()] if L is not None else grp.loc[(grp.mid - 0.5).abs().idxmin()]
            res = kalshi_result(r.ticker)
            if res is None:
                continue
            win = res == "no"
            no_ask = round(1 - r.yes_bid, 2); spread = round((1 - r.no_bid) - r.yes_bid, 2)
            base = {"game_id": g["game_id"], "team": team, "trigger": trig, "player": pn, "stat": stat, "venue": r.ticker, "line_or_strike": r.strike}
            for strat, ok in (("kalshi_take_all", True), ("kalshi_take_tight", spread <= 0.04)):
                if ok:
                    cost = no_ask + fee(no_ask)
                    rows.append({**base, "strategy": strat, "price": no_ask, "stake": round(cost, 4), "filled": 1.0,
                                 "result": "win" if win else "loss", "pnl": round((1 - cost) if win else -cost, 4)})
            # resting NO bid at the mid; queue ahead = size already resting at that NO price
            no_px = round(1 - round(r.mid, 2), 2); yes_px = round(1 - no_px, 2)
            q_ahead = float(r.no_bid_q or 0) if abs(no_px - r.no_bid) < 1e-9 else 0.0
            filled = rest_fill(r.ticker, yes_px, q_ahead, ts_snap, kick) / 100
            rows.append({**base, "strategy": "kalshi_rest_mid", "price": no_px, "stake": round(no_px * filled, 4), "filled": filled,
                         "result": ("win" if win else "loss") if filled > 0 else "unfilled",
                         "pnl": round(((1 - no_px) if win else -no_px) * filled, 4)})
    return "graded"


def grade(games):
    done = json.loads(DONE.read_text()) if DONE.exists() else {}
    rows = []
    now = int(time.time())
    for p in sorted(SNAP.glob("*.json.gz")):
        gid = p.name.split(".")[0]
        if done.get(gid) in ("graded", "no_trigger"):
            continue
        snap = json.load(gzip.open(p, "rt"))
        if now < snap["game"]["kick"] + 5 * 3600:
            continue
        try:
            done[gid] = grade_game(snap, rows)
        except Exception as e:  # keep other games going; retried next run
            print(f"grade {gid} failed: {e}"); done[gid] = "error"
        print(f"grade {gid}: {done[gid]}")
    if rows:
        new = pd.DataFrame(rows)
        old = pd.read_csv(BETS) if BETS.exists() else pd.DataFrame()
        pd.concat([old, new]).to_csv(BETS, index=False)
    DONE.write_text(json.dumps(done, indent=1))


def summarize():
    lines = ["# Paper test: surprise inactive -> teammates' receiving unders", "",
             f"_Updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Prices snapshotted ~T-85 (after inactives); nothing is actually bet._", ""]
    done = json.loads(DONE.read_text()) if DONE.exists() else {}
    lines.append(f"Games snapshotted: {len(list(SNAP.glob('*.json.gz')))} · graded: {sum(v == 'graded' for v in done.values())} with a trigger, "
                 f"{sum(v == 'no_trigger' for v in done.values())} without · pending: {sum(v in ('waiting', 'error') for v in done.values())}")
    lines.append("")
    if BETS.exists():
        b = pd.read_csv(BETS)
        lines += ["| strategy | bets | fills | win rate | staked (units) | P&L (units) | ROI |", "|---|---|---|---|---|---|---|"]
        for s, d in b.groupby("strategy"):
            f = d[d.filled > 0]
            st = f.stake.sum(); pnl = f.pnl.sum()
            wr = (f.result == "win").sum() / max(1, (f.result.isin(["win", "loss"])).sum())
            lines.append(f"| {s} | {len(d)} | {len(f)} | {wr:.1%} | {st:.1f} | {pnl:+.2f} | {pnl / st if st else 0:+.1%} |")
        # which venue would have been best for the same prop
        lines += ["", "Kalshi bets stake the contract cost, so ROI is per dollar risked. Book/exchange bets stake 1 unit at American odds.", ""]
    (ROOT / "TARGETS_FREED.md").write_text("\n".join(lines))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("\n\n" + "\n".join(lines))


if __name__ == "__main__":
    games = schedule()
    steps = sys.argv[1:] or ["snapshot", "grade", "summarize"]
    if "snapshot" in steps: snapshot(games, os.environ.get("ODDS_API_KEY"))
    if "grade" in steps: grade(games)
    if "summarize" in steps: summarize()
