"""Pull NFL odds from The Odds API into data/odds_raw/ (gzipped JSON, resumable), then load into DuckDB.

Usage (key read from .env as ODDS_API_KEY):
  uv run python scripts/pull_odds.py live-test            # free tier: a couple of upcoming games, real parse + load
  uv run python scripts/pull_odds.py discover             # historical event IDs per NFL week (1 credit per call)
  uv run python scripts/pull_odds.py pull --pass close    # one pass; see PASSES. --budget caps credits spent this run
  uv run python scripts/pull_odds.py load                 # parse everything in data/odds_raw into odds_* tables

Credit rules (docs): historical event odds cost 10 x [markets RETURNED] x [regions]; empty responses are free.
So we request every market and only pay for what books actually posted. Snapshots are "closest at or before `date`".
"""
import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

import duckdb
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/odds_raw"
DB = ROOT / "data/nfl.duckdb"
API = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
PROPS_START = datetime(2023, 5, 3, 6, tzinfo=timezone.utc)  # props history starts here
LINES_START = datetime(2020, 6, 6, tzinfo=timezone.utc)

PROP_MARKETS = [
    "player_pass_yds", "player_pass_tds", "player_pass_attempts", "player_pass_completions", "player_pass_interceptions",
    "player_pass_longest_completion", "player_pass_rush_yds", "player_pass_yds_q1",
    "player_rush_yds", "player_rush_attempts", "player_rush_longest", "player_rush_tds", "player_rush_reception_yds",
    "player_rush_reception_tds", "player_receptions", "player_reception_yds", "player_reception_longest", "player_reception_tds",
    "player_pass_rush_reception_yds", "player_pass_rush_reception_tds",
    "player_anytime_td", "player_1st_td", "player_last_td", "player_tds_over",
    "player_field_goals", "player_kicking_points", "player_pats",
    "player_sacks", "player_solo_tackles", "player_tackles_assists", "player_assists", "player_defensive_interceptions",
]
ALT_MARKETS = [m + "_alternate" for m in (
    "player_pass_yds", "player_pass_tds", "player_pass_attempts", "player_pass_completions", "player_pass_interceptions",
    "player_rush_yds", "player_rush_attempts", "player_rush_tds", "player_receptions", "player_reception_yds",
    "player_reception_tds", "player_rush_reception_yds", "player_pass_rush_yds", "player_field_goals", "player_kicking_points",
    "player_sacks", "player_tackles_assists")]
LINE_MARKETS = ["h2h", "spreads", "totals"]
PERIOD_MARKETS = [f"{m}_{p}" for p in ("q1", "q2", "q3", "q4", "h1", "h2")
                  for m in ("h2h", "spreads", "totals", "team_totals", "alternate_spreads", "alternate_totals")]

# Each pass = (snapshot offset before kickoff, regions, market set, earliest kickoff). Ordered by value per credit.
PASSES = {
    "close":      dict(offset=timedelta(minutes=10), regions="us,us2,us_ex", markets=PROP_MARKETS + LINE_MARKETS, start=PROPS_START),
    "close_alt":  dict(offset=timedelta(minutes=10), regions="us",           markets=ALT_MARKETS, start=PROPS_START),
    "day_before": dict(offset=timedelta(hours=24),   regions="us,us_ex",     markets=PROP_MARKETS + LINE_MARKETS, start=PROPS_START),
    "open":       dict(offset=timedelta(hours=96),   regions="us,us_ex",     markets=PROP_MARKETS + LINE_MARKETS, start=PROPS_START),
    "periods_close": dict(offset=timedelta(minutes=10), regions="us,us_ex", markets=PERIOD_MARKETS, start=PROPS_START),
    "periods_open":  dict(offset=timedelta(hours=96),   regions="us,us_ex", markets=PERIOD_MARKETS, start=PROPS_START),
    "lines_close": dict(offset=timedelta(minutes=10), regions="us,eu",       markets=LINE_MARKETS, start=LINES_START, end=PROPS_START),
    "lines_open":  dict(offset=timedelta(hours=144), regions="us,eu",        markets=LINE_MARKETS, start=LINES_START, end=PROPS_START),
}

TEAM = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC", "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    "Washington Football Team": "WAS", "Washington Redskins": "WAS",
}


def load_key():
    key = os.environ.get("ODDS_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ODDS_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit("ODDS_API_KEY not set (put it in .env)")
    return key


class NotFound(Exception):
    pass


class Client:
    def __init__(self, key, budget=None):
        self.key, self.budget, self.spent, self.remaining = key, budget, 0, None
        self.lock = Lock()
        self.s = requests.Session()

    def get(self, path, **params):
        if self.budget is not None and self.spent >= self.budget:
            raise RuntimeError(f"budget of {self.budget} credits reached")
        params["apiKey"] = self.key
        for attempt in range(6):
            r = self.s.get(f"{API}{path}", params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(2 ** attempt); continue
            with self.lock:
                self.spent += int(r.headers.get("x-requests-last", 0) or 0)
                if r.headers.get("x-requests-remaining"):
                    self.remaining = int(float(r.headers["x-requests-remaining"]))
            if r.status_code == 404:
                raise NotFound(r.text[:200])
            if r.status_code != 200:
                raise RuntimeError(f"{r.status_code} {path}: {r.text[:300]}")
            return r.json()
        raise RuntimeError(f"rate limited: {path}")


def write_gz(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(obj, f)
    tmp.replace(path)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def nfl_games():
    con = duckdb.connect(str(DB), read_only=True)
    return con.execute("""SELECT game_id, season, week, gameday, gametime, home_team, away_team
                          FROM schedules WHERE season >= 2020 ORDER BY gameday, gametime""").df()


def match_game(games, home, away, commence):
    h, a = TEAM.get(home), TEAM.get(away)
    cand = games[(games.home_team == h) & (games.away_team == a)]
    if cand.empty:
        return None
    d = commence.date()
    cand = cand.assign(dd=(cand.gameday.astype("datetime64[ns]") - datetime(d.year, d.month, d.day)).abs())
    best = cand.sort_values("dd").iloc[0]
    return best.game_id if best.dd <= timedelta(days=2) else None


# ------------------------------------------------------------------ discover historical event IDs
def discover(client):
    games = nfl_games()
    now = datetime.now(timezone.utc)
    weeks = games.groupby(["season", "week"]).gameday.agg(["min", "max"]).reset_index()
    out = RAW / "events"
    for w in weeks.itertuples():
        first = datetime.fromisoformat(str(w.min)).replace(tzinfo=timezone.utc)
        last = datetime.fromisoformat(str(w.max)).replace(tzinfo=timezone.utc) + timedelta(days=1, hours=12)
        snap = first - timedelta(days=2)  # events are listed well before kickoff
        if snap < LINES_START or snap > now:
            continue
        path = out / f"{w.season}_{w.week:02d}.json.gz"
        if path.exists():
            continue
        data = client.get(f"/historical/sports/{SPORT}/events", date=iso(snap),
                          commenceTimeFrom=iso(first - timedelta(hours=12)), commenceTimeTo=iso(last))
        for ev in data.get("data", []):
            ev["game_id"] = match_game(games, ev["home_team"], ev["away_team"], datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")))
        write_gz(path, data)
        n = len(data.get("data", [])); miss = sum(1 for e in data.get("data", []) if not e["game_id"])
        print(f"{w.season} wk{w.week:02d}: {n} events, {miss} unmatched | spent {client.spent}, remaining {client.remaining}")


def all_events():
    evs = {}
    for p in sorted((RAW / "events").glob("*.json.gz")):
        for ev in json.load(gzip.open(p, "rt")).get("data", []):
            if ev.get("game_id"):
                evs[ev["id"]] = ev
    return list(evs.values())


# ------------------------------------------------------------------ pull a pass
def pull(client, pass_name, workers=6, limit=None):
    cfg = PASSES[pass_name]
    now = datetime.now(timezone.utc)
    todo = []
    for ev in all_events():
        ko = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        if ko < cfg["start"] or ko > now or ("end" in cfg and ko >= cfg["end"]):
            continue
        path = RAW / pass_name / f"{ev['id']}.json.gz"
        if not path.exists():
            todo.append((ev, ko - cfg["offset"], path))
    todo = todo[:limit] if limit else todo
    print(f"pass {pass_name}: {len(todo)} events to fetch ({cfg['regions']}, {len(cfg['markets'])} markets requested)")

    def one(item):
        ev, snap, path = item
        data = client.get(f"/historical/sports/{SPORT}/events/{ev['id']}/odds", regions=cfg["regions"],
                          markets=",".join(cfg["markets"]), date=iso(snap), oddsFormat="american")
        data["game_id"] = ev["game_id"]; data["pass"] = pass_name
        write_gz(path, data)

    done = missing = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, it) for it in todo]
        for f in as_completed(futs):
            try:
                f.result(); done += 1
            except NotFound:
                missing += 1; continue
            except RuntimeError as e:
                print("stop:", e); ex.shutdown(cancel_futures=True); break
            if done % 50 == 0:
                print(f"  {done}/{len(todo)} | spent {client.spent} | remaining {client.remaining}")
    print(f"pass {pass_name}: fetched {done}, not found {missing}, spent {client.spent}, remaining {client.remaining}")


# ------------------------------------------------------------------ live test (free tier)
def live_test(client, n_events=2):
    events = client.get(f"/sports/{SPORT}/events")  # free
    now = datetime.now(timezone.utc)
    games = nfl_games()
    upcoming = [e for e in events if datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) > now][:n_events]
    for ev in upcoming:
        ev["game_id"] = match_game(games, ev["home_team"], ev["away_team"], datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")))
        before = client.spent
        odds = client.get(f"/sports/{SPORT}/events/{ev['id']}/odds", regions="us,us_ex",
                          markets=",".join(PROP_MARKETS + LINE_MARKETS), oddsFormat="american")
        wrapped = {"timestamp": iso(now), "data": odds, "game_id": ev["game_id"], "pass": "live_test"}
        write_gz(RAW / "live_test" / f"{ev['id']}.json.gz", wrapped)
        books = [b["key"] for b in odds.get("bookmakers", [])]
        mk = sorted({m["key"] for b in odds.get("bookmakers", []) for m in b["markets"]})
        print(f"{ev['away_team']} @ {ev['home_team']} -> {ev['game_id']}: cost {client.spent - before}, "
              f"{len(books)} books {books}\n  {len(mk)} markets: {mk}")
    print(f"live test spent {client.spent}, remaining {client.remaining}")


# ------------------------------------------------------------------ load raw -> DuckDB
LOAD_SQL = """
CREATE OR REPLACE TABLE odds_raw AS
WITH f AS (
  SELECT game_id, pass, CAST(timestamp AS TIMESTAMPTZ) AS snapshot_ts, data
  FROM read_json({globs}, union_by_name = true, maximum_object_size = 100000000,
                 columns = {{game_id: 'VARCHAR', pass: 'VARCHAR', timestamp: 'VARCHAR',
                            data: 'STRUCT(id VARCHAR, commence_time VARCHAR, bookmakers STRUCT(key VARCHAR, markets STRUCT(key VARCHAR, last_update VARCHAR, outcomes STRUCT(name VARCHAR, description VARCHAR, price DOUBLE, point DOUBLE)[])[])[])'}})
  WHERE pass IS NOT NULL
), b AS (SELECT game_id, pass, snapshot_ts, data.id AS event_id, CAST(data.commence_time AS TIMESTAMPTZ) AS commence_time,
                unnest(data.bookmakers) AS bk FROM f),
   m AS (SELECT * EXCLUDE (bk), bk.key AS bookmaker, unnest(bk.markets) AS mk FROM b),
   o AS (SELECT * EXCLUDE (mk), mk.key AS market, CAST(mk.last_update AS TIMESTAMPTZ) AS last_update, unnest(mk.outcomes) AS oc FROM m)
SELECT game_id, event_id, pass, snapshot_ts, commence_time, bookmaker, market, last_update,
       oc.description AS player, oc.name AS outcome, oc.point AS point, CAST(oc.price AS INTEGER) AS price
FROM o;
CREATE OR REPLACE VIEW odds_props AS SELECT * FROM odds_raw WHERE market LIKE 'player_%';
CREATE OR REPLACE VIEW odds_lines AS SELECT * FROM odds_raw WHERE market IN ('h2h', 'spreads', 'totals');
"""


def load():
    con = duckdb.connect(str(DB))
    dirs = [d for d in RAW.iterdir() if d.is_dir() and d.name != "events" and any(d.glob("*.json.gz"))]
    globs = "[" + ", ".join(f"'{(d / '*.json.gz').as_posix()}'" for d in dirs) + "]"
    con.execute(LOAD_SQL.format(globs=globs))
    print(con.execute("SELECT pass, count(DISTINCT event_id) events, count(DISTINCT bookmaker) books, count(DISTINCT market) markets, count(*) n_rows FROM odds_raw GROUP BY 1 ORDER BY 1").df())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["live-test", "discover", "pull", "load"])
    ap.add_argument("--pass", dest="pass_name", choices=list(PASSES))
    ap.add_argument("--budget", type=int, help="max credits to spend this run")
    ap.add_argument("--limit", type=int, help="max events this run (for sampling cost)")
    a = ap.parse_args()
    if a.cmd == "load":
        load(); sys.exit()
    c = Client(load_key(), a.budget)
    {"live-test": lambda: live_test(c), "discover": lambda: discover(c),
     "pull": lambda: pull(c, a.pass_name, limit=a.limit)}[a.cmd]()
