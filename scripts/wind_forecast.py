"""Idea #2: bet unders off the *forecast* wind, at the total available when the forecast was made.

Forecasts: Open-Meteo Previous Runs API, GFS (archived from March 2021): wind predicted 1/2/4/6 days ahead of kickoff.
Totals: The Odds API snapshots (consensus of books + best under price, plus exchanges), and nflverse closes.
Writes data/weather/*.json (cache), table game_wind_forecast, and results/wind_forecast.json.
"""
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
WX = ROOT / "data/weather"; WX.mkdir(parents=True, exist_ok=True)
STADIUMS = {  # outdoor NFL stadiums (lat, lon)
    "NYC01": (40.8135, -74.0745), "KAN00": (39.0489, -94.4839), "BUF00": (42.7738, -78.7870), "PHI00": (39.9008, -75.1675),
    "SFO01": (37.4030, -121.9700), "TAM00": (27.9759, -82.5033), "BAL00": (39.2780, -76.6227), "SEA00": (47.5952, -122.3316),
    "DEN00": (39.7439, -105.0201), "CIN00": (39.0955, -84.5161), "BOS00": (42.0909, -71.2643), "CHI98": (41.8623, -87.6167),
    "PIT00": (40.4468, -80.0158), "CAR00": (35.2258, -80.8528), "MIA00": (25.9580, -80.2389), "GNB00": (44.5013, -88.0622),
    "NAS00": (36.1665, -86.7713), "CLE00": (41.5061, -81.6995), "WAS00": (38.9078, -76.8645), "JAX00": (30.3239, -81.6373),
}
LEADS = [1, 2, 4, 6]
START = "2021-03-25"
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"))


def fetch(sid, lat, lon):
    path = WX / f"{sid}.json"
    if path.exists():
        return json.loads(path.read_text())
    end = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    vars_ = ["wind_speed_10m", "wind_gusts_10m", "precipitation"] + \
            [f"{v}_previous_day{d}" for v in ("wind_speed_10m", "wind_gusts_10m", "precipitation") for d in LEADS]
    for attempt in range(5):
        r = requests.get("https://previous-runs-api.open-meteo.com/v1/forecast", timeout=180, params={
            "latitude": lat, "longitude": lon, "start_date": START, "end_date": end, "hourly": ",".join(vars_),
            "models": "gfs_seamless", "wind_speed_unit": "mph", "precipitation_unit": "inch", "timezone": "UTC"})
        if r.status_code == 200:
            path.write_text(r.text); return r.json()
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{sid}: {r.status_code} {r.text[:200]}")


# ------------------------------------------------------------------ forecasts per game (kickoff hour .. +3h mean)
games = con.execute(f"""
    SELECT g.game_id, g.season, g.week, g.gameday, g.gametime, g.home_team, g.away_team, g.total, g.total_line AS nfl_close_total,
           g.ou_result, g.wind_mph AS obs_wind, s.stadium_id
    FROM games g JOIN schedules s USING (game_id)
    WHERE g.gameday >= DATE '{START}' + INTERVAL 7 DAY AND s.stadium_id IN ({",".join("'" + k + "'" for k in STADIUMS)})
""").df()
et = ZoneInfo("America/New_York")
games["kick_utc"] = [datetime.fromisoformat(f"{d} {t}").replace(tzinfo=et).astimezone(timezone.utc).replace(tzinfo=None)
                     for d, t in zip(games.gameday.astype(str), games.gametime)]
rows = []
for sid, (lat, lon) in STADIUMS.items():
    js = fetch(sid, lat, lon)
    h = pd.DataFrame(js["hourly"]); h["time"] = pd.to_datetime(h.time)
    h = h.set_index("time")
    for g in games[games.stadium_id == sid].itertuples():
        k0 = pd.Timestamp(g.kick_utc).floor("h")
        win = h.loc[k0: k0 + pd.Timedelta(hours=3)]
        rec = {"game_id": g.game_id}
        for d in [0] + LEADS:
            sfx = "" if d == 0 else f"_previous_day{d}"
            rec[f"wind_d{d}"] = win[f"wind_speed_10m{sfx}"].mean()
            rec[f"gust_d{d}"] = win[f"wind_gusts_10m{sfx}"].mean()
            rec[f"precip_d{d}"] = win[f"precipitation{sfx}"].sum()
        rows.append(rec)
    print(f"{sid}: {int((games.stadium_id == sid).sum())} games")
fc = games.merge(pd.DataFrame(rows), on="game_id")
con.execute("CREATE OR REPLACE TABLE game_wind_forecast AS SELECT * FROM fc")

# ------------------------------------------------------------------ totals available at each snapshot
tot = con.execute("""
    WITH t AS (
      SELECT game_id, pass, bookmaker, point, max(price) FILTER (WHERE outcome = 'Under') AS under_price,
             max(price) FILTER (WHERE outcome = 'Over') AS over_price
      FROM odds_raw WHERE market = 'totals' GROUP BY ALL),
    books AS (SELECT * FROM t WHERE bookmaker NOT IN ('kalshi','polymarket','novig','prophetx','betopenly')),
    cons AS (SELECT game_id, pass, median(point) AS cons_total, count(DISTINCT bookmaker) nb FROM books GROUP BY ALL)
    SELECT c.game_id, c.pass, c.cons_total, c.nb,
           max(b.under_price) FILTER (WHERE b.point = c.cons_total) AS best_under,
           max(b.under_price) FILTER (WHERE b.point = c.cons_total AND b.bookmaker IN ('draftkings','fanduel','betmgm','williamhill_us')) AS major_under,
           max(e.under_price) FILTER (WHERE e.point = c.cons_total) AS exch_under
    FROM cons c LEFT JOIN books b USING (game_id, pass)
    LEFT JOIN (SELECT * FROM t WHERE bookmaker IN ('kalshi','novig','prophetx','polymarket')) e USING (game_id, pass)
    GROUP BY ALL""").df()
tot["pass"] = tot["pass"].replace({"lines_open": "open6d", "lines_close": "close"})
wide = tot.pivot_table(index="game_id", columns="pass", values=["cons_total", "best_under", "major_under", "exch_under"], aggfunc="first")
wide.columns = [f"{a}_{b}" for a, b in wide.columns]
df = fc.merge(wide.reset_index(), on="game_id", how="left")


def pay(o):
    return np.where(o < 0, 100 / -o, o / 100)


def evalset(d, line_col, price_col, fallback=-110):
    d = d.dropna(subset=[line_col])
    under = np.where(d.total < d[line_col], 1.0, np.where(d.total > d[line_col], 0.0, np.nan))
    price = d[price_col].fillna(fallback).values if price_col in d else np.full(len(d), fallback)
    ok = ~np.isnan(under)
    prof = np.where(under[ok] == 1, pay(price[ok]), -1.0)
    return {"n": int(ok.sum()), "under_pct": round(float(under[ok].mean()), 4) if ok.sum() else None,
            "roi": round(float(prof.mean()), 4) if ok.sum() else None,
            "se": round(float(prof.std() / math.sqrt(ok.sum())), 4) if ok.sum() > 1 else None,
            "avg_line": round(float(d[line_col].mean()), 2) if len(d) else None, "avg_total": round(float(d.total.mean()), 2) if len(d) else None}


R = {"games": len(df), "forecast_skill": {}}
for d in [0] + LEADS:
    ok = df[[f"wind_d{d}", "obs_wind"]].dropna()
    R["forecast_skill"][f"day{d}"] = {"corr_with_observed": round(float(ok.corr().iloc[0, 1]), 3), "n": len(ok),
                                      "mean_abs_err_mph": round(float((ok[f"wind_d{d}"] - ok.obs_wind).abs().mean()), 2)}

# snapshot line x forecast lead that was available at that moment
PAIRS = [("open6d", 6, "cons_total_open6d", "best_under_open6d"),       # 2021-2022 lines 6 days out
         ("open4d", 4, "cons_total_open", "best_under_open"),           # 2023+ lines 4 days out
         ("day_before", 1, "cons_total_day_before", "best_under_day_before"),
         ("close (odds api)", 1, "cons_total_close", "best_under_close"),
         ("close (nflverse, -110)", 1, "nfl_close_total", "__none__")]
R["by_snapshot"] = {}
for lab, lead, line, price in PAIRS:
    out = {}
    for th in (0, 10, 12, 15, 18):
        sub = df[df[f"wind_d{lead}"] >= th] if th else df
        out[f"fcst wind >= {th}" if th else "all games"] = evalset(sub, line, price)
    # the useful comparison: did windy-forecast games' totals drop between this snapshot and the close?
    windy = df[df[f"wind_d{lead}"] >= 12].dropna(subset=[line, "nfl_close_total"])
    out["line move to close, windy fcst"] = round(float((windy.nfl_close_total - windy[line]).mean()), 2) if len(windy) else None
    calm = df[df[f"wind_d{lead}"] < 8].dropna(subset=[line, "nfl_close_total"])
    out["line move to close, calm fcst"] = round(float((calm.nfl_close_total - calm[line]).mean()), 2) if len(calm) else None
    R["by_snapshot"][lab] = out

# exchange prices for the windy unders (where you'd actually bet)
for lab, line, price in (("day_before", "cons_total_day_before", "exch_under_day_before"), ("open4d", "cons_total_open", "exch_under_open")):
    sub = df[(df.wind_d1 >= 12) if lab == "day_before" else (df.wind_d4 >= 12)].dropna(subset=[price])
    R[f"exchange_under_{lab}_wind12"] = evalset(sub, line, price)

# by season, day-before forecast >= 12 at the nflverse close (largest sample)
R["by_season_d1_ge12_close"] = {int(s): evalset(g[g.wind_d1 >= 12], "nfl_close_total", "__none__") for s, g in df.groupby("season")}
# gusts version
R["gusts_d1"] = {f"gust >= {th}": evalset(df[df.gust_d1 >= th], "nfl_close_total", "__none__") for th in (20, 25, 30)}
(ROOT / "results/wind_forecast.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps(R, indent=1, default=str))
