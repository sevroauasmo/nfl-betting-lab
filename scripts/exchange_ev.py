"""Idea #1: bet exchanges / prediction markets when they're off the sportsbook consensus.

Reference: de-vigged sportsbook prices at the same line (all non-exchange books, >=4 at that point).
Target: Kalshi, Polymarket, Novig, ProphetX, BetOpenly quotes (game lines + props), graded against results.
Kalshi taker fee is applied (0.07 * p * (1 - p) per $1 contract, rounded up to the cent).
Writes results/exchange_ev.json.
"""
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXCH = ["kalshi", "polymarket", "novig", "prophetx", "betopenly"]
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)


def imp(o):
    o = np.asarray(o, float)
    return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))


def roi(win, profit):
    ok = ~np.isnan(win)
    p = profit[ok]
    return {"n": int(ok.sum()), "roi": round(float(p.mean()), 4) if ok.sum() else None,
            "se": round(float(p.std() / math.sqrt(ok.sum())), 4) if ok.sum() > 1 else None}


# ------------------------------------------------------------------ game lines: one row per (game, pass, venue, market, side, point)
q = con.execute("""
    SELECT o.game_id, left(o.game_id, 4)::INT AS season, o.pass, o.bookmaker, o.market, o.outcome, o.point, o.price,
           g.home_team, g.away_team, g.result, g.total
    FROM odds_raw o JOIN games g USING (game_id)
    WHERE o.market IN ('h2h', 'spreads', 'totals') AND o.pass IN ('open', 'day_before', 'close') AND o.price IS NOT NULL
""").df()
TEAM = json.loads(json.dumps({  # full name -> nflverse code, as in pull_odds.py
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF", "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB", "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC", "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF",
    "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}))
q["side"] = np.where(q.market == "totals", q.outcome.str.lower(), np.where(q.outcome.map(TEAM) == q.home_team, "home", "away"))
q = q[q.side.isin(["home", "away", "over", "under"])]
q["point"] = q.point.fillna(0.0)
# grade each side
res = q.result.astype(float); tot = q.total.astype(float)
margin_side = np.where(q.side == "home", res, -res)                       # side's final margin
w = np.select([q.market == "h2h", q.market == "spreads", q.side == "over", q.side == "under"],
              [np.sign(margin_side), np.sign(margin_side + q.point), np.sign(tot - q.point), np.sign(q.point - tot)], np.nan)
q["win"] = np.where(w > 0, 1.0, np.where(w < 0, 0.0, np.nan))           # nan = push/tie
q["p_raw"] = imp(q.price)
q["exch"] = q.bookmaker.isin(EXCH)

# pair sides to de-vig each book's quote
opp = {"home": "away", "away": "home", "over": "under", "under": "over"}
k = ["game_id", "pass", "bookmaker", "market"]
q["opp_side"] = q.side.map(opp)
q["opp_point"] = np.where(q.market == "spreads", -q.point, q.point)
m = q.merge(q[k + ["side", "point", "p_raw"]].rename(columns={"side": "opp_side", "point": "opp_point", "p_raw": "p_opp"}),
            on=k + ["opp_side", "opp_point"], how="left")
m["p_fair"] = m.p_raw / (m.p_raw + m.p_opp)

ref = (m[~m.exch & m.p_fair.notna()].groupby(["game_id", "pass", "market", "side", "point"])
       .agg(ref=("p_fair", "mean"), n_books=("bookmaker", "nunique")).reset_index())
ref = ref[ref.n_books >= 4]
x = m[m.exch].merge(ref, on=["game_id", "pass", "market", "side", "point"])
# Kalshi: price p per $1 contract plus fee ceil(0.07 * p * (1-p) * 100)/100
cost = x.p_raw + np.where(x.bookmaker == "kalshi", np.ceil(0.07 * x.p_raw * (1 - x.p_raw) * 100) / 100, 0.0)
x["cost"] = cost
x["ev"] = x.ref / x.cost - 1
x["profit"] = np.where(x.win == 1, 1 / x.cost - 1, np.where(x.win == 0, -1.0, 0.0))
x["hold_venue"] = x.p_raw + x.p_opp - 1

R = {"coverage": x.groupby(["bookmaker", "season"]).game_id.nunique().reset_index().to_dict("records"),
     "venue_hold": x.groupby("bookmaker").hold_venue.mean().round(4).to_dict()}
R["lines_ev"] = {}
for th in (0.0, 0.01, 0.02, 0.03, 0.05):
    d = x[x.ev >= th]
    R["lines_ev"][f"EV>={int(th*100)}%"] = {
        "all": roi(d.win.values, d.profit.values),
        "by_venue": {b: roi(g.win.values, g.profit.values) for b, g in d.groupby("bookmaker")},
        "by_market": {b: roi(g.win.values, g.profit.values) for b, g in d.groupby("market")},
        "by_pass": {b: roi(g.win.values, g.profit.values) for b, g in d.groupby("pass")},
        "by_season": {int(b): roi(g.win.values, g.profit.values) for b, g in d.groupby("season")},
        "avg_claimed_ev": round(float(d.ev.mean()), 4) if len(d) else None}
# baseline: every exchange quote regardless of EV (what does just trading there cost?)
R["lines_all_quotes"] = {b: roi(g.win.values, g.profit.values) for b, g in x.groupby("bookmaker")}

# calibration of the reference itself: are books' de-vigged probabilities right?
cal = x.drop_duplicates(["game_id", "pass", "market", "side", "point"]).dropna(subset=["win"])
cal = cal.assign(b=pd.cut(cal.ref, [0, .2, .35, .45, .55, .65, .8, 1]))
R["ref_calibration"] = cal.groupby("b", observed=True).agg(n=("win", "size"), ref=("ref", "mean"), actual=("win", "mean")).round(4).reset_index().assign(b=lambda d: d.b.astype(str)).to_dict("records")

# ------------------------------------------------------------------ props on exchanges vs book consensus (same point)
pr = con.execute("""
    SELECT game_id, left(game_id,4)::INT season, pass, bookmaker, market, player_key, point, over_price, under_price, actual
    FROM prop_results WHERE over_price IS NOT NULL AND under_price IS NOT NULL AND point IS NOT NULL AND actual IS NOT NULL
      AND pass IN ('open','day_before','close') AND market NOT LIKE '%alternate'
""").df()
pr["po"], pr["pu"] = imp(pr.over_price), imp(pr.under_price)
pr = pr[(pr.po + pr.pu - 1).between(-0.02, 0.25)]
pr["fair"] = pr.po / (pr.po + pr.pu)
pk = ["game_id", "pass", "market", "player_key", "point"]
pref = pr[~pr.bookmaker.isin(EXCH)].groupby(pk).agg(ref=("fair", "mean"), n_books=("bookmaker", "nunique")).reset_index()
pref = pref[pref.n_books >= 4]
px = pr[pr.bookmaker.isin(EXCH)].merge(pref, on=pk)
px["over"] = np.where(px.actual > px.point, 1.0, np.where(px.actual < px.point, 0.0, np.nan))
rows = []
for side in ("over", "under"):
    c = px.po if side == "over" else px.pu
    c = c + np.where(px.bookmaker == "kalshi", np.ceil(0.07 * c * (1 - c) * 100) / 100, 0.0)
    p = px.ref if side == "over" else 1 - px.ref
    win = px.over if side == "over" else 1 - px.over
    rows.append(pd.DataFrame({"season": px.season, "venue": px.bookmaker, "market": px.market, "pass": px["pass"], "side": side,
                              "ev": p / c - 1, "win": win, "profit": np.where(win == 1, 1 / c - 1, np.where(win == 0, -1.0, 0.0))}))
pb = pd.concat(rows)
R["props_ev"] = {}
for th in (0.0, 0.02, 0.04, 0.06):
    d = pb[pb.ev >= th]
    R["props_ev"][f"EV>={int(th*100)}%"] = {"all": roi(d.win.values, d.profit.values),
                                            "by_venue": {b: roi(g.win.values, g.profit.values) for b, g in d.groupby("venue")},
                                            "by_side": {b: roi(g.win.values, g.profit.values) for b, g in d.groupby("side")},
                                            "by_pass": {b: roi(g.win.values, g.profit.values) for b, g in d.groupby("pass")},
                                            "by_season": {int(b): roi(g.win.values, g.profit.values) for b, g in d.groupby("season")}}

(ROOT / "results/exchange_ev.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps({k: R[k] for k in ("venue_hold", "lines_all_quotes")}, indent=1))
for k in ("lines_ev", "props_ev"):
    print("\n==", k)
    for t, v in R[k].items():
        print(t, v["all"], "| claimed" if "avg_claimed_ev" in v else "", v.get("avg_claimed_ev", ""))
        for grp in [g for g in v if g.startswith("by_")]:
            print("   ", grp, {a: (b["n"], b["roi"]) for a, b in v[grp].items()})
print("\nref calibration", pd.DataFrame(R["ref_calibration"]).to_string(index=False))
