"""Quarter / half markets: are books' period lines (derived from the full-game line) mispriced?

Lines: The Odds API periods_close / periods_open snapshots (2023-05+), consensus main line + de-vigged fair prob per market.
Outcomes: game_periods (points by quarter from play-by-play, 1999+).
Model: for each game, P(period over line) / P(side covers) estimated from *earlier seasons only* by nearest neighbours on
the full-game closing total and spread (no odds needed for training, so it can use 2011+). Compared with the book price.
Writes results/quarters.json.
"""
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
EXCH = ("kalshi", "polymarket", "novig", "prophetx", "betopenly")
payout = lambda o: np.where(np.asarray(o, float) < 0, 100 / -np.asarray(o, float), np.asarray(o, float) / 100)  # noqa: E731
imp = lambda o: np.where(np.asarray(o, float) < 0, -np.asarray(o, float) / (-np.asarray(o, float) + 100), 100 / (np.asarray(o, float) + 100))  # noqa: E731
R: dict = {}

# ------------------------------------------------------------------ outcomes per period
gp = con.execute("SELECT * FROM game_periods WHERE season >= 2011").df()
P_ = {"q1": ["q1"], "q2": ["q2"], "q3": ["q3"], "q4": ["q4"], "h1": ["q1", "q2"], "h2": ["q3", "q4"]}  # books grade h2/q4 incl. OT? see below
for p, qs in P_.items():
    gp[f"{p}_h"] = sum(gp[f"{q}_h"] for q in qs); gp[f"{p}_a"] = sum(gp[f"{q}_a"] for q in qs)
# US books settle 2nd-half and 4th-quarter markets including overtime
for p in ("h2", "q4"):
    gp[f"{p}_h"] = gp[f"{p}_h"] + gp.ot_h.fillna(0); gp[f"{p}_a"] = gp[f"{p}_a"] + gp.ot_a.fillna(0)
shares = {p: round(float((gp[f"{p}_h"] + gp[f"{p}_a"]).sum() / gp.total.sum()), 4) for p in P_}
R["scoring_share_2011_25"] = shares

# ------------------------------------------------------------------ period lines (consensus + fair) from odds_raw
o = con.execute(f"""
    SELECT o.game_id, left(o.game_id, 4)::INT AS season, o.pass, o.bookmaker, o.market, o.outcome, o.point, o.price, g.home_team
    FROM odds_raw o JOIN games g USING (game_id)
    WHERE o.pass IN ('periods_close', 'periods_open') AND regexp_matches(o.market, '^(spreads|totals)_(q[1-4]|h[12])$')
""").df()
TEAM = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF", "Carolina Panthers": "CAR",
        "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
        "Detroit Lions": "DET", "Green Bay Packers": "GB", "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
        "Kansas City Chiefs": "KC", "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC", "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ",
        "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF",
        "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}
o["kind"] = o.market.str.split("_").str[0]; o["period"] = o.market.str.split("_").str[1]
o["side"] = np.where(o.kind == "totals", o.outcome.str.lower(), np.where(o.outcome.map(TEAM) == o.home_team, "home", "away"))
o = o[o.side.isin(["over", "under", "home", "away"])]
# home-perspective point for spreads so both sides share one key
o["line"] = np.where(o.kind == "spreads", np.where(o.side == "home", o.point, -o.point), o.point)
w = o.pivot_table(index=["game_id", "season", "pass", "bookmaker", "kind", "period", "line"], columns="side", values="price", aggfunc="first").reset_index()
w["a_side"] = np.where(w.kind == "totals", w.get("over"), w.get("home"))
w["b_side"] = np.where(w.kind == "totals", w.get("under"), w.get("away"))
w = w.dropna(subset=["a_side", "b_side"])
w["pa"], w["pb"] = imp(w.a_side), imp(w.b_side)
w = w[(w.pa + w.pb - 1).between(-0.02, 0.3)]
w["fair_a"] = w.pa / (w.pa + w.pb)
w["exch"] = w.bookmaker.isin(EXCH)
w["dist"] = (w.fair_a - .5).abs()
key = ["game_id", "season", "pass", "kind", "period"]
main = w.sort_values("dist").drop_duplicates(key + ["bookmaker"])
cons = main[~main.exch].groupby(key).agg(line=("line", "median"), n_books=("bookmaker", "nunique")).reset_index()
at = main.merge(cons[key + ["line"]], on=key + ["line"])
L = at[~at.exch].groupby(key + ["line"]).agg(fair_a=("fair_a", "mean"), best_a=("a_side", "max"), best_b=("b_side", "max")).reset_index()
L = L.merge(cons[key + ["n_books"]], on=key)
ex = at[at.exch].groupby(key + ["line"]).agg(ex_a=("a_side", "max"), ex_b=("b_side", "max")).reset_index()
L = L.merge(ex, on=key + ["line"], how="left")
R["line_coverage"] = L.groupby(["pass", "kind", "period"]).size().unstack(["kind"]).fillna(0).astype(int).reset_index().to_dict("records")

# grade: a-side = over (totals) / home covers (spreads, home-perspective line = home points given: home_margin + line > 0)
G = gp[["game_id", "season", "week", "total_line", "spread_line", "roof", "wind_mph"] + [f"{p}_{s}" for p in P_ for s in "ha"]]
L = L.merge(G, on=["game_id", "season"])
L["pts"] = np.select([L.period == p for p in P_], [L[f"{p}_h"] + L[f"{p}_a"] for p in P_])
L["margin"] = np.select([L.period == p for p in P_], [L[f"{p}_h"] - L[f"{p}_a"] for p in P_])
val = np.where(L.kind == "totals", L.pts - L.line, L.margin + L.line)
L["a_win"] = np.where(val > 0, 1.0, np.where(val < 0, 0.0, np.nan))

# ------------------------------------------------------------------ 1. raw bias by period
bias = []
for (ps, kd, pr), d in L.groupby(["pass", "kind", "period"]):
    d = d.dropna(subset=["a_win"])
    if len(d) < 50:
        continue
    ra = np.where(d.a_win == 1, payout(d.best_a), -1.0); rb = np.where(d.a_win == 0, payout(d.best_b), -1.0)
    bias.append({"pass": ps, "market": f"{kd}_{pr}", "n": len(d), "a_rate": round(d.a_win.mean(), 4), "fair_a": round(d.fair_a.mean(), 4),
                 "roi_a_best": round(ra.mean(), 4), "roi_b_best": round(rb.mean(), 4),
                 "push_rate": round(float(np.isnan(np.where(np.where(d.kind == 'totals', d.pts - d.line, d.margin + d.line) == 0, np.nan, 1)).mean()), 4),
                 "by_season_a_rate": {int(s): round(x.a_win.mean(), 4) for s, x in d.groupby("season")}})
R["bias"] = bias

# ------------------------------------------------------------------ 2. neighbour model from earlier seasons' outcomes
def nn_prob(train, tl, sp, kind, period, line, k=400):
    """P(a-side wins) among the k training games closest in (total_line, spread_line)."""
    d = ((train.total_line - tl) / 3.0) ** 2 + ((train.spread_line - sp) / 2.5) ** 2
    nb = train.iloc[np.argpartition(d.values, min(k, len(d) - 1))[:k]]
    if kind == "totals":
        v = nb[f"{period}_h"] + nb[f"{period}_a"] - line
    else:
        v = nb[f"{period}_h"] - nb[f"{period}_a"] + line
    wins, losses = (v > 0).sum(), (v < 0).sum()
    return wins / (wins + losses) if wins + losses else np.nan


M = L[L["pass"] == "periods_close"].dropna(subset=["a_win"]).copy()
M["p_model"] = np.nan
for s in sorted(M.season.unique()):
    train = gp[(gp.season < s) & (gp.season >= s - 8)].dropna(subset=["total_line", "spread_line"])
    idx = M.index[M.season == s]
    M.loc[idx, "p_model"] = [nn_prob(train, r.total_line, r.spread_line, r.kind, r.period, r.line) for r in M.loc[idx].itertuples()]
M["edge_a"] = M.p_model - M.fair_a
R["model_vs_market_brier"] = {f"{kd}_{pr}": {"n": len(d), "brier_market": round(float(((d.fair_a - d.a_win) ** 2).mean()), 5),
                                             "brier_model": round(float(((d.p_model - d.a_win) ** 2).mean()), 5)}
                              for (kd, pr), d in M.groupby(["kind", "period"])}


def sim(d, pa_col, pb_col, label):
    out = {}
    for th in (0.0, 0.02, 0.04, 0.06, 0.08):
        ea = d.p_model * (1 + payout(d[pa_col])) - 1; eb = (1 - d.p_model) * (1 + payout(d[pb_col])) - 1
        a, b = d[ea >= th], d[eb >= th]
        win = np.r_[a.a_win, 1 - b.a_win]; od = np.r_[a[pa_col], b[pb_col]]; ss = np.r_[a.season, b.season]
        mk = np.r_[a.kind + "_" + a.period, b.kind + "_" + b.period]
        pr = np.where(win == 1, payout(od), -1.0)
        out[f"EV>={int(th*100)}%"] = {"n": len(pr), "roi": round(pr.mean(), 4) if len(pr) else None,
                                      "se": round(pr.std() / math.sqrt(len(pr)), 4) if len(pr) > 1 else None,
                                      "by_season": {int(s): [int((ss == s).sum()), round(pr[ss == s].mean(), 4)] for s in np.unique(ss)},
                                      "by_market": {m_: [int((mk == m_).sum()), round(pr[mk == m_].mean(), 4)] for m_ in np.unique(mk)}}
    R[label] = out


sim(M, "best_a", "best_b", "model_roi_best_book")
sim(M.dropna(subset=["ex_a", "ex_b"]), "ex_a", "ex_b", "model_roi_exchange")

# ------------------------------------------------------------------ 3. situational residuals (actual vs fair), consensus close
S = M.copy()
S["resid"] = S.a_win - S.fair_a
S["abs_spread"] = S.spread_line.abs()
S["windy"] = (S.wind_mph.fillna(0) >= 12) & S.roof.isin(["outdoors", "open"])
S["dome"] = S.roof.isin(["dome", "closed"])
sit = []
for (kd, pr), d in S.groupby(["kind", "period"]):
    for lab, m in {"big spread (7.5+)": d.abs_spread >= 7.5, "pick'em-ish (<=3)": d.abs_spread <= 3, "high total (48+)": d.total_line >= 48,
                   "low total (<=41)": d.total_line <= 41, "windy 12+ mph": d.windy, "dome": d.dome}.items():
        x = d[m]
        if len(x) >= 60:
            sit.append({"market": f"{kd}_{pr}", "situation": lab, "n": len(x), "resid": round(x.resid.mean(), 4),
                        "t": round(x.resid.mean() / (x.resid.std() / math.sqrt(len(x))), 2)})
R["situational"] = sorted(sit, key=lambda r: -abs(r["t"]))[:30]

(ROOT / "results/quarters.json").write_text(json.dumps(R, indent=1, default=str))
pd.set_option("display.width", 220)
print("scoring share:", shares)
print(pd.DataFrame(R["line_coverage"]).to_string(index=False))
print(pd.DataFrame(bias).drop(columns="by_season_a_rate").to_string(index=False))
print(json.dumps(R["model_vs_market_brier"], indent=0))
for lab in ("model_roi_best_book", "model_roi_exchange"):
    print(lab)
    for k, v in R[lab].items():
        print("  ", k, v["n"], v["roi"], v["se"], v["by_season"], v["by_market"])
print(pd.DataFrame(R["situational"]).to_string(index=False))
