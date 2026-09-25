"""Low-liquidity unders: are unders on thin, low-usage player props mispriced while star props are efficient?

We have no handle data, so liquidity is proxied by: the size of the line, how many books post the prop, whether the
exchanges (Novig/ProphetX) list it, the player's trailing snap share, and his league-wide line rank that week.
RB/WR/TE/FB, main lines only, markets: rec yds, rush yds, receptions, rush attempts, rush+rec yds.
Measured at four snapshots (T-2h, T-80m after inactives, T-45m, close ~T-15m).
Win rate is not the test (thin props often have lines like 0.5 receptions priced -250); the test is actual under rate vs
the book's no-vig fair probability, and ROI at median / best / exchange price. Discovery 2023-24, confirmation 2025-26.
Writes results/low_liquidity.json.
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
EXCHANGES = ("novig", "prophetx")
MARKETS = ["player_reception_yds", "player_rush_yds", "player_receptions", "player_rush_attempts", "player_rush_reception_yds"]
PASSES = ["news_t120", "news_t80", "news_t45", "close"]
rng = np.random.default_rng(3)
R: dict = {}


def imp(o):
    o = np.asarray(o, float)
    return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))


raw = con.execute(f"""
    SELECT game_id, left(game_id, 4)::INT AS season, pass, bookmaker, market, player_key, player, player_id,
           CASE WHEN position = 'FB' THEN 'RB' ELSE position END AS position, team, point, over_price, under_price, actual
    FROM prop_results
    WHERE pass IN ({",".join("'" + p + "'" for p in PASSES)}) AND position IN ('RB', 'WR', 'TE', 'FB')
      AND market IN ({",".join("'" + m + "'" for m in MARKETS)})
      AND point IS NOT NULL AND under_price IS NOT NULL AND over_price IS NOT NULL AND actual IS NOT NULL
""").df()
raw = raw[~((raw.market == "player_rush_yds") & (raw.position != "RB")) & ~((raw.market == "player_rush_attempts") & (raw.position != "RB"))]
raw["pu"], raw["po"] = imp(raw.under_price), imp(raw.over_price)
raw = raw[(raw.pu + raw.po - 1).between(0, .25)]  # drop broken quotes
raw["fair_u"] = raw.pu / (raw.pu + raw.po)
raw["u_dec"] = 1 / raw.pu
raw["exch"] = raw.bookmaker.isin(EXCHANGES)
key = ["pass", "game_id", "market", "player_key"]
bk = raw[~raw.exch]
mode_pt = (bk.groupby(key + ["point"]).size().rename("n").reset_index()
           .sort_values("n", ascending=False).drop_duplicates(key).drop(columns="n"))
nbooks = bk.groupby(key).bookmaker.nunique().rename("n_books_any").reset_index()  # books posting any line for him
at = bk.merge(mode_pt, on=key + ["point"])
b = at.groupby(key + ["point"]).agg(season=("season", "first"), player=("player", "first"), player_id=("player_id", "first"),
                                    position=("position", "first"), team=("team", "first"), actual=("actual", "first"),
                                    fair_u=("fair_u", "median"), u_med=("u_dec", "median"), u_best=("u_dec", "max")).reset_index()
ex = raw[raw.exch].merge(mode_pt, on=key + ["point"]).groupby(key).u_dec.max().rename("u_exch").reset_index()
b = b.merge(nbooks, on=key).merge(ex, on=key, how="left")
b["listed_on_exchange"] = b.u_exch.notna()
b["win"] = np.where(b.actual < b.point, 1.0, np.where(b.actual > b.point, 0.0, np.nan))
b = b.dropna(subset=["win"])
for c in ["med", "best", "exch"]:
    b[f"pnl_{c}"] = np.where(b[f"u_{c}"].isna(), np.nan, np.where(b.win == 1, b[f"u_{c}"] - 1, -1.0))

# ------------------------------------------------------------------ liquidity proxies
g = con.execute("SELECT game_id, week, gameday FROM games WHERE season >= 2022").df()
b = b.merge(g, on="game_id", how="left")
b["wk_rank"] = b.groupby(["pass", "season", "week", "market"]).point.rank(ascending=False, method="first")
b["tier"] = np.select([b.wk_rank <= 24, b.wk_rank <= 64], ["star (top 24 line)", "starter (25-64)"], "role/bench (65+)")
cuts = {"player_reception_yds": [0, 14.5, 24.5, 39.5, 59.5, 999], "player_rush_yds": [0, 19.5, 39.5, 59.5, 999],
        "player_receptions": [0, 1.5, 2.5, 3.5, 5.5, 99], "player_rush_attempts": [0, 7.5, 12.5, 16.5, 99],
        "player_rush_reception_yds": [0, 29.5, 49.5, 69.5, 999]}
b["line_bucket"] = ""
for m, c in cuts.items():
    s = b.market == m
    b.loc[s, "line_bucket"] = pd.cut(b.loc[s, "point"], c).astype(str)
b["books_bucket"] = pd.cut(b.n_books_any, [0, 3, 6, 9, 99], labels=["1-3 books", "4-6", "7-9", "10+"]).astype(str)
sn = con.execute("""
    SELECT s.game_id, p.gsis_id AS player_id, s.offense_pct, g.gameday
    FROM snap_counts s JOIN players p ON p.pfr_id = s.pfr_player_id JOIN games g ON g.game_id = s.game_id WHERE s.season >= 2022
""").df().sort_values(["player_id", "gameday"])
sn["snap3"] = sn.groupby("player_id").offense_pct.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
b = b.merge(sn[["game_id", "player_id", "snap3"]], on=["game_id", "player_id"], how="left")
b["snap_bucket"] = pd.cut(b.snap3, [-.01, .35, .6, .8, 1.01], labels=["<35% snaps", "35-60%", "60-80%", "80%+"]).astype(str)
print(f"{(b['pass'] == 'close').sum():,} close bets in {b.game_id.nunique()} games")


def evaluate(d, label, price="pnl_med", boot=1000):
    d = d.dropna(subset=[price])
    if len(d) < 40:
        return None
    gs = d.groupby("game_id").agg(p=(price, "sum"), n=(price, "size"), w=("win", "sum"), f=("fair_u", "sum"))
    idx = rng.integers(0, len(gs), (boot, len(gs)))
    nb = gs.n.values[idx].sum(1)
    rois = gs.p.values[idx].sum(1) / nb
    bias = (gs.w.values[idx].sum(1) - gs.f.values[idx].sum(1)) / nb
    return {"label": label, "bets": int(gs.n.sum()), "under": round(gs.w.sum() / gs.n.sum(), 3),
            "fair": round(gs.f.sum() / gs.n.sum(), 3), "bias": round((gs.w.sum() - gs.f.sum()) / gs.n.sum(), 4),
            "bias_ci90": [round(float(np.quantile(bias, q)), 3) for q in (.05, .95)],
            "roi": round(gs.p.sum() / gs.n.sum(), 4), "roi_ci90": [round(float(np.quantile(rois, q)), 3) for q in (.05, .95)]}


def table(d, col, title, order=None, split=True, extra_prices=True):
    rows = []
    for v in (order or sorted(d[col].dropna().unique())):
        periods = [("23-24", d.season <= 2024), ("25-26", d.season >= 2025)] if split else [("all", d.season > 0)]
        for per, sel in periods:
            dd = d[(d[col] == v) & sel]
            r = evaluate(dd, f"{v} [{per}]")
            if r and extra_prices:
                r["roi_best"] = (evaluate(dd, "", "pnl_best") or {}).get("roi")
                e = evaluate(dd, "", "pnl_exch"); r["roi_exch"] = e["roi"] if e else None; r["n_exch"] = e["bets"] if e else 0
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    print(f"\n== {title}\n{df.to_string(index=False)}")
    return rows


C = b[b["pass"] == "close"]
R["tier"] = table(C, "tier", "league-wide line rank (all markets), close", ["star (top 24 line)", "starter (25-64)", "role/bench (65+)"])
R["books"] = table(C, "books_bucket", "number of books posting the prop, close", ["1-3 books", "4-6", "7-9", "10+"])
R["snaps"] = table(C, "snap_bucket", "trailing-3 snap share, close", ["<35% snaps", "35-60%", "60-80%", "80%+"])
R["exch_listed"] = table(C, "listed_on_exchange", "listed on Novig/ProphetX? (2024+), close", [True, False])
R["position"] = table(C, "position", "position, close", ["RB", "WR", "TE"])
for m in MARKETS:
    R[f"line_{m}"] = table(C[C.market == m], "line_bucket", f"line size: {m}, close", split=False)

# timing: same bets at T-2h / T-80m / T-45m / close, thin (role/bench) vs star
rows = []
for t in ["star (top 24 line)", "starter (25-64)", "role/bench (65+)"]:
    for p in PASSES:
        r = evaluate(b[(b["pass"] == p) & (b.tier == t)], f"{t} @ {p}")
        if r:
            rows.append(r)
R["timing"] = rows
print("\n== timing\n" + pd.DataFrame(rows).to_string(index=False))

# the claimed strategy: blanket unders on thin props only, bet ~T-45m (after inactives)
thin = (b.tier == "role/bench (65+)") | (b.n_books_any <= 6) | (b.snap3 < .5)
rows = [evaluate(b[(b["pass"] == "news_t45") & thin & (b.season == s)], f"thin, T-45, {s}") for s in sorted(b.season.unique())]
rows += [evaluate(b[(b["pass"] == "news_t45") & ~thin & (b.season == s)], f"not thin, T-45, {s}") for s in sorted(b.season.unique())]
R["thin_strategy"] = [r for r in rows if r]
print("\n== thin-prop blanket unders at T-45 by season (median price)\n" + pd.DataFrame(R["thin_strategy"]).to_string(index=False))

# the "win rate" illusion: under win % vs ROI by fair-probability bucket
C2 = C.assign(fb=pd.cut(C.fair_u, [0, .4, .47, .53, .6, .7, 1]).astype(str))
R["winrate_vs_roi"] = table(C2, "fb", "under win rate vs ROI by book's fair under prob", split=False, extra_prices=False)

(ROOT / "results/low_liquidity.json").write_text(json.dumps(R, indent=1, default=str))

# the one proxy that held in both halves: low trailing snap share. By season, pooled, and by market.
lo = C[C.snap3 < .35]
rows = [evaluate(lo, "snap<35%, all")] + [evaluate(lo[lo.season == s], f"snap<35%, {s}") for s in sorted(lo.season.unique())]
rows += [evaluate(lo[lo.market == m], f"snap<35%, {m}") for m in MARKETS]
rows += [evaluate(lo, "snap<35%, exchange price", "pnl_exch"), evaluate(lo, "snap<35%, best price", "pnl_best")]
R["low_snap"] = [r for r in rows if r]
print("\n== low snap share (<35% trailing 3), close\n" + pd.DataFrame(R["low_snap"]).to_string(index=False))
(ROOT / "results/low_liquidity.json").write_text(json.dumps(R, indent=1, default=str))
