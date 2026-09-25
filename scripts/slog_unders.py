"""Slog unders: pick games likely to be defensive slogs (pre-game info only), then blanket-under the main yardage
prop of every offensive player in those games at equal stakes.

Thesis: retail loves overs, so books shade yardage lines up; the shade should hurt most in games that go
under the yardage everyone expects (low totals, good defenses, bad offenses, short rest, bad weather).

One bet per player per game on his primary yardage market (QB pass yds, RB rush yds, WR/TE rec yds), main line only.
Line = modal closing point across regulated books. Prices: flat -110, median regulated-book under price at that point,
best regulated-book under price, and the exchanges (Novig/ProphetX), which are what the user can actually bet.
All bets in a game are correlated, so standard errors are bootstrapped by game.
Writes results/slog_unders" + ("" if PASS == "close" else "_" + PASS) + ".json.
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
con = duckdb.connect(str(ROOT / "data/nfl.duckdb"), read_only=True)
EXCHANGES = ("novig", "prophetx")
PRIMARY = {"QB": "player_pass_yds", "RB": "player_rush_yds", "FB": "player_rush_yds",
           "WR": "player_reception_yds", "TE": "player_reception_yds"}
rng = np.random.default_rng(7)
import sys
PASS = sys.argv[1] if len(sys.argv) > 1 else "close"
R: dict = {}


def payout(o):
    o = np.asarray(o, float)
    return np.where(o < 0, 100 / -o, o / 100)


# ------------------------------------------------------------------ bets: one per player-game on his primary market
raw = con.execute(f"""
    SELECT game_id, left(game_id, 4)::INT AS season, bookmaker, market, player_key, player, position, team,
           point, under_price, over_price, actual
    FROM prop_results
    WHERE pass = '{PASS}' AND market IN ('player_pass_yds', 'player_rush_yds', 'player_reception_yds')
      AND point IS NOT NULL AND under_price IS NOT NULL AND over_price IS NOT NULL AND actual IS NOT NULL
""").df()
raw = raw[raw.market == raw.position.map(PRIMARY)]
raw["exch"] = raw.bookmaker.isin(EXCHANGES)
raw["under_dec"] = 1 + payout(raw.under_price)  # decimal odds: medians of American odds break across +/-100
books = raw[~raw.exch]
mode_pt = (books.groupby(["game_id", "market", "player_key", "point"]).size().rename("n").reset_index()
           .sort_values("n", ascending=False).drop_duplicates(["game_id", "market", "player_key"]))
key = ["game_id", "market", "player_key", "point"]
at = books.merge(mode_pt[key], on=key)
bets = at.groupby(key).agg(season=("season", "first"), player=("player", "first"), position=("position", "first"),
                           team=("team", "first"), actual=("actual", "first"), n_books=("bookmaker", "size"),
                           under_med=("under_dec", "median"), under_best=("under_dec", "max")).reset_index()
ex = raw[raw.exch].merge(mode_pt[key], on=key).groupby(key).under_dec.max().rename("under_exch").reset_index()
bets = bets.merge(ex, on=key, how="left")
bets = bets[bets.n_books >= 2]
bets["win"] = np.where(bets.actual < bets.point, 1.0, np.where(bets.actual > bets.point, 0.0, np.nan))
bets = bets.dropna(subset=["win"])
# role size: rank players within a game by line so "top N" = the N biggest roles
bets["line_rank"] = bets.groupby(["game_id", "market"]).point.rank(ascending=False, method="first")
for col, name in [("under_med", "med"), ("under_best", "best"), ("under_exch", "exch")]:
    bets[f"pnl_{name}"] = np.where(bets[col].isna(), np.nan, np.where(bets.win == 1, bets[col] - 1, -1.0))
bets["pnl_110"] = np.where(bets.win == 1, 100 / 110, -1.0)

# ------------------------------------------------------------------ pre-game game features
g = con.execute("""
    SELECT game_id, season, week, weekday, gametime, home_team, away_team, total_line, spread_line, total, ou_result,
           roof, temp_f, home_rest, away_rest, div_game
    FROM games WHERE season >= 2021 AND game_type IS NOT NULL
""").df()
wx = con.execute("SELECT game_id, wind_d1, precip_d1 FROM game_wind_forecast").df()
g = g.merge(wx, on="game_id", how="left")

# rolling team offense / defense EPA per play, carried across seasons (EWMA, halflife 8 games), shifted (pre-game)
e = con.execute("""
    SELECT t.game_id, t.season, t.team, t.off_epa, o.off_epa AS def_epa_allowed, g.gameday
    FROM team_game_epa t JOIN team_game_epa o ON o.game_id = t.game_id AND o.team <> t.team
    JOIN games g ON g.game_id = t.game_id WHERE t.season >= 2019
""").df().sort_values(["team", "gameday"])
for c in ["off_epa", "def_epa_allowed"]:
    e[f"{c}_pre"] = e.groupby("team")[c].transform(lambda s: s.shift(1).ewm(halflife=8, min_periods=4).mean())
e = e[["game_id", "team", "off_epa_pre", "def_epa_allowed_pre"]]
for side in ["home", "away"]:
    g = g.merge(e.rename(columns={"team": f"{side}_team", "off_epa_pre": f"{side}_off", "def_epa_allowed_pre": f"{side}_def"}),
                on=["game_id", f"{side}_team"], how="left")

outdoor = g.roof.isin(["outdoors", "open"])
g["tnf"] = g.weekday.eq("Thursday")
g["short_rest"] = g[["home_rest", "away_rest"]].min(axis=1) <= 4
g["bad_wx"] = outdoor & ((g.wind_d1 >= 12) | (g.precip_d1 >= 0.05) | (g.temp_f <= 32))
g["def_sum"] = g.home_def + g.away_def            # EPA allowed; lower = better defenses
g["off_sum"] = g.home_off + g.away_off
g["abs_spread"] = g.spread_line.abs()

# composite slog score, z-scored within season so it is not tuned to any one year's scoring environment
def z(s):
    return (s - s.mean()) / s.std()


gs = g[g.season >= 2023].copy()
for c in ["total_line", "def_sum", "off_sum"]:
    gs[f"z_{c}"] = gs.groupby("season")[c].transform(z)
gs["slog"] = (-gs.z_total_line - 0.5 * gs.z_def_sum - 0.5 * gs.z_off_sum
              + 0.75 * gs.tnf + 0.75 * gs.bad_wx.astype(float))
gs["slog_pct"] = gs.groupby("season").slog.rank(pct=True)

b = bets.merge(gs, on=["game_id", "season"], how="inner")
print(f"{len(b):,} bets in {b.game_id.nunique()} games; {b.groupby('game_id').size().median():.0f} per game (median)")
R["n_bets"], R["n_games"] = len(b), int(b.game_id.nunique())


# ------------------------------------------------------------------ evaluation with game-clustered bootstrap
def evaluate(d, label, price="pnl_med", boot=2000):
    d = d.dropna(subset=[price])
    if d.empty:
        return None
    gsum = d.groupby("game_id").agg(p=(price, "sum"), n=(price, "size"), w=("win", "sum"))
    idx = rng.integers(0, len(gsum), (boot, len(gsum)))
    rois = gsum.p.values[idx].sum(1) / gsum.n.values[idx].sum(1)
    out = {"label": label, "games": len(gsum), "bets": int(gsum.n.sum()), "under_pct": round(gsum.w.sum() / gsum.n.sum(), 4),
           "roi": round(gsum.p.sum() / gsum.n.sum(), 4), "ci90": [round(float(np.quantile(rois, q)), 4) for q in (.05, .95)],
           "p_roi_le_0": round(float((rois <= 0).mean()), 4),
           "game_win_pct": round(float((gsum.p > 0).mean()), 4), "units_per_game": round(float(gsum.p.mean()), 3)}
    return out


def show(rows, title):
    print(f"\n== {title}")
    df = pd.DataFrame([r for r in rows if r])
    print(df.to_string(index=False))
    return df.to_dict("records")


# 1) Baseline: every game, all primary props. Different prices.
R["baseline_prices"] = show([evaluate(b, f"all games, {p}", p) for p in ["pnl_110", "pnl_med", "pnl_best", "pnl_exch"]],
                            "blanket unders, every game")
R["baseline_by_season"] = show([evaluate(b[b.season == s], str(s)) for s in sorted(b.season.unique())], "by season (median price)")
R["baseline_by_pos"] = show([evaluate(b[b.position.isin(ps)], "/".join(ps)) for ps in [["QB"], ["RB", "FB"], ["WR"], ["TE"]]],
                            "by position (median price)")

# 2) Single slog ingredients
tl_q = b.groupby("game_id").total_line.first()
rules = {
    "total <= 41": b.total_line <= 41,
    "total <= 38.5": b.total_line <= 38.5,
    "total >= 48 (contrast)": b.total_line >= 48,
    "TNF": b.tnf,
    "short rest (<=4 days)": b.short_rest,
    "bad weather (outdoor, fcst wind>=12 | precip | <=32F)": b.bad_wx,
    "both D top-third (EPA allowed)": b.def_sum <= b.groupby("season").def_sum.transform(lambda s: s.quantile(1 / 3)),
    "both O bottom-third": b.off_sum <= b.groupby("season").off_sum.transform(lambda s: s.quantile(1 / 3)),
    "divisional": b.div_game,
    "spread <= 3 (close game)": b.abs_spread <= 3,
}
R["ingredients"] = show([evaluate(b[m], k) for k, m in rules.items()], "single slog ingredients (median price)")

# 3) Composite slog score buckets
b["slog_bucket"] = pd.cut(b.slog_pct, [0, .2, .4, .6, .8, .9, 1.0], labels=["0-20", "20-40", "40-60", "60-80", "80-90", "90-100"])
R["slog_buckets"] = show([evaluate(b[b.slog_bucket == q], f"slog pct {q}") for q in b.slog_bucket.cat.categories],
                         "composite slog score buckets (median price)")

# 4) The strategy as described: slog games (top 20% score), 10-13 players = top-N by line among primary props
strat = b.slog_pct > .8
rows = []
for n_players in [6, 10, 12, 99]:
    top = b.groupby("game_id").point.rank(ascending=False, method="first") <= n_players  # crude: across markets
    # better "biggest roles": 2 QBs, top-3 rushers, top-7 receivers per game
    rows.append(evaluate(b[strat & top], f"slog top20%, top {n_players} by line"))
role = ((b.market == "player_pass_yds") & (b.line_rank <= 2)) | ((b.market == "player_rush_yds") & (b.line_rank <= 3)) | \
       ((b.market == "player_reception_yds") & (b.line_rank <= 7))
rows.append(evaluate(b[strat & role], "slog top20%, 2 QB + 3 rush + 7 rec (12 bets)"))
for p in ["pnl_110", "pnl_best", "pnl_exch"]:
    rows.append(evaluate(b[strat & role], f"  ...same at {p}", p))
R["strategy"] = show(rows, "strategy: slog games, ~12 equal-unit unders")

# 5) Holdout: design years 2023-24, test 2025-26 (the score was not fit, but the bucket cut was chosen by eye)
R["holdout"] = show([evaluate(b[strat & role & (b.season <= 2024)], "2023-24 slog top20%, 12 bets"),
                     evaluate(b[strat & role & (b.season >= 2025)], "2025-26 slog top20%, 12 bets"),
                     evaluate(b[~strat & role & (b.season <= 2024)], "2023-24 other games, 12 bets"),
                     evaluate(b[~strat & role & (b.season >= 2025)], "2025-26 other games, 12 bets")],
                    "holdout")

# 6) Does the slog score predict anything beyond the total? logistic-style check: under% by score within total buckets
b["tl_bucket"] = pd.cut(b.total_line, [0, 40.5, 44.5, 48.5, 99], labels=["<=40.5", "41-44.5", "45-48.5", ">=49"])
rows = []
for tb in b.tl_bucket.cat.categories:
    d = b[b.tl_bucket == tb]
    hi = d.slog_pct > d.groupby("season").slog_pct.transform("median")
    rows += [evaluate(d[hi], f"total {tb}, slog above bucket median"), evaluate(d[~hi], f"total {tb}, slog below")]
R["within_total"] = show(rows, "slog score within total buckets (is it just the total?)")

# 7) Compare to simply betting the game total under in the same games (closing line, -110)
gg = gs[gs.slog_pct > .8].dropna(subset=["total"])
gg = gg[gg.total != gg.total_line]
R["game_total_under_in_slog_games"] = {"games": len(gg), "under_pct": round(float((gg.total < gg.total_line).mean()), 4),
                                       "roi_at_110": round(float(np.where(gg.total < gg.total_line, 100 / 110, -1).mean()), 4)}
print("\n== game-total under in slog games:", R["game_total_under_in_slog_games"])

# 8) Per-game P&L distribution for the strategy (variance a bettor would feel)
d = b[strat & role].groupby("game_id").agg(p=("pnl_med", "sum"), n=("pnl_med", "size"), season=("season", "first"),
                                           total=("total", "first"), line=("total_line", "first"))
R["strategy_game_pnl"] = {"mean": round(d.p.mean(), 3), "sd": round(d.p.std(), 3), "worst": round(d.p.min(), 2),
                          "best": round(d.p.max(), 2), "pct_games_up": round(float((d.p > 0).mean()), 3),
                          "corr_with_total_minus_line": round(float(np.corrcoef(d.p, d.total - d.line)[0, 1]), 3)}
print("\n== per-game P&L (units, 12 x 1u):", R["strategy_game_pnl"])
R["strategy_cum"] = (d.reset_index().merge(gs[["game_id", "week"]], on="game_id")
                     .sort_values(["season", "week", "game_id"])
                     .assign(cum=lambda x: x.p.cumsum().round(2))[["game_id", "season", "week", "p", "cum"]]
                     .round(3).to_dict("records"))

(ROOT / "results/slog_unders" + ("" if PASS == "close" else "_" + PASS) + ".json").write_text(json.dumps(R, indent=1, default=str))
