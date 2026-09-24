"""P&L of the rule: when a pass-catcher with >=10% target share is a surprise inactive, bet UNDER on each remaining
teammate's receiving props. Graded at the close (T-10) unless noted. Flat 1-unit stakes."""
import json, math
from pathlib import Path
import duckdb, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
exec(open(ROOT / "scripts/news_window.py").read().split("# ------------------------------------------------------------------ 1. how fast")[0])
payout_ = lambda o: np.where(np.asarray(o, float) < 0, 100 / -np.asarray(o, float), np.asarray(o, float) / 100)  # noqa: E731
W["season"] = W.game_id.str[:4].astype(int)
T = W[W.news == "targets freed"].copy()
T["res"] = np.where(T.actual < T.line_t10, 1.0, np.where(T.actual > T.line_t10, 0.0, np.nan))   # under wins
T = T.dropna(subset=["res"])
# DraftKings / FanDuel single-book prices at the consensus line (what one account would get)
sb = con.execute("""SELECT game_id, market, player_key, bookmaker, point, under_price FROM prop_results
                    WHERE pass='close' AND bookmaker IN ('draftkings','fanduel') AND under_price IS NOT NULL""").df()
sb = sb.pivot_table(index=["game_id", "market", "player_key", "point"], columns="bookmaker", values="under_price", aggfunc="max").reset_index()
T = T.merge(sb.rename(columns={"point": "line_t10"}), on=["game_id", "market", "player_key", "line_t10"], how="left")


def book(d, price_col, label):
    d = d.dropna(subset=[price_col])
    if d.empty:
        return {"venue": label, "bets": 0, "win_rate": None, "avg_odds": None, "units": 0.0, "roi": None}
    pr = np.where(d.res == 1, payout_(d[price_col]), -1.0)
    return {"venue": label, "bets": len(d), "win_rate": round(d.res.mean(), 3), "avg_odds": int(round(d[price_col].median())),
            "units": round(pr.sum(), 1), "roi": round(pr.mean(), 4)}


rows = []
for season, d in list(T.groupby("season")) + [("ALL", T)]:
    tg_ = d.groupby(["game_id", "team"]).ngroups
    for col, lab in (("best_under_t10", "best of ~8 books"), ("draftkings", "DraftKings only"), ("fanduel", "FanDuel only"), ("ex_under_t10", "Novig/ProphetX")):
        r = book(d, col, lab); r.update(season=season, team_games=tg_, fair_under=round(1 - d.fair_t10.mean(), 3)); rows.append(r)
    # flat -110 as a neutral benchmark
    pr = np.where(d.res == 1, 100 / 110, -1.0)
    rows.append({"season": season, "venue": "flat -110", "bets": len(d), "win_rate": round(d.res.mean(), 3), "avg_odds": -110,
                 "units": round(pr.sum(), 1), "roi": round(pr.mean(), 4), "team_games": tg_, "fair_under": round(1 - d.fair_t10.mean(), 3)})
tab = pd.DataFrame(rows)[["season", "venue", "team_games", "bets", "win_rate", "fair_under", "avg_odds", "units", "roi"]]

# one bet per player per game (receiving yards only, else receptions) -> no double counting
one = T.sort_values("market", ascending=False).drop_duplicates(["game_id", "player_key"])
pr1 = np.where(one.res == 1, payout_(one.best_under_t10), -1.0)
by_market = T.groupby("market").apply(lambda d: pd.Series({"bets": len(d), "win_rate": d.res.mean(),
                                                          "roi_best": np.mean(np.where(d.res == 1, payout_(d.best_under_t10), -1.0))})).round(3)
# bankroll path at best price, chronological, 1 unit per bet
g = con.execute("SELECT game_id, gameday FROM games").df()
path = T.merge(g, on="game_id").sort_values("gameday")
path["pnl"] = np.where(path.res == 1, payout_(path.best_under_t10), -1.0)
cum = path.pnl.cumsum(); dd = (cum - cum.cummax()).min()
weekly = path.groupby(["season", path.game_id.str[5:7]]).pnl.sum()
# per team-game results (the real unit of independence)
tgp = path.groupby(["game_id", "team"]).pnl.sum()
out = {"table": tab.to_dict("records"),
       "one_bet_per_player": {"bets": len(one), "win_rate": round(one.res.mean(), 3), "roi_best": round(pr1.mean(), 4), "units": round(pr1.sum(), 1)},
       "by_market": by_market.reset_index().to_dict("records"),
       "bankroll": {"bets": len(path), "units_total": round(cum.iloc[-1], 1), "max_drawdown_units": round(dd, 1),
                    "weeks_with_bets": int(len(weekly)), "winning_weeks": int((weekly > 0).sum()), "avg_bets_per_week": round(len(path) / len(weekly), 1),
                    "team_games": int(len(tgp)), "team_games_profitable": int((tgp > 0).sum()), "worst_team_game_units": round(tgp.min(), 1),
                    "best_team_game_units": round(tgp.max(), 1)},
       "cumulative_units_by_season_end": {int(s): round(cum[path.season == s].iloc[-1], 1) for s in sorted(path.season.unique())}}
(ROOT / "results/targets_freed_pnl.json").write_text(json.dumps(out, indent=1, default=str))
pd.set_option("display.width", 200)
print(tab.to_string(index=False)); print(json.dumps({k: v for k, v in out.items() if k != "table"}, indent=1, default=str))
