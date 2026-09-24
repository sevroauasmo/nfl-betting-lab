"""Price the 'surprise inactive -> teammates' receiving unders' rule on Kalshi's own order book (2025-26).

For each qualifying teammate prop (receptions / receiving yards, sportsbook consensus line L), find the Kalshi ladder
contract "Player: S+" with S = floor(L)+1 (exact equivalent of the under) or, for yards, the nearest strike. Price = the
hourly candle closing just before T-80 (10 min after inactives): taking NO at the ask = 1 - yes_bid (+ taker fee), or a
resting NO bid at 1 - yes_ask (no taker fee; fill not guaranteed). Graded with Kalshi's own settlement.
Control: same pricing for receiving props on no-news team-games. Writes results/kalshi_targets_freed.json.
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from kalshi_fetch import get, kickoffs  # noqa: E402

exec(open(ROOT / "scripts/news_window.py").read().split("# ------------------------------------------------------------------ 1. how fast")[0])
fee = lambda p: np.ceil(0.07 * p * (1 - p) * 100) / 100  # noqa: E731
W["season"] = W.game_id.str[:4].astype(int)
W = W[(W.season >= 2025) & (W.group == "receiving")].copy()
news = W[W.news == "targets freed"].copy()
rng = np.random.default_rng(3)
ctrl_games = W[W.news == "none"].groupby(["game_id", "team"]).ngroups
ctrl_keys = W[W.news == "none"][["game_id", "team"]].drop_duplicates().sample(n=min(120, ctrl_games), random_state=3)
ctrl = W[W.news == "none"].merge(ctrl_keys, on=["game_id", "team"])

# Kalshi markets for these games
mk = pd.read_parquet(ROOT / "data/kalshi/markets.parquet")
mk = mk[mk.series.isin(["KXNFLREC", "KXNFLRECYDS"])].copy()
look = kickoffs()
g_code = {}
for code, (gid, kts) in look.items():
    g_code.setdefault(gid, []).append((code, kts))
mk["code"] = mk.event_ticker.str.split("-").str[1]
code2game = {c: (gid, k) for gid, lst in g_code.items() for c, k in lst}
mk["game_id"] = mk.code.map(lambda c: code2game.get(c, (None, None))[0])
mk["kick"] = mk.code.map(lambda c: code2game.get(c, (None, None))[1])
mk["pname"] = mk.yes_sub_title.str.split(":").str[0].map(lambda s: re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", re.sub(r"[^a-z ]", "", str(s).lower())).strip())
mk["strike"] = mk.yes_sub_title.str.extract(r":\s*([\d.]+)\+")[0].astype(float)
mk["market"] = mk.series.map({"KXNFLREC": "player_receptions", "KXNFLRECYDS": "player_reception_yds"})
mk = mk.dropna(subset=["game_id", "strike"])


def match(df):
    m = df.merge(mk, left_on=["game_id", "market", "player_key"], right_on=["game_id", "market", "pname"], suffixes=("", "_k"))
    m["want"] = np.floor(m.line_t10) + 1
    m["gap"] = (m.strike - m.want).abs()
    m = m.sort_values("gap").drop_duplicates(["game_id", "market", "player_key"])
    return m[m.gap <= np.where(m.market == "player_receptions", 0, 5)]      # receptions exact; yards within 5


def price(m):
    rows = []
    for r in m.itertuples():
        path = "/historical/markets/{}/candlesticks" if r.src == "hist" else f"/series/{r.series}/markets/{{}}/candlesticks"
        t_entry = int(r.kick) - 80 * 60
        d = get(path.format(r.ticker), start_ts=t_entry - 10 * 60, end_ts=t_entry, period_interval=1) or {}   # T-90..T-80, 1-min candles
        cs = [c for c in d.get("candlesticks", []) if c["end_period_ts"] <= t_entry]
        if not cs:
            continue
        c = cs[-1]
        v = lambda side: float((c.get(side) or {}).get("close_dollars", (c.get(side) or {}).get("close")) or "nan")  # noqa: E731
        yb, ya = v("yes_bid"), v("yes_ask")
        rows.append({"ticker": r.ticker, "game_id": r.game_id, "team": r.team, "player": r.player_key, "market": r.market,
                     "line": r.line_t10, "strike": r.strike, "gap": r.gap, "yes_bid": yb, "yes_ask": ya, "result": r.result,
                     "age_min": round((t_entry - c["end_period_ts"]) / 60)})
    return pd.DataFrame(rows)


def evaluate(p, label):
    p = p[(p.yes_bid > 0) & (p.yes_ask < 1) & (p.yes_ask > p.yes_bid)].copy()
    y = (p.result == "yes").astype(float)
    no_ask = 1 - p.yes_bid; no_bid = 1 - p.yes_ask
    taker = np.where(y == 0, (1 - no_ask - fee(no_ask)) / (no_ask + fee(no_ask)), -1.0)
    mid = 1 - (p.yes_bid + p.yes_ask) / 2
    maker = np.where(y == 0, (1 - mid) / mid, -1.0)                     # filled at the mid (between resting and taking)
    def cl(v):
        s = pd.Series(v, index=p.index).groupby([p.game_id, p.team]).agg(["sum", "count"])
        m_ = s["sum"].sum() / s["count"].sum(); r_ = s["sum"] - m_ * s["count"]; n = len(s)
        return round(m_, 4), round(math.sqrt(n / (n - 1) * (r_ ** 2).sum()) / s["count"].sum(), 4) if n > 1 else None, n
    rt, st, n = cl(taker); rm, sm, _ = cl(maker)
    return {"set": label, "contracts": len(p), "team_games": n, "no_win_rate": round(1 - y.mean(), 4),
            "avg_no_ask": round(no_ask.mean(), 3), "avg_spread_c": round(100 * (p.yes_ask - p.yes_bid).mean(), 1),
            "roi_take_no_at_ask": rt, "se_take": st, "roi_no_at_mid": rm, "se_mid": sm,
            "tight_spread_le_4c": {"contracts": int(((p.yes_ask - p.yes_bid) <= 0.04).sum()),
                                   "no_win_rate": round(float(1 - y[((p.yes_ask - p.yes_bid) <= 0.04).values].mean()), 4) if ((p.yes_ask - p.yes_bid) <= 0.04).any() else None,
                                   "roi_take": round(float(np.mean(taker[((p.yes_ask - p.yes_bid) <= 0.04).values])), 4) if ((p.yes_ask - p.yes_bid) <= 0.04).any() else None},
            "median_spread_c": round(100 * float((p.yes_ask - p.yes_bid).median()), 1),
            "by_market": {mk_: round(float(np.mean(taker[(p.market == mk_).values])), 4) for mk_ in p.market.unique()},
            "exact_strike_only_take_roi": round(float(np.mean(taker[(p.gap == 0).values])), 4) if (p.gap == 0).any() else None}


out = {}
for label, df in (("targets freed (news)", news), ("control: no news", ctrl)):
    m = match(df)
    p = price(m)
    out[label] = {"sportsbook_props": len(df), "matched_kalshi_contracts": len(m), "priced": len(p), **evaluate(p, label)}
    p.to_csv(ROOT / f"results/kalshi_tf_{'control' if 'control' in label else 'news'}_1min.csv", index=False)
(ROOT / "results/kalshi_targets_freed_1min.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
