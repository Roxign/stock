"""Momentum without survivorship bias: every TWSE common stock that existed at the time (delisted ones included,
data/rl/full from fulluniverse.py), with the universe chosen point-in-time.

On the first trading day of each month (decision at its close, filled at the next open by stocklab.portfolio):
  universe = the N stocks with the largest average traded value over their last 60 bars, among stocks with at least
             260 bars of history that traded that day (N = 50 / 100 / 200)
  momentum = hold the 20% of the universe with the highest return over their last 250 bars, equal weight
             (the rule fixed in alloc.py / momentum.py: top 20 of about 100)
  benchmark = equal weight of the whole universe, same days, same fees; and 0050
A second run uses only stocks still listed today, to measure how much survivorship flatters the result.

    python -m strategies.rl.momentum_full
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from stocklab import portfolio as pf
from stocklab.trials import log_trial

from .common import AGENT, save
from .fulluniverse import DIR, load_full, stock_list

OUT = __import__("pathlib").Path(__file__).parent / "results" / "momentum_full.json"
PERIODS = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "is", "oos", "full"]


def build(codes):
    data = {}
    for c in codes:
        if (DIR / f"{c}.csv").exists():
            df = load_full(c)
            if len(df) > 60:
                data[c] = df
    return data


def signals(data):
    """Per stock (own bars): 60-bar average traded value, 250-bar return, bars of history; on the union calendar."""
    liq, mom, age = {}, {}, {}
    for c, df in data.items():
        liq[c] = df["value"].rolling(60, min_periods=40).mean()
        mom[c] = df["close"] / df["close"].shift(250) - 1
        age[c] = pd.Series(np.arange(1, len(df) + 1), index=df.index)
    cal = pd.DatetimeIndex(sorted(set().union(*[df.index for df in data.values()])))
    f = lambda d: pd.DataFrame(d).reindex(cal)
    return cal, f(liq), f(mom), f(age)


def weights(cal, liq, mom, age, n, frac=0.2, ew=False):
    reb = cal[~cal.to_period("M").duplicated()]
    reb = reb[reb >= pd.Timestamp("2009-06-01")]
    W = pd.DataFrame(np.nan, index=cal, columns=liq.columns)
    sizes = []
    for d in reb:
        ok = (age.loc[d] >= 260) & liq.loc[d].notna() & mom.loc[d].notna()
        uni = liq.loc[d][ok].nlargest(n).index
        if len(uni) < 10:
            continue
        pick = uni if ew else mom.loc[d, uni].nlargest(max(1, int(round(len(uni) * frac)))).index
        row = pd.Series(0.0, index=liq.columns)
        row[pick] = 1.0 / len(pick)
        W.loc[d] = row
        sizes.append(len(uni))
    W = W.ffill().fillna(0.0)
    return {c: W[c] for c in W.columns}


def excess(a, b):
    ra, rb = a.pct_change().dropna(), b.pct_change().dropna()
    ra, rb = ra.align(rb, join="inner")
    d = ra - rb
    yrs = len(d) / 252
    X = np.column_stack([np.ones(len(rb)), rb.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, ra.to_numpy(), rcond=None)
    res = ra.to_numpy() - X @ coef
    se = math.sqrt(res.var() / len(res) * np.linalg.inv(X.T @ X / len(res))[0, 0])
    return {"excess": float(d.mean() * 252), "t_excess": float(d.mean() / d.std() * math.sqrt(252) * math.sqrt(yrs)),
            "beta": float(coef[1]), "alpha": float(coef[0] * 252), "t_alpha": float(coef[0] / se)}


def main():
    current, delisted, _ = stock_list()
    have = {p.stem for p in DIR.glob("*.csv")}
    every = build(sorted(have))
    alive = {c: df for c, df in every.items() if c in set(current) and df.index[-1] >= pd.Timestamp("2026-09-01")}
    print(f"資料：全部 {len(every)} 檔，其中今天仍上市 {len(alive)} 檔、已下市 {len(every) - len(alive)} 檔", flush=True)
    etf = pf.load_etf()
    res = {"0050": {}}
    for p in PERIODS:
        r = pf.simulate({"0050": etf}, {"0050": pd.Series(1.0, index=etf.index)}, p)
        if r:
            res["0050"][p] = {k: float(r["metrics"][k]) for k in ("cagr", "sharpe", "mdd")}
    print("0050 買進持有  " + "  ".join(f"{p} {v['cagr']:+.0%}/{v['sharpe']:.2f}" for p, v in res["0050"].items()), flush=True)
    for label, data in (("含下市股（無倖存者偏差）", every), ("只用今天仍上市的股票", alive)):
        cal, liq, mom, age = signals(data)
        for n in (50, 100, 200):
            key = f"{label}_top{n}"
            res[key] = {}
            eq = {}
            for rname, kw in (("等權", {"ew": True}), ("動能前 20%", {})):
                W = weights(cal, liq, mom, age, n, **kw)
                res[key][rname] = {}
                for p in PERIODS:
                    r = pf.simulate(data, W, p)
                    if r is None:
                        continue
                    m = r["metrics"]
                    res[key][rname][p] = {k: float(m[k]) for k in ("cagr", "sharpe", "mdd", "turnover")}
                    if p == "full":
                        eq[rname] = r["equity"]
                    log_trial(AGENT, f"momfull_{label}_top{n}_{rname}", {"family": "momentum_full", "survivor_free": label.startswith("含"), "n": n},
                              res[key][rname][p], period=p)
                row = res[key][rname]
                print(f"{label} 前 {n} 大 {rname:<6} " + "  ".join(f"{p} {row[p]['cagr']:+.0%}/{row[p]['sharpe']:.2f}" for p in PERIODS if p in row), flush=True)
            x = excess(eq["動能前 20%"], eq["等權"])
            res[key]["excess_full"] = x
            wins = sum(res[key]["動能前 20%"][f]["cagr"] > res[key]["等權"][f]["cagr"] for f in PERIODS[:8] if f in res[key]["等權"])
            wsh = sum(res[key]["動能前 20%"][f]["sharpe"] > res[key]["等權"][f]["sharpe"] for f in PERIODS[:8] if f in res[key]["等權"])
            res[key]["fold_wins"] = {"cagr": wins, "sharpe": wsh}
            print(f"   └ 2010–今 動能減等權：年化 {x['excess']:+.1%}（t = {x['t_excess']:.2f}），beta {x['beta']:.2f}，扣 beta 後 alpha {x['alpha']:+.1%}（t = {x['t_alpha']:.2f}）；"
                  f"8 段中年化勝 {wins}、Sharpe 勝 {wsh}", flush=True)
            save(OUT, res)


if __name__ == "__main__":
    main()
