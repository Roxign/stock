"""Cross-sectional momentum, checked with the project's own portfolio engine (stocklab.portfolio: one account,
next-open fills, each stock's own fees). The allocator study found that selection (which stocks) carries more
usable information than timing (when): a fully invested portfolio does not fight the market's drift, and 105 stocks
give 105 small bets per rebalance instead of one.

Rule (fixed from the development run in alloc.py): at the close of each rebalance day hold the `top` stocks with the
highest return over the last `look` bars (optionally skipping the most recent `skip` bars), equal weight; benchmark =
equal weight of every stock eligible that day (same rebalance days, same fees).

    python -m strategies.rl.momentum
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stocklab import backtest as bt
from stocklab import portfolio as pf
from stocklab.data import load
from stocklab.trials import log_trial
from stocklab.universe import CODES, EXTRA_CODES

from .common import AGENT, save

OUT = __import__("pathlib").Path(__file__).parent / "results" / "momentum.json"
PERIODS = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "is", "oos"]


def weights(data, look=250, skip=0, top=20, every="M", ew=False):
    close = pd.DataFrame({c: df["close"] for c, df in data.items()}).sort_index()
    mom = pd.DataFrame({c: df["close"].shift(skip) / df["close"].shift(look) - 1 for c, df in data.items()}).reindex(close.index)
    idx = close.index
    if every == "M":
        reb = idx[~idx.to_period("M").duplicated()]
    else:
        reb = idx[::every]
    W = pd.DataFrame(np.nan, index=idx, columns=close.columns)
    for d in reb:
        m = mom.loc[d].dropna()
        m = m[close.loc[d, m.index].notna()]
        if m.empty:
            continue
        pick = m.index if ew else m.nlargest(min(top, len(m))).index
        row = pd.Series(0.0, index=close.columns)
        row[pick] = 1.0 / len(pick)
        W.loc[d] = row
    W = W.ffill().fillna(0.0)
    return {c: W[c] for c in close.columns}


def main():
    res = {}
    unis = {"105 檔": CODES + EXTRA_CODES, "55 檔代表股": EXTRA_CODES, "50 檔成分股": CODES}
    etf = pf.load_etf()
    for uname, codes in unis.items():
        data = {c: load(c) for c in codes}
        rules = {"等權": dict(ew=True), "動能 12 個月・前 20 檔": dict(look=250), "動能 12-1 個月・前 20 檔": dict(look=250, skip=20),
                 "動能 3 個月・前 20 檔": dict(look=60), "動能 12 個月・前 10 檔": dict(look=250, top=10)}
        res[uname] = {}
        for rname, kw in rules.items():
            W = weights(data, **kw)
            res[uname][rname] = {}
            for p in PERIODS:
                r = pf.simulate(data, W, p)
                if r is None:
                    continue
                m = r["metrics"]
                res[uname][rname][p] = {k: float(m[k]) for k in ("cagr", "sharpe", "mdd", "turnover")}
                log_trial(AGENT, f"momentum_{uname}_{rname}", {"family": "momentum", "universe": uname, **kw}, res[uname][rname][p], period=p)
            row = res[uname][rname]
            print(f"{uname} {rname:<18} " + "  ".join(f"{p} {row[p]['cagr']:+.0%}/{row[p]['sharpe']:.2f}" for p in PERIODS if p in row), flush=True)
    res["0050"] = {}
    for p in PERIODS:
        r = pf.simulate({"0050": etf}, {"0050": pd.Series(1.0, index=etf.index)}, p)
        if r:
            res["0050"][p] = {k: float(r["metrics"][k]) for k in ("cagr", "sharpe", "mdd")}
    print("0050 買進持有        " + "  ".join(f"{p} {v['cagr']:+.0%}/{v['sharpe']:.2f}" for p, v in res["0050"].items()), flush=True)
    save(OUT, res)


if __name__ == "__main__":
    main()
