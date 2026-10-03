"""KDJ round: in-sample-only evaluation harness.

All KDJ-round research scripts load data TRUNCATED at 2020-12-31, so nothing after the in-sample period (not even
labels whose 60-bar window would run into 2021) can influence a research decision. Positions are causal, so the
in-sample backtest on truncated data equals the in-sample part of a full-data backtest.
Run from the repo root:  python -m strategies.kdj_macd_dl.research.<script>
"""

import math

import numpy as np
import pandas as pd

from stocklab import backtest as bt
from stocklab.data import load_all

IS_END = "2020-12-31"
_DATA = None
_BASE = None


def data():
    global _DATA
    if _DATA is None:
        _DATA = {c: df.loc[:IS_END] for c, df in load_all().items() if df.index[0] < pd.Timestamp(IS_END)}
    return _DATA


def base():
    global _BASE
    if _BASE is None:
        d = data()
        _BASE = {"bh": {c: bt.run_buy_hold(df, "is") for c, df in d.items()},
                 "dca": {c: bt.run_dca(df, "is") for c, df in d.items()}}
    return _BASE


def row(res, name):
    b = base()
    codes = [c for c, r in res.items() if r]
    m = {c: res[c]["metrics"] for c in codes}
    med = lambda k: float(np.nanmedian([m[c].get(k, math.nan) for c in codes]))
    return {
        "name": name, "n": len(codes),
        "cagr": med("cagr"), "mdd": med("mdd"), "sharpe": med("sharpe"), "expo": med("exposure"),
        "orders": med("orders"), "trades": med("trades"),
        "shBH": float(np.mean([m[c]["sharpe"] > b["bh"][c]["metrics"]["sharpe"] for c in codes])),
        "beatDCA": float(np.mean([m[c]["cagr"] > b["dca"][c]["metrics"]["cagr"] for c in codes])),
    }


def evaluate(pos, name):
    d = data()
    return row({c: bt.run_strategy(df, pos[c], "is") for c, df in d.items()}, name)


def baseline_rows():
    b = base()
    return [row(b["bh"], "buy_hold"), row(b["dca"], "dca")]


def show(rows, path=None):
    df = pd.DataFrame(rows)
    fmt = {k: "{:.1%}".format for k in ["cagr", "mdd", "expo", "shBH", "beatDCA"]}
    fmt["sharpe"] = "{:.2f}".format
    fmt["orders"] = "{:.0f}".format
    fmt["trades"] = "{:.0f}".format
    txt = df.to_string(formatters=fmt, index=False)
    print(txt)
    if path:
        df.to_csv(path, index=False)
    return df
