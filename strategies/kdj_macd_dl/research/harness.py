"""Scratch evaluation harness (mirrors stocklab.runner.leaderboard, without writing position caches)."""
import math
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd

from stocklab import backtest as bt
from stocklab.data import load_all

_DATA = None
_BASE = {}


def data():
    global _DATA
    if _DATA is None:
        _DATA = load_all()
    return _DATA


def base(period):
    if period not in _BASE:
        d = data()
        _BASE[period] = {
            "bh": {c: bt.run_buy_hold(df, period) for c, df in d.items()},
            "dca": {c: bt.run_dca(df, period) for c, df in d.items()},
        }
    return _BASE[period]


def row(res, period, name):
    b = base(period)
    codes = [c for c, r in res.items() if r]
    m = {c: res[c]["metrics"] for c in codes}
    med = lambda k: float(np.nanmedian([m[c].get(k, math.nan) for c in codes]))
    return {
        "name": name, "period": period, "n": len(codes),
        "cagr": med("cagr"), "cagr_mean": float(np.mean([m[c]["cagr"] for c in codes])),
        "mdd": med("mdd"), "sharpe": med("sharpe"), "expo": med("exposure"), "trades": med("trades"),
        "win": med("win"),
        "beatBH": float(np.mean([m[c]["cagr"] > b["bh"][c]["metrics"]["cagr"] for c in codes])),
        "shBH": float(np.mean([m[c]["sharpe"] > b["bh"][c]["metrics"]["sharpe"] for c in codes])),
        "beatDCA": float(np.mean([m[c]["cagr"] > b["dca"][c]["metrics"]["cagr"] for c in codes])),
    }


def evaluate(pos, name, periods=("is",)):
    d = data()
    rows = []
    for p in periods:
        res = {c: bt.run_strategy(df, pos[c], p) for c, df in d.items()}
        rows.append(row(res, p, name))
    return rows


def baseline_rows(periods=("is",)):
    rows = []
    for p in periods:
        b = base(p)
        rows.append(row(b["bh"], p, "buy_hold"))
        rows.append(row(b["dca"], p, "dca"))
    return rows


def show(rows):
    df = pd.DataFrame(rows)
    fmt = {k: "{:.1%}".format for k in ["cagr", "cagr_mean", "mdd", "expo", "win", "beatBH", "shBH", "beatDCA"]}
    fmt["sharpe"] = "{:.2f}".format
    fmt["trades"] = "{:.0f}".format
    print(df.to_string(formatters=fmt, index=False))
    return df
