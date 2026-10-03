"""Portfolio-level backtest: one account that can hold any of the 50 stocks plus cash, benchmarked on the 0050 ETF.

Per-stock strategies map to equal capital slots (weight = exposure / N). Cross-sectional strategies can instead
provide target weights of total equity directly (sum <= 1). Same timing as the single-stock engine: weights decided at
bar t's close are filled at bar t+1's open, and a stock is only traded when its target weight changes.
"""

import numpy as np
import pandas as pd

from . import backtest as bt
from .data import RAW_DIR, download, load

ETF = "0050"
ETF_SELL_FEE = 0.001425 + 0.001  # ETF transaction tax is 0.1%


def load_etf():
    if not (RAW_DIR / f"{ETF}.csv").exists():
        download([ETF])
    return load(ETF)


def per_stock_weights(positions, data, n=None):
    """Equal capital slots: each stock gets 1/n of equity times its own exposure."""
    n = n or len(data)
    return {c: bt.clean_target(p, data[c].index) / n for c, p in positions.items()}


def simulate(data, weights, period, sell_fee=bt.SELL_FEE):
    codes = sorted(weights)
    opens = pd.DataFrame({c: data[c]["open"] for c in codes}).sort_index()
    closes = pd.DataFrame({c: data[c]["close"] for c in codes}).reindex(opens.index)
    idx = opens.index
    b = bt.period_bounds(idx, period)
    if b is None:
        return None
    i0, i1 = b
    W = pd.DataFrame({c: weights[c].reindex(data[c].index).astype(float).fillna(0.0) for c in codes}).reindex(idx)
    target = W.ffill().fillna(0.0).clip(lower=0.0).shift(1).fillna(0.0).to_numpy()
    total = target.sum(axis=1, keepdims=True)
    target = np.where(total > 1 + 1e-9, target / np.where(total > 0, total, 1), target)
    O = opens.to_numpy()
    C = closes.ffill().to_numpy()

    cash, n = float(bt.CAPITAL), len(codes)
    sh, cur = np.zeros(n), np.zeros(n)
    eq = np.empty(i1 - i0 + 1)
    invested = np.empty_like(eq)
    traded = 0.0
    for k, i in enumerate(range(i0, i1 + 1)):
        t, o = target[i], O[i]
        chg = (np.abs(t - cur) > 1e-9) & ~np.isnan(o)
        if chg.any():
            mark = np.where(np.isnan(o), C[i - 1], o)
            equity = cash + np.nansum(sh * mark)
            for j in np.flatnonzero(chg & (t < cur)):
                d = sh[j] if t[j] == 0 else sh[j] - t[j] * equity / o[j]
                cash += d * o[j] * (1 - sell_fee)
                sh[j] -= d
                traded += d * o[j]
            for j in np.flatnonzero(chg & (t > cur)):
                d = min(t[j] * equity / o[j] - sh[j], cash / (o[j] * (1 + bt.BUY_FEE)))
                if d > 0:
                    cash -= d * o[j] * (1 + bt.BUY_FEE)
                    sh[j] += d
                    traded += d * o[j]
            cur[chg] = t[chg]
        held = np.nansum(sh * C[i])
        eq[k] = cash + held
        invested[k] = held / eq[k]
    equity = pd.Series(eq, index=idx[i0 : i1 + 1])
    m = bt.metrics(equity, invested)
    m["turnover"] = traded / float(np.mean(eq)) / m["years"] / 2
    return {"equity": equity, "metrics": m}


def run(data, weights, period):
    return simulate(data, weights, period)


def benchmarks(data, period):
    """0050 buy-and-hold, 0050 定期定額, and an equal-weight buy-and-hold of the 50 stocks."""
    etf = load_etf()
    out = {
        "etf_buy_hold": simulate({ETF: etf}, {ETF: pd.Series(1.0, index=etf.index)}, period, sell_fee=ETF_SELL_FEE),
        "etf_dca": bt.run_dca(etf, period),
        "ew_buy_hold": simulate(data, {c: pd.Series(1.0 / len(data), index=df.index) for c, df in data.items()}, period),
    }
    return out


BENCH_LABELS = {"etf_buy_hold": "0050 買進持有", "etf_dca": "0050 定期定額", "ew_buy_hold": "50 檔等權買進持有"}
