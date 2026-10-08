"""Portfolio-level backtest: one account holding any of the 50 stocks and the ETFs (long or 融券 short) plus cash.

Per-security strategies map to equal capital slots over their own universe (weight = exposure / N). Portfolio
strategies can instead provide target weights of total equity directly (gross sum |w| <= 1, the rest is cash).
Same timing as the single-security engine: weights decided at bar t's close are filled at bar t+1's open, and a
security is only traded when its target weight changes. Each security pays its own costs (stocklab.shortrules.fees).
"""

import numpy as np
import pandas as pd

from . import backtest as bt

ETF = "0050"


def load_etf():
    from .etf import load_etf as _load

    return _load(ETF)


def per_stock_weights(positions, data, n=None):
    """Equal capital slots: each security gets 1/n of equity times its own exposure (n = securities traded)."""
    n = n or len(positions)
    return {c: bt.clean_target(p, data[c].index) / n for c, p in positions.items()}


def simulate(data, weights, period):
    from .runner import fees_of

    codes = sorted(weights)
    opens = pd.DataFrame({c: data[c]["open"] for c in codes}).sort_index()
    closes = pd.DataFrame({c: data[c]["close"] for c in codes}).reindex(opens.index)
    idx = opens.index
    b = bt.period_bounds(idx, period)
    if b is None:
        return None
    i0, i1 = b
    W = pd.DataFrame({c: weights[c].reindex(data[c].index).astype(float).fillna(0.0) for c in codes}).reindex(idx)
    target = W.ffill().fillna(0.0).clip(-1.0, 1.0).shift(1).fillna(0.0).to_numpy()
    gross = np.abs(target).sum(axis=1, keepdims=True)
    target = np.where(gross > 1 + 1e-9, target / np.where(gross > 0, gross, 1), target)
    O = opens.to_numpy()
    C = closes.ffill().to_numpy()
    fees = [fees_of(c) for c in codes]
    buy = np.array([f["buy"] for f in fees])
    sell = np.array([f["sell"] for f in fees])
    short = np.array([f["short"] for f in fees])

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
            want = np.where(chg, t * equity / np.where(np.isnan(o), 1, o), sh)
            for j in np.flatnonzero(chg):  # reductions first: sell longs, cover shorts
                if sh[j] > 0 and want[j] < sh[j]:
                    d = sh[j] if want[j] <= 0 else sh[j] - want[j]
                    cash += d * o[j] * (1 - sell[j])
                    sh[j] -= d
                    traded += d * o[j]
                elif sh[j] < 0 and want[j] > sh[j]:
                    d = -sh[j] if want[j] >= 0 else want[j] - sh[j]
                    cash -= d * o[j] * (1 + buy[j])
                    sh[j] += d
                    traded += d * o[j]
            for j in np.flatnonzero(chg):  # then increases: buy longs (cash permitting), open shorts
                if want[j] > 0 and want[j] > sh[j]:
                    d = min(want[j] - sh[j], max(cash, 0.0) / (o[j] * (1 + buy[j])))
                    cash -= d * o[j] * (1 + buy[j])
                    sh[j] += d
                    traded += d * o[j]
                elif want[j] < 0 and want[j] < sh[j]:
                    d = sh[j] - want[j]
                    cash += d * o[j] * (1 - sell[j] - short[j])
                    sh[j] -= d
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
    from .runner import fees_of
    from .universe import CODES

    etf = load_etf()
    stocks = {c: data[c] for c in CODES if c in data}
    return {
        "etf_buy_hold": simulate({ETF: etf}, {ETF: pd.Series(1.0, index=etf.index)}, period),
        "etf_dca": bt.run_dca(etf, period, fees_of(ETF)),
        "ew_buy_hold": simulate(stocks, {c: pd.Series(1.0 / len(stocks), index=df.index) for c, df in stocks.items()}, period),
    }


BENCH_LABELS = {"etf_buy_hold": "0050 買進持有", "etf_dca": "0050 定期定額", "ew_buy_hold": "50 檔等權買進持有"}
