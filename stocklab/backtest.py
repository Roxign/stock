"""Long-only daily backtester shared by every strategy.

A strategy outputs a target exposure in [0, 1] for each bar, decided with information up to that bar's close.
The engine fills the change at the NEXT bar's open, so a signal can never trade on the bar that produced it.
docs/engine.js mirrors this logic for the web viewer; keep the two in sync.
"""

import math

import numpy as np
import pandas as pd

CAPITAL = 1_000_000
BUY_FEE = 0.001425
SELL_FEE = 0.001425 + 0.003  # commission + securities transaction tax
WARMUP = 60
PERIODS = {
    "full": ("2010-01-01", None),
    "is": ("2010-01-01", "2020-12-31"),
    "oos": ("2021-01-01", None),
}
PERIOD_LABELS = {"full": "全期 2010–今", "is": "樣本內 2010–2020", "oos": "樣本外 2021–今"}


def period_bounds(index, period):
    """Return (i0, i1) positional bounds inclusive, or None if the stock has no tradable bars in the period."""
    start, end = PERIODS[period]
    if len(index) <= WARMUP:
        return None
    i0 = max(index.searchsorted(pd.Timestamp(start)), WARMUP)
    i1 = len(index) - 1 if end is None else index.searchsorted(pd.Timestamp(end), side="right") - 1
    return (i0, i1) if i1 - i0 >= 20 else None


def clean_target(target, index):
    return target.reindex(index).astype(float).fillna(0.0).clip(0.0, 1.0)


def simulate(df, target, i0, i1):
    target = clean_target(target, df.index).to_numpy()
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    n = i1 - i0 + 1
    cash, sh, cur = float(CAPITAL), 0.0, 0.0
    eq = np.empty(n)
    invested = np.zeros(n)
    trades, entry, orders = [], None, 0
    for k, i in enumerate(range(i0, i1 + 1)):
        t = target[i - 1]
        if abs(t - cur) > 1e-9:
            orders += 1
            px = o[i]
            want = t * (cash + sh * px) / px
            d = want - sh
            if d > 0:
                d = min(d, cash / (px * (1 + BUY_FEE)))
                cash -= d * px * (1 + BUY_FEE)
                sh += d
            elif d < 0:
                if t == 0:
                    d = -sh
                cash += -d * px * (1 - SELL_FEE)
                sh += d
            if cur == 0 and t > 0:
                entry = (i, px)
            elif cur > 0 and t == 0 and entry:
                trades.append([entry[0], i, entry[1], px, px * (1 - SELL_FEE) / (entry[1] * (1 + BUY_FEE)) - 1])
                entry = None
            cur = t
        eq[k] = cash + sh * c[i]
        invested[k] = sh * c[i] / eq[k]
    if entry:
        trades.append([entry[0], None, entry[1], c[i1], c[i1] * (1 - SELL_FEE) / (entry[1] * (1 + BUY_FEE)) - 1])
    return pd.Series(eq, index=df.index[i0 : i1 + 1]), invested, trades, orders


def simulate_dca(df, i0, i1):
    """定期定額: split CAPITAL evenly and buy at the open of the first trading day of each month."""
    idx = df.index[i0 : i1 + 1]
    first = ~idx.to_period("M").duplicated()
    amt = CAPITAL / first.sum()
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    cash, sh = float(CAPITAL), 0.0
    eq = np.empty(len(idx))
    flows = []
    for k, i in enumerate(range(i0, i1 + 1)):
        if first[k]:
            sh += amt / (o[i] * (1 + BUY_FEE))
            cash -= amt
            flows.append((idx[k], -amt))
        eq[k] = cash + sh * c[i]
    flows.append((idx[-1], eq[-1] - cash))
    return pd.Series(eq, index=idx), flows


def xirr(flows):
    t0 = flows[0][0]
    t = np.array([(d - t0).days / 365.25 for d, _ in flows])
    f = np.array([v for _, v in flows])
    npv = lambda r: float(np.sum(f / (1 + r) ** t))
    lo, hi = -0.99, 10.0
    if npv(lo) * npv(hi) > 0:
        return math.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def metrics(eq, invested=None, trades=None, orders=None):
    """invested: per-bar fraction of equity held in stock. trades: round trips from flat to flat. orders: position changes."""
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1 / 365.25)
    growth = eq.iloc[-1] / CAPITAL
    ret = eq.pct_change().fillna(eq.iloc[0] / CAPITAL - 1)
    peak = np.maximum.accumulate(np.r_[CAPITAL, eq.to_numpy()])[1:]
    sd = ret.std()
    m = {
        "total": growth - 1,
        "cagr": growth ** (1 / years) - 1,
        "mdd": float((eq.to_numpy() / peak - 1).min()),
        "sharpe": float(ret.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0,
        "vol": float(sd * math.sqrt(252)),
        "years": years,
    }
    if invested is not None:
        m["exposure"] = float(np.mean(invested))
    if orders is not None:
        m["orders"] = orders
    if trades is not None:
        rets = [t[4] for t in trades]
        m["trades"] = len(trades)
        m["win"] = float(np.mean([r > 0 for r in rets])) if rets else math.nan
        m["avg_trade"] = float(np.mean(rets)) if rets else math.nan
        closed = [t for t in trades if t[1] is not None]
        m["avg_hold"] = float(np.mean([t[1] - t[0] for t in closed])) if closed else math.nan
    return m


def run_strategy(df, target, period):
    b = period_bounds(df.index, period)
    if b is None:
        return None
    eq, invested, trades, orders = simulate(df, target, *b)
    return {"equity": eq, "trades": trades, "metrics": metrics(eq, invested, trades, orders)}


def run_buy_hold(df, period):
    return run_strategy(df, pd.Series(1.0, index=df.index), period)


def run_dca(df, period):
    b = period_bounds(df.index, period)
    if b is None:
        return None
    eq, flows = simulate_dca(df, *b)
    m = metrics(eq)
    m["xirr"] = xirr(flows)
    return {"equity": eq, "trades": [], "metrics": m}


def summarize(results_by_code, keys=("cagr", "mdd", "sharpe", "total")):
    """Median of each metric across stocks; results_by_code maps code -> run_* result (or None)."""
    rows = [r["metrics"] for r in results_by_code.values() if r]
    return {k: float(np.nanmedian([r[k] for r in rows])) for k in keys} | {"n": len(rows)}
