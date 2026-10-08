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

# Blocked k-fold cross-validation: 8 contiguous ~2-year folds, each a different market regime.
FOLDS = {
    "f1": ("2010-01-01", "2011-12-31"),
    "f2": ("2012-01-01", "2013-12-31"),
    "f3": ("2014-01-01", "2015-12-31"),
    "f4": ("2016-01-01", "2017-12-31"),
    "f5": ("2018-01-01", "2019-12-31"),
    "f6": ("2020-01-01", "2021-12-31"),
    "f7": ("2022-01-01", "2023-12-31"),
    "f8": ("2024-01-01", None),
}
FOLD_LABELS = {"f1": "2010–11", "f2": "2012–13", "f3": "2014–15", "f4": "2016–17", "f5": "2018–19", "f6": "2020–21",
               "f7": "2022–23", "f8": "2024–今"}
ALL_PERIODS = PERIODS | FOLDS
PERIOD_LABELS |= {f: f"第{i + 1}折 {FOLD_LABELS[f]}" for i, f in enumerate(FOLDS)}


def period_bounds(index, period):
    """Return (i0, i1) positional bounds inclusive, or None if the stock has no tradable bars in the period."""
    start, end = ALL_PERIODS[period]
    if len(index) <= WARMUP:
        return None
    i0 = max(index.searchsorted(pd.Timestamp(start)), WARMUP)
    i1 = len(index) - 1 if end is None else index.searchsorted(pd.Timestamp(end), side="right") - 1
    return (i0, i1) if i1 - i0 >= 20 else None


def clean_target(target, index):
    """Exposure in [-1, 1]: positive = long, negative = short (融券)."""
    return target.reindex(index).astype(float).fillna(0.0).clip(-1.0, 1.0)


def default_fees():
    return {"buy": BUY_FEE, "sell": SELL_FEE, "short": 0.0}


def _trade_ret(side, entry_px, exit_px, f):
    if side > 0:
        return exit_px * (1 - f["sell"]) / (entry_px * (1 + f["buy"])) - 1
    return entry_px * (1 - f["sell"] - f["short"]) / (exit_px * (1 + f["buy"])) - 1


def simulate(df, target, i0, i1, fees=None):
    """fees: {"buy", "sell" (incl. transaction tax), "short" (extra fee on short sales)}; defaults to stock costs."""
    f = fees or default_fees()
    buy, sell, short = f["buy"], f["sell"], f.get("short", 0.0)
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
            if t >= 0 and cur >= 0:
                want = t * (cash + sh * px) / px
                d = want - sh
                if d > 0:
                    d = min(d, cash / (px * (1 + buy)))
                    cash -= d * px * (1 + buy)
                    sh += d
                elif d < 0:
                    if t == 0:
                        d = -sh
                    cash += -d * px * (1 - sell)
                    sh += d
            elif t <= 0 and cur <= 0:
                want = t * (cash + sh * px) / px
                d = want - sh
                if d < 0:
                    cash += -d * px * (1 - sell - short)
                    sh += d
                elif d > 0:
                    if t == 0:
                        d = -sh
                    cash -= d * px * (1 + buy)
                    sh += d
            else:  # crossing zero: close the old side, then open the new side with what is left
                cash += sh * px * (1 - sell) if sh > 0 else sh * px * (1 + buy)
                want = t * cash / px
                if t > 0:
                    want = min(want, cash / (px * (1 + buy)))
                    cash -= want * px * (1 + buy)
                else:
                    cash += -want * px * (1 - sell - short)
                sh = want
            if cur != 0 and (t == 0 or (t > 0) != (cur > 0)) and entry:
                trades.append([entry[0], i, entry[1], px, _trade_ret(entry[2], entry[1], px, f)])
                entry = None
            if t != 0 and (cur == 0 or (t > 0) != (cur > 0)):
                entry = (i, px, 1 if t > 0 else -1)
            cur = t
        eq[k] = cash + sh * c[i]
        invested[k] = sh * c[i] / eq[k]
    if entry:
        trades.append([entry[0], None, entry[1], c[i1], _trade_ret(entry[2], entry[1], c[i1], f)])
    return pd.Series(eq, index=df.index[i0 : i1 + 1]), invested, trades, orders


def enforce_short_rules(code, df, target):
    """The exposure a trader could actually hold under Taiwan's 融券 rules (stocklab/shortrules.py).

    Long exposure passes through. A short is opened or enlarged only if the security is shortable that day and,
    when 平盤以下 is restricted, the open is not below the previous close; open shorts are covered on forced-cover
    days and after a margin call (擔保維持率 below 130% at the close -> cover at the next open).
    Value at bar t = exposure for the fill at bar t+1's open, the same convention as strategy targets.
    """
    t = clean_target(target, df.index)
    if (t >= 0).all():
        return t
    from . import shortrules as sr

    cal = sr.short_calendar(code, df)
    can, must, flat = (cal[k].to_numpy() for k in ("can_open_short", "must_cover", "flat_restricted"))
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    mr = sr.margin_rate(df.index).to_numpy()
    f = sr.fees(code)
    raw = t.to_numpy()
    eff = raw.copy()
    cur, entry, call = 0.0, None, False
    for i in range(1, len(raw)):
        want = raw[i - 1]
        if want < 0:
            if must[i] or call:
                want = 0.0
            elif want < cur and (not can[i] or (flat[i] and o[i] < c[i - 1])):
                want = cur if cur < 0 else 0.0
        if want < 0:
            if cur >= 0:
                entry = o[i]
            elif want < cur:
                entry = (entry * -cur + o[i] * (cur - want)) / -want
        else:
            entry = None
        call = want < 0 and (mr[i] + 1 - f["sell"] - f["short"]) * entry / c[i] < sr.MAINTENANCE
        eff[i - 1] = want
        cur = want
    return pd.Series(eff, index=df.index)


def effective_positions(positions, data):
    return {c: enforce_short_rules(c, data[c], p) for c, p in positions.items()}


def simulate_dca(df, i0, i1, buy_fee=None):
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
            sh += amt / (o[i] * (1 + (BUY_FEE if buy_fee is None else buy_fee)))
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


def run_strategy(df, target, period, fees=None):
    b = period_bounds(df.index, period)
    if b is None:
        return None
    eq, invested, trades, orders = simulate(df, target, *b, fees=fees)
    return {"equity": eq, "trades": trades, "metrics": metrics(eq, invested, trades, orders)}


def run_buy_hold(df, period, fees=None):
    return run_strategy(df, pd.Series(1.0, index=df.index), period, fees)


def run_dca(df, period, fees=None):
    b = period_bounds(df.index, period)
    if b is None:
        return None
    eq, flows = simulate_dca(df, *b, buy_fee=None if fees is None else fees["buy"])
    m = metrics(eq)
    m["xirr"] = xirr(flows)
    return {"equity": eq, "trades": [], "metrics": m}


def summarize(results_by_code, keys=("cagr", "mdd", "sharpe", "total")):
    """Median of each metric across stocks; results_by_code maps code -> run_* result (or None)."""
    rows = [r["metrics"] for r in results_by_code.values() if r]
    return {k: float(np.nanmedian([r[k] for r in rows])) for k in keys} | {"n": len(rows)}
