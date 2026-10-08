"""Controls every model has to beat, so that "held more" or "timed volatility" isn't mistaken for skill.

B1 constant exposure: the strategy's own average exposure in the period, held as one fixed position.
B2 shuffled signal: each stock trades another stock's positions (fixed derangement) — same persistence and exposure,
   same market-wide timing, but no information about the stock itself.
B3 volatility target: exposure min(1, s / 20-day realised vol), quantised to 0.25 steps with hysteresis, with s set so
   the average exposure in the period matches the strategy's.
B1 and B3 are calibrated on the evaluation period itself, which flatters the controls; that is intentional.
"""

import numpy as np
import pandas as pd

from . import backtest as bt

CONTROL_LABELS = {"const": "固定曝險", "shuffle": "別檔訊號", "voltarget": "波動目標"}


def _exec_mean(target, b):
    i0, i1 = b
    return float(target.iloc[i0 - 1 : i1].mean())


def constant_exposure(pos, df, period):
    b = bt.period_bounds(df.index, period)
    if b is None:
        return None
    return pd.Series(round(_exec_mean(bt.clean_target(pos, df.index), b), 2), index=df.index)


def shuffled(positions, data, seed=0):
    codes = sorted(positions)
    perm = codes[:]
    rng = np.random.default_rng(seed)
    while True:
        rng.shuffle(perm)
        if all(a != b for a, b in zip(codes, perm)):
            break
    return {c: bt.clean_target(positions[p], data[p].index).reindex(data[c].index).ffill().fillna(0.0) for c, p in zip(codes, perm)}


def _quantise(raw, step=0.25):
    out = np.empty(len(raw))
    cur = 0.0
    for i, x in enumerate(raw):
        if abs(x - cur) >= 0.75 * step:
            cur = round(x / step) * step
        out[i] = cur
    return out


def vol_target(pos, df, period, n=20):
    b = bt.period_bounds(df.index, period)
    if b is None:
        return None
    want = _exec_mean(bt.clean_target(pos, df.index), b)
    vol = (np.log(df["close"]).diff().rolling(n).std() * np.sqrt(252)).to_numpy()
    i0, i1 = b

    def path(s):
        raw = np.nan_to_num(np.minimum(1.0, s / vol), nan=0.0)
        return _quantise(raw)

    lo, hi = 0.0, 5.0
    for _ in range(16):
        mid = (lo + hi) / 2
        if path(mid)[i0 - 1 : i1].mean() < want:
            lo = mid
        else:
            hi = mid
    return pd.Series(path((lo + hi) / 2), index=df.index)


def evaluate_controls(sid, positions, data, periods):
    """results[f"{sid}~{kind}"][period][code] in the same shape as runner.evaluate."""
    from .runner import fees_of

    data = {c: data[c] for c in positions}
    shuf = bt.effective_positions(shuffled(positions, data), data)
    out = {f"{sid}~{k}": {p: {} for p in periods} for k in CONTROL_LABELS}

    def run(c, df, target, p):
        return None if target is None else bt.run_strategy(df, bt.enforce_short_rules(c, df, target), p, fees_of(c))

    for p in periods:
        for c, df in data.items():
            out[f"{sid}~const"][p][c] = run(c, df, constant_exposure(positions[c], df, p), p)
            out[f"{sid}~shuffle"][p][c] = run(c, df, shuf[c], p)
            out[f"{sid}~voltarget"][p][c] = run(c, df, vol_target(positions[c], df, p), p)
    return out
