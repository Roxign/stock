import numpy as np
import pandas as pd

from harness import data, evaluate
from base_rules import state_machine
from stocklab.indicators import macd, tw_kdj as tw_kd


def base_rule(df, s=5):
    kd = tw_kd(df, 9 * s, 3 * s, 3 * s)
    m = macd(df["close"], 12 * s, 26 * s, 9 * s)
    k, d, dif, osc = kd["k"], kd["d"], m["dif"], m["osc"]
    return pd.Series(state_machine((k > d) | (osc > 0), (k < d) & (osc < 0) & (dif < 0)), index=df.index)


def to_pos(p, thr, mode="q", span=1, lo=0.2, hi=0.4, fallback=None, hyst=0.0):
    ps = p.ewm(span=span, adjust=False).mean() if span > 1 else p
    ps = ps.where(p.notna())
    v = ps.to_numpy()
    if mode == "q":
        tlo = thr[f"q{int(lo*100)}"].to_numpy()
        thi = thr[f"q{int(hi*100)}"].to_numpy()
    else:  # absolute
        tlo = np.full(len(v), lo)
        thi = np.full(len(v), hi)
    out = np.zeros(len(v))
    cur = None
    for i in range(len(v)):
        if np.isnan(v[i]):
            out[i] = np.nan
            cur = None
            continue
        tgt = 1.0 if v[i] >= thi[i] else (0.5 if v[i] >= tlo[i] else 0.0)
        if cur is not None and hyst > 0 and tgt != cur:
            # require margin to change state
            if tgt > cur and v[i] < (thi[i] if tgt == 1 else tlo[i]) + hyst:
                tgt = cur
            elif tgt < cur and v[i] > (thi[i] if cur == 1 else tlo[i]) - hyst:
                tgt = cur
        out[i] = tgt
        cur = tgt
    s = pd.Series(out, index=p.index)
    if fallback is not None:
        s = s.fillna(fallback)
    return s.fillna(0.0)


def eval_out(out, name, periods=("is",), **kw):
    d = data()
    pos = {}
    for c, df in d.items():
        fb = base_rule(df)
        pos[c] = to_pos(out["preds"][c], out["thr"][c], fallback=fb, **kw)
    return evaluate(pos, name, periods)


def rule_pos(df, name):
    kind, s = name.split("_x")
    s = int(s)
    kd = tw_kd(df, 9 * s, 3 * s, 3 * s); mm = macd(df["close"], 12 * s, 26 * s, 9 * s)
    k, dd, dif, osc = kd["k"], kd["d"], mm["dif"], mm["osc"]
    if kind == "either":
        return pd.Series(state_machine((k > dd) | (osc > 0), (k < dd) & (osc < 0) & (dif < 0)), index=df.index)
    if kind == "both":
        return pd.Series(state_machine((k > dd) & (osc > 0), (k < dd) & (osc < 0)), index=df.index)
    raise ValueError(name)
