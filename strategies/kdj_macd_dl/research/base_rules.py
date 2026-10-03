import numpy as np
import pandas as pd

from harness import baseline_rows, data, evaluate, show
from stocklab.indicators import macd, tw_kd


def ind(df):
    kd = tw_kd(df)
    m = macd(df["close"])
    return pd.concat([kd, m], axis=1)


def state_machine(enter, exit_):
    enter = enter.to_numpy()
    exit_ = exit_.to_numpy()
    pos = np.zeros(len(enter))
    cur = 0.0
    for i in range(len(enter)):
        if cur == 0 and enter[i]:
            cur = 1.0
        elif cur == 1 and exit_[i]:
            cur = 0.0
        pos[i] = cur
    return pos


def rules(df):
    x = ind(df)
    k, d, dif, sig, osc = x["k"], x["d"], x["dif"], x["macd"], x["osc"]
    kup = (k > d) & (k.shift() <= d.shift())
    kdn = (k < d) & (k.shift() >= d.shift())
    out = {}
    # R1: KD golden cross & OSC rising; exit KD death cross
    out["R1_kdx_oscup"] = state_machine(kup & (osc > osc.shift()), kdn)
    # R2: hold while OSC>0, enter needs K>D
    out["R2_osc_pos"] = state_machine((osc > 0) & (k > d), osc < 0)
    # R3: both bullish to enter, both bearish to exit
    out["R3_both"] = state_machine((k > d) & (osc > 0), (k < d) & (osc < 0))
    # R4: KD golden cross below 50 & DIF>MACD ... exit OSC<0
    out["R4_kdlow_osc"] = state_machine(kup & (d < 50) & (osc > osc.shift()), (osc < 0) & (k < d))
    # R5: KD golden cross with DIF>0 (uptrend) exit KD death cross or DIF<0
    out["R5_kdx_difpos"] = state_machine(kup & (dif > 0), kdn | (dif < 0))
    # R6: R3 with DIF>0 trend filter
    out["R6_both_dif"] = state_machine((k > d) & (osc > 0) & (dif > 0), ((k < d) & (osc < 0)) | (dif < 0))
    # R7: DIF>MACD only (pure MACD cross)
    out["R7_macdx"] = (osc > 0).astype(float).to_numpy()
    return {n: pd.Series(v, index=df.index) for n, v in out.items()}


if __name__ == "__main__":
    d = data()
    allr = {c: rules(df) for c, df in d.items()}
    rows = baseline_rows(("is",))
    for name in next(iter(allr.values())):
        rows += evaluate({c: allr[c][name] for c in d}, name, ("is",))
    show(rows)
