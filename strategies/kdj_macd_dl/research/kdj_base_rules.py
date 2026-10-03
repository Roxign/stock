"""KDJ round, experiment K1/K2: KDJ (+MACD) base rules, in-sample only.

  python -m strategies.kdj_macd_dl.research.kdj_base_rules

Scale s multiplies every parameter: KDJ(9s,3s,3s), MACD(12s,26s,9s); s=5 is the 'weekly-equivalent' setting.
Note: J - D = 3(K - D) and J - K = 2(K - D), so "J crosses D", "J crosses K" and "K crosses D" are the SAME event.
J-specific information is therefore only J's level beyond the 0..100 band and J's own turning points.
"""

import sys

import numpy as np
import pandas as pd

from stocklab.indicators import macd, tw_kdj

from .kdj_harness import baseline_rows, data, evaluate, show


def state_machine(enter, exit_):
    enter, exit_ = np.asarray(enter, bool), np.asarray(exit_, bool)
    pos = np.zeros(len(enter))
    cur = 0.0
    for i in range(len(enter)):
        if cur == 0 and enter[i]:
            cur = 1.0
        elif cur == 1 and exit_[i]:
            cur = 0.0
        pos[i] = cur
    return pos


def signals(df, s):
    x = tw_kdj(df, 9 * s, 3 * s, 3 * s)
    m = macd(df["close"], 12 * s, 26 * s, 9 * s)
    k, d, j, dif, osc = x["k"], x["d"], x["j"], m["dif"], m["osc"]
    j1 = j.shift()
    return dict(
        kbull=k > d, kbear=k < d, mbull=osc > 0, mbear=osc < 0, difpos=dif > 0,
        jup=j > j1, jdn=j < j1, oscup=osc > osc.shift(), oscdn=osc < osc.shift(),
        jreb=(j1 < 0) & (j >= 0),          # J climbs back above 0 (leaves the oversold zone)
        jfall=(j1 > 100) & (j <= 100),     # J drops back below 100 (leaves the overbought zone)
        jlow=j < 0, jhigh=j > 100, jlt100=j < 100,
    )


RULES = {
    # previous round's K/D rules (for comparison)
    "KD_both":        lambda g: (g["kbull"] & g["mbull"], g["kbear"] & g["mbear"]),
    "KD_either":      lambda g: (g["kbull"] | g["mbull"], g["kbear"] & g["mbear"]),
    # J turning points instead of the K/D cross
    "Jturn_both":     lambda g: (g["jup"] & g["mbull"], g["jdn"] & g["mbear"]),
    "Jturn_oscturn":  lambda g: (g["jup"] & g["oscup"], g["jdn"] & g["oscdn"]),
    # J extremes
    "Jext_pure":      lambda g: (g["jreb"], g["jfall"]),                                  # textbook J<0 buy / J>100 sell
    "Jreb_MACDexit":  lambda g: (g["jreb"] & g["difpos"], g["kbear"] & g["mbear"]),       # oversold rebound in MACD uptrend
    "both_noChase":   lambda g: (g["kbull"] & g["mbull"] & g["jlt100"], g["kbear"] & g["mbear"]),
    "both_JobExit":   lambda g: (g["kbull"] & g["mbull"], (g["kbear"] & g["mbear"]) | (g["jfall"] & g["oscdn"])),
    "both_or_Jreb":   lambda g: ((g["kbull"] & g["mbull"]) | g["jreb"], g["kbear"] & g["mbear"]),
    "either_or_Jreb": lambda g: (g["kbull"] | g["mbull"] | g["jreb"], g["kbear"] & g["mbear"]),
}
SCALES = (1, 2, 3, 5)


def rule_positions(name, s):
    out = {}
    for c, df in data().items():
        en, ex = RULES[name](signals(df, s))
        out[c] = pd.Series(state_machine(en.to_numpy(), ex.to_numpy()), index=df.index)
    return out


if __name__ == "__main__":
    rows = baseline_rows()
    for name in RULES:
        for s in SCALES:
            rows.append(evaluate(rule_positions(name, s), f"{name} x{s}"))
            # floor version (0.5 core position) is what the DL variants build on
            if s == 5:
                rows.append(evaluate({c: 0.5 + 0.5 * p for c, p in rule_positions(name, s).items()}, f"{name} x{s} +floor"))
    show(rows, sys.argv[1] if len(sys.argv) > 1 else None)
