"""KDJ round, experiment K2b: J-extreme rules combined with MACD confirmation (in-sample only).

  python -m strategies.kdj_macd_dl.research.kdj_base_rules2

K1 showed that the textbook J-extreme rule (buy when J climbs back above 0, sell when it drops back below 100)
at weekly-equivalent speed had the best floor variant, but it uses no MACD. Here MACD is added as confirmation.
Thresholds stay at the textbook 0 / 100 (not tuned).
"""

import sys

import pandas as pd

from .kdj_base_rules import signals, state_machine
from .kdj_harness import baseline_rows, data, evaluate, show

RULES = {
    "Jext_pure":        lambda g: (g["jreb"], g["jfall"]),
    # MACD momentum must confirm the J turn on the same bar
    "Jext_OSCconfirm":  lambda g: (g["jreb"] & g["oscup"], g["jfall"] & g["oscdn"]),
    # buy the oversold rebound, sell only on a KD+MACD trend break
    "Jreb_trendexit":   lambda g: (g["jreb"], g["kbear"] & g["mbear"]),
    # buy the oversold rebound, sell on overbought exhaustion OR trend break
    "Jext_or_trend":    lambda g: (g["jreb"], g["jfall"] | (g["kbear"] & g["mbear"])),
    # J rebound or KD+MACD bullish to buy; overbought exhaustion confirmed by falling OSC, or trend break, to sell
    "JextMACD_full":    lambda g: (g["jreb"] | (g["kbull"] & g["mbull"]), (g["jfall"] & g["oscdn"]) | (g["kbear"] & g["mbear"])),
    # J extremes, but don't sell overbought while MACD is still above its signal line and rising
    "Jext_MACDhold":    lambda g: (g["jreb"], g["jfall"] & ~(g["mbull"] & g["oscup"])),
}
SCALES = (2, 3, 5)


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
            p = rule_positions(name, s)
            rows.append(evaluate(p, f"{name} x{s}"))
            rows.append(evaluate({c: 0.5 + 0.5 * v for c, v in p.items()}, f"{name} x{s} +floor"))
    show(rows, sys.argv[1] if len(sys.argv) > 1 else None)
