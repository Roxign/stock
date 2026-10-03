"""KDJ round, K2c: stability of the leading base-rule candidates across the two in-sample halves.

  python -m strategies.kdj_macd_dl.research.kdj_base_halves
"""

import numpy as np

from stocklab import backtest as bt

from . import kdj_base_rules as r1
from . import kdj_base_rules2 as r2
from .kdj_harness import data

bt.PERIODS["h1"] = ("2010-01-01", "2015-06-30")   # runtime-only additions (stocklab files are not modified)
bt.PERIODS["h2"] = ("2015-07-01", "2020-12-31")

CANDS = [("KD_both", r1, 5), ("KD_both", r1, 3), ("Jext_pure", r1, 5), ("Jext_OSCconfirm", r2, 5),
         ("Jext_MACDhold", r2, 5), ("Jext_MACDhold", r2, 3), ("Jturn_both", r1, 5)]


def med(res, k):
    return float(np.nanmedian([r["metrics"][k] for r in res.values() if r]))


if __name__ == "__main__":
    d = data()
    for per in ("h1", "h2"):
        bh = {c: bt.run_buy_hold(df, per) for c, df in d.items()}
        print(f"--- {per} {bt.PERIODS[per]}  buy&hold CAGR {med(bh, 'cagr'):.1%} Sharpe {med(bh, 'sharpe'):.2f}")
        for name, mod, s in CANDS:
            pos = mod.rule_positions(name, s)
            for tag, p in (("0/1", pos), ("floor", {c: 0.5 + 0.5 * v for c, v in pos.items()})):
                res = {c: bt.run_strategy(df, p[c], per) for c, df in d.items()}
                print(f"{name:>16} x{s} {tag:>5}: CAGR {med(res, 'cagr'):6.1%}  MDD {med(res, 'mdd'):6.1%}  "
                      f"Sharpe {med(res, 'sharpe'):5.2f}  expo {med(res, 'exposure'):5.1%}")
