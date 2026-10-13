"""One-shot out-of-sample check (2021-) of the configurations picked on the development years, frozen beforehand:

  fqi_gbm_g95              best timing skill vs. a constant position on 2013-2020 (t = +1.6)
  fqi_ridge_g95_keep80     best equal-weight Sharpe among the per-stock learners on 2013-2020
  imitate_fee30_gbm_keep80 best equal-weight portfolio Sharpe on 2013-2020 (1.25 vs 1.15)
  ppo_rel                  the PPO agent (relative-to-buy-and-hold reward)
  direct_log_smooth        the best direct-reinforcement agent by median CAGR

Models are retrained every January on everything before it, exactly as in development. Reported on all 105 stocks
and on the 55 outside today's 0050 (less survivorship bias).

    python -m strategies.rl.final_oos
"""

from __future__ import annotations

import numpy as np

from . import direct, fqi, imitate, ppo
from .common import OOS_YEARS, evaluate, line, load_panel, save
from .direct import POS_DIR

OUT = __import__("pathlib").Path(__file__).parent / "results" / "final_oos.json"


def main():
    P = load_panel()
    runs = {
        "fqi_gbm_g95": lambda: fqi.fqi(P, 0.95, model="gbm", years=OOS_YEARS),
        "fqi_ridge_g95_keep80": lambda: fqi.fqi(P, 0.95, rule="keep80", years=OOS_YEARS),
        "imitate_fee30_gbm_keep80": lambda: imitate.run(P, 30, "gbm", 0.8, years=OOS_YEARS)[0],
        "ppo_rel": lambda: ppo.run(P, ppo.Cfg("rel", reward="rel"), years=OOS_YEARS)[1],
        "direct_log_smooth": lambda: direct.run(P, direct.Cfg("log_smooth", smooth=0.8), years=OOS_YEARS, verbose=False)[1],
    }
    res = {}
    extra = np.flatnonzero(~P.is50)
    for name, fn in runs.items():
        pos = fn()
        np.save(POS_DIR / f"{name}_oos.npy", pos)
        m = evaluate(P, pos, OOS_YEARS, name=f"final_{name}", config={"name": name}, log=True, period="oos")
        mx = evaluate(P, pos, OOS_YEARS, cols=extra)
        res[name] = {"all": {k: v for k, v in m.items() if not isinstance(v, dict)} | {"ew_sharpe": m["ew"]["sharpe"], "ew_bh_sharpe": m["ew_bh"]["sharpe"]},
                     "extra55": {k: v for k, v in mx.items() if not isinstance(v, dict)} | {"ew_sharpe": mx["ew"]["sharpe"], "ew_bh_sharpe": mx["ew_bh"]["sharpe"]}}
        print(line(f"[樣本外] {name}", m), flush=True)
        print(line("   只看 55 檔代表股", mx), flush=True)
        save(OUT, res)


if __name__ == "__main__":
    main()
