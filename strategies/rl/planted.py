"""Is the learner the bottleneck, or the data? Plant a signal of known strength into the real returns and see which
learners find it and turn it into profit.

R' = R + beta * z * sd(R), where z is a standardised feature the learners can see (a slow one, the distance from the
60-day MA, or a fast one, yesterday's return) and beta sets the planted daily IC rho = corr(z, R'). Everything else
(real returns, fees, walk-forward years) stays the same, so the only change is a signal we know is there.

    python -m strategies.rl.planted
"""

from __future__ import annotations

import copy
import time

import numpy as np

from .common import evaluate, line, load_panel, save
from . import direct, fqi, ppo

OUT = __import__("pathlib").Path(__file__).parent / "results" / "planted.json"


def plant(P, feat, rho, seed=0):
    Q = copy.copy(P)
    x = P.X[:, :, P.fi[feat]].astype(np.float64)
    use = P.warm & P.ok
    mu = np.array([x[use[:, n], n].mean() if use[:, n].any() else 0 for n in range(x.shape[1])])
    sd = np.array([x[use[:, n], n].std() + 1e-9 if use[:, n].any() else 1 for n in range(x.shape[1])])
    z = (x - mu) / sd
    sr = np.array([P.R[use[:, n], n].std() if use[:, n].any() else 0 for n in range(x.shape[1])])
    beta = rho / np.sqrt(1 - rho ** 2)
    Q.R = (P.R + beta * z * sr[None, :] * use).astype(np.float32)
    return Q


def main():
    P = load_panel()
    res = {}
    for feat, label in (("ma60", "慢訊號（距季線）"), ("r1", "快訊號（昨日漲跌）")):
        for rho in (0.0, 0.01, 0.02, 0.03, 0.05, 0.1):
            Q = plant(P, feat, rho)
            out = {}
            t0 = time.time()
            _, pos = direct.run(Q, direct.Cfg(f"planted_{feat}_{rho}"), verbose=False)
            out["direct"] = evaluate(Q, pos)
            out["ridge_h1"] = evaluate(Q, fqi.sup(Q, 1, "ridge", "zero"))
            out["ridge_h20"] = evaluate(Q, fqi.sup(Q, 20, "ridge", "zero"))
            if rho in (0.02, 0.05):
                _, pos = ppo.run(Q, ppo.Cfg(f"planted_{feat}_{rho}", reward="rel"))
                out["ppo_rel"] = evaluate(Q, pos)
            res[f"{feat}_{rho}"] = {k: {kk: v for kk, v in m.items() if not isinstance(v, dict)} | {"ew_sharpe": m["ew"]["sharpe"]} for k, m in out.items()}
            for k, m in out.items():
                print(line(f"{label} IC {rho:.2f} {k}", m), flush=True)
            print(f"   ({time.time() - t0:.0f}s)", flush=True)
            save(OUT, res)


if __name__ == "__main__":
    main()
