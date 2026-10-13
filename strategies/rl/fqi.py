"""Two learners that use the fact that our orders don't move the market, so the reward of EVERY action is known for
every past day (full counterfactual information), not just the action an agent happened to take:

  sup   supervised baseline: ridge / gradient boosting predict the forward H-bar return; hold while the prediction
        is above an entry threshold, leave when it falls below an exit threshold (hysteresis against fees)
  fqi   fitted Q-iteration (Ernst et al. 2005) for long/flat with fees: G_a(x) = value of holding a (0 or 1) today
        and acting optimally afterwards, discount gamma; the policy switches only when the value gap beats the fee

Thresholds: "zero" = act on the sign of the expected value (what a profit maximiser does), or "keepXX" = hold the
top XX% of the training-year predictions (the forced-budget push: the model must rank days, not just stay long).

    python -m strategies.rl.fqi
"""

from __future__ import annotations

import sys
import time

import numpy as np

from .common import DEV_YEARS, evaluate, forward, line, load_panel, train_rows
from .direct import POS_DIR
from .panel import GROUPS

ALL = [f for g in ("tech", "xs", "chips", "mkt") for f in GROUPS[g]]


def _std(P, cols, tr):
    m = P.warm[tr]
    x = P.X[tr][m][:, cols].astype(np.float64)
    mu, sd = x.mean(0), x.std(0) + 1e-9
    return lambda X: np.clip((X[..., cols] - mu) / sd, -5, 5)


def _ridge(X, y, lam=1.0):
    Xa = np.column_stack([X, np.ones(len(X))])
    A = Xa.T @ Xa + lam * np.diag(np.r_[np.ones(X.shape[1]), 0.0]) * len(X) * 1e-3
    w = np.linalg.solve(A, Xa.T @ y)
    return lambda Z: Z @ w[:-1] + w[-1]


def _gbm(X, y):
    from sklearn.ensemble import HistGradientBoostingRegressor

    g = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=300, l2_regularization=1.0,
                                      random_state=0).fit(X, y)
    return g.predict


def hysteresis(score, enter, leave, use):
    """Positions from a score: go long when score > enter, go flat when score < leave (per stock, sequential)."""
    T, N = score.shape
    pos = np.zeros((T, N), np.float32)
    cur = np.zeros(N, np.float32)
    for t in range(T):
        s = score[t]
        cur = np.where(use[t], np.where(cur > 0, (s >= leave).astype(np.float32), (s > enter).astype(np.float32)), cur)
        pos[t] = cur
    return pos


def sup(P, h=20, model="ridge", rule="zero", groups=("tech", "xs", "chips", "mkt"), years=DEV_YEARS):
    cols = P.cols([f for g in groups for f in GROUPS[g]])
    y_all = forward(P, h)
    score = np.full(P.R.shape, np.nan)
    thr = {}
    for year in years:
        tr = train_rows(P, year, h)
        f = _std(P, cols, tr)
        m = P.warm[tr] & np.isfinite(y_all[tr])
        ti, ni = np.nonzero(m)
        if model == "gbm":
            k = (tr[ti] % 3) == 0
            ti, ni = ti[k], ni[k]
        X, y = f(P.X[tr[ti], ni]), y_all[tr[ti], ni]
        fit = (_ridge if model == "ridge" else _gbm)(X, y)
        te = np.flatnonzero(P.year == year)
        score[te] = fit(f(P.X[te]).reshape(-1, len(cols))).reshape(len(te), -1)
        ptr = fit(X)
        if rule == "zero":
            thr[year] = (0.0, 0.0)
        else:
            q = int(rule[4:]) / 100
            hi = np.quantile(ptr, 1 - q)
            thr[year] = (hi, np.quantile(ptr, max(0.0, 1 - q - 0.1)))
    use = P.warm & P.ok
    pos = np.zeros(P.R.shape, np.float32)
    rows = np.flatnonzero(np.isin(P.year, years))
    enter = np.array([thr[y][0] for y in P.year[rows]])[:, None]
    leave = np.array([thr[y][1] for y in P.year[rows]])[:, None]
    s = np.nan_to_num(score[rows], nan=-1e9)
    T = len(rows)
    cur = np.zeros(P.R.shape[1], np.float32)
    for k in range(T):
        st = s[k]
        cur = np.where(use[rows[k]], np.where(cur > 0, (st >= leave[k]).astype(np.float32), (st > enter[k]).astype(np.float32)), cur)
        pos[rows[k]] = cur
    return pos


def fqi(P, gamma=0.95, iters=25, model="ridge", groups=("tech", "xs", "chips", "mkt"), years=DEV_YEARS, rule="zero"):
    """Fitted Q-iteration for long/flat with fees; G1 - G0 is the value of being long, net of the future."""
    cols = P.cols([f for g in groups for f in GROUPS[g]])
    lr = np.log1p(P.R.astype(np.float64))
    cb, cs = -np.log1p(-P.buy), -np.log1p(-P.sell)
    pos = np.zeros(P.R.shape, np.float32)
    use = P.warm & P.ok
    cur = np.zeros(P.R.shape[1], np.float32)
    for year in years:
        tr = train_rows(P, year, 1)
        f = _std(P, cols, tr)
        m = use[tr]
        nxt = np.r_[tr[1:], tr[-1]]
        m &= use[nxt]
        ti, ni = np.nonzero(m)
        if model == "gbm":
            k = (tr[ti] % 2) == 0
            ti, ni = ti[k], ni[k]
        X = f(P.X[tr[ti], ni])
        Xn = f(P.X[nxt[ti], ni])
        r = lr[tr[ti], ni]
        g0 = g1 = lambda Z: np.zeros(len(Z))
        fitter = _ridge if model == "ridge" else _gbm
        for _ in range(iters if model == "ridge" else 6):
            a0, a1 = g0(Xn), g1(Xn)
            v_from0 = np.maximum(a0, a1 - cb[ni])          # flat today -> best of stay flat / buy tomorrow
            v_from1 = np.maximum(a1, a0 - cs[ni])          # long today -> best of stay / sell
            g0, g1 = fitter(X, gamma * v_from0), fitter(X, r + gamma * v_from1)
        te = np.flatnonzero(P.year == year)
        Z = f(P.X[te]).reshape(-1, len(cols))
        d = (g1(Z) - g0(Z)).reshape(len(te), -1)
        if rule != "zero":
            q = int(rule[4:]) / 100
            dtr = g1(X) - g0(X)
            d = d - np.quantile(dtr, 1 - q)
        for k, t in enumerate(te):
            # flat: buy when the long value beats flat by more than the buy fee; long: sell when flat beats long by the sell fee
            cur = np.where(use[t], np.where(cur > 0, (d[k] > -cs[None, :][0]).astype(np.float32), (d[k] > cb).astype(np.float32)), cur)
            pos[t] = cur
    return pos


RUNS = [
    ("sup_h5_ridge_zero", lambda P: sup(P, 5, "ridge", "zero")),
    ("sup_h20_ridge_zero", lambda P: sup(P, 20, "ridge", "zero")),
    ("sup_h60_ridge_zero", lambda P: sup(P, 60, "ridge", "zero")),
    ("sup_h20_ridge_keep80", lambda P: sup(P, 20, "ridge", "keep80")),
    ("sup_h20_ridge_keep50", lambda P: sup(P, 20, "ridge", "keep50")),
    ("sup_h60_ridge_keep80", lambda P: sup(P, 60, "ridge", "keep80")),
    ("sup_h20_gbm_keep80", lambda P: sup(P, 20, "gbm", "keep80")),
    ("sup_h60_mkt_keep80", lambda P: sup(P, 60, "ridge", "keep80", groups=("mkt",))),
    ("sup_h20_mkt_keep80", lambda P: sup(P, 20, "ridge", "keep80", groups=("mkt",))),
    ("fqi_ridge_g95", lambda P: fqi(P, 0.95)),
    ("fqi_ridge_g99", lambda P: fqi(P, 0.99, iters=60)),
    ("fqi_ridge_g95_keep80", lambda P: fqi(P, 0.95, rule="keep80")),
    ("fqi_gbm_g95", lambda P: fqi(P, 0.95, model="gbm")),
    ("fqi_mkt_g95_keep80", lambda P: fqi(P, 0.95, rule="keep80", groups=("mkt",))),
]


if __name__ == "__main__":
    P = load_panel()
    pick = sys.argv[1:]
    POS_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in RUNS:
        if pick and name not in pick:
            continue
        t0 = time.time()
        pos = fn(P)
        m = evaluate(P, pos, name=name, config={"family": name.split("_")[0], "name": name}, log=True)
        print(line(name, m), f"({time.time() - t0:.0f}s)", flush=True)
        print(line("  └ 別檔訊號 B2", evaluate(P, np.roll(pos, 1, axis=1))), flush=True)
        np.save(POS_DIR / f"{name}_dev.npy", pos)
