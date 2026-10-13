"""Learn from the perfect trader: label every past day with what a hindsight-optimal long/flat trader held (DP with
fees; inflated fees give smoother "swing" labels), then train a classifier to recognise those days from today's
features only, walk-forward. The out-of-sample AUC says how recognisable good holding periods are in advance.

    python -m strategies.rl.imitate
"""

from __future__ import annotations

import time

import numpy as np

from .common import DEV_YEARS, evaluate, line, load_panel, save, train_rows
from .direct import POS_DIR
from .panel import GROUPS

OUT = __import__("pathlib").Path(__file__).parent / "results" / "imitate.json"
ALL = [f for g in ("tech", "xs", "chips", "mkt") for f in GROUPS[g]]


def oracle_labels(P, mult):
    """Hindsight-optimal long/flat positions per stock (log-wealth DP with fees x mult, backtracked)."""
    use = P.warm & P.ok
    lr = np.log1p(np.where(use, P.R, 0.0).astype(np.float64))
    b, s = np.log1p(-P.buy * mult), np.log1p(-P.sell * mult)
    T, N = lr.shape
    v0, v1 = np.zeros(N), np.full(N, -1e18)
    c0 = np.zeros((T, N), bool)  # flat at t came from long at t-1 (a sell)
    c1 = np.zeros((T, N), bool)  # long at t came from flat at t-1 (a buy)
    for t in range(T):
        n0 = np.maximum(v0, v1 + s)
        c0[t] = v1 + s > v0
        n1 = np.maximum(v1, v0 + b)
        c1[t] = v0 + b > v1
        v0, v1 = n0, n1 + lr[t]
    lab = np.zeros((T, N), np.float32)
    st = (v1 + s > v0).astype(int)
    for t in range(T - 1, -1, -1):
        lab[t] = st
        st = np.where(st == 1, np.where(c1[t], 0, 1), np.where(c0[t], 1, 0))
    return lab


def auc(p, y):
    o = np.argsort(p)
    r = np.empty(len(p))
    r[o] = np.arange(1, len(p) + 1)
    pos = y > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def run(P, mult, model="gbm", keep=None, purge=120, years=DEV_YEARS):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    y = oracle_labels(P, mult)
    cols = P.cols(ALL)
    use = P.warm & P.ok
    prob = np.zeros(P.R.shape, np.float32)
    aucs, thr = {}, {}
    for year in years:
        tr = train_rows(P, year, purge)
        m = use[tr]
        ti, ni = np.nonzero(m)
        k = (tr[ti] % 3) == 0
        ti, ni = ti[k], ni[k]
        X = P.X[tr[ti], ni][:, cols].astype(np.float64)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        X = np.clip((X - mu) / sd, -5, 5)
        Y = y[tr[ti], ni]
        clf = (HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=300, random_state=0)
               if model == "gbm" else LogisticRegression(C=0.1, max_iter=300)).fit(X, Y)
        te = np.flatnonzero(P.year == year)
        Z = np.clip((P.X[te][:, :, cols] - mu) / sd, -5, 5).reshape(-1, len(cols))
        prob[te] = clf.predict_proba(Z)[:, 1].reshape(len(te), -1)
        mt = use[te]
        aucs[year] = auc(prob[te][mt], y[te][mt])
        ptr = clf.predict_proba(X)[:, 1]
        thr[year] = np.quantile(ptr, 1 - keep) if keep else 0.5
    rows = np.flatnonzero(np.isin(P.year, years))
    pos = np.zeros(P.R.shape, np.float32)
    cur = np.zeros(P.R.shape[1], np.float32)
    for t in rows:
        th = thr[P.year[t]]
        cur = np.where(use[t], np.where(cur > 0, (prob[t] > th - 0.05).astype(np.float32), (prob[t] > th + 0.05).astype(np.float32)), cur)
        pos[t] = cur
    base = float(np.mean(y[rows][use[rows]]))
    return pos, aucs, base


def main():
    P = load_panel()
    res = {}
    for mult in (1, 10, 30):
        for model in ("logit", "gbm"):
            for keep in (None, 0.8):
                t0 = time.time()
                pos, aucs, base = run(P, mult, model, keep)
                name = f"imitate_fee{mult}_{model}_{'p50' if keep is None else 'keep80'}"
                m = evaluate(P, pos, name=name, config={"family": "imitate", "fee_mult": mult, "model": model, "keep": keep}, log=True)
                res[name] = {"auc": aucs, "oracle_long_share": base} | {k: v for k, v in m.items() if not isinstance(v, dict)} | {"ew_sharpe": m["ew"]["sharpe"]}
                print(line(name, m), f"AUC {np.nanmean(list(aucs.values())):.3f} 完美交易者持股 {base:.0%} ({time.time() - t0:.0f}s)", flush=True)
                np.save(POS_DIR / f"{name}_dev.npy", pos)
                save(OUT, res)


if __name__ == "__main__":
    main()
