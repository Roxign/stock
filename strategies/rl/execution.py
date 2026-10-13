"""Use the weak, fast signal where it costs nothing: 定期定額 has to buy every month anyway, so the fee is paid no matter
which day the order goes in. The policy may wait inside the month for a day the model likes better (or buy on the
last trading day at the latest); the baseline buys at the open of the first trading day, like stocklab's DCA.

Monthly amounts are equal, so final wealth is proportional to the shares bought: the gain is
sum(1 / fill price) / sum(1 / first-day price) - 1, computed per stock (and for 0050) over the test years.

    python -m strategies.rl.execution          # development years 2013-2020
    python -m strategies.rl.execution oos      # the frozen rules on 2021-
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from stocklab.etf import load_etf

from .common import DEV_YEARS, OOS_YEARS, forward, load_panel, save, train_rows
from .panel import FEATS, GROUPS, _market, _stock_frame

OUT = __import__("pathlib").Path(__file__).parent / "results" / "execution.json"
ALL = [f for g in ("tech", "xs", "chips", "mkt") for f in GROUPS[g]]


def ridge_scores(P, h, years, extra_X=None):
    """Walk-forward ridge prediction of the forward h-bar return for every stock-day (and for extra_X [T, F])."""
    cols = P.cols(ALL)
    y_all = forward(P, h)
    S = np.full(P.R.shape, np.nan)
    E = np.full(len(P.R), np.nan) if extra_X is not None else None
    cuts = {}
    for year in years:
        tr = train_rows(P, year, h)
        m = P.warm[tr] & np.isfinite(y_all[tr])
        ti, ni = np.nonzero(m)
        X = P.X[tr[ti], ni][:, cols].astype(np.float64)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = np.clip((X - mu) / sd, -5, 5)
        y = y_all[tr[ti], ni]
        Za = np.column_stack([Z, np.ones(len(Z))])
        w = np.linalg.solve(Za.T @ Za + np.diag(np.r_[np.ones(len(cols)), 0]) * len(Z) * 1e-3, Za.T @ y)
        f = lambda A: np.clip((A - mu) / sd, -5, 5) @ w[:-1] + w[-1]
        te = np.flatnonzero(P.year == year)
        S[te] = f(P.X[te][:, :, cols].reshape(-1, len(cols))).reshape(len(te), -1)
        if E is not None:
            E[te] = f(extra_X[te][:, cols])
        cuts[year] = np.quantile(f(X), [0.5, 0.7, 0.9])
    return S, E, cuts


def months(dates, rows):
    """Groups of calendar rows per month inside `rows`."""
    per = pd.PeriodIndex(dates[rows], freq="M")
    return [rows[per == p] for p in per.unique()]


def plan(O, ok, score, cut, mrows):
    """Fill index per month: the first day whose decision (previous close) scores >= cut, else the month's last day.
    The decision for a fill on day d is made at the close of d-1 (score row d-1)."""
    fills = []
    for rows in mrows:
        d = None
        for r in rows:
            if not ok[r]:
                continue
            if cut is None or (r - 1 >= 0 and np.isfinite(score[r - 1]) and score[r - 1] >= cut):
                d = r
                break
        if d is None:
            good = [r for r in rows if ok[r]]
            d = good[-1] if good else None
        fills.append(d)
    return fills


def gain(O, fills, base):
    a = sum(1 / O[f] for f in fills if f is not None)
    b = sum(1 / O[f] for f in base if f is not None)
    return a / b - 1 if b else np.nan


def etf_features(P):
    """Panel-compatible features for 0050 (own technical features + the market block; no chips, cross-section 0.5)."""
    df = load_etf("0050")
    f = _stock_frame(df).reindex(P.dates)
    X = np.zeros((len(P.dates), len(FEATS)), np.float32)
    for j, n in enumerate(FEATS):
        if n in f:
            X[:, j] = f[n].to_numpy()
        elif n.startswith("xs_"):
            X[:, j] = 0.5
        elif n.startswith("m_") or n in GROUPS["mkt"]:
            X[:, j] = P.X[:, 0, j]
    X[~np.isfinite(X)] = 0
    return X, df["open"].reindex(P.dates).to_numpy()


def main(period="dev"):
    P = load_panel()
    years = DEV_YEARS if period == "dev" else OOS_YEARS
    rows = np.flatnonzero(np.isin(P.year, years))
    mrows = months(P.dates, rows)
    Xe, Oe = etf_features(P)
    oke = np.isfinite(Oe)
    use = P.warm & P.ok
    res = {}
    for h in (1, 5, 20):
        S, E, cuts = ridge_scores(P, h, years, Xe)
        for qi, q in enumerate((0.5, 0.7, 0.9)):
            g_st, g_rand = [], []
            for n in range(P.R.shape[1]):
                ok = use[:, n] & np.isfinite(P.open[:, n])
                if ok[rows].sum() < 500:
                    continue
                base = plan(P.open[:, n], ok, None, None, mrows)
                # per-year cut from that year's training predictions
                fills = []
                for mr in mrows:
                    fills += plan(P.open[:, n], ok, S[:, n], cuts[P.year[mr[0]]][qi], [mr])
                g_st.append(gain(P.open[:, n], fills, base))
            base_e = plan(Oe, oke, None, None, mrows)
            fills_e = []
            for mr in mrows:
                fills_e += plan(Oe, oke, E, cuts[P.year[mr[0]]][qi], [mr])
            ge = gain(Oe, fills_e, base_e)
            key = f"h{h}_top{int((1 - q) * 100)}"
            res[key] = {"stocks_median": float(np.nanmedian(g_st)), "stocks_mean": float(np.nanmean(g_st)),
                        "stocks_win": float(np.nanmean(np.array(g_st) > 0)), "etf0050": float(ge)}
            print(f"  預測 {h:>2} 天、等到前 {int((1 - q) * 100)}% 的日子才買：個股中位 {res[key]['stocks_median']:+.2%}（{res[key]['stocks_win']:.0%} 檔變好），0050 {ge:+.2%}", flush=True)
    # reference points: last trading day of the month, a random day, and the hindsight-cheapest open of the month
    for label, pick in (("月底才買", lambda rs: rs[-1]), ("每月最便宜那天（事後）", None)):
        g_st = []
        for n in range(P.R.shape[1]):
            ok = use[:, n] & np.isfinite(P.open[:, n])
            if ok[rows].sum() < 500:
                continue
            base = plan(P.open[:, n], ok, None, None, mrows)
            fills = []
            for mr in mrows:
                good = [r for r in mr if ok[r]]
                if not good:
                    fills.append(None)
                elif pick is None:
                    fills.append(min(good, key=lambda r: P.open[r, n]))
                else:
                    fills.append(pick(good))
            g_st.append(gain(P.open[:, n], fills, base))
        res[label] = float(np.nanmedian(g_st))
        print(f"  {label}：個股中位 {res[label]:+.2%}")
    old = {}
    if OUT.exists():
        import json

        old = json.loads(OUT.read_text(encoding="utf-8"))
    save(OUT, old | {period: res})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
