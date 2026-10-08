"""Mandatory non-DL comparisons on the SAME inputs, labels, splits and walk-forward schedule as core.py:
  ridge     ridge regression on the per-stock row [20-day window, snapshot, market] (alpha picked on the purged
            validation tail)
  hgb       HistGradientBoostingRegressor on the same row (early stopping on the purged validation tail)
  hgb_peer  hgb + hand-made peer features: industry average (excluding the stock) of the 5/20/60-day vol-normalised
            returns, TSMC (2330) last-day / 5 / 20-day returns, the stock's cross-sectional rank of its 20/60-day return
Used by the research scripts only (not by the published positions).
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from . import core

PEER_COLS = ["ind_r5", "ind_r20", "ind_r60", "tsm_last", "tsm_r5", "tsm_r20", "rank_r20", "rank_r60"]


def peer_features(P):
    X, AV, ind = P["X"], P["AV"], P["ind"]
    nw = core.L * len(core.WIN_COLS)
    snap = {c: X[:, :, nw + i] for i, c in enumerate(core.SNAP_COLS)}
    T, N = AV.shape
    out = np.full((T, N, len(PEER_COLS)), np.nan, np.float32)
    for k, col in enumerate(("r5", "r20", "r60")):
        v = np.where(AV, snap[col], np.nan)
        for g in np.unique(ind):
            members = np.flatnonzero(ind == g)
            vs = v[:, members]
            tot = np.nansum(vs, 1, keepdims=True)
            cnt = (~np.isnan(vs)).sum(1, keepdims=True)
            loo = (tot - np.nan_to_num(vs)) / np.maximum(cnt - (~np.isnan(vs)), 1)
            loo[np.broadcast_to(cnt, vs.shape) - (~np.isnan(vs)) <= 0] = np.nan
            out[:, members, k] = loo
    j = core.CODES.index("2330")
    w_last = X[:, j, nw - len(core.WIN_COLS): nw]              # last day's [on, id, rng, vol]
    out[:, :, 3] = (w_last[:, 0] + w_last[:, 1])[:, None]
    out[:, :, 4] = snap["r5"][:, j][:, None]
    out[:, :, 5] = snap["r20"][:, j][:, None]
    for k, col in ((6, "r20"), (7, "r60")):
        v = pd.DataFrame(np.where(AV, snap[col], np.nan))
        out[:, :, k] = v.rank(axis=1, pct=True).to_numpy()
    return out


def rows_matrix(P, rows, peer=None):
    X = P["X"][rows]                                           # (r, N, FIN)
    M = np.broadcast_to(P["M"][rows][:, None, :], (len(rows), core.N, core.FM))
    parts = [X, M]
    if peer is not None:
        parts.append(peer[rows])
    Z = np.concatenate(parts, -1)
    ok = P["AV"][rows]
    y = P["Y"][rows]
    return Z, y, ok


def _fit_predict(P, kind, tr, va, te, peer, seed=0, stride=2):
    tr = tr[::stride]
    Ztr, ytr, oktr = rows_matrix(P, tr, peer)
    m = oktr & ~np.isnan(ytr)
    Xtr, Ytr = Ztr[m], np.clip(ytr[m], -3, 3)
    Zva, yva, okva = rows_matrix(P, va, peer)
    mv = okva & ~np.isnan(yva)
    Xva, Yva = Zva[mv], np.clip(yva[mv], -3, 3)
    Zte, _, okte = rows_matrix(P, te, peer)
    if kind == "ridge":
        mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        mu = np.nan_to_num(mu)
        f = lambda A: np.clip(np.nan_to_num((A - mu) / sd), -5, 5)
        best = None
        for a in (1.0, 10.0, 100.0, 1000.0, 10000.0):
            r = Ridge(alpha=a).fit(f(Xtr), Ytr)
            e = float(np.mean((r.predict(f(Xva)) - Yva) ** 2))
            if best is None or e < best[0]:
                best = (e, r)
        model, pre = best[1], f
    else:
        model = HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                              min_samples_leaf=400, l2_regularization=1.0, early_stopping=True,
                                              n_iter_no_change=20, random_state=seed)
        model.fit(Xtr, Ytr, X_val=Xva, y_val=Yva)
        pre = lambda A: A
    out = np.full(okte.shape, np.nan, np.float32)
    flat = Zte.reshape(-1, Zte.shape[-1])
    out.reshape(-1)[okte.reshape(-1)] = model.predict(pre(flat[okte.reshape(-1)]))
    return out


def walk_forward(data, kind, years, verbose=False):
    P = core.build_panel(data)
    peer = peer_features(P) if kind == "hgb_peer" else None
    k = "ridge" if kind == "ridge" else "hgb"
    dates = P["dates"]
    pred = np.full((len(dates), core.N), np.nan, np.float32)
    for yr in years:
        c0 = int(dates.searchsorted(pd.Timestamp(f"{yr}-01-01")))
        c1 = int(dates.searchsorted(pd.Timestamp(f"{yr + 1}-01-01")))
        tr, va = core.split_rows(P, core.eligible_before(P, c0))
        t = time.time()
        pred[c0:c1] = _fit_predict(P, k, tr, va, np.arange(c0, c1), peer)
        if verbose:
            print(f"  {kind} {yr}: {time.time() - t:.1f}s", flush=True)
    return {0: pred}


def purged_cv(data, folds, kind, train_end=None, verbose=False):
    P = core.build_panel(data)
    peer = peer_features(P) if kind == "hgb_peer" else None
    k = "ridge" if kind == "ridge" else "hgb"
    dates = P["dates"]
    T = len(dates)
    lim = T if train_end is None else int(dates.searchsorted(pd.Timestamp(train_end), side="right"))
    ye = core.label_end(P, np.arange(T))
    has = (P["AV"] & ~np.isnan(P["Y"])).sum(axis=1) >= 5
    pred = np.full((T, core.N), np.nan, np.float32)
    for i, (f, (a, b)) in enumerate(folds.items()):
        f0 = int(dates.searchsorted(pd.Timestamp(a)))
        f1 = T if b is None else int(dates.searchsorted(pd.Timestamp(b), side="right"))
        rows = np.arange(T)
        ok = has & (ye < lim) & ((ye < f0) | (rows > f1 - 1 + core.H)) & ((rows < f0) | (rows >= f1))
        tr, va = core.split_rows(P, rows[ok])
        lo = f0 if i else 0
        pred[lo:f1] = _fit_predict(P, k, tr, va, np.arange(lo, f1), peer)
        if verbose:
            print(f"  cv {kind} {f}", flush=True)
    return {0: pred}
