"""Research-phase version of ../core.py used by the exploratory scripts in this folder (wf.py, meta.py, exp*.py).

Differences from the production core.py: configurable feature groups / lags, configurable MLP size, optional sample
weights, and the purge in wf.py uses each stock's own label end date (production additionally waits until every
stock's label for that date has ended, which is stricter). Numbers therefore differ slightly (~0.01-0.03 Sharpe).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from stocklab.indicators import atr, macd, tw_kdj as tw_kd

SLOW = 5


def indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    kd = tw_kd(df)
    m = macd(c)
    kdw = tw_kd(df, 9 * SLOW, 3 * SLOW, 3 * SLOW)
    mw = macd(c, 12 * SLOW, 26 * SLOW, 9 * SLOW)
    a = atr(df)
    return pd.DataFrame({
        "k": kd["k"], "d": kd["d"], "dif": m["dif"], "macd": m["macd"], "osc": m["osc"],
        "kw": kdw["k"], "dw": kdw["d"], "difw": mw["dif"], "macdw": mw["macd"], "oscw": mw["osc"],
        "atr": a, "close": c, "open": df["open"],
    }, index=df.index)


def feature_matrix(ind: pd.DataFrame, lags=(0, 1, 2, 3, 4), slow_lags=(0, 5), groups=("kd", "macd", "slow")):
    cols, names = [], []
    k = ind["k"].to_numpy() / 100 - 0.5
    d = ind["d"].to_numpy() / 100 - 0.5
    a = ind["atr"].to_numpy()
    a = np.where(a > 0, a, np.nan)
    dif = ind["dif"].to_numpy() / a
    osc = ind["osc"].to_numpy() / a
    kw = ind["kw"].to_numpy() / 100 - 0.5
    dw = ind["dw"].to_numpy() / 100 - 0.5
    difw = ind["difw"].to_numpy() / a
    oscw = ind["oscw"].to_numpy() / a

    def lag(x, L):
        if L == 0:
            return x
        out = np.full_like(x, np.nan)
        out[L:] = x[:-L]
        return out

    if "kd" in groups:
        for L in lags:
            cols += [lag(k, L), lag(d, L)]
            names += [f"k_{L}", f"d_{L}"]
    if "macd" in groups:
        for L in lags:
            cols += [lag(dif, L), lag(osc, L)]
            names += [f"dif_{L}", f"osc_{L}"]
    if "slow" in groups:
        for L in slow_lags:
            cols += [lag(kw, L), lag(dw, L), lag(difw, L), lag(oscw, L)]
            names += [f"kw_{L}", f"dw_{L}", f"difw_{L}", f"oscw_{L}"]
    return np.column_stack(cols).astype(np.float32), names


class MLP(torch.nn.Module):
    def __init__(self, n_in, hidden=(16, 8)):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_ensemble(X, y, w=None, hidden=(16, 8), seeds=(0, 1, 2), epochs=20, batch=512, lr=2e-3, wd=1e-4, seed_base=0):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.astype(np.float32))
    wt = torch.from_numpy((np.ones(len(y)) if w is None else w).astype(np.float32))
    models = []
    for s in seeds:
        g = torch.Generator().manual_seed(1000 * seed_base + s)
        torch.manual_seed(1000 * seed_base + s)
        net = MLP(X.shape[1], hidden)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
        n = len(yt)
        for _ in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch):
                idx = perm[i : i + batch]
                opt.zero_grad()
                loss = (lossf(net(Xt[idx]), yt[idx]) * wt[idx]).sum() / wt[idx].sum()
                loss.backward()
                opt.step()
        models.append({k: v.detach().numpy().copy() for k, v in net.state_dict().items()})
    return models


def predict_ensemble(models, X):
    out = np.zeros(len(X))
    for sd in models:
        h = X.astype(np.float64)
        keys = sorted({k.rsplit(".", 1)[0] for k in sd}, key=lambda s: int(s.split(".")[1]))
        for j, key in enumerate(keys):
            h = h @ sd[f"{key}.weight"].T.astype(np.float64) + sd[f"{key}.bias"].astype(np.float64)
            if j < len(keys) - 1:
                h = np.maximum(h, 0.0)
        out += 1.0 / (1.0 + np.exp(-h[:, 0]))
    return out / len(models)


def fwd_label(ind: pd.DataFrame, horizon: int):
    o = ind["open"].to_numpy()
    n = len(o)
    ret = np.full(n, np.nan)
    end = np.arange(n) + 1 + horizon
    ok = end < n
    ret[ok] = o[end[ok]] / o[np.arange(n)[ok] + 1] - 1
    return ret, end


def standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return mu.astype(np.float32), sd.astype(np.float32)


def standardize_apply(X, mu, sd, clip=5.0):
    Z = (X - mu) / sd
    return np.clip(np.nan_to_num(Z, nan=0.0), -clip, clip).astype(np.float32)
