"""KD + MACD features, small-MLP walk-forward training and position mapping for the kd_macd_dl family.

Everything here is causal:
  * features at bar t only use bars <= t (EMA / KD recursions seeded at each stock's first bar);
  * the model used for calendar year Y is trained only on samples whose label window ends before Jan 1 of Y
    (purge), normalised with statistics of that training set only;
  * training is deterministic (fixed seeds, fixed sample order, deterministic torch ops, 2 threads), so truncating
    the data at any date reproduces exactly the same positions before that date.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch

from stocklab.indicators import atr, macd, tw_kd

# ---------------------------------------------------------------- configuration (one config for all stocks)

SLOW = 5                      # "weekly-equivalent" KD / MACD: KD(45,15,15), MACD(60,130,45)
LAGS = (0, 1, 2, 3, 4)        # daily KD / MACD window fed to the network
SLOW_LAGS = (0, 5)            # slow KD / MACD: today and one week ago
HORIZON = 60                  # label horizon in bars (~3 months)
HIDDEN = (16, 8)              # MLP hidden layers
SEEDS = (0, 1, 2)             # ensemble of 3 nets
EPOCHS = 10
BATCH = 512
LR = 2e-3
WEIGHT_DECAY = 1e-4
WARM = 150                    # bars per stock skipped for training (slow EMA warm-up)
FIRST_YEAR = 2012             # first year with a model (trained on 2010-08..2011 labels); earlier years use fallback
SMOOTH = 10                   # EMA span applied to the ensemble probability before thresholding
Q_LO, Q_HI = 0.2, 0.4         # thresholds = these quantiles of the model's in-training predictions


# ---------------------------------------------------------------- indicators / features

def indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    kd = tw_kd(df)
    m = macd(c)
    kdw = tw_kd(df, 9 * SLOW, 3 * SLOW, 3 * SLOW)
    mw = macd(c, 12 * SLOW, 26 * SLOW, 9 * SLOW)
    return pd.DataFrame({
        "k": kd["k"], "d": kd["d"], "dif": m["dif"], "osc": m["osc"],
        "kw": kdw["k"], "dw": kdw["d"], "difw": mw["dif"], "oscw": mw["osc"],
        "atr": atr(df), "open": df["open"],
    }, index=df.index)


def feature_matrix(ind: pd.DataFrame):
    """(X float32 [n, 28], names). KD mapped to [-0.5, 0.5]; MACD lines divided by ATR(14) (scale-free)."""
    a = ind["atr"].to_numpy()
    a = np.where(a > 0, a, np.nan)
    base = {
        "k": ind["k"].to_numpy() / 100 - 0.5, "d": ind["d"].to_numpy() / 100 - 0.5,
        "dif": ind["dif"].to_numpy() / a, "osc": ind["osc"].to_numpy() / a,
        "kw": ind["kw"].to_numpy() / 100 - 0.5, "dw": ind["dw"].to_numpy() / 100 - 0.5,
        "difw": ind["difw"].to_numpy() / a, "oscw": ind["oscw"].to_numpy() / a,
    }

    def lag(x, L):
        if L == 0:
            return x
        out = np.full_like(x, np.nan)
        out[L:] = x[:-L]
        return out

    cols, names = [], []
    for L in LAGS:
        for f in ("k", "d"):
            cols.append(lag(base[f], L)); names.append(f"{f}_{L}")
    for L in LAGS:
        for f in ("dif", "osc"):
            cols.append(lag(base[f], L)); names.append(f"{f}_{L}")
    for L in SLOW_LAGS:
        for f in ("kw", "dw", "difw", "oscw"):
            cols.append(lag(base[f], L)); names.append(f"{f}_{L}")
    return np.column_stack(cols).astype(np.float32), names


def base_rule(ind: pd.DataFrame) -> np.ndarray:
    """Weekly-equivalent KD + MACD state machine: long when K>D AND OSC>0, flat when K<D AND OSC<0, else hold."""
    k, d, osc = ind["kw"].to_numpy(), ind["dw"].to_numpy(), ind["oscw"].to_numpy()
    pos = np.zeros(len(k))
    cur = 0.0
    for i in range(len(k)):
        if cur == 0.0 and k[i] > d[i] and osc[i] > 0:
            cur = 1.0
        elif cur == 1.0 and k[i] < d[i] and osc[i] < 0:
            cur = 0.0
        pos[i] = cur
    return pos


# ---------------------------------------------------------------- model

class MLP(torch.nn.Module):
    def __init__(self, n_in, hidden=HIDDEN):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def n_params(n_in, hidden=HIDDEN):
    p, prev = 0, n_in
    for h in list(hidden) + [1]:
        p += prev * h + h
        prev = h
    return p


def train_ensemble(X, y, seed_base):
    """Deterministic training of len(SEEDS) small MLPs (BCE loss, AdamW). Returns list of numpy layer lists."""
    torch.set_num_threads(2)
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        return _train(X, y, seed_base)
    finally:
        torch.use_deterministic_algorithms(was_deterministic)


def _train(X, y, seed_base):
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.astype(np.float32))
    lossf = torch.nn.BCEWithLogitsLoss()
    models = []
    for s in SEEDS:
        g = torch.Generator().manual_seed(1000 * seed_base + s)
        torch.manual_seed(1000 * seed_base + s)
        net = MLP(X.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        n = len(yt)
        for _ in range(EPOCHS):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, BATCH):
                idx = perm[i : i + BATCH]
                opt.zero_grad()
                loss = lossf(net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        lin = [m for m in net.net if isinstance(m, torch.nn.Linear)]
        models.append([(m.weight.detach().numpy().astype(np.float64).copy(), m.bias.detach().numpy().astype(np.float64).copy()) for m in lin])
    return models


def predict_ensemble(models, Z):
    """Mean sigmoid probability of the ensemble in float64 numpy (same maths as the PowerLanguage port)."""
    out = np.zeros(len(Z))
    for layers in models:
        h = Z.astype(np.float64)
        for j, (W, b) in enumerate(layers):
            h = h @ W.T + b
            if j < len(layers) - 1:
                h = np.maximum(h, 0.0)
        out += 1.0 / (1.0 + np.exp(-h[:, 0]))
    return out / len(models)


def standardize_fit(X):
    mu = X.mean(axis=0, dtype=np.float64)
    sd = X.std(axis=0, dtype=np.float64)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return mu, sd


def standardize_apply(X, mu, sd, clip=5.0):
    Z = (X.astype(np.float64) - mu) / sd
    return np.clip(np.nan_to_num(Z, nan=0.0), -clip, clip).astype(np.float32)


# ---------------------------------------------------------------- walk-forward

def _prepare(data):
    """Per-stock features + cross-sectional labels. Label for (stock, t): open(t+1+H)/open(t+1)-1 beats the median
    of the same quantity across all stocks with a label on that date. A label is 'known' from the date at which the
    LAST stock's window on that date has ended (so the cross-sectional median never peeks past the purge date)."""
    stocks = []
    for code in sorted(data):
        df = data[code]
        ind = indicator_frame(df)
        X, names = feature_matrix(ind)
        o = df["open"].to_numpy()
        n = len(df)
        t = np.arange(n)
        end = t + 1 + HORIZON
        ok = end < n
        ret = np.full(n, np.nan)
        ret[ok] = o[end[ok]] / o[t[ok] + 1] - 1
        end_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        end_date[ok] = df.index.values[end[ok]]
        stocks.append(dict(code=code, index=df.index, ind=ind, X=X, ret=ret, end_date=end_date))
    ret_panel = pd.concat({s["code"]: pd.Series(s["ret"], index=s["index"]) for s in stocks}, axis=1)
    end_panel = pd.concat({s["code"]: pd.Series(s["end_date"], index=s["index"]) for s in stocks}, axis=1)
    present = pd.concat({s["code"]: pd.Series(1, index=s["index"]) for s in stocks}, axis=1).notna().sum(axis=1)
    med = ret_panel.median(axis=1)
    known = end_panel.max(axis=1)              # all labels of that date are complete by this date
    known = known.where(ret_panel.notna().sum(axis=1) == present)  # a trading stock's window still open -> unknown
    for s in stocks:
        m = med.reindex(s["index"]).to_numpy()
        y = (s["ret"] > m).astype(np.float32)
        y[np.isnan(s["ret"]) | np.isnan(m)] = np.nan
        s["y"] = y
        s["known"] = known.reindex(s["index"]).to_numpy(dtype="datetime64[ns]")
    return stocks, names


def walk_forward(data):
    """Train one ensemble per calendar year (FIRST_YEAR .. last year in data) and predict that year.

    Returns (signals, models): signals[code] = DataFrame(p, ps, q_lo, q_hi) indexed like data[code] (NaN before
    FIRST_YEAR); models[year] = dict(models, mu, sd, q_lo, q_hi, n_train, train_end)."""
    stocks, names = _prepare(data)
    last_year = max(s["index"][-1].year for s in stocks)
    p_all = {s["code"]: np.full(len(s["index"]), np.nan) for s in stocks}
    qlo_all = {s["code"]: np.full(len(s["index"]), np.nan) for s in stocks}
    qhi_all = {s["code"]: np.full(len(s["index"]), np.nan) for s in stocks}
    models = {}
    for Y in range(FIRST_YEAR, last_year + 1):
        cut = np.datetime64(f"{Y}-01-01")
        nxt = np.datetime64(f"{Y + 1}-01-01")
        Xs, ys = [], []
        for s in stocks:
            m = (np.arange(len(s["y"])) >= WARM) & ~np.isnan(s["y"]) & (s["known"] < cut) & ~np.isnan(s["X"]).any(axis=1)
            Xs.append(s["X"][m]); ys.append(s["y"][m])
        Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
        if len(ytr) < 1000:
            continue
        mu, sd = standardize_fit(Xtr)
        Ztr = standardize_apply(Xtr, mu, sd)
        ens = train_ensemble(Ztr, ytr, seed_base=Y)
        ptr = predict_ensemble(ens, Ztr)
        q_lo, q_hi = np.quantile(ptr, [Q_LO, Q_HI])
        models[Y] = dict(models=ens, mu=mu, sd=sd, q_lo=float(q_lo), q_hi=float(q_hi), n_train=int(len(ytr)),
                         names=names)
        for s in stocks:
            dates = s["index"].values
            m = (dates >= cut) & (dates < nxt)
            if m.any():
                p_all[s["code"]][m] = predict_ensemble(ens, standardize_apply(s["X"][m], mu, sd))
                qlo_all[s["code"]][m] = q_lo
                qhi_all[s["code"]][m] = q_hi
    signals = {}
    for s in stocks:
        p = pd.Series(p_all[s["code"]], index=s["index"])
        ps = p.ewm(span=SMOOTH, adjust=False).mean().where(p.notna())
        signals[s["code"]] = pd.DataFrame({"p": p, "ps": ps, "q_lo": qlo_all[s["code"]], "q_hi": qhi_all[s["code"]],
                                           "base": base_rule(s["ind"])}, index=s["index"])
    return signals, models


_MEMO: dict = {}


def _fingerprint(data):
    h = hashlib.md5()
    for code in sorted(data):
        df = data[code]
        h.update(code.encode())
        h.update(np.ascontiguousarray(df[["open", "high", "low", "close"]].to_numpy(np.float64)).tobytes())
        h.update(np.ascontiguousarray(df.index.values.astype("int64")).tobytes())
    return h.hexdigest()


def cached_walk_forward(data):
    """walk_forward() memoised on the exact input data (both published variants share one training run)."""
    key = _fingerprint(data)
    if key not in _MEMO:
        _MEMO.clear()
        _MEMO[key] = walk_forward(data)
    return _MEMO[key]
