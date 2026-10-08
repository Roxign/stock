"""dl_market_state core: one MARKET-level model sets a common exposure g_t for all 50 stocks (E3 in
research/dl_literature.md).

Pipeline
  market_frame(data)      ~35 market-level features per TW trading day (prices of the 50 stocks in `data`, plus
                          stocklab.external market / chip / derivative / global series cut at data_end(data)) and the
                          labels: forward H-day log return and drawdown of the equal-weight (EW) open-to-open index.
  fit_model(kind, ...)    ridge / logistic / HistGradientBoosting / tiny MLP trained on a utility objective.
  walk_forward(...)       yearly expanding-window retraining (first model 2012, trained on 2008-07..2011 labels).
  purged_cv(...)          purged + embargoed blocked k-fold (stocklab/cv.py "purged" mode).
  exposure_path(...)      raw exposure -> EMA smoothing -> 0.25-step quantisation with hysteresis -> floor.

Timing: the feature row of bar t uses closes up to t and external data published before t+1's open (loaders'
default at_close=False, end=data_end(data)); the label of bar t starts at t+1's open (the fill) and ends at the open
H bars later. A training sample is usable for a model from the first bar after its label is fully known.
"""

import hashlib
import math

import numpy as np
import pandas as pd

from stocklab import external as ext

# ------------------------------------------------------------------------------------------------ constants
FIRST_YEAR = 2012          # first walk-forward model (trained on labels known by 2011-12-31)
TRAIN_START = "2008-07-01"  # first feature row used for training (longest look-back is 120 bars)
EMBARGO = 5                # extra bars after a CV fold that are kept out of training
VAL_BARS = 252             # early-stopping validation tail of each MLP training window
STEP = 0.25                # exposure grid
EXT_COLS = ["taiex_tr", "taiex_turnover", "mkt_foreign_net", "mkt_trust_net", "mkt_dealer_net", "mkt_margin_value",
            "mkt_margin_lots", "mkt_short_lots", "pc_ratio_oi", "pc_ratio_vol", "fut_foreign_net_oi",
            "opt_foreign_call_net_oi", "opt_foreign_put_net_oi", "sox", "spx", "vix", "us10y", "us3m", "dxy",
            "usdtwd", "n225", "kospi"]

GROUPS = {
    "price": ["mom5", "mom20", "mom60", "mom120", "vol20", "volratio", "dd240", "breadth60", "breadth120",
              "breadth240", "disp20", "corr60", "turn", "tx_rel60"],
    "chips": ["fgn5", "fgn20", "trust20", "dealer20", "marg20", "marg_turn", "short_ratio"],
    "deriv": ["pc_oi", "pc_vol5"],
    "fut": ["fut_fgn", "fut_fgn_chg5", "opt_fgn", "has_fut"],  # TAIFEX institutional positions: only from 2018-06-05
    "global": ["sox1", "sox5", "sox20", "spx60", "vix", "vix_chg", "us10y_chg60", "term", "dxy20", "twd20",
               "n225_20", "kospi_20"],
}
FEATURE_SETS = {
    "all": ("price", "chips", "deriv", "global"),
    "all_fut": ("price", "chips", "deriv", "global", "fut"),
    "price": ("price",),
    "price_chips": ("price", "chips", "deriv"),
    "price_global": ("price", "global"),
}


def feature_names(fset="all"):
    return [n for g in FEATURE_SETS[fset] for n in GROUPS[g]]


# ------------------------------------------------------------------------------------------------ features / labels

def _lr(s, k):
    return np.log(s / s.shift(k))


def _z(s, n=250, minp=120):
    """Causal rolling z-score: removes the slow drift of level series (margin/turnover, P/C ratio, term spread)."""
    return (s - s.rolling(n, min_periods=minp).mean()) / s.rolling(n, min_periods=minp).std()


def _panel(data, field, idx):
    return pd.DataFrame({c: df[field] for c, df in data.items()}).reindex(idx)


def market_frame(data, horizons=(20, 60)):
    """Return (features, aux). aux columns: R{H} forward log return of the EW open index from t+1's open to the open H
    bars later, DD{H} its worst point within those H bars, known{H} the date that label is fully known (NaT if not
    yet), sig60 the EW 60-bar daily volatility, n the number of stocks trading."""
    idx = pd.DatetimeIndex(sorted(set().union(*[df.index for df in data.values()])))
    end = ext.data_end(data)
    C, O = _panel(data, "close", idx), _panel(data, "open", idx)
    r = C / C.shift(1) - 1
    ro = O / O.shift(1) - 1
    rew = r.mean(axis=1).fillna(0.0)
    P = (1 + rew).cumprod()
    lr = np.log1p(rew)
    sig20 = lr.rolling(20, min_periods=15).std()
    sig60 = lr.rolling(60, min_periods=40).std()
    sig120 = lr.rolling(120, min_periods=60).std()
    F = pd.DataFrame(index=idx)
    for k in (5, 20, 60, 120):
        F[f"mom{k}"] = _lr(P, k) / (sig60 * math.sqrt(k))
    F["vol20"] = np.log(sig20 * math.sqrt(252))
    F["volratio"] = np.log(sig20 / sig120)
    F["dd240"] = np.log(P / P.rolling(240, min_periods=60).max())
    for n in (60, 120, 240):
        m = C.rolling(n, min_periods=min(n, 120)).mean()
        valid = C.notna() & m.notna()
        F[f"breadth{n}"] = ((C > m) & valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
    F["disp20"] = np.log(r.std(axis=1).rolling(20, min_periods=15).mean())
    var_i = r.rolling(60, min_periods=40).var()
    n_i = var_i.notna().sum(axis=1)
    s_sum, v_sum = np.sqrt(var_i).sum(axis=1), var_i.sum(axis=1)
    var_ew = rew.rolling(60, min_periods=40).var()
    F["corr60"] = ((n_i ** 2 * var_ew - v_sum) / (s_sum ** 2 - v_sum)).clip(-1, 1)

    M = ext.load_market(EXT_COLS, index=idx, end=end)
    # taiex_turnover is 0 on 52 days (all of 2017-12, 2018-09, 2026-02: FinMind gaps) -> fill those days with the
    # traded value of the 50 stocks times the median market/50-stock ratio of the previous 120 valid days (causal)
    tv = M["taiex_turnover"].where(M["taiex_turnover"] > 0)
    v50 = (C * _panel(data, "volume", idx)).sum(axis=1, min_count=1)
    ratio = (tv / v50).rolling(120, min_periods=20).median().ffill()
    tv = tv.fillna(v50 * ratio)
    F["turn"] = np.log(tv.rolling(20, min_periods=10).mean() / tv.rolling(120, min_periods=60).mean())
    F["tx_rel60"] = _lr(M["taiex_tr"], 60) - _lr(P, 60)
    tv5, tv20 = tv.rolling(5, min_periods=3).sum(), tv.rolling(20, min_periods=10).sum()
    F["fgn5"] = M["mkt_foreign_net"].rolling(5, min_periods=3).sum() / tv5
    F["fgn20"] = M["mkt_foreign_net"].rolling(20, min_periods=10).sum() / tv20
    F["trust20"] = M["mkt_trust_net"].rolling(20, min_periods=10).sum() / tv20
    F["dealer20"] = M["mkt_dealer_net"].rolling(20, min_periods=10).sum() / tv20
    mv = M["mkt_margin_value"]
    F["marg20"] = _lr(mv, 20)
    F["marg_turn"] = _z(np.log(mv / tv.rolling(60, min_periods=30).mean()))
    F["short_ratio"] = _z(M["mkt_short_lots"] / M["mkt_margin_lots"])
    F["pc_oi"] = _z(np.log(M["pc_ratio_oi"]))
    F["pc_vol5"] = _z(np.log(M["pc_ratio_vol"].rolling(5, min_periods=3).mean()))
    fut = M["fut_foreign_net_oi"]
    F["fut_fgn"] = _z(fut, 250, 60)
    F["fut_fgn_chg5"] = (fut - fut.shift(5)) / 1e4
    F["opt_fgn"] = (M["opt_foreign_call_net_oi"] - M["opt_foreign_put_net_oi"]) / 1e4
    F["has_fut"] = F["fut_fgn"].notna().astype(float)   # missing-data mask; the values are imputed by the training mean
    F["sox1"], F["sox5"], F["sox20"] = _lr(M["sox"], 1), _lr(M["sox"], 5), _lr(M["sox"], 20)
    F["spx60"] = _lr(M["spx"], 60)
    F["vix"] = np.log(M["vix"])
    F["vix_chg"] = np.log(M["vix"] / M["vix"].rolling(20, min_periods=10).mean())
    F["us10y_chg60"] = M["us10y"] - M["us10y"].shift(60)
    F["term"] = _z(M["us10y"] - M["us3m"])
    F["dxy20"] = _lr(M["dxy"], 20)
    F["twd20"] = _lr(M["usdtwd"], 20)
    F["n225_20"], F["kospi_20"] = _lr(M["n225"], 20), _lr(M["kospi"], 20)
    F = F.replace([np.inf, -np.inf], np.nan)

    # labels on the EW open-to-open index (entry = t+1's open)
    EWO = (1 + ro.mean(axis=1).fillna(0.0)).cumprod().to_numpy()
    n = len(idx)
    aux = pd.DataFrame(index=idx)
    aux["sig60"] = sig60
    aux["n"] = C.notna().sum(axis=1)
    lo = np.log(EWO)
    for H in horizons:
        R = np.full(n, np.nan)
        D = np.full(n, np.nan)
        known = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        for t in range(n - 1 - H):
            path = lo[t + 2 : t + 2 + H] - lo[t + 1]
            R[t] = path[-1]
            D[t] = min(path.min(), 0.0)
            known[t] = idx.values[t + 1 + H]
        aux[f"R{H}"], aux[f"DD{H}"], aux[f"known{H}"] = R, D, known
    return F, aux


def fingerprint(data):
    h = hashlib.md5()
    for code in sorted(data):
        df = data[code]
        h.update(code.encode())
        h.update(np.ascontiguousarray(df[["open", "close"]].to_numpy(np.float64)).tobytes())
        h.update(np.ascontiguousarray(df.index.values.astype("int64")).tobytes())
    return h.hexdigest()


_FRAME_MEMO: dict = {}


def cached_frame(data):
    key = fingerprint(data)
    if key not in _FRAME_MEMO:
        _FRAME_MEMO.clear()
        _FRAME_MEMO[key] = market_frame(data)
    return _FRAME_MEMO[key]


# ------------------------------------------------------------------------------------------------ models

def standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return mu, sd


def standardize_apply(X, mu, sd, clip=5.0):
    Z = np.clip((X - mu) / sd, -clip, clip)
    return np.nan_to_num(Z, nan=0.0)      # missing input = training mean


class Model:
    """Fitted model mapping standardized features (+ the bar's EW volatility) to a raw exposure in [0, 1]."""

    def __init__(self, kind, cfg):
        self.kind, self.cfg = kind, cfg

    # Campbell & Thompson style: w = (mu_bar + s (mu_hat - mu_bar)) / (gamma * second moment), clipped to [0, 1]
    def _ct(self, mu_hat, sig):
        c = self.cfg
        mu = self.mu_bar + c.get("shrink", 1.0) * (mu_hat - self.mu_bar)
        if c.get("risk", "const") == "cond":
            m2 = c["H"] * sig ** 2 + self.mu_bar ** 2
        else:
            m2 = np.full(len(mu), self.m2_bar)
        return np.clip(mu / (c.get("gamma", 3.0) * m2), 0.0, 1.0)

    def predict(self, Z, sig):
        k = self.kind
        if k == "ridge":
            return self._ct(Z @ self.beta + self.b0, sig)
        if k == "gbdt":
            return self._ct(self.est.predict(Z), sig)
        if k == "logit":
            p = 1.0 / (1.0 + np.exp(-(Z @ self.beta + self.b0)))
            return self._ct(self.mu_bar + self.slope * (p - self.p_bar), sig)
        if k == "mlp":
            return np.mean([_mlp_forward(layers, Z) for layers in self.nets], axis=0)
        if k == "const":
            return np.full(len(Z), self.cfg.get("level", 1.0))
        raise ValueError(k)

    def score(self, Z):
        """Unmapped prediction (for IC diagnostics)."""
        k = self.kind
        if k in ("ridge", "logit"):
            return Z @ self.beta + self.b0
        if k == "gbdt":
            return self.est.predict(Z)
        if k == "mlp":
            return np.mean([_mlp_logit(layers, Z) for layers in self.nets], axis=0)
        return np.zeros(len(Z))


def _ridge(Z, y, alpha):
    n, d = Z.shape
    zm, ym = Z.mean(axis=0), y.mean()
    Zc = Z - zm
    beta = np.linalg.solve(Zc.T @ Zc + alpha * n * np.eye(d), Zc.T @ (y - ym))
    return beta, ym - zm @ beta


def _logit(Z, yb, C, iters=200):
    """L2 logistic regression by Newton steps; penalty 1/(C n) per coefficient (sklearn-like scaling)."""
    n, d = Z.shape
    X = np.c_[np.ones(n), Z]
    w = np.zeros(d + 1)
    lam = np.r_[0.0, np.full(d, 1.0 / C)]
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (p - yb) / n + lam * w / n
        Hm = (X * (p * (1 - p))[:, None]).T @ X / n + np.diag(lam) / n
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w[1:], w[0]


def fit_model(kind, cfg, Z, R, sig, seed_base=0, val_mask=None):
    """Z standardized features, R forward H-day log return labels, sig EW daily vol of each sample."""
    m = Model(kind, cfg)
    simple = np.expm1(R)
    m.mu_bar = float(R.mean())
    m.m2_bar = float(np.mean(R ** 2))
    if kind == "ridge":
        m.beta, m.b0 = _ridge(Z, R, cfg.get("alpha", 1.0))
    elif kind == "logit":
        yb = (R > cfg.get("thr", 0.0)).astype(float)
        m.beta, m.b0 = _logit(Z, yb, cfg.get("C", 0.01))
        p = 1.0 / (1.0 + np.exp(-(Z @ m.beta + m.b0)))
        m.p_bar = float(p.mean())
        # map probability to an expected return with the training-set regression of R on p (one slope)
        pc = p - p.mean()
        m.slope = float(pc @ (R - R.mean()) / max(pc @ pc, 1e-12))
    elif kind == "gbdt":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m.est = HistGradientBoostingRegressor(
            max_depth=cfg.get("depth", 2), max_iter=cfg.get("iters", 150), learning_rate=cfg.get("lr", 0.03),
            min_samples_leaf=cfg.get("leaf", 200), l2_regularization=cfg.get("l2", 1.0), early_stopping=False,
            random_state=seed_base).fit(Z, R)
    elif kind == "mlp":
        m.nets = train_mlp(Z, simple, cfg, seed_base, val_mask)
    elif kind == "const":
        pass
    else:
        raise ValueError(kind)
    return m


# ------------------------------------------------------------------------------------------------ tiny MLP, utility loss

def _mlp_logit(layers, Z):
    h = Z.astype(np.float64)
    for j, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if j < len(layers) - 1:
            h = np.tanh(h)
    return h[:, 0]


def _mlp_forward(layers, Z):
    return 1.0 / (1.0 + np.exp(-_mlp_logit(layers, Z)))


def n_params(n_in, hidden):
    p, prev = 0, n_in
    for h in list(hidden) + [1]:
        p += prev * h + h
        prev = h
    return p


def train_mlp(Z, R, cfg, seed_base, val_mask=None):
    """Exposure net g = sigmoid(f(x)); loss = -mean(g R - gamma/2 g^2 R^2) (+ weight decay), R = forward simple return.
    Per seed: find the best epoch on the purged validation tail (val_mask), then refit on all samples for that many
    epochs. Deterministic on CPU."""
    import torch

    torch.set_num_threads(2)
    hidden = cfg.get("hidden", (16,))
    gamma, epochs, lr = cfg.get("gamma", 3.0), cfg.get("epochs", 150), cfg.get("lr", 3e-3)
    wd, noise, batch = cfg.get("wd", 1e-2), cfg.get("noise", 0.1), cfg.get("batch", 256)
    seeds = cfg.get("seeds", (0, 1, 2, 3, 4))
    bias0 = cfg.get("bias0", 1.0)
    if cfg.get("prior_init"):
        # shrink toward the unconditional optimum: an untrained net outputs g0 = E[R] / (gamma E[R^2]) of the training set
        g0 = float(np.clip(np.mean(R) / (gamma * np.mean(R ** 2)), 0.05, 0.95))
        bias0 = math.log(g0 / (1 - g0))
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)

    def make(seed):
        torch.manual_seed(seed)
        mods, d = [], Z.shape[1]
        for h in hidden:
            mods += [torch.nn.Linear(d, h), torch.nn.Tanh()]
            d = h
        mods.append(torch.nn.Linear(d, 1))
        net = torch.nn.Sequential(*mods)
        with torch.no_grad():           # start near the unconditional optimum g ~ sigmoid(b)
            net[-1].weight.mul_(0.1)
            net[-1].bias.fill_(bias0)
        return net

    def utility(net, X, y):
        g = torch.sigmoid(net(X)[:, 0])
        return (g * y - 0.5 * gamma * g * g * y * y).mean()

    def run(X, y, seed, n_ep, Xv=None, yv=None):
        net = make(seed)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        gen = torch.Generator().manual_seed(seed)
        hist = []
        n = len(y)
        for _ in range(n_ep):
            net.train()
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, batch):
                b = perm[i : i + batch]
                xb = X[b]
                if noise:
                    xb = xb + noise * torch.randn(xb.shape, generator=gen)
                opt.zero_grad()
                loss = -utility(net, xb, y[b])
                loss.backward()
                opt.step()
            if Xv is not None:
                net.eval()
                with torch.no_grad():
                    hist.append(float(utility(net, Xv, yv)))
        return net, hist

    try:
        X = torch.tensor(Z, dtype=torch.float32)
        y = torch.tensor(R, dtype=torch.float32)
        nets = []
        for s in seeds:
            seed = 7919 * seed_base + s
            n_ep = epochs
            if val_mask is not None and val_mask.sum() > 50 and (~val_mask).sum() > 200:
                tr = torch.tensor(~val_mask)
                va = torch.tensor(val_mask)
                _, hist = run(X[tr], y[tr], seed, epochs, X[va], y[va])
                n_ep = int(np.argmax(hist)) + 1
            net, _ = run(X, y, seed, n_ep)
            lin = [mm for mm in net if isinstance(mm, torch.nn.Linear)]
            nets.append([(mm.weight.detach().numpy().astype(np.float64).copy(),
                          mm.bias.detach().numpy().astype(np.float64).copy()) for mm in lin])
        return nets
    finally:
        torch.use_deterministic_algorithms(prev)


# ------------------------------------------------------------------------------------------------ training sets

def _design(F, aux, fset, H):
    names = feature_names(fset)
    X = F[names].to_numpy(np.float64)
    R = aux[f"R{H}"].to_numpy()
    known = aux[f"known{H}"].to_numpy()
    dates = F.index.values
    # features of the long-window group must exist; missing derivative-position data is allowed (imputed + masked)
    core = [i for i, n in enumerate(names) if n not in GROUPS["fut"]]
    ok_feat = ~np.isnan(X[:, core]).any(axis=1)
    trainable = ok_feat & ~np.isnan(R) & (dates >= np.datetime64(TRAIN_START))
    return names, X, R, known, dates, trainable


def _fit_on(kind, cfg, X, R, sig, rows, seed_base):
    mu, sd = standardize_fit(X[rows])
    Z = standardize_apply(X, mu, sd)
    val = None
    if kind == "mlp":
        idx = np.flatnonzero(rows)
        H = cfg["H"]
        # validation tail = last VAL_BARS training rows; purge the H + EMBARGO rows before it from the fit part
        vstart = idx[-VAL_BARS] if len(idx) > VAL_BARS + 300 else None
        if vstart is not None:
            sub = np.zeros(len(idx), bool)
            sub[np.searchsorted(idx, vstart):] = True
            keep = (idx >= vstart) | (idx < vstart - H - EMBARGO)
            idx, sub = idx[keep], sub[keep]
            val = sub
        rows_idx = idx
    else:
        rows_idx = np.flatnonzero(rows)
    model = fit_model(kind, cfg, Z[rows_idx], R[rows_idx], sig[rows_idx], seed_base, val)
    return model, Z


def walk_forward(F, aux, kind, cfg, fset="all", first_year=FIRST_YEAR, last_train=None, fallback=1.0):
    """Raw exposure per bar from yearly retrained models (decision year Y uses labels known before Y-01-01).
    Bars before first_year get `fallback`. last_train: stop retraining after this year (dev runs end at 2020)."""
    H = cfg["H"]
    names, X, R, known, dates, trainable = _design(F, aux, fset, H)
    sig = aux["sig60"].to_numpy()
    years = pd.DatetimeIndex(dates).year
    raw = np.full(len(dates), np.nan)
    score = np.full(len(dates), np.nan)
    raw[years < first_year] = fallback
    models = {}
    last_year = years.max() if last_train is None else min(years.max(), last_train)
    for Y in range(first_year, last_year + 1):
        cut = np.datetime64(f"{Y}-01-01")
        rows = trainable & (known < cut)
        model, Z = _fit_on(kind, cfg, X, R, sig, rows, Y)
        m = years == Y
        raw[m] = model.predict(Z[m], sig[m])
        score[m] = model.score(Z[m])
        models[Y] = model
    return pd.Series(raw, index=F.index), pd.Series(score, index=F.index), models


def purged_cv(F, aux, kind, cfg, folds, fset="all", train_end=None):
    """Raw exposure per bar where each fold's DECISION bars (fill = next bar inside the fold) come from a model trained
    without that fold: samples whose [t, label known] overlaps the fold are purged and the H + EMBARGO bars after the
    fold are embargoed. train_end (dev only): also drop samples whose label is known after this date.
    Returns a dict fold -> full raw path of that fold's model (the caller smooths it and keeps the fold's bars)."""
    H = cfg["H"]
    names, X, R, known, dates, trainable = _design(F, aux, fset, H)
    sig = aux["sig60"].to_numpy()
    out = {}
    for f, (start, end) in folds.items():
        fs = np.datetime64(pd.Timestamp(start))
        fe = dates[-1] if end is None else np.datetime64(pd.Timestamp(end))
        overlap = (dates <= fe) & (known >= fs)
        j = np.searchsorted(dates, fe, side="right")
        emb = np.zeros(len(dates), bool)
        emb[j : j + H + EMBARGO] = True
        rows = trainable & ~overlap & ~emb
        if train_end is not None:
            rows &= known <= np.datetime64(pd.Timestamp(train_end))
        model, Z = _fit_on(kind, cfg, X, R, sig, rows, pd.Timestamp(start).year)
        out[f] = pd.Series(model.predict(Z, sig), index=F.index)
    return out


# ------------------------------------------------------------------------------------------------ exposure mapping

def quantise(raw, step=STEP, band=0.75):
    out = np.empty(len(raw))
    cur = np.nan
    for i, x in enumerate(raw):
        if np.isnan(x):
            out[i] = cur
            continue
        if np.isnan(cur) or abs(x - cur) >= band * step:
            cur = round(x / step) * step
        out[i] = cur
    return out


def exposure_path(raw, smooth=10, floor=0.0, step=STEP, band=0.75):
    """raw exposure in [0, 1] -> EMA(smooth) -> floor + (1 - floor) * x -> quantised to `step` with hysteresis."""
    x = raw.ffill().fillna(1.0)
    if smooth and smooth > 1:
        x = x.ewm(span=smooth, adjust=False).mean()
    x = floor + (1.0 - floor) * x.clip(0.0, 1.0)
    q = quantise(x.to_numpy(), step, band)
    return pd.Series(np.clip(q, floor, 1.0), index=raw.index)


def cv_exposure(paths, folds, index, **kw):
    """Assemble fold-wise exposure: each bar's value comes from the model of the fold that contains its fill date."""
    dates = index.values
    fill = np.r_[dates[1:], dates[-1:]]
    g = pd.Series(np.nan, index=index)
    for f, (start, end) in folds.items():
        fs = np.datetime64(pd.Timestamp(start))
        fe = dates[-1] if end is None else np.datetime64(pd.Timestamp(end))
        m = (fill >= fs) & (fill <= fe)
        g[m] = exposure_path(paths[f], **kw).to_numpy()[m]
    return g.ffill().fillna(1.0)


def to_positions(g, data, mode="common", trend=None):
    """Common exposure g (indexed by the union calendar) -> per-stock target exposure.
    mode common: w_i = g;  trend: w_i = g * (0.5 + 0.5 trend_i)  (trend: dict code -> 0/1 Series)."""
    out = {}
    for c, df in data.items():
        gi = g.reindex(df.index).ffill().fillna(1.0)
        if mode == "trend":
            gi = gi * (0.5 + 0.5 * trend[c].reindex(df.index).fillna(0.0))
        out[c] = gi.clip(0.0, 1.0)
    return out
