"""E1: cost-aware direct position network (Deep Momentum Network style, long-only). See RESEARCH.md.

Pipeline
  prepare(data)        per-stock features (price/volume only) + market context (equal-weight index of the 50 plus
                       TAIEX, SOX, Nasdaq, TSM ADR, VIX, USD/TWD aligned by stocklab.external) and the next-period
                       return r_t = open[t+2] / open[t+1] - 1 that a target decided at t's close earns.
  walk_forward(...)    one ensemble per calendar year (expanding window, first model 2012), early-stopped on a purged
                       12-month validation tail; returns the raw ensemble position per bar.
  purged_cv(...)       same training per regime fold of stocklab.backtest.FOLDS without that fold (evaluation only).
  to_positions(...)    partial adjustment (EMA) + no-trade band quantised to discrete steps.

The network maps the feature vector of (stock, day) to w = floor + (1 - floor) * sigmoid(z). It is trained on windows
of WIN consecutive days of one stock; the loss is minus the annualised Sharpe ratio (or a mean-variance utility) of
the pooled daily returns of the batch, R_t = w_t r_t - BUY_FEE (dw_t)+ - SELL_FEE (dw_t)- - alpha |dw_t|,
dw_t = w_t - w_{t-1} (costs of the real engine, plus a turnover penalty). All seeds train at once as one stacked
network (independent parameters, shared mini-batches), deterministically.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
import torch

from stocklab import external as ext
from stocklab.backtest import BUY_FEE, SELL_FEE

VOL_SPAN = 60                      # EWM span of the daily-vol estimate used to normalise returns
RET_K = (1, 5, 20, 60, 120, 250)   # vol-normalised return horizons
MACD_PAIRS = ((8, 24), (16, 48), (32, 96))
CUM_K = (20, 60)                   # overnight / intraday cumulative-return horizons
MKT_COLS = ("taiex", "sox", "nasdaq", "tsm", "vix", "usdtwd")
FIRST_YEAR = 2012                  # first walk-forward model (trained on 2009-2010, validated on 2011)
FALLBACK = 1.0                     # position before the first model / without features: hold (= buy-and-hold)
VAL_DAYS = 365                     # validation tail (calendar days) for early stopping
EMBARGO = 5                        # bars between the training part and the validation tail / a CV fold
CV_EMBARGO = 21                    # bars embargoed after a CV fold

DEFAULT = dict(
    hidden=(32, 16), dropout=0.1, floor=0.0, objective="sharpe", gamma=2.0, alpha=0.001,
    win=63, batch=64, lr=1e-3, wd=1e-3, epochs=40, patience=8, seeds=(0, 1, 2, 3, 4), features="all",
)


# ---------------------------------------------------------------- features

def _vol(lr):
    return lr.ewm(span=VOL_SPAN, min_periods=20).std()


def _vn_ret(close, sig, k):
    return np.log(close / close.shift(k)) / (sig * math.sqrt(k))


def stock_features(df: pd.DataFrame) -> pd.DataFrame:
    c, o, h, v = df["close"], df["open"], df["high"], df["volume"]
    lr = np.log(c).diff()
    sig = _vol(lr)
    f = {}
    for k in RET_K:
        f[f"r{k}"] = _vn_ret(c, sig, k)
    for s, l in MACD_PAIRS:
        f[f"macd{s}_{l}"] = (c.ewm(span=s, adjust=False).mean() - c.ewm(span=l, adjust=False).mean()) / (c * sig)
    on = np.log(o / c.shift(1))
    intra = np.log(c / o)
    for k in CUM_K:
        f[f"on{k}"] = on.rolling(k).sum() / (sig * math.sqrt(k))
        f[f"id{k}"] = intra.rolling(k).sum() / (sig * math.sqrt(k))
    v20 = lr.rolling(20).std()
    v60 = lr.rolling(60).std()
    f["lvol20"] = np.log(v20 * math.sqrt(252))
    f["vratio"] = np.log(v20 / v60)
    f["hi52"] = c / h.rolling(250, min_periods=60).max() - 1
    f["volu"] = np.log(v.rolling(5).mean() / v.rolling(60).mean())
    out = pd.DataFrame(f, index=df.index)
    out["_sig"] = sig
    return out.replace([np.inf, -np.inf], np.nan)


def ew_close(data) -> pd.Series:
    """Equal-weight index of every stock trading on a date (mean of that day's close-to-close returns)."""
    rets = pd.DataFrame({c: df["close"].pct_change() for c, df in data.items()}).sort_index()
    return (1 + rets.mean(axis=1).fillna(0.0)).cumprod()


def market_features(data) -> pd.DataFrame:
    """Market context on the union of the stocks' dates; external series cut at the last date of `data`."""
    ew = ew_close(data)
    idx = ew.index
    m = ext.load_market(list(MKT_COLS), index=idx, end=ext.data_end(data))
    f = {}
    lr = np.log(ew).diff()
    sig = _vol(lr)
    for k in (5, 20, 60, 250):
        f[f"ew_r{k}"] = _vn_ret(ew, sig, k)
    f["ew_lvol20"] = np.log(lr.rolling(20).std() * math.sqrt(252))
    f["ew_hi52"] = ew / ew.rolling(250, min_periods=60).max() - 1
    f["ew_macd"] = (ew.ewm(span=16, adjust=False).mean() - ew.ewm(span=48, adjust=False).mean()) / (ew * sig)

    def vn(col, k):
        s = m[col].ffill()
        return _vn_ret(s, _vol(np.log(s).diff()), k)

    f["taiex_r20"] = vn("taiex", 20)
    f["sox_r1"] = vn("sox", 1)
    f["sox_r20"] = vn("sox", 20)
    f["nasdaq_r60"] = vn("nasdaq", 60)
    f["tsm_r1"] = vn("tsm", 1)
    vix = m["vix"].ffill()
    f["lvix"] = np.log(vix)
    f["vix_rel"] = np.log(vix / vix.rolling(60).mean())
    f["usdtwd_r20"] = vn("usdtwd", 20)
    return pd.DataFrame(f, index=idx).replace([np.inf, -np.inf], np.nan)


FEATURE_GROUPS = {
    "stock": [f"r{k}" for k in RET_K] + [f"macd{s}_{l}" for s, l in MACD_PAIRS]
             + [f"{p}{k}" for k in CUM_K for p in ("on", "id")] + ["lvol20", "vratio", "hi52", "volu"],
    "ew": ["ew_r5", "ew_r20", "ew_r60", "ew_r250", "ew_lvol20", "ew_hi52", "ew_macd"],
    "ext": ["taiex_r20", "sox_r1", "sox_r20", "nasdaq_r60", "tsm_r1", "lvix", "vix_rel", "usdtwd_r20"],
}
FEATURE_SETS = {
    "all": FEATURE_GROUPS["stock"] + FEATURE_GROUPS["ew"] + FEATURE_GROUPS["ext"],
    "price": FEATURE_GROUPS["stock"] + FEATURE_GROUPS["ew"],       # no external data
    "stock": FEATURE_GROUPS["stock"],                              # the stock's own prices only
    "momvol": ["r20", "r60", "r120", "r250", "lvol20", "ew_r60", "ew_lvol20"],   # vol-timed momentum replica
}


def feature_names(fset="all"):
    return list(FEATURE_SETS[fset])


def prepare(data, fset="all"):
    """List of per-stock dicts: code, index, X (n x d float64, NaN allowed), r (next-period return), d2 (date of the
    bar t+2 whose open closes the label window), sig (daily vol), core (feature rows usable at all)."""
    names = feature_names(fset)
    mkt = market_features(data)
    stocks = []
    for code in sorted(data):
        df = data[code]
        sf = stock_features(df)
        feats = sf.join(mkt.reindex(df.index))
        X = feats[names].to_numpy(np.float64)
        o = df["open"].to_numpy()
        n = len(df)
        r = np.full(n, np.nan)
        r[: n - 2] = o[2:] / o[1:-1] - 1
        d2 = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        d2[: n - 2] = df.index.values[2:]
        core = sf["_sig"].notna().to_numpy() & sf["r1"].notna().to_numpy()
        stocks.append(dict(code=code, index=df.index, X=X, r=r, d2=d2, sig=sf["_sig"].to_numpy(), core=core))
    return stocks, names


def fingerprint(data):
    h = hashlib.md5()
    for code in sorted(data):
        df = data[code]
        h.update(code.encode())
        h.update(np.ascontiguousarray(df[["open", "high", "low", "close", "volume"]].to_numpy(np.float64)).tobytes())
        h.update(np.ascontiguousarray(df.index.values.astype("int64")).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- model

class StackedMLP(torch.nn.Module):
    """S independent MLPs evaluated in one batched matmul: x (N, d) -> logits (S, N)."""

    def __init__(self, n_seeds, d, hidden, dropout, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        sizes = [d, *hidden, 1]
        self.W = torch.nn.ParameterList()
        self.b = torch.nn.ParameterList()
        for a, b in zip(sizes[:-1], sizes[1:]):
            bound = 1.0 / math.sqrt(a)
            self.W.append(torch.nn.Parameter((torch.rand(n_seeds, a, b, generator=g) * 2 - 1) * bound))
            self.b.append(torch.nn.Parameter((torch.rand(n_seeds, 1, b, generator=g) * 2 - 1) * bound))
        self.dropout = dropout

    def forward(self, x):
        h = x.unsqueeze(0).expand(self.W[0].shape[0], -1, -1)
        last = len(self.W) - 1
        for j, (W, b) in enumerate(zip(self.W, self.b)):
            h = torch.baddbmm(b, h, W)
            if j < last:
                h = torch.relu(h)
                if self.training and self.dropout > 0:
                    h = torch.nn.functional.dropout(h, self.dropout, True)
        return h[..., 0]


def n_params(d, hidden):
    sizes = [d, *hidden, 1]
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def _pos(logits, floor):
    return floor + (1 - floor) * torch.sigmoid(logits)


def _objective(R, mask, objective, gamma):
    """R, mask (S, M) -> per-seed loss (S,). Annualised."""
    n = mask.sum(dim=1).clamp(min=1)
    mu = (R * mask).sum(dim=1) / n
    var = (((R - mu[:, None]) ** 2) * mask).sum(dim=1) / n
    if objective == "sharpe":
        return -mu / torch.sqrt(var + 1e-10) * math.sqrt(252)
    if objective == "log":          # expected log growth (Kelly): the CAGR of the position net of costs
        return -(torch.log1p(R) * mask).sum(dim=1) / n * 252
    return -(mu - 0.5 * gamma * var) * 252


def _returns(w, r, pair, alpha):
    """w (S, B, L), r (B, L), pair (B, L) True where w[t-1] is the same stock's previous bar."""
    dw = torch.zeros_like(w)
    dw[..., 1:] = (w[..., 1:] - w[..., :-1]) * pair[:, 1:]
    cost = BUY_FEE * torch.relu(dw) + SELL_FEE * torch.relu(-dw) + alpha * dw.abs()
    return w * r - cost


def _rows(stocks, masks):
    """Concatenate the masked rows of every stock (in date order); brk marks rows that don't follow the previous row's
    bar of the same stock (stock boundary or gap), so no dw is charged across them."""
    X, r, brk, sig = [], [], [], []
    for s, m in zip(stocks, masks):
        i = np.flatnonzero(m)
        if not len(i):
            continue
        X.append(s["X"][i]); r.append(s["r"][i]); sig.append(s["sig"][i])
        b = np.ones(len(i), bool)
        b[1:] = np.diff(i) != 1
        brk.append(b)
    if not X:
        return None
    return dict(X=np.concatenate(X), r=np.concatenate(r), brk=np.concatenate(brk), sig=np.concatenate(sig))


def standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    return mu, np.where(sd > 1e-8, sd, 1.0)


def standardize_apply(X, mu, sd, clip=5.0):
    return np.clip(np.nan_to_num((X - mu) / sd, nan=0.0), -clip, clip).astype(np.float32)


def _seq_eval(net, Z, r, brk, cfg):
    """Per-seed validation loss on full sequences (no dropout)."""
    with torch.no_grad():
        net.eval()
        w = _pos(net(torch.from_numpy(Z)), cfg["floor"])          # (S, N)
        pair = torch.from_numpy(~brk).float()[None, :]
        R = _returns(w[:, None, :], torch.from_numpy(r).float()[None, :], pair, cfg["alpha"])[:, 0, :]
        return _objective(R, torch.ones_like(R), cfg["objective"], cfg["gamma"]).numpy()


def best_constant(r, cfg):
    """Constant position in [floor, 1] maximising the training objective (log growth or mean-variance utility)."""
    r = r[np.isfinite(r)]
    grid = np.linspace(cfg["floor"], 1.0, 101)
    if cfg["objective"] == "log":
        score = [np.mean(np.log1p(w * r)) for w in grid]
    else:
        score = [w * r.mean() - 0.5 * cfg["gamma"] * w * w * r.var() for w in grid]
    return float(grid[int(np.argmax(score))])


def train(tr, va, cfg, seed):
    """Train len(cfg['seeds']) stacked nets on rows `tr`, early-stop each on rows `va`. Returns (params, info)."""
    torch.set_num_threads(2)
    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        return _train(tr, va, cfg, seed)
    finally:
        torch.use_deterministic_algorithms(was)


def _train(tr, va, cfg, seed):
    S = len(cfg["seeds"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = StackedMLP(S, tr["Z"].shape[1], cfg["hidden"], cfg["dropout"], seed)
    if cfg.get("init_const") and cfg["objective"] != "sharpe":
        # start every net at the best CONSTANT position of the training rows (zero last layer, bias = its logit), so
        # the network departs from "always hold w*" only where that helps; early stopping can fall back to it.
        w0 = best_constant(tr["r"], cfg)
        u = min(max((w0 - cfg["floor"]) / (1 - cfg["floor"]), 0.02), 0.98)
        with torch.no_grad():
            net.W[-1].zero_()
            net.b[-1].fill_(math.log(u / (1 - u)))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    Zt = torch.from_numpy(tr["Z"])
    rt = torch.from_numpy(np.nan_to_num(tr["r"])).float()
    okt = torch.from_numpy(np.isfinite(tr["r"]))
    brk = tr["brk"]
    N, L = len(rt), cfg["win"]
    best = _seq_eval(net, va["Z"], va["r"], va["brk"], cfg)
    best_state = [p.detach().clone() for p in net.parameters()]
    best_ep = np.zeros(S, int)
    stale = np.zeros(S, int)
    hist = []
    for ep in range(1, cfg["epochs"] + 1):
        net.train()
        off = int(rng.integers(L))
        starts = np.arange(off, N - 1, L)
        rng.shuffle(starts)
        for k in range(0, len(starts), cfg["batch"]):
            st = starts[k : k + cfg["batch"]]
            idx = st[:, None] + np.arange(L)[None, :]
            valid = idx < N
            idx = np.minimum(idx, N - 1)
            it = torch.from_numpy(idx)
            pair = torch.from_numpy(valid & ~brk[idx]).float()
            pair[:, 0] = 0.0
            mask = torch.from_numpy(valid) & okt[it]
            logits = net(Zt[it.reshape(-1)]).reshape(S, *idx.shape)
            w = _pos(logits, cfg["floor"])
            R = _returns(w, rt[it], pair, cfg["alpha"])
            loss = _objective(R.reshape(S, -1), mask.reshape(1, -1).float().expand(S, -1), cfg["objective"], cfg["gamma"])
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
        v = _seq_eval(net, va["Z"], va["r"], va["brk"], cfg)
        hist.append(v)
        imp = v < best - 1e-6
        if imp.any():
            it_imp = torch.from_numpy(imp)
            with torch.no_grad():
                for bp, p in zip(best_state, net.parameters()):
                    bp[it_imp] = p[it_imp]
            best[imp] = v[imp]
            best_ep[imp] = ep
        stale = np.where(imp, 0, stale + 1)
        if (stale >= cfg["patience"]).all():
            break
    params = [p.numpy().astype(np.float64) for p in best_state]
    return params, dict(best_epoch=best_ep.tolist(), val_loss=best.tolist(), epochs_run=len(hist))


def predict(params, Z, floor):
    """Per-seed positions (S, N) in float64 numpy: same maths as StackedMLP.eval(). params = [W0, W1, .., b0, b1, ..]
    (the order of StackedMLP.parameters()); W_j (S, in, out), b_j (S, 1, out)."""
    k = len(params) // 2
    Ws, bs = params[:k], params[k:]
    h = np.broadcast_to(Z.astype(np.float64), (Ws[0].shape[0],) + Z.shape)
    for j in range(k):
        h = h @ Ws[j] + bs[j]
        if j < k - 1:
            h = np.maximum(h, 0.0)
    return floor + (1 - floor) / (1 + np.exp(-h[..., 0]))


# ---------------------------------------------------------------- walk-forward / purged CV

def _ts(x):
    return pd.Timestamp(x).to_datetime64().astype("datetime64[ns]")


def _trainable(s):
    return s["core"] & np.isfinite(s["r"]) & np.isfinite(s["X"]).all(axis=1)


def _split_block(stocks, base_masks, vstart=None, vend=None, embargo=EMBARGO):
    """Split training masks into (fit, val). val = rows dated in [vstart, vend) (default: the last VAL_DAYS of the
    training rows). Fit rows exclude the block, every row whose label window (t .. t+2 bars) reaches into it from
    `embargo` bars before it, and the `embargo` bars after it."""
    if vstart is None:
        last = max(s["index"].values[m].max() for s, m in zip(stocks, base_masks) if m.any())
        vstart = last - np.timedelta64(VAL_DAYS, "D")
    vend = np.datetime64("2262-01-01", "ns") if vend is None else vend
    fit, val = [], []
    for s, m in zip(stocks, base_masks):
        d = s["index"].values
        inb = (d >= vstart) & (d < vend)
        j0 = np.searchsorted(d, vstart)                  # first bar on/after vstart
        j1 = np.searchsorted(d, vend)                    # first bar on/after vend
        lo = d[max(j0 - embargo, 0)] if j0 < len(d) else vstart
        block = (s["d2"] >= lo) & (np.arange(len(d)) < j1 + embargo)
        fit.append(m & ~block); val.append(m & inb)
    return fit, val


def fit_model(stocks, masks, cfg, seed, vstart=None, vend=None):
    """Standardise on the fit rows, train with early stopping on the validation block (default: the last 12 months
    of the training rows). Returns model dict."""
    fit_m, val_m = _split_block(stocks, masks, vstart, vend)
    tr = _rows(stocks, fit_m)
    va = _rows(stocks, val_m)
    if cfg.get("vn_loss"):      # DMN-style: train on vol-scaled returns r * (2% / sigma_t) -> direction, not vol timing
        for d in (tr, va):
            d["r"] = d["r"] * (0.02 / d["sig"])
    mu, sd = standardize_fit(tr["X"])
    tr["Z"] = standardize_apply(tr["X"], mu, sd)
    va["Z"] = standardize_apply(va["X"], mu, sd)
    params, info = train(tr, va, cfg, seed)
    per_seed = predict(params, tr["Z"], cfg["floor"])
    ens = per_seed.mean(axis=0)
    return dict(params=params, mu=mu, sd=sd, n_fit=len(tr["r"]), n_val=len(va["r"]),
                ref=np.quantile(ens, np.linspace(0, 1, 1001)), ref_mean=float(ens.mean()),
                ref_mean_seed=per_seed.mean(axis=1), **info)


def apply_model(model, s, floor):
    """Per-seed positions (S, n) for every bar of stock s (NaN where the stock has no core features)."""
    Z = standardize_apply(s["X"], model["mu"], model["sd"])
    p = predict(model["params"], Z, floor)
    p[:, ~s["core"]] = np.nan
    return p


def _put(out, model, s, m, floor):
    """Write model outputs for stock s on bar mask m: per-seed positions, plus the ensemble mean's percentile among
    the model's own training-row outputs ('pct') and the ensemble mean divided by its training mean ('wn')."""
    p = apply_model(model, s, floor)[:, m]
    o = out[s["code"]]
    o["seeds"][:, m] = p
    w = p.mean(axis=0)
    o["pct"][m] = np.where(np.isnan(w), np.nan, np.searchsorted(model["ref"], w) / len(model["ref"]))
    o["wn"][m] = (w - floor) / max(model["ref_mean"] - floor, 1e-6)
    o["wns"][:, m] = (p - floor) / np.maximum(model["ref_mean_seed"][:, None] - floor, 1e-6)


def _new_out(stocks, S):
    return {s["code"]: {"seeds": np.full((S, len(s["index"])), np.nan), "pct": np.full(len(s["index"]), np.nan),
                        "wn": np.full(len(s["index"]), np.nan), "wns": np.full((S, len(s["index"])), np.nan)}
            for s in stocks}


def _frames(stocks, out):
    raw = {}
    for s in stocks:
        o = out[s["code"]]
        df = pd.DataFrame({f"s{i}": a for i, a in enumerate(o["seeds"])}, index=s["index"])
        df.insert(0, "w", o["seeds"].mean(axis=0))
        df.insert(1, "pct", o["pct"])
        df.insert(2, "wn", o["wn"])
        for i, a in enumerate(o["wns"]):
            df[f"wn{i}"] = a          # per-seed 'wn' (each seed's output / its own training mean)
        raw[s["code"]] = df
    return raw


def walk_forward(data, cfg=None, stocks=None, verbose=False):
    """Yearly expanding-window retraining. Returns (raw, models): raw[code] = DataFrame with 'w' (ensemble mean
    position, NaN before FIRST_YEAR) and one column per seed 's0'..; models[year] = model dict."""
    cfg = {**DEFAULT, **(cfg or {})}
    if stocks is None:
        stocks, _ = prepare(data, cfg["features"])
    last_year = max(s["index"][-1].year for s in stocks)
    out = _new_out(stocks, len(cfg["seeds"]))
    models = {}
    for Y in range(FIRST_YEAR, last_year + 1):
        cut = _ts(f"{Y}-01-01")
        nxt = _ts(f"{Y + 1}-01-01")
        masks = [_trainable(s) & (s["d2"] < cut) for s in stocks]
        model = fit_model(stocks, masks, cfg, seed=1000 * Y + cfg["seeds"][0])
        models[Y] = model
        if verbose:
            print(Y, model["n_fit"], model["n_val"], model["best_epoch"], np.round(model["val_loss"], 3))
        for s in stocks:
            d = s["index"].values
            m = (d >= cut) & (d < nxt)
            if m.any():
                _put(out, model, s, m, cfg["floor"])
    return _frames(stocks, out), models


def purged_cv(data, folds, cfg=None, stocks=None, verbose=False):
    """For each fold: train on every trainable row outside it (2009-.. warm-up included), purging rows whose label
    window [t, t+2] overlaps the fold and embargoing CV_EMBARGO bars after it; early stopping on the last 12 months
    of the remaining rows. The fold model is applied to every bar; its output is kept on the fold's decision bars
    (bars whose fill, the next bar's open, lies in the fold). Evaluation only -- not tradable."""
    cfg = {**DEFAULT, **(cfg or {})}
    if stocks is None:
        stocks, _ = prepare(data, cfg["features"])
    last = max(s["index"].values[-1] for s in stocks)
    by_fold, models = {}, {}
    for f, (start, end) in folds.items():
        fs, fe = _ts(start), (last if end is None else _ts(end))
        masks = []
        for s in stocks:
            d = s["index"].values
            ok = _trainable(s)
            overlap = (d <= fe) & (s["d2"] >= fs)
            emb = np.zeros(len(d), bool)
            j = np.searchsorted(d, fe, side="right")
            emb[j : j + CV_EMBARGO] = True
            masks.append(ok & ~overlap & ~emb)
        model = fit_model(stocks, masks, cfg, seed=1000 * pd.Timestamp(start).year + cfg["seeds"][0])
        models[f] = model
        if verbose:
            print(f, model["n_fit"], model["n_val"], model["best_epoch"])
        out = _new_out(stocks, len(cfg["seeds"]))
        for s in stocks:
            _put(out, model, s, np.ones(len(s["index"]), bool), cfg["floor"])
        raw = _frames(stocks, out)
        for s in stocks:
            d = s["index"].values
            fill = np.r_[d[1:], d[-1:]]               # the last bar counts as its own fill date
            raw[s["code"]]["in_fold"] = (fill >= fs) & (fill <= fe)
        by_fold[f] = raw
    return by_fold, models


def target(df, col, floor):
    """Raw target in [floor, 1] from a model-output frame: 'w' (the network's own position; also 's0'.. per seed),
    'pct' (percentile among the model's training-row outputs) or 'wn' (output / its training mean, capped at 1)."""
    if col == "pct":
        return floor + (1 - floor) * df["pct"]
    if col == "pct2":       # full above the training median, scaled down below it
        return floor + (1 - floor) * (2 * df["pct"]).clip(0.0, 1.0)
    if col.startswith("wn"):  # 'wn' (ensemble) or 'wn0'.. (one seed)
        return floor + (1 - floor) * df[col].clip(0.0, 1.0)
    return df[col]


def cv_to_positions(by_fold, col="w", **mapping):
    """Each fold's positions come from its own model's full-history path (smoothing and no-trade band warmed up by
    that model, never by a model trained on the fold), kept on the fold's decision bars; other bars get FALLBACK."""
    out = {}
    for f, raw in by_fold.items():
        for code, df in raw.items():
            p = to_positions(target(df, col, mapping["floor"]), **mapping)
            cur = out.setdefault(code, pd.Series(FALLBACK, index=df.index))
            cur[df["in_fold"].to_numpy()] = p[df["in_fold"].to_numpy()]
    return out


# ---------------------------------------------------------------- position mapping

def quantise(x, step, lo, hi, band):
    """No-trade band: move to the nearest multiple of `step` only when the target is >= band away from the current
    position. NaN input keeps the current position."""
    out = np.empty(len(x))
    cur = np.nan
    for i, v in enumerate(x):
        if not np.isnan(v):
            if np.isnan(cur) or abs(v - cur) >= band:
                cur = min(max(round(v / step) * step, lo), hi)
        out[i] = cur
    return out


def to_positions(raw_w: pd.Series, floor, span=10, step=0.25, band=None, fallback=FALLBACK):
    """raw ensemble position -> tradable exposure: before the first model the fallback; then partial adjustment
    (EMA with kappa = 2 / (span + 1)) and a no-trade band quantised to `step`."""
    band = 0.75 * step if band is None else band
    x = raw_w.fillna(fallback)
    if span and span > 1:
        x = x.ewm(span=span, adjust=False).mean()
    q = quantise(x.to_numpy(), step, floor, 1.0, band)
    return pd.Series(np.nan_to_num(q, nan=fallback), index=raw_w.index)
