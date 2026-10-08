"""E2 (research/dl_literature.md): low-turnover model on Taiwan-specific information.

Inputs per stock and bar t (all public before bar t+1's open, via stocklab.external with end=data_end(data)):
  月營收 (from the raw monthly series, available on revenue_avail(): first TW trading day >= the 10th of the next month
  + 1 trading day), 三大法人 net buy / volume (from 2012-05, NaN before), 外資持股 changes, 融資 / 融券 / 借券,
  PER / PBR / 殖利率 percentiles vs the stock's own past 5 years, price context and a few market-wide series.
Label: ABSOLUTE log return from the open of bar t+1 to the open of bar t+1+H (H = 20 bars ~ one revenue cycle).
Decisions: only on monthly revenue release days (one per month, the same calendar day for every stock), held until
the next one -> at most 12 trades per stock and year.
Models: HistGradientBoosting, a small MLP with a 4-d stock embedding, ridge -- trained on all 50 stocks pooled.
Walk-forward: one model per calendar year from FIRST_YEAR, expanding window, training samples whose label window
(+ embargo) ends before Jan 1; early stopping on a purged validation tail (last 12 months of the window).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from stocklab import external as ext
from stocklab.universe import CODES

H = 20                      # label horizon in bars (open t+1 -> open t+1+H)
EMBARGO = 5                 # extra bars between a training label's end and the first prediction / validation bar
FIRST_YEAR = 2012           # first model (trained on labels ending before 2012); earlier years: fallback
VAL_MONTHS = 12             # early-stopping tail of each training window
SEEDS = (0, 1, 2)
STRIDE = 3                  # training rows: every STRIDE-th bar of each stock (offset by seed -> bagging)
CODE_ID = {c: i for i, c in enumerate(sorted(CODES))}

PRICE = ["r20", "r60", "r120", "r250", "vol20", "hi52", "vratio"]
REVENUE = ["yoy1", "yoy3", "yoy12", "accel", "mom_sa", "high12", "high_all", "rev_age"]
FLOW = ["f5", "f20", "f60", "t20", "t60", "d20"]
HOLD = ["fr20", "fr60", "mg20", "mg_z", "short_ratio", "sbl20", "sbl_z"]
VALUE = ["per_pct", "pbr_pct", "dy_pct"]
MARKET = ["m_r20", "m_r60", "m_f20", "m_vix", "m_sox20", "m_margin20", "ew_r60"]
XSEC = ["cs_yoy3", "cs_r60", "cs_f20"]
GROUPS = {"price": PRICE, "revenue": REVENUE, "flow": FLOW, "hold": HOLD, "value": VALUE, "market": MARKET,
          "xsec": XSEC}
ALL = [f for g in GROUPS.values() for f in g]


# ---------------------------------------------------------------- features

def _logret(s, n):
    return np.log(s / s.shift(n))


def revenue_features(code, end):
    """Monthly revenue features computed on the raw monthly series (each month uses only months <= itself), indexed by
    the month's availability date. Revenue <= 0 (金控 net revenue can be negative) gives NaN growth."""
    raw = ext.stock_raw(code, "revenue")
    raw = raw[raw.index <= pd.Timestamp(end)]
    if raw.empty:
        return pd.DataFrame(columns=REVENUE[:-1] + ["month_id"])
    rev = raw["revenue"].where(raw["revenue"] > 0).to_numpy()
    per = pd.PeriodIndex.from_fields(year=raw["year"].astype(int), month=raw["month"].astype(int), freq="M")
    s = pd.Series(rev, index=per)
    s = s[~s.index.duplicated(keep="last")]
    full = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))   # gaps -> NaN, keeps lags honest
    lg = np.log(full)
    s3 = full.rolling(3, min_periods=3).sum()
    s12 = full.rolling(12, min_periods=12).sum()
    f = pd.DataFrame(index=full.index)
    f["yoy1"] = (lg - lg.shift(12)).clip(-1.5, 1.5)
    f["yoy3"] = np.log(s3 / s3.shift(12)).clip(-1.5, 1.5)
    f["yoy12"] = np.log(s12 / s12.shift(12)).clip(-1.5, 1.5)
    f["accel"] = f["yoy3"] - f["yoy12"]
    mom = lg - lg.shift(1)
    # seasonality-adjusted month-on-month: this month's mom minus the median mom of the same calendar month in the
    # previous 3 years (only past years)
    hist = pd.concat([mom.shift(12 * k) for k in (1, 2, 3)], axis=1)
    f["mom_sa"] = (mom - hist.median(axis=1, skipna=True).where(hist.notna().sum(axis=1) >= 2)).clip(-1.5, 1.5)
    f["high12"] = (lg - lg.shift(1).rolling(12, min_periods=10).max()).clip(-1.5, 1.0)
    f["high_all"] = (lg - lg.shift(1).expanding(min_periods=24).max()).clip(-2.0, 1.0)
    f = f.reindex(s.index)
    f["month_id"] = s.index.year * 12 + s.index.month
    avail = pd.Series(raw.index, index=per)
    avail = avail[~avail.index.duplicated(keep="last")]
    f.index = pd.DatetimeIndex(avail.reindex(f.index).to_numpy()).as_unit("ns")
    return f


def _pct_rank(s, window=1250, min_periods=250):
    return s.rolling(window, min_periods=min_periods).rank(pct=True)


def _zscore(s, window=250, min_periods=120):
    m = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return ((s - m) / sd.where(sd > 0)).clip(-4, 4)


def market_frame(end):
    cal = ext.calendar(end)
    m = ext.load_market(["taiex_tr", "mkt_foreign_net", "taiex_turnover", "vix", "sox", "mkt_margin_value"],
                        index=cal, end=end)
    out = pd.DataFrame(index=cal)
    out["m_r20"] = _logret(m["taiex_tr"], 20)
    out["m_r60"] = _logret(m["taiex_tr"], 60)
    out["m_f20"] = (m["mkt_foreign_net"].rolling(20, min_periods=15).sum()
                    / m["taiex_turnover"].rolling(20, min_periods=15).sum())
    out["m_vix"] = np.log(m["vix"])
    out["m_sox20"] = _logret(m["sox"], 20)
    out["m_margin20"] = _logret(m["mkt_margin_value"], 20)
    return out


def stock_frame(code, df, end, mkt):
    idx = df.index
    c, v = df["close"], df["volume"].astype(float)
    lr = np.log(c).diff()
    out = pd.DataFrame(index=idx)
    for n in (20, 60, 120, 250):
        out[f"r{n}"] = _logret(c, n)
    out["vol20"] = lr.rolling(20).std() * np.sqrt(252)
    out["hi52"] = np.log(c / c.rolling(250, min_periods=120).max())
    out["vratio"] = np.log(v.rolling(20).mean() / v.rolling(120, min_periods=60).mean())

    rf = revenue_features(code, end)
    if len(rf):
        al = ext.align(rf, idx, tol=ext.MONTHLY_TOL)
        for k in REVENUE[:-1]:
            out[k] = al[k].to_numpy()
        # months since the revenue month (0 right after release ~ 1.3 a month later); NaN if stale
        now_id = idx.year * 12 + idx.month
        out["rev_age"] = (now_id - al["month_id"].to_numpy()).astype(float)
    else:
        for k in REVENUE:
            out[k] = np.nan

    f = ext.load_stock_fields(code, ["foreign_net", "trust_net", "dealer_net", "foreign_ratio", "shares_issued",
                                     "margin_balance", "short_balance", "sbl_balance", "per", "pbr",
                                     "dividend_yield"], index=idx, end=end)
    vol_sum = {n: v.rolling(n, min_periods=int(0.6 * n)).sum() for n in (5, 20, 60)}
    for key, col, ns in (("f", "foreign_net", (5, 20, 60)), ("t", "trust_net", (20, 60)), ("d", "dealer_net", (20,))):
        for n in ns:
            out[f"{key}{n}"] = (f[col].rolling(n, min_periods=int(0.6 * n)).sum() / vol_sum[n]).clip(-1, 1)
    out["fr20"] = f["foreign_ratio"].diff(20)
    out["fr60"] = f["foreign_ratio"].diff(60)
    sh = f["shares_issued"]
    mb = f["margin_balance"]
    eps = 1e-4 * sh
    out["mg20"] = np.log((mb + eps) / (mb.shift(20) + eps.shift(20))).clip(-2, 2)
    out["mg_z"] = _zscore(mb / sh)
    out["short_ratio"] = (f["short_balance"] / mb.where(mb > 0)).clip(0, 2)
    sbl = f["sbl_balance"] / sh
    out["sbl20"] = (sbl - sbl.shift(20)) * 100.0
    out["sbl_z"] = _zscore(sbl)
    out["per_pct"] = _pct_rank(f["per"])
    out["pbr_pct"] = _pct_rank(f["pbr"])
    out["dy_pct"] = _pct_rank(f["dividend_yield"])

    mk = mkt.reindex(idx)
    for k in MARKET:
        if k in mk:
            out[k] = mk[k].to_numpy()
    return out


def decision_days(end):
    """Revenue release days (the availability date of each month's revenue): one decision per month."""
    cal = ext.calendar(end)
    months = pd.period_range(cal[0].to_period("M") - 1, cal[-1].to_period("M"), freq="M")
    days = [ext.revenue_avail(p.year, p.month, cal) for p in months]
    return pd.DatetimeIndex(sorted({d for d in days if pd.notna(d)}))


def prepare(data, h=H):
    """Per-stock arrays: X (bars x ALL), y (label, horizon h), known (label end + embargo date), dec (decision mask)."""
    end = ext.data_end(data)
    mkt = market_frame(end)
    frames = {c: stock_frame(c, df, end, mkt) for c, df in data.items()}
    # equal-weight 50-stock 60-day return and cross-sectional ranks (only stocks with a bar that day)
    for key, src in (("ew_r60", "r60"),):
        panel = pd.DataFrame({c: fr[src] for c, fr in frames.items()})
        ew = panel.mean(axis=1, skipna=True).where(panel.notna().sum(axis=1) >= 10)
        for c, fr in frames.items():
            fr[key] = ew.reindex(fr.index).to_numpy()
    for key, src in (("cs_yoy3", "yoy3"), ("cs_r60", "r60"), ("cs_f20", "f20")):
        panel = pd.DataFrame({c: fr[src] for c, fr in frames.items()})
        rk = panel.rank(axis=1, pct=True)
        for c, fr in frames.items():
            fr[key] = rk[c].reindex(fr.index).to_numpy()
    dec_days = decision_days(end)
    stocks = []
    for c, df in data.items():
        idx = df.index
        o = df["open"].to_numpy(float)
        n = len(idx)
        y = np.full(n, np.nan)
        known = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        if n > h + 1 + EMBARGO:
            y[: n - h - 1] = np.log(o[h + 1:] / o[1: n - h])
            known[: n - h - 1 - EMBARGO] = idx.values[h + 1 + EMBARGO:]
        # decision bar = first bar of the stock on/after each release day
        pos = np.unique(idx.searchsorted(dec_days[dec_days >= idx[0]]))
        dec = np.zeros(n, bool)
        dec[pos[pos < n]] = True
        X = frames[c][ALL].to_numpy(np.float64)
        stocks.append({"code": c, "index": idx, "X": X, "y": y, "known": known, "dec": dec,
                       "sid": CODE_ID.get(c, 0), "h": h})
    return stocks


def fingerprint(data):
    h = hashlib.md5()
    for code in sorted(data):
        df = data[code]
        h.update(code.encode())
        h.update(np.ascontiguousarray(df[["open", "close"]].to_numpy(np.float64)).tobytes())
        h.update(np.ascontiguousarray(df.index.values.astype("int64")).tobytes())
    return h.hexdigest()


_PREP: dict = {}


def cached_prepare(data, h=H):
    key = (fingerprint(data), h)
    if key not in _PREP:
        _PREP.clear()
        _PREP[key] = prepare(data, h)
    return _PREP[key]


# ---------------------------------------------------------------- models

def _usable(s, cols=None):
    """Rows with a label and the core inputs (price + revenue growth) present; flows/valuation may be NaN.
    The same rows for every column subset, so ablations differ only in their inputs."""
    return ~np.isnan(s["y"]) & ~np.isnat(s["known"]) & _predictable(s)


def _predictable(s):
    X = s["X"]
    core = [ALL.index(k) for k in ("r60", "vol20", "yoy3", "yoy12")]
    return ~np.isnan(X[:, core]).any(axis=1)


class Standardizer:
    """Train-only mean/sd, NaN -> 0 after scaling, plus missing indicators for the flow and valuation groups."""

    def __init__(self, X, cols=ALL):
        self.mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        self.mu = np.nan_to_num(self.mu)
        self.miss_cols = [list(cols).index(k) for k in ("f20", "per_pct") if k in cols]

    def __call__(self, X):
        Z = np.clip((X - self.mu) / self.sd, -5, 5)
        miss = np.isnan(X[:, self.miss_cols]).astype(np.float64)
        return np.concatenate([np.nan_to_num(Z), miss], axis=1).astype(np.float32)


def fit_gbdt(Xtr, ytr, Xva, yva, seed, params=None):
    from sklearn.ensemble import HistGradientBoostingRegressor

    p = dict(learning_rate=0.03, max_iter=300, max_leaf_nodes=15, min_samples_leaf=400, l2_regularization=1.0,
             max_features=0.7)
    p |= params or {}
    from threadpoolctl import threadpool_limits

    with threadpool_limits(2):
        m = HistGradientBoostingRegressor(loss="squared_error", early_stopping=False, random_state=seed, **p)
        m.fit(Xtr, ytr)
        if Xva is not None and len(yva) > 100:
            errs = [np.mean((yva - pv) ** 2) for pv in m.staged_predict(Xva)]
            best = max(int(np.argmin(errs)) + 1, 20)
            if best < p["max_iter"]:
                m = HistGradientBoostingRegressor(loss="squared_error", early_stopping=False, random_state=seed,
                                                  **(p | {"max_iter": best}))
                m.fit(Xtr, ytr)
    return m


class _Gbdt:
    """Wraps a fitted GBDT with the columns that were entirely missing in its training set (set to 0 so they are
    constant, e.g. 三大法人 before 2012-05 for the 2012 model)."""

    def __init__(self, Xtr, ytr, Xva, yva, seed, params):
        self.dead = np.isnan(Xtr).all(axis=0)
        self.m = fit_gbdt(self._fix(Xtr), ytr, None if Xva is None else self._fix(Xva), yva, seed, params)

    def _fix(self, X):
        if not self.dead.any():
            return X
        X = X.copy()
        X[:, self.dead] = 0.0
        return X

    def predict(self, X):
        from threadpoolctl import threadpool_limits

        with threadpool_limits(2):
            return self.m.predict(self._fix(X))


def fit_ridge(Ztr, ytr, alpha=100.0):
    from sklearn.linear_model import Ridge

    return Ridge(alpha=alpha).fit(Ztr, ytr)


HIDDEN = (32, 16)
EMB_DIM = 4


def mlp_n_params(n_in, hidden=HIDDEN, n_codes=len(CODES), emb=EMB_DIM):
    p, prev = n_codes * emb, n_in + emb
    for h in list(hidden) + [1]:
        p += prev * h + h
        prev = h
    return p


def fit_mlp(Ztr, ytr, sid_tr, Zva, yva, sid_va, seed, epochs=40, lr=1e-3, wd=1e-3, batch=512, patience=6,
            dropout=0.1, hidden=HIDDEN, emb=EMB_DIM, huber=0.05):
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    class Net(torch.nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.emb = torch.nn.Embedding(len(CODES), emb) if emb else None
            if emb:
                torch.nn.init.normal_(self.emb.weight, std=0.01)
            layers, prev = [], n_in + emb
            for h in hidden:
                layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
                prev = h
            layers.append(torch.nn.Linear(prev, 1))
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x, sid):
            if self.emb is not None:
                x = torch.cat([x, self.emb(sid)], dim=1)
            return self.net(x).squeeze(-1)

    net = Net(Ztr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.HuberLoss(delta=huber)
    Xt, yt, st = torch.from_numpy(Ztr), torch.from_numpy(ytr.astype(np.float32)), torch.from_numpy(sid_tr.astype(np.int64))
    have_val = Zva is not None and len(yva) > 100
    if have_val:
        Xv, yv, sv = torch.from_numpy(Zva), torch.from_numpy(yva.astype(np.float32)), torch.from_numpy(sid_va.astype(np.int64))
    best, best_state, bad = np.inf, None, 0
    n = len(yt)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            b = perm[i: i + batch]
            opt.zero_grad()
            loss = lossf(net(Xt[b], st[b]), yt[b])
            loss.backward()
            opt.step()
        if have_val:
            net.eval()
            with torch.no_grad():
                vl = float(lossf(net(Xv, sv), yv))
            if vl < best - 1e-7:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def predict_mlp(net, Z, sid):
    import torch

    with torch.no_grad():
        return net(torch.from_numpy(Z), torch.from_numpy(sid.astype(np.int64))).numpy().astype(np.float64)


# ---------------------------------------------------------------- training sets

def _rows(stocks, masks, cols, stride=1, offset=0):
    """Concatenate (X[:, cols], y, sid) over stocks for the given boolean masks, every `stride`-th bar."""
    ci = [ALL.index(k) for k in cols]
    Xs, ys, ss = [], [], []
    for s, m in zip(stocks, masks):
        m = m.copy()
        if stride > 1:
            keep = np.zeros(len(m), bool)
            keep[offset % stride:: stride] = True
            m &= keep
        Xs.append(s["X"][m][:, ci]); ys.append(s["y"][m]); ss.append(np.full(int(m.sum()), s["sid"]))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss)


def train_predict(stocks, train_masks, val_masks, pred_masks, kind="gbdt", cols=ALL, seeds=SEEDS, stride=STRIDE,
                  params=None):
    """Fit `kind` on the training rows (early stopping on the validation rows), predict pred_masks rows.
    Returns per-stock arrays of predictions (NaN outside pred_masks) and a dict describing the fitted ensemble."""
    ci = [ALL.index(k) for k in cols]
    preds = [np.full(len(s["index"]), np.nan) for s in stocks]
    params = dict(params or {})
    outs = []
    fitted = []
    for k, seed in enumerate(seeds):
        Xtr, ytr, str_ = _rows(stocks, train_masks, cols, stride, offset=k)
        Xva, yva, sva = _rows(stocks, val_masks, cols, 1) if val_masks is not None else (None, None, None)
        lo, hi = np.quantile(ytr, [0.01, 0.99])
        ytr = np.clip(ytr, lo, hi)
        if yva is not None:
            yva = np.clip(yva, lo, hi)
        if kind == "gbdt":
            m = _Gbdt(Xtr, ytr, Xva, yva, seed, params.get("gbdt"))
            fitted.append(("gbdt", m, None))
        elif kind == "ridge":
            st = Standardizer(Xtr, cols)
            m = fit_ridge(st(Xtr), ytr, params.get("alpha", 100.0))
            fitted.append(("ridge", m, st))
        elif kind == "mlp":
            st = Standardizer(Xtr, cols)
            m = fit_mlp(st(Xtr), ytr, str_, st(Xva) if Xva is not None else None, yva, sva, seed,
                        **params.get("mlp", {}))
            fitted.append(("mlp", m, st))
        else:
            raise ValueError(kind)
        if len(seeds) == 1 and kind == "ridge":
            break
    for s, pm, out in zip(stocks, pred_masks, preds):
        if not pm.any():
            continue
        X = s["X"][pm][:, ci]
        sid = np.full(len(X), s["sid"])
        acc = np.zeros(len(X))
        for kd, m, st in fitted:
            if kd == "gbdt":
                acc += m.predict(X)
            elif kd == "ridge":
                acc += m.predict(st(X))
            else:
                acc += predict_mlp(m, st(X), sid)
        out[pm] = acc / len(fitted)
    return preds, {"n_train": int(len(ytr)), "fitted": fitted, "cols": list(cols)}


def _ts(x):
    return np.datetime64(pd.Timestamp(x), "ns")


def walk_forward(stocks, kind="gbdt", cols=ALL, seeds=SEEDS, first_year=FIRST_YEAR, stride=STRIDE, params=None,
                 pred_all=False, keep_models=False):
    """One model per calendar year Y (expanding window, labels known before Jan 1 of Y), predicting Y's decision
    bars (all bars if pred_all). Returns (pred: dict code -> np.array, info: dict year -> fit info)."""
    last_year = max(pd.Timestamp(s["index"][-1]).year for s in stocks)
    pred = {s["code"]: np.full(len(s["index"]), np.nan) for s in stocks}
    info = {}
    usable = [_usable(s, cols) for s in stocks]
    okpred = [_predictable(s) for s in stocks]
    for Y in range(first_year, last_year + 1):
        cut = _ts(f"{Y}-01-01")
        nxt = _ts(f"{Y + 1}-01-01")
        vstart = _ts(pd.Timestamp(f"{Y}-01-01") - pd.DateOffset(months=VAL_MONTHS))
        trm, vam, pm = [], [], []
        for s, u, okp in zip(stocks, usable, okpred):
            d = s["index"].values
            inwin = u & (s["known"] < cut)
            trm.append(inwin & (s["known"] < vstart))            # train part ends (label+embargo) before the val tail
            vam.append(inwin & (d >= vstart))
            sel = (d >= cut) & (d < nxt) & okp
            pm.append(sel if pred_all else sel & s["dec"])
        if sum(int(m.sum()) for m in trm) < 3000:
            continue
        p, fi = train_predict(stocks, trm, vam, pm, kind, cols, seeds, stride, params)
        for s, arr in zip(stocks, p):
            m = ~np.isnan(arr)
            pred[s["code"]][m] = arr[m]
        info[Y] = fi if keep_models else {"n_train": fi["n_train"]}
    return pred, info


def purged_cv(stocks, folds, kind="gbdt", cols=ALL, seeds=SEEDS, stride=STRIDE, params=None, pred_all=False):
    """For each fold [start, end]: train on every usable sample whose [t, label end + embargo] does not overlap the fold
    and that is not within H + EMBARGO bars after it; early-stopping tail = the last 12 months of training data before
    the fold (or, for the first fold, after it). Predict the fold's decision bars plus the last decision bar before it
    (whose holding period runs into the fold). NOT tradable (later data trains earlier folds)."""
    last = max(s["index"].values[-1] for s in stocks)
    pred = {s["code"]: np.full(len(s["index"]), np.nan) for s in stocks}
    fold_of = {s["code"]: np.full(len(s["index"]), "", dtype=object) for s in stocks}
    usable = [_usable(s, cols) for s in stocks]
    okpred = [_predictable(s) for s in stocks]
    info = {}
    for f, (start, end) in folds.items():
        fs = _ts(start)
        fe = last if end is None else _ts(end)
        trm, pm = [], []
        for s, u, okp in zip(stocks, usable, okpred):
            d = s["index"].values
            overlap = (d <= fe) & (s["known"] >= fs)
            j = np.searchsorted(d, fe, side="right")
            emb = np.zeros(len(d), bool)
            emb[j: j + s.get("h", H) + EMBARGO] = True
            trm.append(u & ~overlap & ~emb)
            # decision bars whose holding period touches the fold: decisions inside the fold (fill inside it), plus the
            # last decision before the fold
            inf = (d >= fs) & (d <= fe)
            dec_idx = np.flatnonzero(s["dec"])
            sel = inf & s["dec"]
            before = dec_idx[d[dec_idx] < fs]
            if len(before):
                sel[before[-1]] = True
            if pred_all:
                sel = inf.copy()
                if len(before):
                    sel[before[-1]] = True
            pm.append(sel & okp)
        # validation tail: the last VAL_MONTHS of training data before the fold, purged from the remaining train rows
        tr_dates = np.concatenate([s["index"].values[m] for s, m in zip(stocks, trm)])
        before_fold = tr_dates[tr_dates < fs]
        if len(before_fold) > 0 and (pd.Timestamp(before_fold.max()) - pd.Timestamp(tr_dates.min())).days > 3 * 365:
            vend = before_fold.max()
            vstart = _ts(pd.Timestamp(vend) - pd.DateOffset(months=VAL_MONTHS))
            vam = [m & (s["index"].values >= vstart) & (s["index"].values <= vend) for s, m in zip(stocks, trm)]
            trm2 = [m & ~((s["index"].values <= vend) & (s["known"] >= vstart)) & ~v for s, m, v in zip(stocks, trm, vam)]
        else:   # first fold: use the 12 months right after the fold's embargo as the validation tail
            after = tr_dates[tr_dates > fe]
            vstart = after.min()
            vend = _ts(pd.Timestamp(vstart) + pd.DateOffset(months=VAL_MONTHS))
            vam = [m & (s["index"].values >= vstart) & (s["index"].values <= vend) for s, m in zip(stocks, trm)]
            j_after = [np.searchsorted(s["index"].values, vend, side="right") for s in stocks]
            trm2 = []
            for s, m, v, j in zip(stocks, trm, vam, j_after):
                mm = m & ~v & ~((s["index"].values <= vend) & (s["known"] >= vstart))
                mm[j: j + s.get("h", H) + EMBARGO] = False
                trm2.append(mm)
        p, fi = train_predict(stocks, trm2, vam, pm, kind, cols, seeds, stride, params)
        for s, arr, sel in zip(stocks, p, pm):
            m = ~np.isnan(arr)
            pred[s["code"]][m] = arr[m]
            fold_of[s["code"]][m] = f
        info[f] = {"n_train": fi["n_train"]}
    return pred, fold_of, info


# ---------------------------------------------------------------- positions

def hold_steps(pred, dec, index):
    """Predictions on decision bars carried forward until the next decision bar (NaN before the first)."""
    v = np.where(dec, np.where(np.isnan(pred), -1e9, pred), np.nan)
    s = pd.Series(v, index=index).ffill()
    return s.where(s > -1e8)


def steps_frame(stocks, pred):
    return pd.DataFrame({s["code"]: hold_steps(pred[s["code"]], s["dec"], s["index"]) for s in stocks})


def abs_positions(stocks, pred, floor=0.5, thr=0.0, fallback=1.0):
    """Per-stock exposure: 1 when the absolute forecast at the last revenue day is > thr, else `floor`;
    `fallback` before the stock's first forecast."""
    out = {}
    for s in stocks:
        st = hold_steps(pred[s["code"]], s["dec"], s["index"])
        out[s["code"]] = pd.Series(np.where(st.isna(), fallback, np.where(st > thr, 1.0, floor)), index=s["index"])
    return out


def topk_weights(stocks, pred, K=20, B=10, gate=None):
    """Cross-sectional portfolio: on each revenue day, keep held names whose rank is still <= K + B, then fill up to K
    names with the best-ranked ones (rank <= K); each held name gets 1/K of equity (rest cash). gate (optional
    DataFrame of bools, same shape) removes names regardless of rank (e.g. negative absolute forecast).
    Before the first forecast: equal weight over the stocks that have a bar."""
    P = steps_frame(stocks, pred)
    Pf = P.to_numpy()
    rk = P.rank(axis=1, ascending=False, method="first").to_numpy()
    G = None if gate is None else gate.reindex(index=P.index, columns=P.columns).fillna(False).to_numpy()
    filled = P.ffill().to_numpy()
    changed = np.r_[True, ~((filled[1:] == filled[:-1]) | (np.isnan(filled[1:]) & np.isnan(filled[:-1]))).all(axis=1)]
    held = np.zeros(P.shape[1], bool)
    W = np.zeros(P.shape)
    has = ~np.isnan(Pf).all(axis=1)
    for i in range(len(P)):
        if not has[i]:
            continue
        if changed[i]:
            r = rk[i]
            ok = ~np.isnan(r) if G is None else (~np.isnan(r) & G[i])
            new = held & ok & (r <= K + B)
            for j in np.argsort(np.where(np.isnan(r), 1e9, r)):
                if new.sum() >= K or np.isnan(r[j]) or r[j] > K:
                    break
                if ok[j]:
                    new[j] = True
            held = new
        W[i] = held / K
    W = pd.DataFrame(W, index=P.index, columns=P.columns)
    alive = pd.DataFrame({s["code"]: pd.Series(1.0, index=s["index"]) for s in stocks}).reindex(P.index)
    ew = alive.div(alive.notna().sum(axis=1), axis=0).fillna(0.0)
    W[~has] = ew[~has]
    return {s["code"]: W[s["code"]].reindex(s["index"]).fillna(0.0) for s in stocks}


def xs_rank(stocks, cols=None):
    """Gu-Kelly-Xiu style inputs: every stock-level column replaced by its cross-sectional percentile rank on the same
    date (mapped to [-0.5, 0.5]; NaN stays NaN), market columns unchanged. Same-date information only."""
    cols = cols or [k for k in ALL if k not in MARKET]
    out = [dict(s) for s in stocks]
    for k in cols:
        j = ALL.index(k)
        panel = pd.DataFrame({s["code"]: pd.Series(s["X"][:, j], index=s["index"]) for s in stocks})
        R = panel.rank(axis=1, pct=True) - 0.5
        R = R.where(panel.notna().sum(axis=1) >= 5)
        for s, o in zip(stocks, out):
            if o["X"] is s["X"]:
                o["X"] = s["X"].copy()
            o["X"][:, j] = R[s["code"]].reindex(s["index"]).to_numpy()
    return out
