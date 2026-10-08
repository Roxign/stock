"""Trend-scanning / triple-barrier labels + a small causal TCN over raw daily sequences (dl_trend_labels).

Pipeline (research/dl_literature.md E5):
  * per-day input channels (all causal, bar t uses bars <= t): volatility-scaled close / overnight / intraday returns,
    candle shape (KLEN, KUP, KLOW), volume z-score, KDJ, MACD, distances to the 5/20/60/120/240-day MAs, volatility
    level, and three channels of an equal-weighted "market" built from every stock in `data` (multi-stock input);
    the network reads the last T=64 days of these plus the log price relative to the latest close;
  * labels at bar t (trade filled at the open of t+1):
      trend scanning (Lopez de Prado 2020): OLS of log price [open(t+1), close(t+1..t+L)] on time for
      L = 20, 30, ..., 120; the L with the largest |t-value| wins; label = sign, sample weight = min(|t|, 30);
      triple barrier: path close(t+1..t+80) / open(t+1), barriers +-1.5 * sigma20 * sqrt(80) -> up / timeout / down
      (the first research round used L = 10..60 and 40 bars / 1.0 sigma, see RESEARCH.md);
    the longest look-ahead is LABEL_END = 120 bars, so a sample is usable for training only once bar t+120+5
    (embargo) is before the training cut (`known`);
  * model (MODEL = "tcnx"): TCN (1x1 input conv, 4 residual causal conv layers with dilation 1/2/4/8, last-step +
    mean pooling) plus a one-layer branch on 39 tabular summary features, with a trend-scanning head (weighted BCE)
    and a triple-barrier head (3-class CE); 3 seeds, each trained on a different 1-in-3 subsample of the days,
    early stopping on a purged 12-month validation tail of the training window;
  * walk-forward: one ensemble per calendar year (first 2012), trained only on samples known before Jan 1;
    channel normalisation and the decision thresholds (quantiles of the ensemble's predictions on the validation
    tail) come from that training window only;
  * positions: state machine on the smoothed probabilities (see positions_from).

Everything is deterministic (fixed seeds and sample order, deterministic torch ops, 2 threads), so truncating the data
at any date reproduces exactly the same positions before that date. purged_cv() is the purged k-fold CV for
evaluate.py --cv and never feeds the published positions.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view as swv

from stocklab.indicators import macd, sma, tw_kdj

# ---------------------------------------------------------------- configuration
T = 64                           # days of history the network reads
TS_LS = tuple(range(20, 121, 10))  # trend-scanning look-ahead spans (research: 10..60 step 5 was weaker, RESEARCH.md)
TS_WCAP = 30.0                   # cap of the |t| sample weight
TS_WSCALE = 12.0                 # fixed divisor (~ mean capped |t|) so weights average ~1
TB_H, TB_K = 80, 1.5             # triple barrier: horizon and barrier width in sigma20 * sqrt(H) (research: 40 / 1.0)
LABEL_END = max(max(TS_LS), TB_H)
EMBARGO = 5
PURGE = LABEL_END + EMBARGO      # a sample at bar t is "known" at the date of bar t + 125
WARM = 130                       # bars per stock before the first sample (MA / KDJ / vol warm-up)
FIRST_YEAR = 2012                # first walk-forward model (trained on 2008-2011); 2010-2011 use FALLBACK
FALLBACK = 1.0                   # exposure before a stock has a model (2010-2011, first WARM bars of a new listing)
VAL_MONTHS = 12                  # early-stopping tail of each training window
SEEDS = (0, 1, 2)
STRIDE = 3                       # seed s trains on samples with (bar + s) % STRIDE == 0
HIDDEN = 16
DILATIONS = (1, 2, 4, 8)
KERNEL = 3
DROPOUT = 0.1
BATCH = 256
LR = 5e-4
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 12
PATIENCE = 3
TASK = "both"                    # loss: "ts" (trend scanning only), "tb" (triple barrier only), "both"
SMOOTH = 5                       # EMA span on the ensemble probabilities

MA_N = (5, 20, 60, 120, 240)
CHANNELS = ("ret", "on", "id", "klen", "kup", "klow", "vol", "k", "d", "j", "macd", "osc",
            *(f"ma{n}" for n in MA_N), "lsig", "m_ret", "m_on", "m_ma60")
DROP_CHANNELS = ()               # research ablations only (published model uses every channel)
N_IN = len(CHANNELS) + 1         # + log price relative to the latest close (built when windows are gathered)
N_TAB = len(CHANNELS) + 18       # tabular summary features (tabular())


# ---------------------------------------------------------------- features

def market_frame(data):
    """Equal-weighted market of every stock in `data` on each date (mean log returns of the stocks trading that day)."""
    r = {c: np.log(df["close"]).diff() for c, df in data.items()}
    o = {c: np.log(df["open"] / df["close"].shift()) for c, df in data.items()}
    ret = pd.concat(r, axis=1).sort_index().mean(axis=1).fillna(0.0)
    on = pd.concat(o, axis=1).sort_index().mean(axis=1).fillna(0.0)
    sig = ret.ewm(span=40, adjust=False).std().fillna(0.01).clip(lower=0.003)
    lvl = ret.cumsum()
    ma60 = (lvl - np.log(np.exp(lvl).rolling(60, min_periods=20).mean())) / (sig * math.sqrt(20))
    return pd.DataFrame({"m_ret": ret / sig, "m_on": on / sig, "m_ma60": ma60})


def stock_frame(df, mkt):
    """Per-day channels of one stock (dates x CHANNELS) plus helpers logc / sig (daily log-return volatility)."""
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    lc = np.log(c)
    r = lc.diff()
    sig = r.ewm(span=40, adjust=False).std().fillna(0.02).clip(lower=0.004)
    f = pd.DataFrame(index=df.index)
    f["ret"] = r / sig
    f["on"] = np.log(o / c.shift()) / sig
    f["id"] = np.log(c / o) / sig                                    # KMID
    f["klen"] = (h - l) / o / sig
    f["kup"] = (h - np.maximum(o, c)) / o / sig
    f["klow"] = (np.minimum(o, c) - l) / o / sig
    lv = np.log1p(v.astype(float))
    f["vol"] = lv - lv.rolling(60, min_periods=20).median()
    kdj = tw_kdj(df)
    f["k"], f["d"], f["j"] = kdj["k"] / 100 - 0.5, kdj["d"] / 100 - 0.5, kdj["j"] / 100 - 0.5
    m = macd(c)
    f["macd"] = m["dif"] / c / sig
    f["osc"] = m["osc"] / c / sig
    for n in MA_N:
        f[f"ma{n}"] = np.log(c / sma(c, n)) / (sig * math.sqrt(min(n, 60)))
    f["lsig"] = np.log(sig)
    f = f.join(mkt.reindex(df.index))
    return f[list(CHANNELS)], lc, sig


def tabular(f, lc, sig, mkt_ret):
    """Hand-made summary features of the same inputs for the logistic / GBDT baselines (dates x 38)."""
    r = lc.diff()
    g = f.copy()
    for n in (5, 10, 20, 40, 60):
        g[f"mom{n}"] = r.rolling(n).sum() / (sig * math.sqrt(n))
    for n in (20, 60):
        g[f"on{n}"] = f["on"].rolling(n).mean()
        g[f"id{n}"] = f["id"].rolling(n).mean()
        g[f"dd{n}"] = (lc - lc.rolling(n).max()) / sig
        g[f"up{n}"] = (lc - lc.rolling(n).min()) / sig
        g[f"m_mom{n}"] = mkt_ret.reindex(f.index).rolling(n).mean()
    g["vol5"] = f["vol"].rolling(5).mean()
    g["vol20"] = f["vol"].rolling(20).mean()
    g["klen20"] = f["klen"].rolling(20).mean()
    return g


# ---------------------------------------------------------------- labels

def trend_scan(df):
    """(t-value, span) of the most significant forward trend at each bar; NaN where the longest span is unavailable."""
    o = np.log(df["open"].to_numpy(np.float64))
    c = np.log(df["close"].to_numpy(np.float64))
    n = len(c)
    best_t = np.full(n, np.nan)
    best_l = np.full(n, np.nan)
    for L in TS_LS:
        m = n - 1 - L
        if m <= 0:
            continue
        x = np.arange(L + 1, dtype=float)
        xc = x - x.mean()
        sxx = (xc ** 2).sum()
        Y = np.column_stack([o[1 : m + 1], swv(c[1:], L)[:m]])     # row t: open(t+1), close(t+1..t+L)
        Y = Y - Y.mean(axis=1, keepdims=True)
        b = (Y * xc).sum(axis=1) / sxx
        res = ((Y - b[:, None] * xc) ** 2).sum(axis=1) / (L - 1)
        tv = b / np.sqrt(res / sxx + 1e-12)
        upd = np.isnan(best_t[:m]) | (np.abs(tv) > np.abs(best_t[:m]))
        best_t[:m][upd] = tv[upd]
        best_l[:m][upd] = L
    best_t[max(n - 1 - max(TS_LS), 0):] = np.nan
    best_l[np.isnan(best_t)] = np.nan
    return best_t, best_l


def triple_barrier(df, h=None, k=None):
    """0 = lower barrier first, 1 = neither within h bars, 2 = upper barrier first (close path vs open(t+1))."""
    h, k = h or TB_H, k or TB_K
    o = np.log(df["open"].to_numpy(np.float64))
    c = np.log(df["close"].to_numpy(np.float64))
    sig = pd.Series(c).diff().rolling(20).std().to_numpy()
    n = len(c)
    out = np.full(n, np.nan)
    m = n - 1 - h
    if m <= 0:
        return out
    W = swv(c[1:], h)[:m] - o[1 : m + 1, None]
    b = k * sig[:m] * math.sqrt(h)
    up = W >= b[:, None]
    dn = W <= -b[:, None]
    fu = np.where(up.any(axis=1), up.argmax(axis=1), h + 1)
    fd = np.where(dn.any(axis=1), dn.argmax(axis=1), h + 1)
    lab = np.where(fu < fd, 2.0, np.where(fd < fu, 0.0, 1.0))
    lab[np.isnan(sig[:m])] = np.nan
    out[:m] = lab
    return out


# ---------------------------------------------------------------- panel (all stocks stacked)

def prepare(data):
    """Stack every stock into one panel of rows (stock, bar). Returns a dict of global arrays + per-stock slices."""
    mkt = market_frame(data)
    codes = sorted(data)
    F, G, LC, SG, D, TT, TL, TB, FW, KN, LOC, SL = [], [], [], [], [], [], [], [], [], [], [], {}
    off = 0
    for code in codes:
        df = data[code]
        n = len(df)
        f, lc, sig = stock_frame(df, mkt)
        g = tabular(f, lc, sig, mkt["m_ret"])
        tv, tl = trend_scan(df)
        tb = triple_barrier(df)
        o = df["open"].to_numpy(np.float64)
        fw = np.full(n, np.nan)
        if n > TB_H + 1:
            fw[: n - TB_H - 1] = np.log(o[TB_H + 1 :] / o[1 : n - TB_H])
        d = df.index.values.astype("datetime64[ns]")
        kn = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        if n > PURGE:
            kn[: n - PURGE] = d[PURGE:]
        F.append(f.to_numpy(np.float32)); G.append(g.to_numpy(np.float32))
        LC.append(lc.to_numpy(np.float64)); SG.append(sig.to_numpy(np.float64)); D.append(d)
        TT.append(tv); TL.append(tl); TB.append(tb); FW.append(fw); KN.append(kn); LOC.append(np.arange(n))
        SL[code] = slice(off, off + n)
        off += n
    P = dict(codes=codes, sl=SL, F=np.concatenate(F), G=np.concatenate(G), logc=np.concatenate(LC),
             sig=np.concatenate(SG), date=np.concatenate(D), ts_t=np.concatenate(TT), ts_l=np.concatenate(TL),
             tb=np.concatenate(TB), fwd=np.concatenate(FW), known=np.concatenate(KN), loc=np.concatenate(LOC),
             gnames=list(g.columns))
    P["usable"] = P["loc"] >= max(WARM, T - 1)                             # a full window inside the stock
    P["labelled"] = P["usable"] & ~np.isnan(P["ts_t"]) & ~np.isnan(P["tb"]) & ~np.isnat(P["known"])
    P["ts_w"] = (np.minimum(np.abs(np.nan_to_num(P["ts_t"])), TS_WCAP) / TS_WSCALE).astype(np.float32)
    P["ts_y"] = (np.nan_to_num(P["ts_t"]) > 0).astype(np.float32)
    P["tb_y"] = np.nan_to_num(P["tb"], nan=1.0).astype(np.int64)
    return P


OFFS = np.arange(-T + 1, 1)


def gather(Fn, P, idx):
    """(B, T, N_IN) float32 windows ending at global rows idx (normalised channels + relative log price)."""
    w = idx[:, None] + OFFS
    X = Fn[w]
    rel = (P["logc"][w] - P["logc"][idx][:, None]) / (P["sig"][idx][:, None] * math.sqrt(20))
    return np.concatenate([X, np.clip(rel, -5, 5).astype(np.float32)[:, :, None]], axis=2)


def norm_fit(X):
    mu = np.nanmean(X, axis=0, dtype=np.float64)
    sd = np.nanstd(X, axis=0, dtype=np.float64)
    return mu, np.where(sd > 1e-8, sd, 1.0)


def norm_apply(X, mu, sd, clip=5.0):
    return np.clip(np.nan_to_num((X - mu) / sd, nan=0.0), -clip, clip).astype(np.float32)


# ---------------------------------------------------------------- models

class TCN(torch.nn.Module):
    def __init__(self, c_in=N_IN, h=HIDDEN, dil=DILATIONS, k=KERNEL, drop=DROPOUT):
        super().__init__()
        self.inp = torch.nn.Conv1d(c_in, h, 1)
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(h, h, k, dilation=d) for d in dil])
        self.pads = [(k - 1) * d for d in dil]
        self.drop = torch.nn.Dropout(drop)
        self.head = torch.nn.Linear(2 * h, 4)      # [trend-scan logit, triple-barrier logits down/flat/up]

    def forward(self, x):
        z = self.inp(x.transpose(1, 2))
        for p, cv in zip(self.pads, self.convs):
            z = z + torch.relu(cv(torch.nn.functional.pad(z, (p, 0))))
        return self.head(self.drop(torch.cat([z[:, :, -1], z.mean(dim=2)], dim=1)))


class TCNX(TCN):
    """TCN trunk + a small branch on the tabular summary features (the GBDT baseline's inputs)."""

    def __init__(self, c_in=N_IN, h=HIDDEN, n_tab=N_TAB, **kw):
        super().__init__(c_in, h, **kw)
        self.tab = torch.nn.Linear(n_tab, h)
        self.head = torch.nn.Linear(3 * h, 4)

    def forward(self, x, g):
        z = self.inp(x.transpose(1, 2))
        for p, cv in zip(self.pads, self.convs):
            z = z + torch.relu(cv(torch.nn.functional.pad(z, (p, 0))))
        u = torch.relu(self.tab(g))
        return self.head(self.drop(torch.cat([z[:, :, -1], z.mean(dim=2), u], dim=1)))


class GRU(torch.nn.Module):
    def __init__(self, c_in=N_IN, h=HIDDEN, drop=DROPOUT):
        super().__init__()
        self.rnn = torch.nn.GRU(c_in, h, batch_first=True)
        self.drop = torch.nn.Dropout(drop)
        self.head = torch.nn.Linear(h, 4)

    def forward(self, x):
        _, hn = self.rnn(x)
        return self.head(self.drop(hn[0]))


ARCH = {"tcn": TCN, "tcnx": TCNX, "gru": GRU}


def n_params(arch="tcn"):
    return sum(p.numel() for p in ARCH[arch]().parameters())


def _loss(out, ya, wa, yb, task):
    la = (torch.nn.functional.binary_cross_entropy_with_logits(out[:, 0], ya, reduction="none") * wa).mean()
    lb = torch.nn.functional.cross_entropy(out[:, 1:], yb)
    return {"ts": la, "tb": lb, "both": la + lb}[task]


def _inputs(Fn, Gn, P, idx):
    x = torch.from_numpy(gather(Fn, P, idx))
    return (x,) if Gn is None else (x, torch.from_numpy(Gn[idx]))


def _predict_net(net, Fn, P, idx, batch=4096, Gn=None):
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            outs.append(net(*_inputs(Fn, Gn, P, idx[i : i + batch])))
    out = torch.cat(outs) if outs else torch.zeros((0, 4))
    pa = torch.sigmoid(out[:, 0]).numpy().astype(np.float64)
    pb = torch.softmax(out[:, 1:], dim=1).numpy().astype(np.float64)
    return pa, pb


def train_nn(P, Fn, tr, va, seed_base, arch="tcn", task=TASK, seeds=SEEDS, stride=STRIDE, max_epochs=MAX_EPOCHS,
             Gn=None):
    """Train len(seeds) nets (seed s on the rows of `tr` with (bar + s) % stride == 0), each early-stopped on `va`.
    Returns (predict(idx) -> (p_ts_up, p_tb[down, flat, up]), info)."""
    torch.set_num_threads(2)
    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        nets, info = [], []
        yva = (torch.from_numpy(P["ts_y"][va]), torch.from_numpy(P["ts_w"][va]), torch.from_numpy(P["tb_y"][va]))
        Gn = Gn if arch == "tcnx" else None
        Xva = _inputs(Fn, Gn, P, va)
        for s in seeds:
            sub = tr[(P["loc"][tr] + s) % stride == 0]
            torch.manual_seed(1000 * seed_base + s)
            g = torch.Generator().manual_seed(1000 * seed_base + s)
            net = ARCH[arch]()
            opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            best, best_ep, best_state, bad, hist = math.inf, 0, None, 0, []
            ya_all, wa_all, yb_all = P["ts_y"][sub], P["ts_w"][sub], P["tb_y"][sub]
            for ep in range(max_epochs):
                net.train()
                perm = torch.randperm(len(sub), generator=g).numpy()
                for i in range(0, len(sub), BATCH):
                    j = perm[i : i + BATCH]
                    x = _inputs(Fn, Gn, P, sub[j])
                    opt.zero_grad()
                    loss = _loss(net(*x), torch.from_numpy(ya_all[j]), torch.from_numpy(wa_all[j]),
                                 torch.from_numpy(yb_all[j]), task)
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    vl = float(_loss(net(*Xva), *yva, task))
                hist.append(round(vl, 4))
                if vl < best - 1e-5:
                    best, best_ep, bad = vl, ep + 1, 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
            net.load_state_dict(best_state)
            nets.append(net)
            info.append({"seed": s, "epochs": best_ep, "val_loss": best, "n": int(len(sub)), "hist": hist})
    finally:
        torch.use_deterministic_algorithms(was)

    def predict(idx):
        pa, pb = np.zeros(len(idx)), np.zeros((len(idx), 3))
        for net in nets:
            a, b = _predict_net(net, Fn, P, idx, Gn=Gn)
            pa += a / len(nets)
            pb += b / len(nets)
        return pa, pb

    predict.nets = nets
    return predict, info


def train_tab(P, Gn, tr, va, seed_base, kind="hgb", task=TASK):
    """Same labels on the tabular summary features: logistic regression or HistGradientBoosting (no early stopping:
    fixed settings; the validation tail is only reported). Returns (predict, info) like train_nn."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    tr = np.concatenate([tr, va])               # tabular models do not early-stop: train on the whole window
    X, w = Gn[tr], P["ts_w"][tr]
    if kind == "hgb":
        mk = lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                                    min_samples_leaf=400, l2_regularization=1.0,
                                                    early_stopping=False, random_state=seed_base)
    else:
        mk = lambda: LogisticRegression(C=0.05, max_iter=500)
    ma = mk().fit(X, P["ts_y"][tr], sample_weight=w) if task in ("ts", "both") else None
    mb = mk().fit(X, P["tb_y"][tr]) if task in ("tb", "both") else None

    def predict(idx):
        Z = Gn[idx]
        pa = ma.predict_proba(Z)[:, 1] if ma is not None else np.full(len(idx), 0.5)
        pb = mb.predict_proba(Z) if mb is not None else np.full((len(idx), 3), 1 / 3)
        return pa, pb

    return predict, [{"n": int(len(tr))}]


# ---------------------------------------------------------------- walk-forward

def _date(s):
    return np.datetime64(pd.Timestamp(s).to_datetime64(), "ns")


def _normalisers(P, tr):
    mu, sd = norm_fit(P["F"][tr])
    gmu, gsd = norm_fit(P["G"][tr])
    Fn, Gn = norm_apply(P["F"], mu, sd), norm_apply(P["G"], gmu, gsd)
    for ch in DROP_CHANNELS:                    # ablations: a dropped channel is fed as zeros
        Fn[:, CHANNELS.index(ch)] = 0.0
        if ch in P["gnames"]:
            Gn[:, P["gnames"].index(ch)] = 0.0
    return Fn, Gn


def fit_window(P, tr_mask, seed_base, model="tcn", task=TASK):
    """Train on the rows of tr_mask (labelled rows only), using its last VAL_MONTHS months as a purged validation tail.
    Returns (predict, info, vstart)."""
    rows = np.flatnonzero(tr_mask & P["labelled"])
    last = P["date"][rows].max()
    vstart = _date(pd.Timestamp(last) - pd.DateOffset(months=VAL_MONTHS))
    va = rows[P["date"][rows] >= vstart]
    tr = rows[P["known"][rows] < vstart]        # purge: training labels end before the validation tail starts
    Fn, Gn = _normalisers(P, tr)
    if model in ARCH:
        pred, info = train_nn(P, Fn, tr, va, seed_base, arch=model, task=task, Gn=Gn)
    else:
        pred, info = train_tab(P, Gn, tr, va, seed_base, kind=model, task=task)
    return pred, info, vstart


def _signal_cols(pa, pb, signal):
    if signal == "ts":
        return pa, 1.0 - pa
    if signal == "tb":
        return pb[:, 2], pb[:, 0]
    return 0.5 * (pa + pb[:, 2]), 0.5 * ((1.0 - pa) + pb[:, 0])  # "mix"


def walk_forward(data=None, model=None, task=TASK, P=None, first_year=FIRST_YEAR, verbose=False):
    """One model per calendar year Y (first_year .. last year), trained on rows known before Jan 1 of Y, predicting
    every usable row of Y. Also predicts the rows between the validation start and Jan 1 (threshold calibration).
    Returns dict(P, pa, pb (NaN where no model), year (model year per row), calib {Y: (codes, pa, pb)}, info)."""
    P = P if P is not None else prepare(data)
    model = model or MODEL
    N = len(P["date"])
    pa, pb, yr = np.full(N, np.nan), np.full((N, 3), np.nan), np.zeros(N, int)
    calib, infos = {}, {}
    years = pd.DatetimeIndex(P["date"]).year.to_numpy()
    for Y in range(first_year, int(years.max()) + 1):
        cut = _date(f"{Y}-01-01")
        trm = P["labelled"] & (P["known"] < cut)
        if trm.sum() < 5000:
            continue
        pred, info, vstart = fit_window(P, trm, Y, model, task)
        test = np.flatnonzero(P["usable"] & (years == Y))
        cal = np.flatnonzero(P["usable"] & (P["date"] >= vstart) & (P["date"] < cut))
        a, b = pred(np.concatenate([cal, test]))
        pa[test], pb[test], yr[test] = a[len(cal):], b[len(cal):], Y
        calib[Y] = (cal, a[: len(cal)], b[: len(cal)])
        infos[Y] = info
        if verbose:
            print(Y, info, flush=True)
    return dict(P=P, pa=pa, pb=pb, year=yr, calib=calib, info=infos)


# ---------------------------------------------------------------- positions

def _ema_by_stock(P, x, rows, span):
    """EMA of x along rows (sorted global rows), restarting at stock boundaries and at gaps in rows."""
    out = np.empty(len(rows))
    a = 2.0 / (span + 1)
    prev_row, acc = -10, 0.0
    for i, (r, v) in enumerate(zip(rows, x)):
        acc = v if r != prev_row + 1 or P["loc"][r] == 0 else acc + a * (v - acc)
        out[i] = acc
        prev_row = r
    return out


def thresholds(P, calib, signal, span, q_in, q_out):
    """{Y: (theta_in, theta_out)}: quantiles of the smoothed calibration-tail predictions of model Y."""
    th = {}
    for Y, (cal, a, b) in calib.items():
        up, dn = _signal_cols(a, b, signal)
        th[Y] = (float(np.quantile(_ema_by_stock(P, up, cal, span), q_in)),
                 float(np.quantile(_ema_by_stock(P, dn, cal, span), q_out)))
    return th


def state_machine(up, dn, th_in, th_out, floor, min_hold, fallback=FALLBACK):
    """1.0 after P(up) > theta_in, `floor` after P(down) > theta_out; no switch within min_hold bars of the last
    one. Bars without a model get `fallback`; the machine starts in the fully-invested state."""
    n = len(up)
    pos = np.full(n, fallback, dtype=float)
    state, held = 1.0, min_hold
    for i in range(n):
        if np.isnan(up[i]):
            continue
        if i == 0 or np.isnan(up[i - 1]):
            state, held = 1.0, min_hold          # (re)start after a gap
        if held >= min_hold:
            if state < 1.0 and up[i] > th_in[i]:
                state, held = 1.0, 0
            elif state == 1.0 and dn[i] > th_out[i]:
                state, held = floor, 0
        pos[i] = state
        held += 1
    return pos


MODEL = "tcnx"                   # published architecture: TCN trunk + tabular branch
DEFAULT_MAP = dict(signal="tb", span=SMOOTH, q_in=0.5, q_out=0.8, floor=0.5, min_hold=20)  # frozen (RESEARCH.md)


def positions_from(wf, data, signal="tb", span=SMOOTH, q_in=0.5, q_out=0.8, floor=0.5, min_hold=20):
    """Per-stock exposure Series from walk-forward (or CV) predictions."""
    P = wf["P"]
    th = thresholds(P, wf["calib"], signal, span, q_in, q_out)
    up_all, dn_all = _signal_cols(wf["pa"], wf["pb"], signal)
    out = {}
    for code in P["codes"]:
        sl = P["sl"][code]
        up, dn, yr = up_all[sl], dn_all[sl], wf["year"][sl]
        ok = ~np.isnan(up)
        a = 2.0 / (span + 1)
        ups, dns = np.full(len(up), np.nan), np.full(len(up), np.nan)
        acc_u = acc_d = None
        for i in np.flatnonzero(ok):
            acc_u = up[i] if acc_u is None or not ok[i - 1] else acc_u + a * (up[i] - acc_u)
            acc_d = dn[i] if acc_d is None or not ok[i - 1] else acc_d + a * (dn[i] - acc_d)
            ups[i], dns[i] = acc_u, acc_d
        th_in = np.array([th[y][0] if y in th else np.nan for y in yr])
        th_out = np.array([th[y][1] if y in th else np.nan for y in yr])
        pos = state_machine(ups, dns, th_in, th_out, floor, min_hold)
        out[code] = pd.Series(pos, index=data[code].index)
    return out


# ---------------------------------------------------------------- purged k-fold CV (evaluation only)

def purged_cv(data=None, folds=None, model=None, task=TASK, P=None, verbose=False):
    """Purged blocked k-fold CV (Lopez de Prado 2018 ch. 7). NOT tradable: fold f is predicted by a model trained on
    every other fold (later ones included). Training rows: labelled rows whose span [t, t+PURGE bars] does not
    overlap the fold, minus an embargo of the PURGE (= LABEL_END + 5 = 125) bars after the fold (the label windows of
    the fold's last samples reach that far past the fold, and the input windows of the next T bars contain fold
    bars; the selection-stage CV in RESEARCH.md used a T = 64-bar embargo). Early stopping
    on the last VAL_MONTHS of that training set (purged), thresholds from the usable rows of that tail outside the
    fold, seed base = fold start year. Each fold model predicts its decision bars (fill date inside the fold) and the
    60 bars before them (state-machine burn-in, inside the purge zone).
    Returns {fold: dict shaped like walk_forward (year = 1 on the fold's rows)}."""
    P = P if P is not None else prepare(data)
    model = model or MODEL
    N = len(P["date"])
    last = P["date"].max()
    out = {}
    for f, (start, end) in folds.items():
        fs, fe = _date(start), (last if end is None else _date(end))
        overlap = (P["date"] <= fe) & (P["known"] >= fs)
        emb = np.zeros(N, bool)
        test = []
        for code in P["codes"]:
            sl = P["sl"][code]
            d = P["date"][sl]
            k = np.searchsorted(d, fe, side="right")
            emb[sl.start + k : sl.start + k + max(T, PURGE)] = True
            fill = np.r_[d[1:], d[-1:]]
            m = np.flatnonzero((fill >= fs) & (fill <= fe))
            if len(m):
                rows = np.arange(max(m[0] - 60, 0), m[-1] + 1) + sl.start
                test.append(rows[P["usable"][rows]])
        test = np.concatenate(test) if test else np.zeros(0, int)
        trm = P["labelled"] & ~overlap & ~emb
        if trm.sum() < 5000 or not len(test):
            continue
        pred, info, vstart = fit_window(P, trm, pd.Timestamp(fs).year, model, task)
        infold = (P["date"] >= fs) & (P["date"] <= fe)
        cal = np.flatnonzero(P["usable"] & (P["date"] >= vstart) & ~infold & ~emb)
        a, b = pred(np.concatenate([cal, test]))
        pa, pb, yr = np.full(N, np.nan), np.full((N, 3), np.nan), np.zeros(N, int)
        pa[test], pb[test], yr[test] = a[len(cal):], b[len(cal):], 1
        out[f] = dict(P=P, pa=pa, pb=pb, year=yr, calib={1: (cal, a[: len(cal)], b[: len(cal)])}, info=info,
                      bounds=(fs, fe))
        if verbose:
            print(f, info, flush=True)
    return out


def cv_positions_from(cvr, data, **kw):
    """State-machine positions of each fold model, kept only on that fold's decision bars (fill date in the fold)."""
    out = {c: pd.Series(FALLBACK, index=df.index) for c, df in data.items()}
    for f, r in cvr.items():
        fs, fe = r["bounds"]
        pos = positions_from(r, data, **kw)
        for code, p in pos.items():
            d = p.index.values
            fill = np.r_[d[1:], d[-1:]]
            m = (fill >= fs) & (fill <= fe)
            out[code][m] = p[m]
    return out


# ---------------------------------------------------------------- caching

def fingerprint(data):
    h = hashlib.md5()
    for code in sorted(data):
        df = data[code]
        h.update(code.encode())
        h.update(np.ascontiguousarray(df[["open", "high", "low", "close", "volume"]].to_numpy(np.float64)).tobytes())
        h.update(np.ascontiguousarray(df.index.values.astype("int64")).tobytes())
    return h.hexdigest()
