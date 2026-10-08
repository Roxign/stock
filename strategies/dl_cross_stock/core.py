"""E4 cross-stock attention model (research/dl_literature.md section 6, E4).

One sample = one trading date. Every stock of the 50 gets a token built from a 20-day window of daily features plus
multi-horizon snapshot features; a market vector gates the stock inputs (MASTER-style) and is added as an extra
token; one layer of multi-head self-attention mixes the stocks of that date (stocks without data are masked); a shared
head predicts each stock's ABSOLUTE vol-normalised forward return
    y[t, i] = ln(O[t+1+H] / O[t+1]) / (sigma[t] * sqrt(H)),  H = 20,
optionally with a ListNet ranking auxiliary loss. Walk-forward yearly retraining (expanding window, purge = label
window + embargo, early stopping on a purged 12-month validation tail, normalisation from training rows only), 3 seeds.

Everything a date's features use is known before the next TW open (prices up to t's close, US closes of calendar
date t, TWSE evening statistics of t, monthly revenue from its legal deadline) via stocklab.external with
end=data_end(data), so the truncation lookahead check also truncates the external inputs.

Trained models are memoised in-process on a fingerprint of the exact training arrays (dates before the model's
cutoff), so positions() and weights() share one training run and a truncated-data rerun reuses a model only when
its training inputs are bit-identical.
"""

from __future__ import annotations

import hashlib
import math
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from stocklab import external as ext
from stocklab.universe import CODES

torch.set_num_threads(2)

H = 20               # label horizon in bars
L = 20               # window length of the temporal encoder
EMBARGO = 5          # extra bars between training labels and validation / test
VAL_DAYS = 250       # early-stopping tail of the training window
MIN_BARS = 60        # a stock needs this many bars before it is used
FIRST_YEAR = 2012    # first walk-forward model (trained on 2008-2011)
SEEDS = (0, 1, 2)
N = len(CODES)

WIN_COLS = ["w_on", "w_id", "w_rng", "w_vol"]
SNAP_COLS = ["r5", "r20", "r60", "r120", "r250", "macd8", "macd16", "macd32", "on20", "on60", "id20", "id60",
             "lvol", "vratio", "hi52", "vlm", "fr20", "fr60", "mg20", "ryoy", "pbrz"]
MKT_COLS = ["ew5", "ew20", "ew60", "ewvol", "breadth", "disp", "sox1", "sox5", "sox20", "tsm1", "tsm5", "vix",
            "vixd", "twd20", "us10y20", "ff5", "turn", "pcr"]
CHIP_FIELDS = ["foreign_ratio", "margin_balance", "shares_issued", "revenue_yoy", "pbr"]
MKT_FIELDS = ["sox", "tsm", "vix", "usdtwd", "us10y", "mkt_foreign_net", "taiex_turnover", "pc_ratio_oi"]
FIN = L * len(WIN_COLS) + len(SNAP_COLS)   # per-stock input width (80 + 21)
FM = len(MKT_COLS)


# ------------------------------------------------------------------------------------------------ features

def industry_classes():
    """Static industry class per code: the 6 largest TWSE 產業別 among the 50, everything else 'other'."""
    lab = ext.industry(CODES).fillna("other")
    counts = lab.value_counts()
    top = sorted(counts.index, key=lambda k: (-counts[k], k))[:6]
    cls = {k: i for i, k in enumerate(top)}
    return np.array([cls.get(lab[c], 6) for c in CODES], dtype=np.int64), lab


def _stock_features(df, chips):
    o, h, l, c, v = (df[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    lc = np.log(c)
    lr = lc.diff()
    sig = lr.ewm(span=60, min_periods=20).std().clip(lower=0.004)
    on = np.log(o / c.shift(1))
    intr = np.log(c / o)
    vma60 = v.rolling(60, min_periods=20).mean()
    win = pd.DataFrame({
        "w_on": (on / sig).clip(-5, 5),
        "w_id": (intr / sig).clip(-5, 5),
        "w_rng": (np.log(h / l) / sig).clip(0, 8),
        "w_vol": np.log((v + 1) / (vma60 + 1)).clip(-4, 4),
    }, index=df.index)
    s = {}
    for k in (5, 20, 60, 120, 250):
        s[f"r{k}"] = (lc - lc.shift(k)) / (sig * math.sqrt(k))
    for a, b in ((8, 24), (16, 48), (32, 96)):
        q = (c.ewm(span=a, adjust=False).mean() - c.ewm(span=b, adjust=False).mean()) / c.rolling(63, min_periods=20).std()
        s[f"macd{a}"] = q / q.rolling(252, min_periods=60).std()
    for k in (20, 60):
        s[f"on{k}"] = on.rolling(k).sum() / (sig * math.sqrt(k))
        s[f"id{k}"] = intr.rolling(k).sum() / (sig * math.sqrt(k))
    s["lvol"] = np.log(sig * math.sqrt(252))
    s["vratio"] = np.log(lr.rolling(20).std() / lr.rolling(120, min_periods=60).std())
    s["hi52"] = lc - lc.rolling(250, min_periods=60).max()
    s["vlm"] = np.log((v.rolling(5).mean() + 1) / (vma60 + 1))
    fr = chips["foreign_ratio"]
    s["fr20"] = fr - fr.shift(20)
    s["fr60"] = fr - fr.shift(60)
    mg = chips["margin_balance"] / chips["shares_issued"] * 100
    s["mg20"] = mg - mg.shift(20)
    yoy = chips["revenue_yoy"]
    s["ryoy"] = (np.sign(yoy) * np.log1p(yoy.abs())).clip(-2, 2)
    lp = np.log(chips["pbr"].where(chips["pbr"] > 0))
    s["pbrz"] = (lp - lp.rolling(750, min_periods=250).mean()) / lp.rolling(750, min_periods=250).std()
    snap = pd.DataFrame(s, index=df.index)[SNAP_COLS].replace([np.inf, -np.inf], np.nan)
    y = (np.log(o.shift(-(H + 1))) - np.log(o.shift(-1))) / (sig * math.sqrt(H))
    return win, snap, y, sig


def _market_features(dates, close_panel, lr_panel, end):
    m = {}
    ew = lr_panel.mean(axis=1, skipna=True).fillna(0.0)
    ew_sig = ew.ewm(span=60, min_periods=20).std().clip(lower=0.003)
    cum = ew.cumsum()
    for k in (5, 20, 60):
        m[f"ew{k}"] = (cum - cum.shift(k)) / (ew_sig * math.sqrt(k))
    m["ewvol"] = np.log(ew_sig * math.sqrt(252))
    sma60 = close_panel.rolling(60, min_periods=40).mean()
    above = (close_panel > sma60).where(sma60.notna())
    m["breadth"] = above.mean(axis=1, skipna=True)
    r20 = np.log(close_panel / close_panel.shift(20))
    m["disp"] = r20.std(axis=1, skipna=True)
    x = ext.load_market(MKT_FIELDS, index=dates, end=end)
    for name, ks in (("sox", (1, 5, 20)), ("tsm", (1, 5))):
        lx = np.log(x[name])
        sg = lx.diff().ewm(span=60, min_periods=20).std()
        for k in ks:
            m[f"{name}{k}"] = (lx - lx.shift(k)) / (sg * math.sqrt(k))
    m["vix"] = np.log(x["vix"])
    m["vixd"] = np.log(x["vix"] / x["vix"].rolling(20, min_periods=10).mean())
    m["twd20"] = np.log(x["usdtwd"] / x["usdtwd"].shift(20))
    m["us10y20"] = x["us10y"] - x["us10y"].shift(20)
    to = x["taiex_turnover"].fillna(0.0)
    m["ff5"] = x["mkt_foreign_net"].fillna(0.0).rolling(5).sum() / to.rolling(5).sum().replace(0, np.nan)
    m["turn"] = np.log(to.rolling(5).mean() / to.rolling(60, min_periods=20).mean().replace(0, np.nan))
    m["pcr"] = np.log(x["pc_ratio_oi"])
    out = pd.DataFrame(m, index=dates)[MKT_COLS].replace([np.inf, -np.inf], np.nan)
    return out


def fingerprint(data):
    h = hashlib.md5()
    for code in sorted(data):
        df = data[code]
        h.update(code.encode())
        h.update(np.asarray(df.index.values, dtype="datetime64[ns]").tobytes())
        h.update(np.ascontiguousarray(df[["open", "close"]].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()


_PANELS: dict = {}


@np.errstate(divide="ignore", invalid="ignore")
def build_panel(data):
    """All model arrays on one date grid (union of the stocks' bars), 50 fixed stock slots (CODES order)."""
    key = fingerprint(data)
    if key in _PANELS:
        return _PANELS[key]
    end = ext.data_end(data)
    dates = pd.DatetimeIndex(sorted(set().union(*[df.index for df in data.values()])))
    T = len(dates)
    Wd = np.zeros((T, N, len(WIN_COLS)), np.float32)
    S = np.full((T, N, len(SNAP_COLS)), np.nan, np.float32)
    Y = np.full((T, N), np.nan, np.float32)
    YEND = np.full((T, N), np.iinfo(np.int64).max // 2, np.int64)
    SIG = np.full((T, N), np.nan)
    AV = np.zeros((T, N), bool)
    BAR = np.zeros((T, N), bool)
    closes, lrs = {}, {}
    for j, code in enumerate(CODES):
        if code not in data:
            continue
        df = data[code]
        ii = dates.get_indexer(df.index)
        chips = ext.load_stock_fields(code, CHIP_FIELDS, index=df.index, end=end)
        win, snap, y, sig = _stock_features(df, chips)
        Wd[ii, j] = np.nan_to_num(win.to_numpy(np.float32))
        listed = dates >= df.index[0]
        sf = snap.reindex(dates).ffill(limit=10).to_numpy(np.float32, copy=True)
        sf[~listed] = np.nan
        S[:, j] = sf
        Y[ii, j] = y.to_numpy(np.float32)
        ye = np.full(len(df), YEND[0, 0])
        ye[: max(len(df) - H - 1, 0)] = ii[H + 1:]
        YEND[ii, j] = ye
        SIG[ii, j] = sig.to_numpy()
        BAR[ii, j] = True
        AV[ii, j] = (np.arange(len(df)) >= MIN_BARS) & sig.notna().to_numpy()
        closes[code] = df["close"]
        lrs[code] = np.log(df["close"]).diff()
    cp = pd.DataFrame(closes).reindex(dates).ffill(limit=10)
    lp = pd.DataFrame(lrs).reindex(dates)
    M = _market_features(dates, cp, lp, end).to_numpy(np.float32)
    # flattened windows: X[t, j] = [Wd[t-L+1..t, j, :] (oldest first), S[t, j]]
    win = np.lib.stride_tricks.sliding_window_view(Wd, L, axis=0)        # (T-L+1, N, C, L)
    win = np.ascontiguousarray(win.transpose(0, 1, 3, 2)).reshape(T - L + 1, N, L * len(WIN_COLS))
    X = np.zeros((T, N, FIN), np.float32)
    X[L - 1:, :, : L * len(WIN_COLS)] = win
    X[:, :, L * len(WIN_COLS):] = S
    ind, _ = industry_classes()
    P = {"dates": dates, "X": X, "M": M, "Y": Y, "YEND": YEND, "SIG": SIG, "AV": AV, "BAR": BAR, "ind": ind,
         "codes": list(CODES), "key": key}
    _PANELS.clear()
    _PANELS[key] = P
    return P


# ------------------------------------------------------------------------------------------------ models

class CrossStockNet(nn.Module):
    """Shared per-stock MLP encoder over the flattened 20-day window + snapshot features, market gate on the inputs,
    market token (added to every stock and attendable), one layer of cross-stock multi-head self-attention
    (attn=False: the 'pooled' ablation without any cross-stock mixing), FFN, shared regression head, optional linear
    skip path from the raw inputs."""

    def __init__(self, fin=FIN, fm=FM, d=32, hid=32, emb=8, iemb=4, heads=4, ffn=48, attn=True, drop=0.1, skip=False):
        super().__init__()
        self.lin = nn.Linear(fin + fm, 1) if skip else None
        self.gate = nn.Linear(fm, fin)
        self.emb = nn.Embedding(N, emb)
        self.iemb = nn.Embedding(7, iemb)
        self.enc = nn.Sequential(nn.Linear(fin + emb + iemb, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, d))
        self.mkt = nn.Linear(fm, d)
        self.use_attn = attn
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True) if attn else None
        self.alpha = nn.Parameter(torch.zeros(1)) if attn else None
        self.ln1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(drop), nn.Linear(ffn, d))
        self.ln2 = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x, m, ids, ind, mask, need_weights=False):
        lin = self.lin(torch.cat([x, m.unsqueeze(1).expand(-1, x.shape[1], -1)], -1)).squeeze(-1) if self.lin is not None else 0
        x = x * (2 * torch.sigmoid(self.gate(m))).unsqueeze(1)
        mt = self.mkt(m).unsqueeze(1)
        h = self.enc(torch.cat([x, self.emb(ids), self.iemb(ind)], -1)) + mt
        w = None
        if self.use_attn:
            # stocks attend to the market token and to every other stock of the same date (padding masked); the
            # attention residual is scaled by a learnable scalar that starts at 0 (ReZero), so training starts
            # from the pooled model and only adds cross-stock mixing if it lowers the loss
            tok = torch.cat([mt, h], 1)
            kpm = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], 1)
            a, w = self.attn(tok, tok, tok, key_padding_mask=kpm, need_weights=need_weights)
            h = h + self.alpha * self.drop(a[:, 1:])
        h = self.ln1(h)
        h = self.ln2(h + self.drop(self.ffn(h)))
        out = self.head(h).squeeze(-1) + lin
        return (out, w) if need_weights else out


class FlatNet(nn.Module):
    """Tan et al. (2023)-style flat cross-section: every stock's snapshot features + market vector -> one hidden
    layer -> one output per stock slot."""

    def __init__(self, fs=len(SNAP_COLS), fm=FM, hid=16, drop=0.1):
        super().__init__()
        self.l1 = nn.Linear(N * fs + fm, hid)
        self.l2 = nn.Linear(hid, N)
        self.drop = nn.Dropout(drop)
        self.fs = fs

    def forward(self, x, m, ids, ind, mask):
        s = x[:, :, -self.fs:] * mask.unsqueeze(-1)
        z = torch.cat([s.flatten(1), m], 1)
        return self.l2(self.drop(torch.relu(self.l1(z))))


KINDS = ("attn", "pooled", "shuffle", "flat")


def make_model(kind, **kw):
    if kind == "flat":
        return FlatNet(**{k: v for k, v in kw.items() if k in ("hid", "drop")})
    return CrossStockNet(attn=kind != "pooled", **kw)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------------------------------------ training

DEFAULT_CFG = {"lr": 1e-3, "wd": 1e-3, "batch": 32, "epochs": 30, "patience": 5, "rank": 0.5, "clip": 3.0,
               "stride": 1, "model": {}, "val_days": VAL_DAYS}


def _stats(P, rows):
    """Mean / std per input column from training rows only (window columns share stats per channel)."""
    X, AV, M = P["X"], P["AV"], P["M"]
    xr = X[rows][AV[rows]]                                   # (n, FIN)
    nw = L * len(WIN_COLS)
    w = xr[:, :nw].reshape(-1, L, len(WIN_COLS))
    wm, ws = np.nanmean(w, axis=(0, 1)), np.nanstd(w, axis=(0, 1))
    sm, ss = np.nanmean(xr[:, nw:], axis=0), np.nanstd(xr[:, nw:], axis=0)
    mu = np.r_[np.tile(wm, L), sm].astype(np.float32)
    sd = np.r_[np.tile(ws, L), ss].astype(np.float32)
    mm, ms = np.nanmean(M[rows], axis=0), np.nanstd(M[rows], axis=0)
    fix = lambda a, b: (np.nan_to_num(a).astype(np.float32), np.where(np.isfinite(b) & (b > 1e-8), b, 1.0).astype(np.float32))
    mu, sd = fix(mu, sd)
    mm, ms = fix(mm, ms)
    return mu, sd, mm, ms


class Batcher:
    def __init__(self, P, stats, kind, clip):
        self.X = torch.from_numpy(P["X"])
        self.M = torch.from_numpy(P["M"])
        self.Y = torch.from_numpy(np.clip(P["Y"], -clip, clip))
        self.AV = torch.from_numpy(P["AV"])
        self.ind = torch.from_numpy(P["ind"])
        mu, sd, mm, ms = (torch.from_numpy(a) for a in stats)
        self.mu, self.sd, self.mm, self.ms = mu, sd, mm, ms
        self.kind = kind

    def get(self, rows, gen=None):
        r = torch.from_numpy(np.asarray(rows))
        x = ((self.X[r] - self.mu) / self.sd).clamp(-5, 5).nan_to_num(0.0)
        m = ((self.M[r] - self.mm) / self.ms).clamp(-5, 5).nan_to_num(0.0)
        mask = self.AV[r]
        x = x * mask.unsqueeze(-1)
        B = len(r)
        ids = torch.arange(N).expand(B, N)
        if self.kind == "shuffle":
            ids = torch.argsort(torch.rand(B, N, generator=gen), dim=1)
        ind = self.ind[ids]
        return x, m, ids, ind, mask, self.Y[r]


def _loss(pred, y, mask, rank_w):
    ok = mask & ~torch.isnan(y)
    yv = torch.nan_to_num(y)
    mse = ((pred - yv) ** 2 * ok).sum() / ok.sum().clamp(min=1)
    if rank_w <= 0:
        return mse
    neg = torch.finfo(pred.dtype).min / 4
    lp = torch.log_softmax(torch.where(ok, pred, torch.full_like(pred, neg)), dim=1)
    pt = torch.softmax(torch.where(ok, yv, torch.full_like(yv, neg)), dim=1)
    keep = ok.sum(1) >= 5
    ln = -(pt * torch.where(ok, lp, torch.zeros_like(lp))).sum(1)
    ln = (ln[keep] - torch.log(ok.sum(1)[keep].float())).mean() if keep.any() else pred.sum() * 0
    return mse + rank_w * ln


def split_rows(P, eligible, val_days=VAL_DAYS):
    """eligible: sorted array of date rows usable for training (label windows already purged against the test set).
    Returns (train, val): val = the last VAL_DAYS eligible dates; train = eligible dates whose label window ends at least
    EMBARGO bars before the first validation date."""
    eligible = np.asarray(eligible)
    if len(eligible) < 2 * val_days:
        return eligible, eligible[-val_days // 2:]
    val = eligible[-val_days:]
    v0 = val[0]
    train = eligible[(label_end(P, eligible) < v0 - EMBARGO) & (eligible < v0)]
    return train, val


def label_end(P, rows):
    """Last bar row any used label of each date row reaches (-1 if the date has no label)."""
    ok = P["AV"][rows] & ~np.isnan(P["Y"][rows])
    return np.where(ok, P["YEND"][rows], -1).max(axis=1)


def fit(P, kind, train, val, seed, cfg=None):
    cfg = DEFAULT_CFG | (cfg or {})
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    stats = _stats(P, train)
    bat = Batcher(P, stats, kind, cfg["clip"])
    model = make_model(kind, **cfg["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best, best_state, bad, best_ep = math.inf, None, 0, 0
    tr = train[:: cfg["stride"]]
    for ep in range(cfg["epochs"]):
        model.train()
        perm = rng.permutation(tr)
        for b in range(0, len(perm), cfg["batch"]):
            x, m, ids, ind, mask, y = bat.get(perm[b: b + cfg["batch"]], gen)
            loss = _loss(model(x, m, ids, ind, mask), y, mask, cfg["rank"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vl = _eval_loss(model, bat, val)
        if vl < best - 1e-5:
            best, bad, best_ep = vl, 0, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "stats": stats, "val_loss": best, "epochs": best_ep + 1, "kind": kind}


def _eval_loss(model, bat, rows):
    model.eval()
    tot, n = 0.0, 0
    gen = torch.Generator().manual_seed(12345)
    with torch.inference_mode():
        for b in range(0, len(rows), 256):
            x, m, ids, ind, mask, y = bat.get(rows[b: b + 256], gen)
            p = model(x, m, ids, ind, mask)
            ok = mask & ~torch.isnan(y)
            tot += float((((p - torch.nan_to_num(y)) ** 2) * ok).sum())
            n += int(ok.sum())
    return tot / max(n, 1)


def predict(P, fitted, rows):
    bat = Batcher(P, fitted["stats"], fitted["kind"], 99.0)
    model = fitted["model"]
    out = np.full((len(rows), N), np.nan, np.float32)
    gen = torch.Generator().manual_seed(777)
    with torch.inference_mode():
        for b in range(0, len(rows), 256):
            x, m, ids, ind, mask, _ = bat.get(rows[b: b + 256], gen)
            p = model(x, m, ids, ind, mask).numpy()
            out[b: b + 256] = np.where(mask.numpy(), p, np.nan)
    return out


# ------------------------------------------------------------------------------------------------ walk-forward / CV

_MODELS: dict = {}


def _train_key(P, rows_upto, kind, seed, cfg, tag):
    """Fingerprint of everything a model trained on dates < rows_upto can see."""
    h = hashlib.md5()
    h.update(repr((kind, seed, sorted((cfg or {}).items()), tag, H, L, EMBARGO, VAL_DAYS)).encode())
    sl = slice(0, rows_upto)
    y = np.where(P["YEND"][sl] < rows_upto, P["Y"][sl], np.nan)
    for a in (P["X"][sl], P["M"][sl], y, P["AV"][sl], P["dates"].values[sl].astype("datetime64[ns]")):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def eligible_before(P, cut_row, lo_row=0):
    """Date rows < cut_row whose every label window (over available stocks) ends >= EMBARGO bars before cut_row."""
    rows = np.arange(lo_row, cut_row)
    has = (P["AV"][rows] & ~np.isnan(P["Y"][rows])).sum(axis=1) >= 5
    return rows[has & (label_end(P, rows) < cut_row - EMBARGO)]


def walk_forward(data, kind="attn", seeds=SEEDS, cfg=None, years=None, verbose=False):
    """Returns (pred[s] arrays (T, N) per seed, info). Model for year Y is trained on dates whose labels end before
    Y-01-01 (minus embargo) and predicts every date in year Y."""
    P = build_panel(data)
    dates = P["dates"]
    if years is None:
        years = range(FIRST_YEAR, dates[-1].year + 1)
    preds = {s: np.full((len(dates), N), np.nan, np.float32) for s in seeds}
    info = []
    for yr in years:
        c0 = int(dates.searchsorted(pd.Timestamp(f"{yr}-01-01")))
        c1 = int(dates.searchsorted(pd.Timestamp(f"{yr + 1}-01-01")))
        if c0 >= len(dates) or c1 <= c0:
            continue
        el = eligible_before(P, c0)
        tr, va = split_rows(P, el, (DEFAULT_CFG | (cfg or {}))["val_days"])
        for s in seeds:
            key = _train_key(P, c0, kind, s, cfg, "wf")
            t = time.time()
            if key not in _MODELS:
                _MODELS[key] = fit(P, kind, tr, va, s, cfg)
            f = _MODELS[key]
            preds[s][c0:c1] = predict(P, f, np.arange(c0, c1))
            info.append({"year": yr, "seed": s, "n_train": len(tr), "n_val": len(va), "epochs": f["epochs"],
                         "val_loss": f["val_loss"], "sec": time.time() - t})
            if verbose:
                print(f"  {kind} {yr} seed {s}: train {len(tr)} val {len(va)} ep {f['epochs']} "
                      f"val {f['val_loss']:.4f} {time.time() - t:.1f}s", flush=True)
    return preds, info


def purged_cv(data, folds, kind="attn", seeds=SEEDS, cfg=None, train_end=None, verbose=False):
    """Purged blocked k-fold: fold f is predicted by a model trained on every other date whose label window does not
    overlap f and that does not start within H bars after f (embargo). train_end: optional last date usable for
    training (labels must end before it), e.g. '2020-12-31' for in-sample-only development."""
    P = build_panel(data)
    dates = P["dates"]
    T = len(dates)
    lim = T if train_end is None else int(dates.searchsorted(pd.Timestamp(train_end), side="right"))
    preds = {s: np.full((T, N), np.nan, np.float32) for s in seeds}
    info = []
    ye_all = label_end(P, np.arange(T))
    has = (P["AV"] & ~np.isnan(P["Y"])).sum(axis=1) >= 5
    for f, (a, b) in folds.items():
        f0 = int(dates.searchsorted(pd.Timestamp(a)))
        f1 = T if b is None else int(dates.searchsorted(pd.Timestamp(b), side="right"))
        if f0 >= T or f1 <= f0:
            continue
        rows = np.arange(T)
        ok = has & (ye_all < lim) & ((ye_all < f0) | (rows > f1 - 1 + H)) & ((rows < f0) | (rows >= f1))
        el = rows[ok]
        tr, va = split_rows(P, el, (DEFAULT_CFG | (cfg or {}))["val_days"])
        lo = f0 if f != list(folds)[0] else 0           # first fold also predicts the warm-up years
        for s in seeds:
            t = time.time()
            fitted = fit(P, kind, tr, va, s, cfg)
            preds[s][lo:f1] = predict(P, fitted, np.arange(lo, f1))
            info.append({"fold": f, "seed": s, "n_train": len(tr), "epochs": fitted["epochs"],
                         "val_loss": fitted["val_loss"], "sec": time.time() - t})
            if verbose:
                print(f"  cv {kind} {f} seed {s}: train {len(tr)} ep {fitted['epochs']} {time.time() - t:.1f}s", flush=True)
    return preds, info


def ensemble(preds):
    if len(preds) == 1:
        return next(iter(preds.values()))
    with warnings.catch_warnings():                 # all-NaN cells (stock without data) stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(list(preds.values())), axis=0)


# ------------------------------------------------------------------------------------------------ positions

POS_CFG = {"gamma": 2.0, "span": 20, "band": 0.5, "fallback": 1.0}


def _hysteresis(raw, band, step=0.25):
    """No-trade band: the held exposure (multiple of `step`) moves to the rounded target only when the raw target is at
    least `band` away from it."""
    out = np.empty(len(raw))
    cur = None
    for i, x in enumerate(raw):
        if np.isnan(x):
            out[i] = np.nan
            continue
        if cur is None or abs(x - cur) >= band:
            cur = float(np.clip(round(x / step) * step, 0, 1))
        out[i] = cur
    return out


def smoothed(P, pred, span):
    """EMA of predictions on each stock's own bars (dates x codes DataFrame)."""
    df = pd.DataFrame(pred, index=P["dates"], columns=P["codes"]).where(P["BAR"])
    return df.apply(lambda s: s.dropna().ewm(span=span, min_periods=1).mean().reindex(s.index)) if span > 1 else df


def positions_from(P, pred, data, gamma=None, span=None, band=None, fallback=None):
    c = POS_CFG | {k: v for k, v in dict(gamma=gamma, span=span, band=band, fallback=fallback).items() if v is not None}
    z = smoothed(P, pred, c["span"])
    sig = pd.DataFrame(P["SIG"], index=P["dates"], columns=P["codes"])
    want = (z / (c["gamma"] * sig * math.sqrt(H))).clip(0, 1)
    out = {}
    for code, df in data.items():
        w = want[code].reindex(df.index).to_numpy()
        q = _hysteresis(w, c["band"])
        out[code] = pd.Series(np.where(np.isnan(q), c["fallback"], q), index=df.index)
    return out


def weights_from(P, pred, data, k=10, span=None, keep_mult=2, min_score=0.0, score="z"):
    """Portfolio weights: on the first bar of each month hold the k stocks with the best smoothed predicted
    vol-normalised return (> min_score), 1/k of equity each; held stocks stay while still in the top keep_mult*k and
    above min_score; empty slots are cash. Before the first model: equal weight in all available stocks."""
    span = span or POS_CFG["span"]
    z = smoothed(P, pred, span)
    if score == "ret":                         # expected return instead of vol-normalised return
        z = z * P["SIG"]
    dates = P["dates"]
    month = dates.to_period("M")
    first = np.r_[True, month[1:] != month[:-1]]
    W = np.zeros((len(dates), N))
    held: list[int] = []
    Z = z.to_numpy()
    AV = P["AV"]
    for t in range(len(dates)):
        if not first[t]:
            W[t] = W[t - 1]
            continue
        zt = Z[t]
        ok = AV[t] & ~np.isnan(zt)
        if not ok.any():                      # no model yet: equal weight in every available stock
            held = list(np.flatnonzero(AV[t]))
            wt = 1.0 / len(held) if held else 0.0
        else:
            order = [j for j in np.argsort(-np.where(ok, zt, -np.inf)) if ok[j] and zt[j] > min_score]
            top = set(order[: keep_mult * k])
            held = [j for j in held if j in top][:k]
            for j in order:
                if len(held) >= k:
                    break
                if j not in held:
                    held.append(j)
            wt = 1.0 / k
        W[t] = 0.0
        if held:
            W[t, held] = wt
    out = {}
    for j, code in enumerate(P["codes"]):
        if code in data:
            out[code] = pd.Series(W[:, j], index=dates).reindex(data[code].index).fillna(0.0)
    return out
