"""KDJ + MACD (+ Taiwan MA lines) features, small-MLP walk-forward training and the KDJ base rule (kdj_macd_dl).

The published model uses FEATURE_SET = "kdj_ma_atr" (48 features, 3 x MLP 48-16-8-1 = 2,787 parameters); the other
feature sets are kept so the research scripts in research/kdj_*.py can be re-run.

Everything here is causal:
  * features at bar t only use bars <= t (EMA / KDJ recursions seeded at each stock's first bar);
  * the model used for calendar year Y is trained only on samples whose label window ends before Jan 1 of Y
    (purge), normalised with statistics of that training set only;
  * training is deterministic (fixed seeds, fixed sample order, deterministic torch ops, 2 threads), so truncating
    the data at any date reproduces exactly the same positions before that date.
purged_cv() at the bottom is the exception on purpose: purged k-fold cross-validation for evaluate.py --cv only
(each fold traded by a model trained on all OTHER folds, later ones included); it never feeds the published positions.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch

from stocklab.indicators import MA_PERIODS, atr, macd, sma, tw_kdj

# ---------------------------------------------------------------- configuration (one config for all stocks)

SLOW = 5                      # "weekly-equivalent" KDJ / MACD: KDJ(45,15,15), MACD(60,130,45)
LAGS = (0, 1, 2, 3, 4)        # daily KDJ / MACD window fed to the network
SLOW_LAGS = (0, 5)            # slow KDJ / MACD: today and one week ago
SINCE_CAP = 60                # "bars since J extreme" features are capped at this many bars
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

# feature groups (see feature_matrix); the published model uses FEATURE_SET
GROUPS = {
    "kd": "daily K, D (lags 0-4)",
    "j": "daily J (lags 0-4)",
    "macd": "daily DIF/ATR, OSC/ATR (lags 0-4)",
    "kdw": "slow K, D (lags 0, 5)",
    "jw": "slow J (lags 0, 5)",
    "macdw": "slow DIF/ATR, OSC/ATR (lags 0, 5)",
    "jx": "J extremes: J above 100 / below 0 (daily, slow) and bars since the last J>100 / J<0 (daily, slow)",
    "ma_dist": "close / MA_n - 1 for the Taiwan MA lines n = 5, 10, 20, 60, 120, 240",
    "ma_slope": "MA_n / MA_n[5 bars ago] - 1 for n = 20, 60, 120, 240",
    "ma_align": "MA alignment score (adjacent MA pairs in bullish order, -1..1), full bullish / bearish alignment flags",
    "ma_dist_atr": "(close - MA_n) / ATR(14) for n = 5, 10, 20, 60, 120, 240  (published)",
    "ma_slope_atr": "(MA_n - MA_n[5 bars ago]) / ATR(14) for n = 20, 60, 120, 240  (published)",
}
MA_N = tuple(MA_PERIODS.values())           # 5, 10, 20, 60, 120, 240 (週/雙週/月/季/半年/年線)
MA_SLOPE_N = (20, 60, 120, 240)
MA_SLOPE_LAG = 5
MA_TIE = 1e-6                               # relative tolerance for "MA_a equals MA_b" in the alignment score
GROUP_ALIAS = {"ma": ("ma_dist", "ma_slope", "ma_align"), "ma_atr": ("ma_dist_atr", "ma_slope_atr", "ma_align")}
FEATURE_SETS = {
    "kd": ("kd", "macd", "kdw", "macdw"),                       # previous round (28)
    "kdj": ("kd", "j", "macd", "kdw", "jw", "macdw"),           # + J lines (35)
    "jd": ("jd", "macd", "jdw", "macdw"),                       # J and D instead of K and D (28; same linear span as kd)
    "kdjx": ("kd", "j", "macd", "kdw", "jw", "macdw", "jx"),    # + J lines + J-extreme features (43)
    "kdx": ("kd", "macd", "kdw", "macdw", "jx"),                # K/D + J-extreme features, no raw J (36)
    "j_only": ("j", "macd", "jw", "macdw"),                     # J replaces K and D (21)
    "macd_only": ("macd", "macdw"),                             # no KDJ at all (14)
    "kdj_ma": ("kd", "j", "macd", "kdw", "jw", "macdw", "ma"),          # KDJ + MACD + Taiwan MA lines (48)
    "kdjx_ma": ("kd", "j", "macd", "kdw", "jw", "macdw", "jx", "ma"),   # everything (56)
    "kdj_madist": ("kd", "j", "macd", "kdw", "jw", "macdw", "ma_dist"),  # + only close/MA distances (41)
    "kdj_maslope": ("kd", "j", "macd", "kdw", "jw", "macdw", "ma_slope", "ma_align"),  # + MA slopes/alignment (42)
    "ma_only": ("ma",),                                                  # MA lines alone (13)
    "macd_ma": ("macd", "macdw", "ma"),                                  # MACD + MA, no KDJ (27)
    "kdj_ma_atr": ("kd", "j", "macd", "kdw", "jw", "macdw", "ma_atr"),  # MA features in ATR units (48)
    "kdj_madist_atr": ("kd", "j", "macd", "kdw", "jw", "macdw", "ma_dist_atr"),  # (41)
    "ma_atr_only": ("ma_atr",),                                          # (13)
    "kd_ma_atr": ("kd", "macd", "kdw", "macdw", "ma_atr"),              # final set without J (41)
    "kdjx_ma_atr": ("kd", "j", "macd", "kdw", "jw", "macdw", "jx", "ma_atr"),  # (56)
    "macd_ma_atr": ("macd", "macdw", "ma_atr"),                          # final set without KDJ (27)
}
FEATURE_SET = "kdj_ma_atr"    # frozen after the in-sample research (RESEARCH.md K3/K4): 48 features


# ---------------------------------------------------------------- indicators / features

def indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    kdj = tw_kdj(df)
    m = macd(c)
    kdjw = tw_kdj(df, 9 * SLOW, 3 * SLOW, 3 * SLOW)
    mw = macd(c, 12 * SLOW, 26 * SLOW, 9 * SLOW)
    return pd.DataFrame({
        "k": kdj["k"], "d": kdj["d"], "j": kdj["j"], "dif": m["dif"], "osc": m["osc"],
        "kw": kdjw["k"], "dw": kdjw["d"], "jw": kdjw["j"], "difw": mw["dif"], "oscw": mw["osc"],
        "atr": atr(df), "open": df["open"], "close": c,
        **{f"ma{n}": sma(c, n) for n in MA_N},
    }, index=df.index)


def ma_features(ind: pd.DataFrame) -> dict:
    """Scale-free moving-average features (NaN until the 240-bar MA exists)."""
    c = ind["close"].to_numpy()
    ma = {n: ind[f"ma{n}"].to_numpy() for n in MA_N}
    out = {f"ma_dist_{n}": c / ma[n] - 1 for n in MA_N}
    slope = {f"ma_slope_{n}": ma[n] / _lag(ma[n], MA_SLOPE_LAG) - 1 for n in MA_SLOPE_N}
    # +1 = shorter MA above longer MA, -1 below, 0 = tie within a relative 1e-6 (MA5 == MA10 happens on flat,
    # low-priced stocks; without the tolerance the sign of float rounding noise would decide the feature)
    pairs = [np.where(ma[a] > ma[b] * (1 + MA_TIE), 1.0, np.where(ma[a] < ma[b] * (1 - MA_TIE), -1.0, 0.0))
             for a, b in zip(MA_N[:-1], MA_N[1:])]
    score = sum(pairs) / len(pairs)
    valid = ~np.isnan(ma[MA_N[-1]])
    score = np.where(valid, score, np.nan)
    align = {"ma_align": score,
             "ma_bull_all": np.where(valid, (score == 1).astype(float), np.nan),
             "ma_bear_all": np.where(valid, (score == -1).astype(float), np.nan)}
    # research variant: the same distances / slopes measured in ATR units (volatility-scaled, like MACD/ATR)
    a = ind["atr"].to_numpy()
    a = np.where(a > 0, a, np.nan)
    dist_atr = {f"ma_datr_{n}": (c - ma[n]) / a for n in MA_N}
    slope_atr = {f"ma_satr_{n}": (ma[n] - _lag(ma[n], MA_SLOPE_LAG)) / a for n in MA_SLOPE_N}
    return {"ma_dist": out, "ma_slope": slope, "ma_align": align, "ma_dist_atr": dist_atr, "ma_slope_atr": slope_atr}


def bars_since(cond: np.ndarray, cap: int = SINCE_CAP) -> np.ndarray:
    """Bars since cond was last True (0 on the bar itself), capped at cap (also cap if it never happened)."""
    out = np.empty(len(cond))
    n = cap
    for i, c in enumerate(cond):
        n = 0 if c else min(n + 1, cap)
        out[i] = n
    return out


def _lag(x, L):
    if L == 0:
        return x
    out = np.full_like(x, np.nan)
    out[L:] = x[:-L]
    return out


def feature_matrix(ind: pd.DataFrame, fset: str | tuple = None):
    """(X float32 [n, F], names). K, D, J mapped to x/100 - 0.5; MACD lines divided by ATR(14) (scale-free);
    J extremes as (J-100)+/100 and (-J)+/100; bars-since features divided by SINCE_CAP (0..1)."""
    groups = FEATURE_SETS[fset or FEATURE_SET] if not isinstance(fset, tuple) else fset
    groups = [g2 for g in groups for g2 in GROUP_ALIAS.get(g, (g,))]
    a = ind["atr"].to_numpy()
    a = np.where(a > 0, a, np.nan)
    kdj = lambda col: ind[col].to_numpy() / 100 - 0.5
    base = {
        "k": kdj("k"), "d": kdj("d"), "j": kdj("j"),
        "dif": ind["dif"].to_numpy() / a, "osc": ind["osc"].to_numpy() / a,
        "kw": kdj("kw"), "dw": kdj("dw"), "jw": kdj("jw"),
        "difw": ind["difw"].to_numpy() / a, "oscw": ind["oscw"].to_numpy() / a,
    }
    J, Jw = ind["j"].to_numpy(), ind["jw"].to_numpy()
    jx = {
        "jhi": np.maximum(J - 100, 0) / 100, "jlo": np.maximum(-J, 0) / 100,
        "jhiw": np.maximum(Jw - 100, 0) / 100, "jlow": np.maximum(-Jw, 0) / 100,
        "since_jhi": bars_since(J > 100) / SINCE_CAP, "since_jlo": bars_since(J < 0) / SINCE_CAP,
        "since_jhiw": bars_since(Jw > 100) / SINCE_CAP, "since_jlow": bars_since(Jw < 0) / SINCE_CAP,
    }
    spec = {
        "kd": [(f, L) for L in LAGS for f in ("k", "d")],
        "jd": [(f, L) for L in LAGS for f in ("j", "d")],
        "j": [("j", L) for L in LAGS],
        "macd": [(f, L) for L in LAGS for f in ("dif", "osc")],
        "kdw": [(f, L) for L in SLOW_LAGS for f in ("kw", "dw")],
        "jdw": [(f, L) for L in SLOW_LAGS for f in ("jw", "dw")],
        "jw": [("jw", L) for L in SLOW_LAGS],
        "macdw": [(f, L) for L in SLOW_LAGS for f in ("difw", "oscw")],
    }
    extra = {"jx": jx}
    if any(g.startswith("ma_") for g in groups):
        extra.update(ma_features(ind))
    cols, names = [], []
    for g in groups:
        if g in extra:
            for f, x in extra[g].items():
                cols.append(x); names.append(f)
            continue
        for f, L in spec[g]:
            cols.append(_lag(base[f], L)); names.append(f"{f}_{L}")
    return np.column_stack(cols).astype(np.float32), names


def feature_names(fset=None):
    """Column names of feature_matrix() for a feature set (computed on a tiny dummy frame)."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    dummy = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=idx)
    return feature_matrix(indicator_frame(dummy), fset)[1]


# ---------------------------------------------------------------- base rule (no model)

def base_rule(ind: pd.DataFrame) -> np.ndarray:
    """Weekly-equivalent KDJ(45,15,15) + MACD(60,130,45) state machine ('Jext_MACDhold x5', RESEARCH.md K2):
    enter when slow J climbs back above 0 (J[t-1] < 0 <= J[t], leaving the oversold zone);
    exit when slow J drops back below 100 (J[t-1] > 100 >= J[t], leaving the overbought zone) UNLESS MACD still
    confirms the up-move (OSC > 0 and OSC rising); otherwise hold."""
    j, osc = ind["jw"].to_numpy(), ind["oscw"].to_numpy()
    pos = np.zeros(len(j))
    cur = 0.0
    for i in range(1, len(j)):
        if cur == 0.0 and j[i - 1] < 0 <= j[i]:
            cur = 1.0
        elif cur == 1.0 and j[i - 1] > 100 >= j[i] and not (osc[i] > 0 and osc[i] > osc[i - 1]):
            cur = 0.0
        pos[i] = cur
    return pos


def base_rule_kd(ind: pd.DataFrame) -> np.ndarray:
    """Previous round's K/D rule (for comparison only): enter slow K>D and OSC>0, exit slow K<D and OSC<0."""
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


def train_ensemble(X, y, seed_base, seeds=SEEDS):
    """Deterministic training of len(seeds) small MLPs (BCE loss, AdamW). Returns list of numpy layer lists."""
    torch.set_num_threads(2)
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        return _train(X, y, seed_base, seeds)
    finally:
        torch.use_deterministic_algorithms(was_deterministic)


def _train(X, y, seed_base, seeds):
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.astype(np.float32))
    lossf = torch.nn.BCEWithLogitsLoss()
    models = []
    for s in seeds:
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

def prepare(data, fset=None):
    """Per-stock features + cross-sectional labels. Label for (stock, t): open(t+1+H)/open(t+1)-1 beats the median
    of the same quantity across all stocks with a label on that date. A label is 'known' from the date at which the
    LAST stock's window on that date has ended (so the cross-sectional median never peeks past the purge date)."""
    stocks = []
    names = None
    for code in sorted(data):
        df = data[code]
        ind = indicator_frame(df)
        X, names = feature_matrix(ind, fset)
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


def walk_forward(data, fset=None, trainer=None, seeds=SEEDS, stocks=None):
    """Train one ensemble per calendar year (FIRST_YEAR .. last year in data) and predict that year.

    Returns (signals, models): signals[code] = DataFrame(p, ps, q_lo, q_hi, base) indexed like data[code] (NaN before
    FIRST_YEAR); models[year] = dict(models, mu, sd, q_lo, q_hi, n_train, names).
    trainer(Z, y, seed_base) -> list of layer lists (default: the MLP ensemble) lets research swap in other models."""
    if stocks is None:
        stocks, names = prepare(data, fset)
    else:
        stocks, names = stocks
    trainer = trainer or (lambda Z, y, sb: train_ensemble(Z, y, sb, seeds))
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
        ens = trainer(Ztr, ytr, Y)
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
    """walk_forward() memoised on the exact input data (the published DL variants share one training run)."""
    key = _fingerprint(data)
    if key not in _MEMO:
        _MEMO.clear()
        _MEMO[key] = walk_forward(data)
    return _MEMO[key]


# ---------------------------------------------------------------- purged k-fold cross-validation (evaluation only)

def _trainable(s):
    """Samples usable for training at all: walk_forward's filters without its date cut (label complete, `known`)."""
    return ((np.arange(len(s["y"])) >= WARM) & ~np.isnan(s["y"]) & ~np.isnat(s["known"])
            & ~np.isnan(s["X"]).any(axis=1))


def _ts(x):
    return pd.Timestamp(x).to_datetime64().astype("datetime64[ns]")


def purge_masks(stocks, start, end, last):
    """Per stock (train, purged, embargoed) boolean masks for a test fold [start, end] (end None = last date).

    Label window of sample t = from the first bar after t to `known` (the date the LAST stock's 60-bar window on that
    date closes, so the whole cross-sectional median is inside it). Purge = every trainable sample whose date t or
    label window falls in the fold, i.e. [t, known] overlaps [start, end] (all samples inside the fold, the last
    ~H+1 bars before it, and the fold's last bar whose window starts after it); embargo = the first HORIZON bars of
    each stock after the fold ends (their lagged / EMA / MA features still contain fold bars)."""
    fs, fe = _ts(start), (last if end is None else _ts(end))
    out = []
    for s in stocks:
        d = s["index"].values
        ok = _trainable(s)
        overlap = ok & (d <= fe) & (s["known"] >= fs)
        emb = np.zeros(len(d), bool)
        j = np.searchsorted(d, fe, side="right")
        emb[j : j + HORIZON] = True
        emb &= ok & ~overlap
        out.append((ok & ~overlap & ~emb, overlap, emb))
    return out


def purged_cv(data, folds, trainer=None, seeds=SEEDS, stocks=None):
    """Purged blocked k-fold CV of the published model (Lopez de Prado 2018, ch. 7). NOT tradable: fold f is traded
    with a model that saw every other fold, later ones included; the walk-forward `positions` remain the real record.

    For each fold f = (start, end) of `folds`:
      * train the same 3-net ensemble (same labels, filters, epochs, sample order) on every trainable sample of the
        other folds and of the 2008-2009 warm-up, minus the purge and embargo of purge_masks();
      * mu/sd and the Q_LO/Q_HI thresholds come from that training set only; seed base = first year of the fold
        (so the last fold's model is bit-identical to walk_forward's model of that year);
      * apply the model to every bar of the stock and smooth with the same 10-day EMA, so the EMA entering f is
        warmed up by model f itself (never by a model that was trained on f), then keep it on f's DECISION bars:
        bars whose fill (the next bar's open) lies in f, i.e. the last bar before f and every bar of f but its last.
    Returns (signals, models) shaped like walk_forward; signals also carry the column 'fold'."""
    if stocks is None:
        stocks, names = prepare(data)
    else:
        stocks, names = stocks
    trainer = trainer or (lambda Z, y, sb: train_ensemble(Z, y, sb, seeds))
    last = max(s["index"].values[-1] for s in stocks)
    cols = {s["code"]: {k: np.full(len(s["index"]), np.nan) for k in ("p", "ps", "q_lo", "q_hi")} for s in stocks}
    fold_of = {s["code"]: np.full(len(s["index"]), None, dtype=object) for s in stocks}
    models = {}
    for f, (start, end) in folds.items():
        masks = purge_masks(stocks, start, end, last)
        Xtr = np.concatenate([s["X"][m[0]] for s, m in zip(stocks, masks)])
        ytr = np.concatenate([s["y"][m[0]] for s, m in zip(stocks, masks)])
        mu, sd = standardize_fit(Xtr)
        Ztr = standardize_apply(Xtr, mu, sd)
        ens = trainer(Ztr, ytr, pd.Timestamp(start).year)
        q_lo, q_hi = np.quantile(predict_ensemble(ens, Ztr), [Q_LO, Q_HI])
        models[f] = dict(models=ens, mu=mu, sd=sd, q_lo=float(q_lo), q_hi=float(q_hi), n_train=int(len(ytr)),
                         n_purged=int(sum(m[1].sum() for m in masks)), n_embargo=int(sum(m[2].sum() for m in masks)),
                         names=names)
        fs, fe = _ts(start), (last if end is None else _ts(end))
        for s in stocks:
            d = s["index"].values
            fill = np.r_[d[1:], d[-1:]]            # the last bar of the data counts as its own fill date
            m = (fill >= fs) & (fill <= fe)
            if not m.any():
                continue
            p = pd.Series(predict_ensemble(ens, standardize_apply(s["X"], mu, sd)))
            ps = p.ewm(span=SMOOTH, adjust=False).mean()
            c = cols[s["code"]]
            c["p"][m], c["ps"][m] = p.to_numpy()[m], ps.to_numpy()[m]
            c["q_lo"][m], c["q_hi"][m] = q_lo, q_hi
            fold_of[s["code"]][m] = f
    signals = {s["code"]: pd.DataFrame({**cols[s["code"]], "base": base_rule(s["ind"]), "fold": fold_of[s["code"]]},
                                       index=s["index"]) for s in stocks}
    return signals, models


_CV_MEMO: dict = {}


def cached_purged_cv(data, folds):
    """purged_cv() memoised on the input data and folds (floor and gate share one CV training run)."""
    key = (_fingerprint(data), tuple((f, str(a), str(b)) for f, (a, b) in folds.items()))
    if key not in _CV_MEMO:
        _CV_MEMO.clear()
        _CV_MEMO[key] = purged_cv(data, folds)
    return _CV_MEMO[key]
