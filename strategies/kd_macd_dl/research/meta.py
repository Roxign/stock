"""Meta-labeling experiments: base KD/MACD rule proposes trades, MLP predicts P(trade profitable)."""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from harness import data, evaluate
import core_research as core
from stocklab.backtest import BUY_FEE, SELL_FEE

CACHE = Path(__import__("tempfile").gettempdir()) / "kd_macd_dl_research_cache"
CACHE.mkdir(exist_ok=True)
WARM = 150


def cross_up(a, b):
    return (a > b) & (np.r_[False, a[:-1] <= b[:-1]])


def cross_dn(a, b):
    return (a < b) & (np.r_[False, a[:-1] >= b[:-1]])


def base_signals(ind, rule):
    k, d = ind["k"].to_numpy(), ind["d"].to_numpy()
    osc, dif = ind["osc"].to_numpy(), ind["dif"].to_numpy()
    kw, dw, oscw, difw = ind["kw"].to_numpy(), ind["dw"].to_numpy(), ind["oscw"].to_numpy(), ind["difw"].to_numpy()
    if rule == "both1":
        return (k > d) & (osc > 0), (k < d) & (osc < 0)
    if rule == "kdx_osc":  # KD golden cross + OSC rising ; exit KD death cross + OSC falling
        osc_up = np.r_[False, osc[1:] > osc[:-1]]
        return cross_up(k, d) & osc_up, cross_dn(k, d) & ~osc_up
    if rule == "kdx":  # pure KD cross
        return cross_up(k, d), cross_dn(k, d)
    if rule == "dip":  # KD low-zone golden cross in a slow-MACD uptrend; exit K>80 cross down or slow trend breaks
        return cross_up(k, d) & (d < 40) & (difw > 0), (cross_dn(k, d) & (d > 70)) | (difw < 0) & (oscw < 0)
    if rule == "dip2":  # KD low-zone golden cross (no trend filter); exit KD death cross above 70 or OSC<0 & K<D
        return cross_up(k, d) & (d < 30), (cross_dn(k, d) & (d > 70))
    if rule == "macdx_kd":  # MACD golden cross (OSC turns positive) with K>D; exit OSC turns negative
        return cross_up(osc, np.zeros_like(osc)) & (k > d), (osc < 0)
    raise ValueError(rule)


def trades_from(enter, exit_):
    """Return list of (entry_bar, exit_bar) with exit_bar=None for open trade; signal bars (fills at +1)."""
    out, cur = [], None
    for i in range(len(enter)):
        if cur is None and enter[i]:
            cur = i
        elif cur is not None and exit_[i]:
            out.append((cur, i))
            cur = None
    if cur is not None:
        out.append((cur, None))
    return out


_P = None


def open_panel():
    global _P
    if _P is None:
        _P = pd.concat({c: df["open"] for c, df in data().items()}, axis=1)
    return _P


def build(cfg):
    d = data()
    stocks = []
    for code, df in sorted(d.items()):
        ind = core.indicator_frame(df)
        X, names = core.feature_matrix(ind, cfg.get("lags", (0, 1, 2, 3, 4)), cfg.get("slow_lags", (0, 5)),
                                       cfg.get("groups", ("kd", "macd", "slow")))
        if "_x" in cfg["rule"]:
            from mapping import rule_pos
            bp = rule_pos(df, cfg["rule"]).to_numpy()
            en = (bp == 1) & (np.r_[0, bp[:-1]] == 0)
            ex = (bp == 0) & (np.r_[0, bp[:-1]] == 1)
        else:
            en, ex = base_signals(ind, cfg["rule"])
        tr = trades_from(en, ex)
        o = df["open"].to_numpy()
        n = len(df)
        ev = []
        for (a, b) in tr:
            if b is not None and b + 1 < n and a + 1 < n:
                r = o[b + 1] * (1 - SELL_FEE) / (o[a + 1] * (1 + BUY_FEE)) - 1
                end = df.index.values[b + 1]
            else:
                r, end = np.nan, np.datetime64("NaT")
            if b is not None and b + 1 < n and a + 1 < n and cfg.get("label") == "xs":
                d0, d1 = df.index[a + 1], df.index[b + 1]
                P = open_panel()
                if d0 in P.index and d1 in P.index:
                    med = np.nanmedian(P.loc[d1].to_numpy() / P.loc[d0].to_numpy())
                    r = o[b + 1] / o[a + 1] - med
            ev.append((a, b, r, end))
        stocks.append(dict(code=code, df=df, X=X, trades=ev, dates=df.index.values))
    return stocks, names


def run(cfg, verbose=False):
    key = hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
    path = CACHE / f"meta_{key}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    stocks, names = build(cfg)
    first, last = cfg.get("first_year", 2012), cfg.get("last_year", 2020)
    # event table
    E = []
    for si, s in enumerate(stocks):
        for (a, b, r, end) in s["trades"]:
            E.append((si, a, b, r, end, s["dates"][a]))
    E = pd.DataFrame(E, columns=["si", "a", "b", "r", "end", "date"])
    E["p"] = np.nan
    Xall = np.stack([stocks[si]["X"][a] for si, a in zip(E["si"], E["a"])])
    okX = ~np.isnan(Xall).any(axis=1) & (E["a"].to_numpy() >= WARM)
    for Y in range(first, last + 1):
        cut = np.datetime64(f"{Y}-01-01")
        m = okX & (E["end"].to_numpy() < cut) & ~np.isnan(E["r"].to_numpy())
        if cfg.get("window"):
            m &= E["date"].to_numpy() >= np.datetime64(f"{Y - cfg['window']}-01-01")
        Xtr, r = Xall[m], E["r"].to_numpy()[m]
        y = (r > cfg.get("min_ret", 0.0)).astype(float)
        w = np.abs(r) if cfg.get("weight") == "absret" else None
        mu, sd = core.standardize_fit(Xtr)
        if cfg.get("model", "mlp") == "mlp":
            models = core.train_ensemble(core.standardize_apply(Xtr, mu, sd), y, w, hidden=cfg.get("hidden", (8, 4)),
                                         seeds=tuple(range(cfg.get("ens", 5))), epochs=cfg.get("epochs", 30),
                                         batch=cfg.get("batch", 128), lr=cfg.get("lr", 2e-3), wd=cfg.get("wd", 1e-3), seed_base=Y)
            pred = lambda Z: core.predict_ensemble(models, Z)
        else:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=cfg.get("C", 0.1), max_iter=1000).fit(core.standardize_apply(Xtr, mu, sd), y, sample_weight=w)
            pred = lambda Z: lr.predict_proba(Z)[:, 1]
        mt = (E["date"].to_numpy() >= cut) & (E["date"].to_numpy() < np.datetime64(f"{Y + 1}-01-01"))
        if mt.any():
            E.loc[mt, "p"] = pred(core.standardize_apply(np.nan_to_num(Xall[mt]), mu, sd))
        ptr = pred(core.standardize_apply(Xtr, mu, sd))
        E.loc[mt, "q50"] = np.quantile(ptr, 0.5)
        E.loc[mt, "q33"] = np.quantile(ptr, 0.33)
        E.loc[mt, "q67"] = np.quantile(ptr, 0.67)
        if verbose:
            print(f"  {Y}: n={m.sum()} win={y.mean():.2f} meanR={r.mean():.4f} ptr={ptr.mean():.3f}")
    out = dict(E=E, stocks=[(s["code"], s["df"].index) for s in stocks], cfg=cfg)
    pickle.dump(out, open(path, "wb"))
    return out


def positions(out, mode="abs", lo=0.5, hi=0.55, fallback="base"):
    E = out["E"]
    pos = {}
    for si, (code, idx) in enumerate(out["stocks"]):
        v = np.zeros(len(idx))
        for _, e in E[E["si"] == si].iterrows():
            a, b = int(e["a"]), (len(idx) - 1 if pd.isna(e["b"]) else int(e["b"]))
            p = e["p"]
            if np.isnan(p):
                size = 1.0 if fallback == "base" else 0.0
            elif mode == "abs":
                size = 1.0 if p >= hi else (0.5 if p >= lo else 0.0)
            elif mode == "q":
                size = 1.0 if p >= e[f"q{int(hi*100)}"] else (0.5 if p >= e[f"q{int(lo*100)}"] else 0.0)
            v[a:b] = size  # in position from signal bar a through b-1; exit signal at b -> target 0 at b
        pos[code] = pd.Series(v, index=idx)
    return pos


def base_positions(out):
    pos = {}
    for si, (code, idx) in enumerate(out["stocks"]):
        v = np.zeros(len(idx))
        for _, e in out["E"][out["E"]["si"] == si].iterrows():
            a, b = int(e["a"]), (len(idx) - 1 if pd.isna(e["b"]) else int(e["b"]))
            v[a:b] = 1.0
        pos[code] = pd.Series(v, index=idx)
    return pos


def report(out):
    from sklearn.metrics import roc_auc_score
    E = out["E"].dropna(subset=["p", "r"])
    E = E[E["date"] <= np.datetime64("2020-12-31")]
    auc = roc_auc_score(E["r"] > 0, E["p"])
    q = pd.qcut(E["p"], 5, labels=False, duplicates="drop")
    g = E.groupby(q)["r"].agg(["mean", "count", lambda s: (s > 0).mean()])
    return auc, (g["mean"] * 100).round(2).tolist(), g.iloc[:, 2].round(2).tolist(), len(E)
