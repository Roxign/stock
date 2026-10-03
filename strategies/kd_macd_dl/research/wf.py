"""Walk-forward experiment driver (direct classification) for in-sample research."""
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from harness import data
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
import core_research as core

CACHE = Path(__import__("tempfile").gettempdir()) / "kd_macd_dl_research_cache"
CACHE.mkdir(exist_ok=True)
WARM = 150


def pooled(cfg):
    d = data()
    rows = []
    for ci, (code, df) in enumerate(sorted(d.items())):
        ind = core.indicator_frame(df)
        X, names = core.feature_matrix(ind, cfg.get("lags", (0, 1, 2, 3, 4)), cfg.get("slow_lags", (0, 5)),
                                       cfg.get("groups", ("kd", "macd", "slow")))
        ret, end = core.fwd_label(ind, cfg["H"])
        n = len(df)
        end_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        ok = end < n
        end_date[ok] = df.index.values[end[ok]]
        atrp = (ind["atr"] / ind["close"]).to_numpy()
        if cfg.get("label") == "dd":
            lo = df["low"].to_numpy(); o = df["open"].to_numpy(); H = cfg["H"]
            mn = pd.Series(lo[::-1]).rolling(H, min_periods=H).min().to_numpy()[::-1]  # min low over [t, t+H-1]
            dd = np.full(n, np.nan)
            ok2 = np.arange(n) + 1 + H <= n
            idx = np.arange(n)[ok2]
            dd[idx] = mn[idx + 1] / o[idx + 1] - 1 if True else 0
            ret = np.where(ok, dd, np.nan)
        cond = np.ones(n, dtype=bool)
        if cfg.get("cond"):
            from mapping import rule_pos
            cond = rule_pos(df, cfg["cond"]).to_numpy() == 0
        rows.append(dict(code=code, dates=df.index.values, X=X, ret=ret, end_date=end_date, bar=np.arange(n), atrp=atrp, cond=cond))
    return rows, names


_XS = {}


def xs_median(cfg):
    key = cfg["H"]
    if key not in _XS:
        rows, _ = pooled({**cfg, "label": "pos"})
        s = pd.concat({r["code"]: pd.Series(r["ret"], index=pd.DatetimeIndex(r["dates"])) for r in rows}, axis=1)
        _XS[key] = s.median(axis=1)
    return _XS[key]


def label_of(cfg, r):
    ret = r["ret"]
    kind = cfg.get("label", "pos")
    if kind == "xs":
        med = xs_median(cfg).reindex(pd.DatetimeIndex(r["dates"])).to_numpy()
        y = (ret > med).astype(float)
        y[np.isnan(ret) | np.isnan(med)] = np.nan
        return y
    if kind == "pos":
        y = (ret > 0).astype(float)
    elif kind == "dd":
        y = (ret < -cfg.get("thr", 0.1)).astype(float)
    elif kind == "neg":
        y = (ret < 0).astype(float)
    elif kind == "cost":
        y = (ret > 0.006).astype(float)
    elif kind == "volnorm":  # return beyond +0.5 ATR-equivalent per sqrt horizon
        y = (ret / (r["atrp"] * np.sqrt(cfg["H"])) > cfg.get("thr", 0.0)).astype(float)
    y[np.isnan(ret)] = np.nan
    return y


def run(cfg, verbose=True):
    key = hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
    path = CACHE / f"direct_{key}.pkl"
    if path.exists():
        return _expand(pickle.load(open(path, "rb")))
    t0 = time.time()
    rows, names = pooled(cfg)
    first, last = cfg.get("first_year", 2012), cfg.get("last_year", 2020)
    preds = {r["code"]: np.full(len(r["dates"]), np.nan) for r in rows}
    thr = {r["code"]: np.full((len(r["dates"]), 3), np.nan) for r in rows}
    stride = cfg.get("stride", 1)
    for Y in range(first, last + 1):
        cut = np.datetime64(f"{Y}-01-01")
        start = np.datetime64(f"{Y - cfg['window']}-01-01") if cfg.get("window") else None
        Xs, ys = [], []
        for r in rows:
            y = label_of(cfg, r)
            m = (r["bar"] >= WARM) & (r["end_date"] < cut) & ~np.isnan(y) & (r["bar"] % stride == 0) & r["cond"]
            if start is not None:
                m &= r["dates"] >= start
            m &= ~np.isnan(r["X"]).any(axis=1)
            Xs.append(r["X"][m]); ys.append(y[m])
        Xtr = np.concatenate(Xs); ytr = np.concatenate(ys)
        mu, sd = core.standardize_fit(Xtr)
        Ztr = core.standardize_apply(Xtr, mu, sd)
        if cfg.get("model", "mlp") == "mlp":
            models = core.train_ensemble(Ztr, ytr, hidden=cfg.get("hidden", (16, 8)), seeds=tuple(range(cfg.get("ens", 3))),
                                         epochs=cfg.get("epochs", 10), batch=cfg.get("batch", 512), lr=cfg.get("lr", 2e-3),
                                         wd=cfg.get("wd", 1e-4), seed_base=Y)
            predf = lambda Z: core.predict_ensemble(models, Z)
        else:
            from sklearn.linear_model import LogisticRegression
            lrm = LogisticRegression(C=cfg.get("C", 1.0), max_iter=2000).fit(Ztr, ytr)
            predf = lambda Z: lrm.predict_proba(Z)[:, 1]
        ptr = predf(Ztr)
        q = np.quantile(ptr, [0.2, 0.4, 0.6])
        for r in rows:
            m = (r["dates"] >= cut) & (r["dates"] < np.datetime64(f"{Y + 1}-01-01"))
            if m.any():
                preds[r["code"]][m] = predf(core.standardize_apply(r["X"][m], mu, sd))
                thr[r["code"]][m] = q
        if verbose:
            print(f"  {Y}: n_train={len(ytr)} base={ytr.mean():.3f} p_tr mean={ptr.mean():.3f} q={np.round(q,3)} ({time.time()-t0:.0f}s)")
    out = {"preds": {r["code"]: pd.Series(preds[r["code"]], index=pd.DatetimeIndex(r["dates"])) for r in rows},
           "thr": {r["code"]: pd.DataFrame(thr[r["code"]], index=pd.DatetimeIndex(r["dates"]), columns=["q20", "q40", "q60"]) for r in rows},
           "names": names, "cfg": cfg}
    pickle.dump({"preds": {c: s.to_numpy(np.float32) for c, s in out["preds"].items()},
                 "thr": {c: t.to_numpy(np.float32) for c, t in out["thr"].items()}, "names": names, "cfg": cfg},
                open(path, "wb"))
    return out


def _expand(z):
    d = data()
    return {"preds": {c: pd.Series(v.astype(float), index=d[c].index) for c, v in z["preds"].items()},
            "thr": {c: pd.DataFrame(v.astype(float), index=d[c].index, columns=["q20", "q40", "q60"]) for c, v in z["thr"].items()},
            "names": z["names"], "cfg": z["cfg"]}


def auc_report(out, cfg, years=None):
    """Pooled AUC of predictions vs realised labels for the predicted (out-of-training) rows."""
    from sklearn.metrics import roc_auc_score
    rows, _ = pooled(cfg)
    P, Y, R = [], [], []
    for r in rows:
        p = out["preds"][r["code"]].to_numpy()
        y = label_of(cfg, r)
        m = ~np.isnan(p) & ~np.isnan(y) & r["cond"]
        P.append(p[m]); Y.append(y[m]); R.append(r["ret"][m])
    P, Y, R = map(np.concatenate, (P, Y, R))
    q = pd.qcut(P, 5, labels=False, duplicates="drop")
    by = pd.DataFrame({"q": q, "r": R}).groupby("q")["r"].mean() * 100
    return roc_auc_score(Y, P), by.round(2).to_dict()
