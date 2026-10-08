"""Research harness for the E1 direct-position network (in-sample development; see RESEARCH.md).

  python -m strategies.dl_position.research wf  NAME key=value ...   # walk-forward a config, cache raw outputs
  python -m strategies.dl_position.research map NAME floor span step  # evaluate one position mapping in-sample
  python -m strategies.dl_position.research base KIND ...             # ridge / HGB baselines on the same inputs

Raw model outputs are cached as pickles in DL_POSITION_CACHE (default: the scratch folder of this session; not
committed). Every evaluated configuration is logged with stocklab.trials.log_trial(agent="dl_position").
"""

from __future__ import annotations

import ast
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stocklab import backtest as bt
from stocklab import controls as ctl
from stocklab.data import load_all
from stocklab.trials import log_trial

from . import core

AGENT = "dl_position"
CACHE = Path(os.environ.get("DL_POSITION_CACHE", Path(os.environ.get("TEMP", ".")) / "dl_position_cache"))
IS12 = ("2012-01-01", "2020-12-31")   # in-sample years with a walk-forward model
bt.ALL_PERIODS.setdefault("is12", IS12)   # runtime-only alias so stocklab.controls can calibrate on 2012-2020


def bounds(index, start, end):
    i0 = max(index.searchsorted(pd.Timestamp(start)), bt.WARMUP)
    i1 = index.searchsorted(pd.Timestamp(end), side="right") - 1
    return (i0, i1) if i1 - i0 >= 20 else None


def run_span(data, pos, start, end):
    out = {}
    for c, df in data.items():
        b = bounds(df.index, start, end)
        if b is None:
            continue
        eq, inv, trades, orders = bt.simulate(df, pos[c], *b)
        out[c] = bt.metrics(eq, inv, trades, orders)
    return out


def summary(ms, ref=None):
    def med(k):
        return float(np.nanmedian([m[k] for m in ms.values()]))
    s = {k: med(k) for k in ("cagr", "sharpe", "mdd", "exposure")}
    s["orders_y"] = float(np.median([m["orders"] / m["years"] for m in ms.values()]))
    if ref:
        s["beat_bh_sharpe"] = float(np.mean([ms[c]["sharpe"] > ref[c]["sharpe"] for c in ms if c in ref]))
    s["n"] = len(ms)
    return s


def fmt(name, s):
    extra = f" beatBH {s['beat_bh_sharpe']:.0%}" if "beat_bh_sharpe" in s else ""
    return (f"{name:<34} CAGR {s['cagr']:6.1%} Sharpe {s['sharpe']:5.2f} MDD {s['mdd']:6.1%} "
            f"expo {s['exposure']:5.1%} orders/y {s['orders_y']:5.1f}{extra}")


def bh(data, start, end):
    return run_span(data, {c: pd.Series(1.0, index=df.index) for c, df in data.items()}, start, end)


def positions_from_raw(raw, col="w", **m):
    return {c: core.to_positions(core.target(df, col, m["floor"]), **m) for c, df in raw.items()}


def controls_is(data, pos, period="is"):
    """B1 constant exposure, B2 shuffled signal, B3 vol target on a stocklab period (calibrated on it)."""
    shuf = ctl.shuffled(pos, data)
    res = {"const": {}, "shuffle": {}, "voltarget": {}}
    for c, df in data.items():
        b = bt.period_bounds(df.index, period)
        if b is None:
            continue
        for kind, p in (("const", ctl.constant_exposure(pos[c], df, period)), ("shuffle", shuf[c]),
                        ("voltarget", ctl.vol_target(pos[c], df, period))):
            eq, inv, trades, orders = bt.simulate(df, p, *b)
            res[kind][c] = bt.metrics(eq, inv, trades, orders)
    return res


def evaluate(data, pos, name, config, log=True, with_controls=False, quiet=False):
    """In-sample summary (2010-2020 = stocklab 'is', and 2012-2020 where every year has a model)."""
    out = {}
    for tag, (a, b) in (("is", bt.PERIODS["is"]), ("is12", IS12)):
        ref = bh(data, a, b)
        s = summary(run_span(data, pos, a, b), ref)
        out[tag] = s
        if not quiet:
            print(fmt(f"{name} [{tag}]", s))
    if log:
        log_trial(AGENT, name, config, {**out["is"], **{f"is12_{k}": v for k, v in out["is12"].items()}}, period="is")
    if with_controls:
        for tag in ("is", "is12"):
            ref = bh(data, *bt.ALL_PERIODS[tag])
            for kind, ms in controls_is(data, pos, tag).items():
                s = summary(ms, ref)
                out[f"ctl_{kind}_{tag}"] = s
                print(fmt(f"  ~{kind} [{tag}]", s))
    return out


def print_refs(data):
    for tag in ("is", "is12"):
        a, b = bt.ALL_PERIODS[tag]
        print(fmt(f"buy_hold [{tag}]", summary(bh(data, a, b))))


def save(name, obj):
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(obj, CACHE / f"{name}.pkl")


def load(name):
    return pd.read_pickle(CACHE / f"{name}.pkl")


def parse_kv(args):
    cfg = {}
    for a in args:
        k, v = a.split("=", 1)
        try:
            cfg[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            cfg[k] = v
    return cfg


def cut_is(data, end="2020-12-31"):
    """Data truncated at the end of the in-sample period (development never sees 2021+)."""
    return {c: df.loc[:end] for c, df in data.items() if df.index[0] < pd.Timestamp(end)}


def cmd_wf(name, cfg):
    data = cut_is(load_all())
    t = time.time()
    raw, models = core.walk_forward(data, cfg, verbose=True)
    info = {Y: {k: v for k, v in m.items() if k not in ("params", "mu", "sd")} for Y, m in models.items()}
    save(f"wf_{name}", dict(cfg=cfg, raw=raw, info=info))
    print(f"walk-forward {name}: {time.time() - t:.0f}s")
    return data, raw


def cmd_map(name, floor, spans=(1, 5, 10, 20), steps=(0.25, 0.5), seeds=False, cols=("w",)):
    data = cut_is(load_all())
    obj = load(f"wf_{name}")
    raw, cfg = obj["raw"], obj["cfg"]
    for col in cols:
        for span in spans:
            for step in steps:
                m = dict(floor=floor, span=span, step=step)
                pos = positions_from_raw(raw, col=col, **m)
                evaluate(data, pos, f"{name}_{col}_s{span}_q{step}", {**cfg, **m, "col": col, "model": name})
    if seeds:
        seed_cols = [c for c in next(iter(raw.values())).columns if c.startswith("s")]
        m = dict(floor=floor, span=spans[0], step=steps[0])
        vals = []
        for col in seed_cols:
            pos = positions_from_raw(raw, col=col, **m)
            vals.append(evaluate(data, pos, f"{name}_{col}", {**cfg, **m, "seed_col": col}, quiet=True)["is12"])
        sh = [v["sharpe"] for v in vals]
        cg = [v["cagr"] for v in vals]
        print(f"per-seed is12 Sharpe {np.mean(sh):.3f} +- {np.std(sh):.3f}  CAGR {np.mean(cg):.2%} +- {np.std(cg):.2%}")


def _fwd(s, H):
    """Vol-normalised forward return open[t+1] -> open[t+1+H] (winsorised) and the date its window closes."""
    lr = pd.Series(np.log1p(s["r"]))
    fwd = lr.rolling(H).sum().shift(-(H - 1)).to_numpy()
    y = np.clip(fwd / (s["sig"] * math.sqrt(H)), -5, 5)
    dH = np.full(len(y), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    dH[: len(y) - (H + 1)] = s["index"].values[H + 1 :]
    return y, dH


def baseline_walk_forward(data, kind="ridge", H=20, floor=0.0, fset="all"):
    """Predict-then-map baselines on the same inputs: ridge or HistGradientBoosting regression of the H-day
    vol-normalised forward return, same yearly expanding walk-forward (labels closed before Jan 1, purge = H bars
    + EMBARGO), standardisation from the training rows; prediction -> position by its percentile among the training
    predictions: w = floor + (1 - floor) * F_train(pred)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    stocks, _ = core.prepare(data, fset)
    labels = [_fwd(s, H) for s in stocks]
    last_year = max(s["index"][-1].year for s in stocks)
    out = {s["code"]: np.full(len(s["index"]), np.nan) for s in stocks}
    for Y in range(core.FIRST_YEAR, last_year + 1):
        cut = core._ts(f"{Y}-01-01")
        nxt = core._ts(f"{Y + 1}-01-01")
        Xs, ys = [], []
        for s, (y, dH) in zip(stocks, labels):
            j = np.searchsorted(s["index"].values, cut)
            emb_cut = s["index"].values[max(j - core.EMBARGO, 0)] if j < len(s["index"]) else cut
            m = core._trainable(s) & np.isfinite(y) & (dH < emb_cut)
            Xs.append(s["X"][m]); ys.append(y[m])
        X, y = np.concatenate(Xs), np.concatenate(ys)
        mu, sd = core.standardize_fit(X)
        Z = core.standardize_apply(X, mu, sd)
        if kind == "ridge":
            mdl = Ridge(alpha=10.0).fit(Z, y)
        else:
            mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                min_samples_leaf=500, l2_regularization=1.0, early_stopping=False,
                                                random_state=Y).fit(Z, y)
        ref = np.sort(mdl.predict(Z))
        for s in stocks:
            d = s["index"].values
            m = (d >= cut) & (d < nxt) & s["core"]
            if m.any():
                p = mdl.predict(core.standardize_apply(s["X"][m], mu, sd))
                out[s["code"]][m] = np.searchsorted(ref, p) / len(ref)
    return {s["code"]: pd.DataFrame({"w": floor + (1 - floor) * out[s["code"]], "pct": out[s["code"]]}, index=s["index"])
            for s in stocks}


def cmd_base(kind, H, floor, fset="all"):
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    data = cut_is(load_all())
    t = time.time()
    raw = baseline_walk_forward(data, kind, H, floor, fset)
    name = f"base_{kind}_H{H}_f{floor}_{fset}"
    save(f"wf_{name}", dict(cfg=dict(kind=kind, H=H, floor=floor, features=fset), raw=raw, info={}))
    print(f"{name}: {time.time() - t:.0f}s")
    for col in ("w", "pct2"):
        for span in (5, 10, 20):
            m = dict(floor=floor, span=span, step=0.25)
            evaluate(data, positions_from_raw(raw, col=col, **m), f"{name}_{col}_s{span}",
                     dict(kind=kind, H=H, features=fset, col=col, **m))


IS_FOLDS = ("f1", "f2", "f3", "f4", "f5")


def fold_medians(data, pos, folds=IS_FOLDS):
    out = {}
    for f in folds:
        ms = [r["metrics"] for r in (bt.run_strategy(df, pos[c], f) for c, df in data.items()) if r]
        out[f] = {k: float(np.nanmedian([m[k] for m in ms])) for k in ("sharpe", "cagr", "mdd", "exposure")}
    return out


def print_folds(name, fm):
    print(f"{name:<30}" + "".join(f"  {f}:{v['sharpe']:5.2f}/{v['exposure']:4.0%}" for f, v in fm.items())
          + f"  mean {np.mean([v['sharpe'] for v in fm.values()]):.3f}")


def cmd_cv(name, cfg, mapping, refs=False):
    """Purged CV of a config (trained on full data minus each fold); prints ONLY the in-sample folds f1-f5."""
    data = load_all()
    key = f"cv_{name}"
    col = mapping.pop("col", "w")
    try:
        obj = load(key)
    except FileNotFoundError:
        t = time.time()
        by_fold, models = core.purged_cv(data, dict(bt.FOLDS), cfg, verbose=True)
        obj = dict(cfg=cfg, by_fold={f: {c: df[["w", "pct", "wn", "in_fold"]] for c, df in r.items()}
                                     for f, r in by_fold.items()},
                   info={f: {k: v for k, v in m.items() if k not in ("params", "mu", "sd", "ref")}
                         for f, m in models.items()})
        save(key, obj)
        print(f"purged cv {name}: {time.time() - t:.0f}s")
    if refs:
        from strategies.trend.strategy import sma_half
        print_folds("buy_hold", fold_medians(data, {c: pd.Series(1.0, index=df.index) for c, df in data.items()}))
        print_folds("trend_sma_half", fold_medians(data, sma_half(data)))
    pos = core.cv_to_positions(obj["by_fold"], col=col, **mapping)
    fm = fold_medians(data, pos)
    print_folds(f"{name} {col} {mapping['span']}/{mapping['step']}", fm)
    log_trial(AGENT, f"cv_{name}_{col}_s{mapping['span']}_q{mapping['step']}", {**cfg, **mapping, "col": col, "cv": "purged f1-f5"},
              {"sharpe": float(np.mean([v["sharpe"] for v in fm.values()])),
               **{f"{f}_sharpe": v["sharpe"] for f, v in fm.items()}}, period="cv_is_folds")
    return fm


def cmd_refs():
    """Buy-and-hold, trend_sma_half and (cached) kdj_macd_dl_floor on the in-sample spans."""
    from stocklab.runner import POS_CACHE
    from strategies.trend.strategy import sma_half

    data = cut_is(load_all())
    print_refs(data)
    evaluate(data, sma_half(data), "ref_trend_sma_half", {}, log=False)
    p = POS_CACHE / "kdj_macd_dl_floor.pkl"
    if p.exists():
        evaluate(data, pd.read_pickle(p), "ref_kdj_macd_dl_floor", {}, log=False)


def cmd_ctl(name, m):
    data = cut_is(load_all())
    obj = load(f"wf_{name}")
    pos = positions_from_raw(obj["raw"], **m)
    evaluate(data, pos, f"{name}_ctl", {**obj["cfg"], **m}, log=False, with_controls=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "refs":
        cmd_refs()
    elif cmd == "ctl":
        cmd_ctl(sys.argv[2], parse_kv(sys.argv[3:]))
    elif cmd == "cv":
        kv = parse_kv(sys.argv[3:])
        mapping = {k: kv.pop(k) for k in ("floor", "span", "step", "col") if k in kv}
        refs = kv.pop("refs", False)
        cmd_cv(sys.argv[2], {**kv, "floor": mapping["floor"]}, mapping, refs)
    elif cmd == "base":
        kv = parse_kv(sys.argv[2:])
        cmd_base(kv.get("kind", "ridge"), kv.get("H", 20), kv.get("floor", 0.0), kv.get("features", "all"))
    if cmd == "wf":
        cmd_wf(sys.argv[2], parse_kv(sys.argv[3:]))
    elif cmd == "map":
        kv = parse_kv(sys.argv[3:])
        cmd_map(sys.argv[2], kv.get("floor", 0.0), kv.get("spans", (1, 5, 10, 20)), kv.get("steps", (0.25, 0.5)),
                kv.get("seeds", False), kv.get("cols", ("w",)))
