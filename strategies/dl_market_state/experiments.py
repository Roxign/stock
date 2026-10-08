"""In-sample research harness for dl_market_state (RESEARCH.md tables). Development only looks at 2010-2020:
walk-forward models are retrained up to 2020 and the dev cross-validation (folds f1-f5) trains only on labels known by
2020-12-31. Every backtested configuration is logged to research/trials.jsonl.

  python -m strategies.dl_market_state.experiments ic            # stage 1: walk-forward IC / AUC of each model
  python -m strategies.dl_market_state.experiments pos           # stage 2: positions vs B&H / controls (IS)
  python -m strategies.dl_market_state.experiments cv            # stage 3: dev purged CV on f1-f5
"""

import json
import sys
import time

import numpy as np
import pandas as pd

from stocklab import backtest as bt
from stocklab import controls as ctl
from stocklab import portfolio as pf
from stocklab.data import load_all
from stocklab.trials import log_trial

from . import core

AGENT = "dl_market_state"
DEV_END = 2020
DEV_FOLDS = {f: bt.FOLDS[f] for f in ("f1", "f2", "f3", "f4", "f5")}
PERIODS = ("is",) + tuple(DEV_FOLDS)

MODELS = {
    "ridge_a1": ("ridge", {"alpha": 1.0}),
    "ridge_a10": ("ridge", {"alpha": 10.0}),
    "ridge_a01": ("ridge", {"alpha": 0.1}),
    "logit_c001": ("logit", {"C": 0.01}),
    "logit_c0001": ("logit", {"C": 0.001}),
    "gbdt_d2": ("gbdt", {"depth": 2, "iters": 150, "lr": 0.03, "leaf": 200}),
    "mlp16": ("mlp", {"hidden": (16,), "gamma": 3.0}),
}


def spearman(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 30:
        return np.nan
    ra, rb = pd.Series(a[m]).rank().to_numpy(), pd.Series(b[m]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(score, y):
    m = ~np.isnan(score) & ~np.isnan(y)
    s, y = score[m], y[m].astype(bool)
    if y.all() or (~y).all():
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    n1 = y.sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (~y).sum()))


# ------------------------------------------------------------------------------------------------ evaluation helpers

_BH: dict = {}


def bh_metrics(data):
    if not _BH:
        for p in PERIODS:
            _BH[p] = {c: bt.run_buy_hold(df, p) for c, df in data.items()}
    return _BH


def eval_positions(pos, data, periods=PERIODS, portfolio=True):
    bh = bh_metrics(data)
    out = {}
    for p in periods:
        res = {c: bt.run_strategy(df, pos[c], p) for c, df in data.items()}
        codes = [c for c, r in res.items() if r]
        med = lambda k: float(np.nanmedian([res[c]["metrics"].get(k, np.nan) for c in codes]))
        out[p] = {k: med(k) for k in ("sharpe", "cagr", "mdd", "exposure", "orders")}
        out[p]["beat_bh_sharpe"] = float(np.mean([res[c]["metrics"]["sharpe"] > bh[p][c]["metrics"]["sharpe"]
                                                  for c in codes]))
    if portfolio:
        r = pf.run(data, pf.per_stock_weights(pos, data), "is")
        out["pf_is"] = {k: r["metrics"][k] for k in ("sharpe", "cagr", "mdd", "exposure", "turnover")}
    return out


def controls_is(pos, data, period="is"):
    """B1 constant exposure and B3 vol target (per stock, calibrated to the same average exposure in `period`)."""
    out = {}
    for kind in ("const", "voltarget"):
        res = {}
        for c, df in data.items():
            ctrl = ctl.constant_exposure(pos[c], df, period) if kind == "const" else ctl.vol_target(pos[c], df, period)
            res[c] = None if ctrl is None else bt.run_strategy(df, ctrl, period)
        codes = [c for c, r in res.items() if r]
        out[kind] = {k: float(np.nanmedian([res[c]["metrics"][k] for c in codes])) for k in ("sharpe", "cagr", "mdd")}
    return out


def fmt(m):
    return f"Sh {m['sharpe']:5.2f} CAGR {m['cagr']:6.1%} MDD {m['mdd']:6.1%} exp {m.get('exposure', np.nan):5.0%}"


def market_voltarget(F, aux, target):
    """Market-level vol-target replica: g = min(1, s / EW 20-day vol), s chosen so the IS mean of g equals target."""
    vol = np.exp(F["vol20"])
    m = (F.index >= "2010-01-01") & (F.index <= "2020-12-31")
    lo, hi = 0.0, 2.0
    for _ in range(30):
        s = (lo + hi) / 2
        g = core.exposure_path(pd.Series(np.minimum(1, s / vol), index=F.index), smooth=1)
        if g[m].mean() < target:
            lo = s
        else:
            hi = s
    return g


# ------------------------------------------------------------------------------------------------ stages

def stage_ic(data, horizons=(20, 60), fset="all"):
    F, aux = core.cached_frame(data)
    rows = []
    years = pd.DatetimeIndex(F.index).year
    for H in horizons:
        R = aux[f"R{H}"].to_numpy()
        DD = aux[f"DD{H}"].to_numpy()
        sel = (years >= core.FIRST_YEAR) & (years <= DEV_END)
        base = {"mom120": F["mom120"], "-vol20": -F["vol20"], "breadth120": F["breadth120"], "dd240": F["dd240"]}
        cands = {k: (None, v.to_numpy()) for k, v in base.items()}
        for name, (kind, cfg) in MODELS.items():
            t = time.time()
            raw, score, _ = core.walk_forward(F, aux, kind, dict(cfg, H=H), fset, last_train=DEV_END)
            cands[name] = (raw.to_numpy(), score.to_numpy())
            print(f"  H{H} {name}: {time.time() - t:.1f}s", flush=True)
        for name, (raw, score) in cands.items():
            per_year = [spearman(score[sel & (years == y)], R[sel & (years == y)]) for y in range(core.FIRST_YEAR, DEV_END + 1)]
            row = {"H": H, "model": name, "IC": spearman(score[sel], R[sel]),
                   "AUC_up": auc(score[sel], (R[sel] > 0).astype(float)),
                   "AUC_dd10": auc(-score[sel], (DD[sel] < -0.10).astype(float)),
                   "IC_pos_years": int(np.sum(np.array(per_year) > 0)),
                   "mean_raw": float(np.nanmean(raw[sel])) if raw is not None else np.nan}
            row |= {str(y): v for y, v in zip(range(core.FIRST_YEAR, DEV_END + 1), per_year)}
            rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    return df


def run_config(data, name, kind, cfg, fset="all", smooth=10, floor=0.0, mode="common", log=True, trend=None,
               verbose=True):
    F, aux = core.cached_frame(data)
    raw, score, _ = core.walk_forward(F, aux, kind, cfg, fset, last_train=DEV_END)
    g = core.exposure_path(raw, smooth=smooth, floor=floor)
    pos = core.to_positions(g, data, mode, trend)
    m = eval_positions(pos, data)
    if verbose:
        print(f"{name:<34} IS {fmt(m['is'])} beat {m['is']['beat_bh_sharpe']:4.0%} ord {m['is']['orders']:4.0f} | "
              + " ".join(f"{f} {m[f]['sharpe']:5.2f}" for f in DEV_FOLDS) + f" | pf {m['pf_is']['sharpe']:.2f}",
              flush=True)
    if log:
        config = {"kind": kind, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                  "fset": fset, "smooth": smooth, "floor": floor, "mode": mode, "walk_forward_to": DEV_END}
        metrics = {**m["is"], "fold_sharpe": {f: m[f]["sharpe"] for f in DEV_FOLDS}, "portfolio": m["pf_is"]}
        log_trial(AGENT, name, config, metrics, "is")
    return g, pos, m


def run_cv(data, name, kind, cfg, fset="all", smooth=10, floor=0.0, mode="common", trend=None, log=True):
    """Dev purged CV: folds f1-f5, every model trained only on labels known by 2020-12-31."""
    F, aux = core.cached_frame(data)
    paths = core.purged_cv(F, aux, kind, cfg, DEV_FOLDS, fset, train_end=f"{DEV_END}-12-31")
    g = core.cv_exposure(paths, DEV_FOLDS, F.index, smooth=smooth, floor=floor)
    pos = core.to_positions(g, data, mode, trend)
    m = eval_positions(pos, data, periods=tuple(DEV_FOLDS), portfolio=False)
    bh = eval_positions({c: pd.Series(1.0, index=df.index) for c, df in data.items()}, data, tuple(DEV_FOLDS), False)
    mean_sh = float(np.mean([m[f]["sharpe"] for f in DEV_FOLDS]))
    wins = sum(m[f]["sharpe"] > bh[f]["sharpe"] for f in DEV_FOLDS)
    print(f"{name:<34} CV " + " ".join(f"{f} {m[f]['sharpe']:5.2f}/{m[f]['mdd']:5.0%}/{m[f]['exposure']:3.0%}" for f in DEV_FOLDS)
          + f" | mean {mean_sh:5.2f} wins {wins}/5", flush=True)
    if log:
        config = {"kind": kind, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                  "fset": fset, "smooth": smooth, "floor": floor, "mode": mode, "cv_train_end": DEV_END}
        log_trial(AGENT, name + "~devcv", config,
                  {"sharpe": mean_sh, "fold_sharpe": {f: m[f]["sharpe"] for f in DEV_FOLDS},
                   "fold_mdd": {f: m[f]["mdd"] for f in DEV_FOLDS}, "fold_exposure": {f: m[f]["exposure"] for f in DEV_FOLDS},
                   "folds_beat_bh": wins}, "cv")
    return g, m


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "ic"
    data = load_all()
    if stage == "ic":
        stage_ic(data)
    elif stage == "pos":
        from strategies.trend.strategy import dual_market, sma_half
        for nm, fn in (("trend_dual_market", dual_market), ("trend_sma_half", sma_half)):
            m = eval_positions(fn(data), data)
            print(f"{nm:<34} IS {fmt(m['is'])} beat {m['is']['beat_bh_sharpe']:4.0%} ord {m['is']['orders']:4.0f} | "
                  + " ".join(f"{f} {m[f]['sharpe']:5.2f}" for f in DEV_FOLDS) + f" | pf {m['pf_is']['sharpe']:.2f}")
        bh = {c: pd.Series(1.0, index=df.index) for c, df in data.items()}
        m = eval_positions(bh, data)
        print(f"{'buy_hold':<34} IS {fmt(m['is'])} | " + " ".join(f"{f} {m[f]['sharpe']:5.2f}" for f in DEV_FOLDS)
              + f" | pf {m['pf_is']['sharpe']:.2f}")
        for H in (20, 60):
            for name in ("ridge_a01", "ridge_a1", "logit_c001", "gbdt_d2", "mlp16"):
                kind, cfg = MODELS[name]
                for risk in (("const", "cond") if kind != "mlp" else ("-",)):
                    run_config(data, f"{name}_H{H}_{risk}", kind, dict(cfg, H=H, risk=risk, gamma=3.0))
    elif stage == "pos2":
        F, aux = core.cached_frame(data)
        # market-level vol-target replicas at 70% / 80% average exposure
        for tgt in (0.7, 0.8):
            g = market_voltarget(F, aux, tgt)
            m = eval_positions(core.to_positions(g, data), data)
            print(f"{'mkt_voltarget_' + str(tgt):<34} IS {fmt(m['is'])} beat {m['is']['beat_bh_sharpe']:4.0%} ord {m['is']['orders']:4.0f} | "
                  + " ".join(f"{f} {m[f]['sharpe']:5.2f}" for f in DEV_FOLDS) + f" | pf {m['pf_is']['sharpe']:.2f}")
            log_trial(AGENT, f"mkt_voltarget_{tgt}", {"kind": "voltarget", "target_exposure": tgt},
                      {**m["is"], "fold_sharpe": {f: m[f]["sharpe"] for f in DEV_FOLDS}, "portfolio": m["pf_is"]})
        # feature-set ablations
        for fset in ("price", "price_chips", "price_global"):
            run_config(data, f"ridge_a01_H60_cond_{fset}", "ridge", dict(alpha=0.1, H=60, risk="cond", gamma=3.0), fset)
            run_config(data, f"mlp16_H20_{fset}", "mlp", dict(hidden=(16,), gamma=3.0, H=20), fset)
        # MLP variants
        run_config(data, "mlp16_H20_g5", "mlp", dict(hidden=(16,), gamma=5.0, H=20))
        run_config(data, "mlp16_H60_g5", "mlp", dict(hidden=(16,), gamma=5.0, H=60))
        run_config(data, "mlp8_H20", "mlp", dict(hidden=(8,), gamma=3.0, H=20))
        run_config(data, "mlp16x8_H20", "mlp", dict(hidden=(16, 8), gamma=3.0, H=20))
        run_config(data, "mlp16_H20_wd1e-1", "mlp", dict(hidden=(16,), gamma=3.0, H=20, wd=1e-1))
        # floors
        run_config(data, "ridge_a01_H60_cond_floor05", "ridge", dict(alpha=0.1, H=60, risk="cond", gamma=3.0), floor=0.5)
        run_config(data, "mlp16_H20_floor05", "mlp", dict(hidden=(16,), gamma=3.0, H=20), floor=0.5)
    elif stage == "cv":
        bh = eval_positions({c: pd.Series(1.0, index=df.index) for c, df in data.items()}, data, tuple(DEV_FOLDS), False)
        print(f"{'buy_hold':<34} CV " + " ".join(f"{f} {bh[f]['sharpe']:5.2f}/{bh[f]['mdd']:5.0%}/100%" for f in DEV_FOLDS)
              + f" | mean {np.mean([bh[f]['sharpe'] for f in DEV_FOLDS]):5.2f}")
        shortlist = [
            ("ridge_a01_H60_cond", "ridge", dict(alpha=0.1, H=60, risk="cond", gamma=3.0), "all", 0.0),
            ("ridge_a01_H60_cond_floor05", "ridge", dict(alpha=0.1, H=60, risk="cond", gamma=3.0), "all", 0.5),
            ("ridge_a1_H60_cond", "ridge", dict(alpha=1.0, H=60, risk="cond", gamma=3.0), "all", 0.0),
            ("logit_c001_H20_const", "logit", dict(C=0.01, H=20, risk="const", gamma=3.0), "all", 0.0),
            ("gbdt_d2_H20_cond", "gbdt", dict(depth=2, iters=150, lr=0.03, leaf=200, H=20, risk="cond", gamma=3.0), "all", 0.0),
            ("gbdt_d2_H60_cond", "gbdt", dict(depth=2, iters=150, lr=0.03, leaf=200, H=60, risk="cond", gamma=3.0), "all", 0.0),
            ("mlp16_H20", "mlp", dict(hidden=(16,), gamma=3.0, H=20), "all", 0.0),
            ("mlp8_H20", "mlp", dict(hidden=(8,), gamma=3.0, H=20), "all", 0.0),
            ("mlp16_H60_g5", "mlp", dict(hidden=(16,), gamma=5.0, H=60), "all", 0.0),
            ("mlp16_H20_price_chips", "mlp", dict(hidden=(16,), gamma=3.0, H=20), "price_chips", 0.0),
            ("ridge_a01_H60_cond_price_chips", "ridge", dict(alpha=0.1, H=60, risk="cond", gamma=3.0), "price_chips", 0.0),
        ]
        for name, kind, cfg, fset, floor in shortlist:
            run_cv(data, name, kind, cfg, fset, floor=floor)
    elif stage == "v2":
        # after replacing the drifting level features by 250-day z-scores (RESEARCH.md section 4.4)
        todo = [
            ("v2_ridge_a01_H60_cond", "ridge", dict(alpha=0.1, H=60, risk="cond", gamma=3.0), "all"),
            ("v2_ridge_a01_H20_cond", "ridge", dict(alpha=0.1, H=20, risk="cond", gamma=3.0), "all"),
            ("v2_ridge_a1_H20_cond", "ridge", dict(alpha=1.0, H=20, risk="cond", gamma=3.0), "all"),
            ("v2_logit_c001_H20_const", "logit", dict(C=0.01, H=20, risk="const", gamma=3.0), "all"),
            ("v2_gbdt_d2_H20_cond", "gbdt", dict(depth=2, iters=150, lr=0.03, leaf=200, H=20, risk="cond", gamma=3.0), "all"),
            ("v2_mlp16_H20", "mlp", dict(hidden=(16,), gamma=3.0, H=20), "all"),
            ("v2_mlp16_H60_g5", "mlp", dict(hidden=(16,), gamma=5.0, H=60), "all"),
            ("v2_mlp16_H20_price_chips", "mlp", dict(hidden=(16,), gamma=3.0, H=20), "price_chips"),
            ("v2_ridge_a01_H60_cond_price_chips", "ridge", dict(alpha=0.1, H=60, risk="cond", gamma=3.0), "price_chips"),
        ]
        only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
        for name, kind, cfg, fset in todo:
            if only and name not in only:
                continue
            g, pos, m = run_config(data, name, kind, cfg, fset)
            print("   exposure by year:", g[g.index.year >= 2012].groupby(g.index.year[g.index.year >= 2012]).mean().round(2).to_dict())
            run_cv(data, name, kind, cfg, fset)
    elif stage == "v3":
        # regularised / slower mappings on the drift-free features (floor = never below 50%, E3 in dl_literature.md)
        mlp = dict(hidden=(16,), gamma=3.0, H=20)
        todo = [
            ("v3_mlp16_H20_floor05", "mlp", mlp, "all", 10, 0.5),
            ("v3_mlp16_H20_floor05_s20", "mlp", mlp, "all", 20, 0.5),
            ("v3_mlp16_H20_wd01_floor05", "mlp", dict(mlp, wd=0.1), "all", 10, 0.5),
            ("v3_mlp16_H20_price_chips_floor05", "mlp", mlp, "price_chips", 10, 0.5),
            ("v3_ridge_a1_H20_cond_shrink05_floor05", "ridge", dict(alpha=1.0, H=20, risk="cond", gamma=3.0, shrink=0.5), "all", 10, 0.5),
            ("v3_logit_c001_H20_const_floor05", "logit", dict(C=0.01, H=20, risk="const", gamma=3.0), "all", 10, 0.5),
        ]
        only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
        for name, kind, cfg, fset, sm, fl in todo:
            if only and name not in only:
                continue
            run_config(data, name, kind, cfg, fset, smooth=sm, floor=fl)
            run_cv(data, name, kind, cfg, fset, smooth=sm, floor=fl)
    elif stage == "v4":
        # MLP initialised at the unconditional optimum (an MLP that learns nothing = constant prior exposure)
        mlp = dict(hidden=(16,), gamma=3.0, H=20, prior_init=True)
        todo = [
            ("v4_mlp16_H20_prior", "mlp", mlp, "all", 10, 0.0),
            ("v4_mlp16_H20_prior_floor05", "mlp", mlp, "all", 10, 0.5),
            ("v4_mlp16_H20_prior_price_chips_floor05", "mlp", mlp, "price_chips", 10, 0.5),
            ("v4_mlp16_H60_g5_prior_floor05", "mlp", dict(mlp, H=60, gamma=5.0), "all", 10, 0.5),
        ]
        only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
        for name, kind, cfg, fset, sm, fl in todo:
            if only and name not in only:
                continue
            g, _, _ = run_config(data, name, kind, cfg, fset, smooth=sm, floor=fl)
            print("   exposure by year:", g[g.index.year >= 2012].groupby(g.index.year[g.index.year >= 2012]).mean().round(2).to_dict())
            run_cv(data, name, kind, cfg, fset, smooth=sm, floor=fl)
    elif stage == "v5":
        mlp = dict(hidden=(16,), gamma=3.0, H=20, prior_init=True)
        # TAIFEX foreign futures / options positions exist only from 2018-06: imputed + masked
        run_config(data, "v5_mlp16_H20_prior_floor05_all_fut", "mlp", mlp, "all_fut", floor=0.5)
        run_cv(data, "v5_mlp16_H20_prior_floor05_all_fut", "mlp", mlp, "all_fut", floor=0.5)
        # linear / tree baselines with the same label (H20), the same inputs and the same floor-0.5 mapping
        for name, kind, cfg in (("v5_ridge_a1_H20_cond_floor05", "ridge", dict(alpha=1.0, H=20, risk="cond", gamma=3.0)),
                                ("v5_gbdt_d2_H20_cond_floor05", "gbdt", dict(depth=2, iters=150, lr=0.03, leaf=200, H=20, risk="cond", gamma=3.0))):
            run_config(data, name, kind, cfg, "all", floor=0.5)
            run_cv(data, name, kind, cfg, "all", floor=0.5)
    elif stage == "pos3":
        from strategies.trend.strategy import trend_state
        trend = {c: trend_state(df["close"]) for c, df in data.items()}
        base = dict(hidden=(16,), gamma=3.0, H=20)
        fs = "price_chips"
        g, pos, m = run_config(data, "mlp16_H20_price_chips", "mlp", base, fs, log=False)
        c = controls_is(pos, data)
        print("   controls IS: B1 const", fmt(c["const"]), "| B3 voltarget", fmt(c["voltarget"]))
        print("   exposure by year:", g.groupby(g.index.year).mean().round(2).to_dict())
        run_config(data, "mlp16_H20_price_chips_floor05", "mlp", base, fs, floor=0.5)
        run_cv(data, "mlp16_H20_price_chips_floor05", "mlp", base, fs, floor=0.5)
        run_config(data, "mlp16_H20_price_chips_trend", "mlp", base, fs, mode="trend", trend=trend)
        run_cv(data, "mlp16_H20_price_chips_trend", "mlp", base, fs, mode="trend", trend=trend)
        for off in (5, 10):
            cfg = dict(base, seeds=tuple(range(off, off + 5)))
            run_config(data, f"mlp16_H20_price_chips_seeds{off}", "mlp", cfg, fs)
            run_cv(data, f"mlp16_H20_price_chips_seeds{off}", "mlp", cfg, fs)
