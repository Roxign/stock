"""Final (post-freeze) comparison tables for RESEARCH.md. Run ONCE after freezing strategy.py:

    python -m strategies.dl_market_state.final_eval

Same inputs (all_fut, 39 features), same label (H = 20) and the same exposure mapping (EMA 10, floor 0.5, 0.25 grid)
for the published MLP and its baselines (L2 logistic, ridge, HistGradientBoosting); walk-forward to the end of the
data; IS / OOS / fixed folds per-stock medians; one-account portfolio; purged 8-fold CV of the baselines; MLP seed
groups; references (buy & hold, trend_dual_market, trend_sma_half, kdj_macd_dl_floor from data/positions cache).
Every row is logged to research/trials.jsonl (agent dl_market_state, names final_*).
"""

import time

import numpy as np
import pandas as pd

from stocklab import backtest as bt
from stocklab import portfolio as pf
from stocklab.data import load_all
from stocklab.runner import POS_CACHE
from stocklab.trials import log_trial

from . import core
from . import strategy as S
from .experiments import AGENT, auc, market_voltarget, spearman

FOLDS = dict(bt.FOLDS)
PERIODS = ("is", "oos") + tuple(FOLDS)
BASELINES = {
    "logit_c001": ("logit", dict(C=0.01, H=20, risk="const", gamma=3.0)),
    "ridge_a1": ("ridge", dict(alpha=1.0, H=20, risk="cond", gamma=3.0)),
    "gbdt_d2": ("gbdt", dict(depth=2, iters=150, lr=0.03, leaf=200, H=20, risk="cond", gamma=3.0)),
}


def per_stock(pos, data, periods=PERIODS):
    out = {}
    for p in periods:
        res = {c: bt.run_strategy(df, pos[c].reindex(df.index).ffill().fillna(1.0), p) for c, df in data.items()}
        codes = [c for c, r in res.items() if r]
        out[p] = {k: float(np.nanmedian([res[c]["metrics"][k] for c in codes]))
                  for k in ("sharpe", "cagr", "mdd", "exposure", "orders")}
    return out


def portfolio(pos, data):
    w = pf.per_stock_weights(pos, data)
    return {p: {k: pf.run(data, w, p)["metrics"][k] for k in ("sharpe", "cagr", "mdd", "exposure", "turnover")}
            for p in ("is", "oos")}


def line(name, m, pfm=None):
    s = f"{name:<26}" + " ".join(f"{p} {m[p]['sharpe']:5.2f}/{m[p]['cagr']:6.1%}/{m[p]['mdd']:6.1%}/{m[p]['exposure']:4.0%}"
                                  for p in ("is", "oos"))
    s += " | folds " + " ".join(f"{m[f]['sharpe']:5.2f}" for f in FOLDS)
    if pfm:
        s += " | pf " + " ".join(f"{p} {pfm[p]['sharpe']:4.2f}/{pfm[p]['cagr']:5.1%}/{pfm[p]['mdd']:6.1%}" for p in ("is", "oos"))
    return s


def log(name, config, m, pfm):
    for p in ("is", "oos"):
        log_trial(AGENT, f"final_{name}", config,
                  {**m[p], "fold_sharpe": {f: m[f]["sharpe"] for f in FOLDS}, "portfolio": pfm[p] if pfm else None}, p)


def main():
    t0 = time.time()
    data = load_all()
    F, aux = core.cached_frame(data)
    rows = {}
    years = pd.DatetimeIndex(F.index).year
    R20, DD20 = aux["R20"].to_numpy(), aux["DD20"].to_numpy()

    def run_model(name, kind, cfg, log_it=True):
        raw, score, _ = core.walk_forward(F, aux, kind, cfg, S.FSET)
        g = core.exposure_path(raw, smooth=S.SMOOTH, floor=S.FLOOR)
        pos = core.to_positions(g, data)
        m, pfm = per_stock(pos, data), portfolio(pos, data)
        sc = score.to_numpy()
        diag = {}
        for p, (a, b) in {"is": (core.FIRST_YEAR, 2020), "oos": (2021, 2026)}.items():
            sel = (years >= a) & (years <= b)
            diag[p] = (spearman(sc[sel], R20[sel]), auc(-sc[sel], (DD20[sel] < -0.08).astype(float)))
        print(line(name, m, pfm), flush=True)
        print(f"    IC(score,R20) is {diag['is'][0]:.3f} oos {diag['oos'][0]:.3f} | AUC(DD20<-8%) is {diag['is'][1]:.3f} "
              f"oos {diag['oos'][1]:.3f} | mean g is {g[(years >= 2010) & (years <= 2020)].mean():.2f} "
              f"oos {g[years >= 2021].mean():.2f} | g by year {g.groupby(years).mean().round(2).loc[2010:].to_dict()}",
              flush=True)
        if log_it:
            log(name, {"kind": kind, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                       "fset": S.FSET, "smooth": S.SMOOTH, "floor": S.FLOOR}, m, pfm)
        rows[name] = (m, pfm)
        return g

    # buy & hold and cached references
    bh = {c: pd.Series(1.0, index=df.index) for c, df in data.items()}
    m, pfm = per_stock(bh, data), portfolio(bh, data)
    print(line("buy_hold", m, pfm), flush=True)
    for ref in ("trend_dual_market", "trend_sma_half", "kdj_macd_dl_floor"):
        path = POS_CACHE / f"{ref}.pkl"
        if path.exists():
            pos = pd.read_pickle(path)
            print(line(ref + " (cache)", per_stock(pos, data), portfolio(pos, data)), flush=True)

    # the published model and its seed groups
    g_pub = run_model("mlp16_pub_seeds0-4", S.KIND, S.CFG)
    for off in (5, 10):
        run_model(f"mlp16_seeds{off}-{off + 4}", S.KIND, dict(S.CFG, seeds=tuple(range(off, off + 5))))
    sh = {p: [rows[n][0][p]["sharpe"] for n in rows if n.startswith("mlp16")] for p in ("is", "oos")}
    cg = {p: [rows[n][0][p]["cagr"] for n in rows if n.startswith("mlp16")] for p in ("is", "oos")}
    print("  MLP seed groups: " + " ".join(f"{p} Sharpe {np.mean(sh[p]):.3f}+-{np.std(sh[p]):.3f} CAGR "
                                           f"{np.mean(cg[p]):.2%}+-{np.std(cg[p]):.2%}" for p in ("is", "oos")))

    # baselines: same inputs, same label, same mapping
    for name, (kind, cfg) in BASELINES.items():
        run_model(name, kind, cfg)

    # market-level vol target with the same IS mean exposure as the published model (B3 at index level)
    tgt = float(g_pub[(years >= 2010) & (years <= 2020)].mean())
    gv = market_voltarget(F, aux, tgt)
    pos = core.to_positions(gv, data)
    m, pfm = per_stock(pos, data), portfolio(pos, data)
    print(line(f"mkt_voltarget_{tgt:.2f}", m, pfm), flush=True)
    log(f"mkt_voltarget_{tgt:.2f}", {"kind": "voltarget", "target_exposure": tgt}, m, pfm)

    # purged 8-fold CV of the baselines (the MLP's purged CV comes from evaluate.py --cv)
    print("purged CV (median per-stock Sharpe per fold):")
    for name, (kind, cfg) in BASELINES.items():
        paths = core.purged_cv(F, aux, kind, cfg, FOLDS, S.FSET)
        g = core.cv_exposure(paths, FOLDS, F.index, smooth=S.SMOOTH, floor=S.FLOOR)
        m = per_stock(core.to_positions(g, data), data, tuple(FOLDS))
        mean = float(np.mean([m[f]["sharpe"] for f in FOLDS]))
        print(f"  {name:<14}" + " ".join(f"{f} {m[f]['sharpe']:5.2f}/{m[f]['mdd']:5.0%}/{m[f]['exposure']:3.0%}" for f in FOLDS)
              + f" | mean {mean:.3f}", flush=True)
        log_trial(AGENT, f"final_{name}~cv8", {"kind": kind, **cfg, "fset": S.FSET, "floor": S.FLOOR},
                  {"sharpe": mean, "fold_sharpe": {f: m[f]["sharpe"] for f in FOLDS},
                   "fold_mdd": {f: m[f]["mdd"] for f in FOLDS}}, "cv")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
