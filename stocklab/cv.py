"""Blocked k-fold cross-validation over the regime folds in backtest.FOLDS.

A strategy can be cross-validated three ways:
  fixed   its published parameters evaluated in every fold. No re-fitting, so folds that overlap the data the
          parameters were chosen on (2010-2020) show behaviour across regimes, not unseen performance.
  retune  the strategy entry provides `param_grid` (list of dicts) and `build(params) -> positions fn`. For each fold,
          the parameters with the best average fold-median Sharpe on the OTHER folds are tested on that fold.
  purged  the strategy entry provides `cv_positions(data, folds) -> dict[code, Series]`: each fold's positions come
          from a model trained without that fold, with label windows overlapping the fold purged plus an embargo.
retune and purged let later folds inform earlier ones, so they measure robustness across regimes, not a tradable
record; the walk-forward `positions` remain the tradable version.
"""

import math

import numpy as np

from . import backtest as bt

FOLDS = tuple(bt.FOLDS)


def run_folds(data, positions):
    from .runner import fees_of

    positions = bt.effective_positions(positions, {c: data[c] for c in positions})
    return {f: {c: bt.run_strategy(data[c], p, f, fees_of(c)) for c, p in positions.items()} for f in FOLDS}


def fold_median(results, key="sharpe"):
    vals = [r["metrics"][key] for r in results.values() if r]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.median(vals)) if vals else math.nan


def retune(strategy, data, objective="sharpe"):
    grid = strategy["param_grid"]
    per = [run_folds(data, strategy["build"](params)(data)) for params in grid]
    score = np.array([[fold_median(per[g][f], objective) for f in FOLDS] for g in range(len(grid))])
    chosen, results = {}, {}
    for j, f in enumerate(FOLDS):
        g = int(np.nanargmax(np.nanmean(np.delete(score, j, axis=1), axis=1)))
        chosen[f] = grid[g]
        results[f] = per[g][f]
    return {"mode": "retune", "chosen": chosen, "results": results, "grid_size": len(grid)}


def purged(strategy, data):
    pos = strategy["cv_positions"](data, dict(bt.FOLDS))
    return {"mode": "purged", "chosen": None, "results": run_folds(data, pos)}


def cross_validate(strategy, data):
    from .runner import universe_data

    data = universe_data(strategy, data)
    if strategy.get("cv_positions"):
        return purged(strategy, data)
    if strategy.get("param_grid"):
        return retune(strategy, data)
    return None


def fold_table(results, ids, labels, codes=None):
    """Rows per id: fold-median Sharpe/CAGR/MDD next to buy-and-hold, and how many folds each beats it in.
    results[id][fold][code] must include 'buy_hold'. codes restricts the securities (ids with none are skipped)."""
    if codes is not None:
        keep = set(codes)
        results = {sid: {f: {c: r for c, r in by_c.items() if c in keep} for f, by_c in by_f.items()}
                   for sid, by_f in results.items()}
        ids = [sid for sid in ids if any(results[sid][f] for f in FOLDS)]
    bh = {f: {k: fold_median(results["buy_hold"][f], k) for k in ("sharpe", "cagr", "mdd")} for f in FOLDS}
    rows = []
    for sid in ids:
        r = {"id": sid, "label": labels.get(sid, sid), "folds": {}}
        won = {"sharpe": 0, "cagr": 0, "mdd": 0}
        pairs, pair_wins = 0, 0
        for f in FOLDS:
            m = {k: fold_median(results[sid][f], k) for k in ("sharpe", "cagr", "mdd")}
            r["folds"][f] = m
            for k in won:
                won[k] += int(m[k] > bh[f][k])
            for c, res in results[sid][f].items():
                b = results["buy_hold"][f].get(c)
                if res and b:
                    pairs += 1
                    pair_wins += int(res["metrics"]["sharpe"] > b["metrics"]["sharpe"])
        r |= {"folds_sharpe": won["sharpe"], "folds_cagr": won["cagr"], "folds_mdd": won["mdd"],
              "stock_fold_sharpe": pair_wins / pairs if pairs else math.nan}
        rows.append(r)
    return rows, bh


def print_fold_table(rows, bh):
    print(f"\n=== 交叉驗證：各折 Sharpe 中位數（括號內為買進持有）===")
    head = "".join(f"{bt.FOLD_LABELS[f]:>9}" for f in FOLDS)
    print(f"{'策略':<34}{head}  Sharpe勝折 報酬勝折 回撤小折 檔折Sharpe勝")
    print(f"{'buy_hold':<34}" + "".join(f"{bh[f]['sharpe']:>9.2f}" for f in FOLDS))
    for r in rows:
        cells = "".join(f"{r['folds'][f]['sharpe']:>9.2f}" for f in FOLDS)
        print(f"{r['id']:<34}{cells}  {r['folds_sharpe']:>5}/8 {r['folds_cagr']:>6}/8 {r['folds_mdd']:>6}/8 {r['stock_fold_sharpe']:>9.0%}")
