"""In-sample ablation: does the small MLP add anything beyond its KD+MACD base rule?

  python -m strategies.kd_macd_dl.ablation                 (run from the repo root; in-sample 2010-2020)
  python -m strategies.kd_macd_dl.ablation --period oos    (run ONCE at the end, nothing was tuned on it)

Compares, with the exact production pipeline (same features, labels, purge, thresholds, position mapping):
  mlp        the published model (3 x MLP 28-16-8-1)
  logistic   a logistic regression on the same 28 KD/MACD features (is the non-linearity needed?)
  wrongstock each stock receives the MLP signal of ANOTHER stock (same persistence / exposure statistics, but
             no link to its own KD/MACD) -> isolates "information" from "just being invested more often"
  rule       the base KD+MACD rule alone (floor: 0.5 + 0.5*rule)
Also prints the pooled walk-forward AUC of the model by year.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from stocklab import backtest as bt
from stocklab.data import load_all
from stocklab.runner import leaderboard, print_leaderboard

from . import core
from .strategy import floor_from, gate_from

PERIOD = "is"
YEARS = {"is": (2012, 2020), "oos": (2021, 2026), "full": (2012, 2026)}


def logistic_walk_forward(data):
    orig = core.train_ensemble

    def train_lr(X, y, seed_base):
        m = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
        return [[(m.coef_.astype(np.float64), m.intercept_.astype(np.float64))]]

    core.train_ensemble = train_lr
    try:
        return core.walk_forward(data)
    finally:
        core.train_ensemble = orig


def wrong_stock(signals):
    codes = sorted(signals)
    out = {}
    for i, c in enumerate(codes):
        donor = signals[codes[(i + 7) % len(codes)]]
        s = signals[c].copy()
        for col in ("p", "ps", "q_lo", "q_hi"):
            s[col] = donor[col].reindex(s.index)
        # keep the fallback period identical: donor values only where this stock has its own model output
        mask = signals[c]["ps"].notna()
        s.loc[~mask, ["ps", "q_lo", "q_hi"]] = np.nan
        out[c] = s
    return out


def yearly_auc(data, signals):
    stocks, _ = core._prepare(data)
    rows = []
    for s in stocks:
        p = signals[s["code"]]["p"].to_numpy()
        m = ~np.isnan(p) & ~np.isnan(s["y"])
        rows.append(pd.DataFrame({"p": p[m], "y": s["y"][m], "year": s["index"][m].year}))
    df = pd.concat(rows)
    lo, hi = YEARS[PERIOD]
    df = df[(df["year"] >= lo) & (df["year"] <= hi)]
    by = df.groupby("year").apply(lambda g: roc_auc_score(g["y"], g["p"]))
    return roc_auc_score(df["y"], df["p"]), by


def run(pos_by_name, data):
    results = {}
    for b, fn in (("buy_hold", bt.run_buy_hold), ("dca", bt.run_dca)):
        results[b] = {PERIOD: {c: fn(df, PERIOD) for c, df in data.items()}}
    for name, pos in pos_by_name.items():
        results[name] = {PERIOD: {c: bt.run_strategy(df, pos[c], PERIOD) for c, df in data.items()}}
    print_leaderboard(leaderboard(results, {}, (PERIOD,)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="is", choices=["is", "oos", "full"])
    PERIOD = ap.parse_args().period
    data = load_all()
    sig_mlp, _ = core.walk_forward(data)
    sig_lr, _ = logistic_walk_forward(data)
    sig_wrong = wrong_stock(sig_mlp)
    for name, sig in (("mlp", sig_mlp), ("logistic", sig_lr)):
        auc, by = yearly_auc(data, sig)
        print(f"{name}: pooled walk-forward AUC {YEARS[PERIOD]} = {auc:.4f}; by year: " + ", ".join(f"{y}:{v:.3f}" for y, v in by.items()))
    rule = {c: s["base"] for c, s in sig_mlp.items()}
    run({
        "floor_mlp": floor_from(sig_mlp), "floor_logistic": floor_from(sig_lr), "floor_wrongstock": floor_from(sig_wrong),
        "floor_rule": {c: 0.5 + 0.5 * r for c, r in rule.items()},
        "gate_mlp": gate_from(sig_mlp), "gate_logistic": gate_from(sig_lr), "gate_wrongstock": gate_from(sig_wrong),
        "rule_only": rule,
    }, data)
