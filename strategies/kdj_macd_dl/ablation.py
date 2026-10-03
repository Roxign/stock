"""Ablation: does the small MLP add anything beyond its KDJ+MACD base rule, and do J / the MA lines matter?

  python -m strategies.kdj_macd_dl.ablation                 (run from the repo root; in-sample 2010-2020)
  python -m strategies.kdj_macd_dl.ablation --period oos    (pre-written; run ONCE at the end, nothing tuned on it)

Compares, with the exact production pipeline (labels, purge, thresholds, position mapping, J base rule):
  mlp          the published model (3 x MLP 48-16-8-1 on KDJ + MACD + MA features)
  logistic     logistic regression on the same 48 features (is the non-linearity needed?)
  noJ          the same MLP without any J feature (K, D, MACD, MA only)
  noMA         the same MLP without the MA features (KDJ + MACD only)
  wrongstock   each stock receives the MLP signal of ANOTHER stock (offsets +7 and +19 in code order): same
               persistence / exposure statistics but no link to its own indicators -> separates information from
               "just being invested more often"
  rule         the J base rule alone (0/1) and with a 0.5 floor (no model)
Also prints the pooled walk-forward AUC by year for every model.
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

YEARS = {"is": (2012, 2020), "oos": (2021, 2026), "full": (2012, 2026)}


def lr_trainer(Z, y, seed_base):
    m = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y)
    return [[(m.coef_.astype(np.float64), m.intercept_.astype(np.float64))]]


def wrong_stock(signals, shift):
    codes = sorted(signals)
    out = {}
    for i, c in enumerate(codes):
        donor = signals[codes[(i + shift) % len(codes)]]
        s = signals[c].copy()
        for col in ("p", "ps", "q_lo", "q_hi"):
            s[col] = donor[col].reindex(s.index)
        mask = signals[c]["ps"].notna()     # keep this stock's own fallback period
        s.loc[~mask, ["ps", "q_lo", "q_hi"]] = np.nan
        out[c] = s
    return out


def yearly_auc(stocks, signals, period):
    rows = []
    for s in stocks:
        p = signals[s["code"]]["p"].to_numpy()
        m = ~np.isnan(p) & ~np.isnan(s["y"])
        rows.append(pd.DataFrame({"p": p[m], "y": s["y"][m], "year": s["index"][m].year}))
    df = pd.concat(rows)
    lo, hi = YEARS[period]
    df = df[(df["year"] >= lo) & (df["year"] <= hi)]
    by = {y: roc_auc_score(g["y"], g["p"]) for y, g in df.groupby("year")}
    return roc_auc_score(df["y"], df["p"]), by


def run(pos_by_name, data, period):
    results = {}
    for b, fn in (("buy_hold", bt.run_buy_hold), ("dca", bt.run_dca)):
        results[b] = {period: {c: fn(df, period) for c, df in data.items()}}
    for name, pos in pos_by_name.items():
        results[name] = {period: {c: bt.run_strategy(df, pos[c], period) for c, df in data.items()}}
    print_leaderboard(leaderboard(results, {}, (period,)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="is", choices=["is", "oos", "full"])
    period = ap.parse_args().period
    data = load_all()
    stocks, names = core.prepare(data)
    sigs = {"mlp": core.walk_forward(None, stocks=(stocks, names))[0],
            "logistic": core.walk_forward(None, trainer=lr_trainer, stocks=(stocks, names))[0]}
    for tag, fset in (("noJ", "kd_ma_atr"), ("noMA", "kdj")):
        st = [{**s, "X": core.feature_matrix(s["ind"], fset)[0]} for s in stocks]
        sigs[tag] = core.walk_forward(None, stocks=(st, core.feature_names(fset)))[0]
    for name, sig in sigs.items():
        auc, by = yearly_auc(stocks, sig, period)
        print(f"{name:>9}: pooled walk-forward AUC {YEARS[period]} = {auc:.4f}; by year: "
              + ", ".join(f"{y}:{v:.3f}" for y, v in by.items()))
    rule = {c: s["base"] for c, s in sigs["mlp"].items()}
    pos = {}
    for name, sig in sigs.items():
        pos[f"floor_{name}"] = floor_from(sig)
        pos[f"gate_{name}"] = gate_from(sig)
    for sh in (7, 19):
        w = wrong_stock(sigs["mlp"], sh)
        pos[f"floor_wrongstock{sh}"] = floor_from(w)
        pos[f"gate_wrongstock{sh}"] = gate_from(w)
    pos["rule_floor"] = {c: 0.5 + 0.5 * r for c, r in rule.items()}
    pos["rule_only"] = rule
    run(pos, data, period)
