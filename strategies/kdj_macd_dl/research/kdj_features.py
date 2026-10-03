"""KDJ round, experiment K3: does J add information to the KD+MACD network? (walk-forward AUC, in-sample only)

  python -m strategies.kdj_macd_dl.research.kdj_features [sets] [--lr] [--seeds 3,4,5] [--out file.csv]

Same production pipeline as core.walk_forward (cross-sectional 60-bar label, yearly expanding re-training, purge,
training-window standardisation), on data truncated at 2020-12-31. Reports pooled 2012-2020 AUC and AUC by year.
Labels/indicators are computed once and shared by every feature set; only the feature columns change.
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .. import core
from .kdj_harness import data


def lr_trainer(Z, y, seed_base):
    m = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y)
    return [[(m.coef_.astype(np.float64), m.intercept_.astype(np.float64))]]


def with_features(stocks, fset):
    out, names = [], None
    for s in stocks:
        X, names = core.feature_matrix(s["ind"], fset)
        out.append({**s, "X": X})
    return out, names


def auc_report(stocks, signals, years=(2012, 2020)):
    rows = []
    for s in stocks:
        p = signals[s["code"]]["p"].to_numpy()
        m = ~np.isnan(p) & ~np.isnan(s["y"])
        rows.append(pd.DataFrame({"p": p[m], "y": s["y"][m], "year": s["index"][m].year}))
    df = pd.concat(rows)
    df = df[(df["year"] >= years[0]) & (df["year"] <= years[1])]
    by = {y: roc_auc_score(g["y"], g["p"]) for y, g in df.groupby("year")}
    # mean of per-date cross-sectional AUCs (the label is cross-sectional, so this is the cleanest view)
    return roc_auc_score(df["y"], df["p"]), by


def run(stocks0, fset, model="mlp", seeds=core.SEEDS):
    stocks, names = with_features(stocks0, fset)
    trainer = lr_trainer if model == "lr" else None
    t = time.time()
    sig, models = core.walk_forward(None, trainer=trainer, seeds=seeds, stocks=(stocks, names))
    auc, by = auc_report(stocks, sig)
    return dict(fset=fset, model=model, seeds=",".join(map(str, seeds)), n_feat=len(names),
                params=core.n_params(len(names)) * (len(seeds) if model == "mlp" else 1) if model == "mlp" else len(names) + 1,
                auc=auc, secs=time.time() - t, **{str(y): v for y, v in by.items()}), sig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="?", default="kd,kdj,jd,kdx,kdjx,j_only,macd_only")
    ap.add_argument("--lr", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out")
    a = ap.parse_args()
    stocks0, _ = core.prepare(data(), ("kd",))
    seeds = tuple(int(x) for x in a.seeds.split(","))
    rows = []
    for fs in a.sets.split(","):
        for model in (["mlp", "lr"] if a.lr else ["mlp"]):
            r, _ = run(stocks0, fs, model, seeds)
            rows.append(r)
            print(f"{fs:>10} {model:>3} seeds={r['seeds']} F={r['n_feat']:>2} params={r['params']:>5}  AUC={r['auc']:.4f}  "
                  + " ".join(f"{y}:{r[str(y)]:.3f}" for y in range(2012, 2021)) + f"  ({r['secs']:.0f}s)", flush=True)
    if a.out:
        pd.DataFrame(rows).to_csv(a.out, index=False)
