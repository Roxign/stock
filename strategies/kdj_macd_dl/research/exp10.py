import time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from harness import show, baseline_rows, data, evaluate
from wf import run, pooled, label_of
from mapping import rule_pos
d = data()
BR = {r: {c: rule_pos(df, r) for c, df in d.items()} for r in ("either_x3", "both_x3")}
def per_year(cfg, out):
    rows, _ = pooled(cfg)
    df = []
    for r in rows:
        p = out["preds"][r["code"]].to_numpy(); y = label_of(cfg, r)
        m = ~np.isnan(p) & ~np.isnan(y)
        df.append(pd.DataFrame({"p": p[m], "y": y[m], "yr": pd.DatetimeIndex(r["dates"][m]).year}))
    df = pd.concat(df)
    py = df.groupby("yr").apply(lambda g: roc_auc_score(g["y"], g["p"]))
    return roc_auc_score(df["y"], df["p"]), py.min(), (py > 0.5).sum()
def veto(out, r, span=10):
    pos = {}
    for c, df in d.items():
        b = BR[r][c]; p0 = out["preds"][c]; p = p0.ewm(span=span, adjust=False).mean().where(p0.notna()); q = out["thr"][c]
        v = np.maximum((p >= q["q40"]).astype(float), 0.5 * ((p >= q["q20"]) & (p < q["q40"])).astype(float))
        pos[c] = np.maximum(b, v).where(p.notna(), b)
    return pos
base = dict(H=60, label="xs", hidden=(16, 8), ens=3, epochs=10, stride=1)
grid = {"base": base, "ens5": {**base, "ens": 5}, "slowlags": {**base, "slow_lags": (0, 5, 10, 20)},
        "h32x16": {**base, "hidden": (32, 16)}, "ep5": {**base, "epochs": 5}, "H120": {**base, "H": 120},
        "H20": {**base, "H": 20}, "lags0_2_5_10": {**base, "lags": (0, 2, 5, 10)}, "lr": {**base, "model": "lr"}}
rows = baseline_rows()
for name, cfg in grid.items():
    t = time.time(); out = run(cfg, verbose=False)
    auc, mn, npos = per_year(cfg, out)
    print(f"{name:14s} AUC={auc:.4f} minYearAUC={mn:.3f} yearsAbove.5={npos}/9 ({time.time()-t:.0f}s)", flush=True)
    for r in BR:
        rows += evaluate(veto(out, r), f"{name} veto {r}")
show(rows)
