import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from wf import run, pooled, label_of
def per_year(cfg):
    out = run(cfg, verbose=False)
    rows, _ = pooled(cfg)
    df = []
    for r in rows:
        p = out["preds"][r["code"]].to_numpy(); y = label_of(cfg, r)
        m = ~np.isnan(p) & ~np.isnan(y) & r["cond"]
        df.append(pd.DataFrame({"p": p[m], "y": y[m], "yr": pd.DatetimeIndex(r["dates"][m]).year}))
    df = pd.concat(df)
    return df.groupby("yr").apply(lambda g: round(roc_auc_score(g["y"], g["p"]), 3) if g["y"].nunique() > 1 else np.nan)
base = dict(H=20, label="pos", hidden=(16, 8), ens=3, epochs=10, stride=1)
for cfg in [base, {**base, "H": 60, "label": "xs"}, {**base, "cond": "both_x3", "H": 60, "label": "neg"}, {**base, "H": 5, "label": "xs"}]:
    print({k: v for k, v in cfg.items() if base.get(k) != v}, per_year(cfg).to_dict())
