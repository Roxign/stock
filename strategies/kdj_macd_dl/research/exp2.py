import time
from wf import run, auc_report
base = dict(H=20, label="pos", hidden=(16, 8), ens=3, epochs=10, stride=1)
grid = []
for H in (5, 10, 20, 60):
    for label in ("pos", "xs"):
        grid.append({**base, "H": H, "label": label})
grid += [
    {**base, "hidden": (8,), "epochs": 3},
    {**base, "epochs": 3},
    {**base, "epochs": 30},
    {**base, "groups": ("kd",)},
    {**base, "groups": ("macd",)},
    {**base, "groups": ("slow",)},
    {**base, "label": "volnorm", "thr": 0.0},
    {**base, "window": 4},
]
for cfg in grid:
    t = time.time()
    out = run(cfg, verbose=False)
    auc, q = auc_report(out, cfg)
    print(f"{str({k: v for k, v in cfg.items() if base.get(k) != v}):60s} AUC={auc:.4f} fwdret by pred quintile={q}  ({time.time()-t:.0f}s)", flush=True)
