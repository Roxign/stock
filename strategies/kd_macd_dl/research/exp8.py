import time, numpy as np, pandas as pd
from harness import show, baseline_rows, evaluate
from meta import run, report, positions, base_positions
rows = baseline_rows()
for rule in ("both1", "both_x2", "both_x3", "macdx_kd"):
    first = True
    for label in ("abs", "xs"):
        for model in ("mlp", "lr"):
            cfg = dict(rule=rule, model=model, label=label)
            out = run(cfg)
            auc, qm, qw, n = report(out)
            print(f"{rule:9s} {label} {model:4s} n={n:5d} AUC={auc:.3f} meanret%/quintile={qm}", flush=True)
            if first:
                rows += evaluate(base_positions(out), f"{rule} base"); first = False
            for lo, hi in ((0.33, 0.67), (0.5, 0.67), (0.33, 0.5)):
                rows += evaluate(positions(out, mode="q", lo=lo, hi=hi), f"{rule} {label} {model} q{lo}-{hi}")
show(rows)
