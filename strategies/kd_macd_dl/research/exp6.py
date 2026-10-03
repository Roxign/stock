import time
from wf import run, auc_report
base = dict(H=20, label="neg", hidden=(16, 8), ens=3, epochs=10, stride=1)
for cond in ("both_x2", "both_x3", "either_x3"):
    for H in (20, 60):
        for label in ("neg", "xs"):
            cfg = {**base, "cond": cond, "H": H, "label": label}
            t = time.time()
            out = run(cfg, verbose=False)
            auc, q = auc_report(out, cfg)
            print(f"{cond} H={H} {label:4s} AUC={auc:.4f} fwdret by pred quintile={q} ({time.time()-t:.0f}s)", flush=True)
