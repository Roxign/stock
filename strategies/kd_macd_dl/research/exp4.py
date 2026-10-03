import time
from meta import run, report
for rule in ("both1", "kdx_osc", "kdx", "dip", "dip2", "macdx_kd"):
    for model in ("mlp", "lr"):
        t = time.time()
        cfg = dict(rule=rule, model=model)
        out = run(cfg)
        auc, qm, qw, n = report(out)
        print(f"{rule:9s} {model:4s} n={n:5d} AUC={auc:.3f} meanret%/quintile={qm} win/quintile={qw} ({time.time()-t:.0f}s)", flush=True)
