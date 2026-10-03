import time
from harness import show, baseline_rows
from wf import run, auc_report
from mapping import eval_out
cfg = dict(H=20, label="pos", hidden=(16, 8), ens=3, epochs=10, stride=1)
t = time.time()
out = run(cfg)
print("time", time.time() - t)
print("AUC, mean fwd ret by pred quintile:", auc_report(out, cfg))
rows = baseline_rows()
for lo, hi in [(0.2, 0.4), (0.2, 0.6), (0.4, 0.6)]:
    for span in (1, 5, 10):
        rows += eval_out(out, f"q{lo}-{hi} s{span}", lo=lo, hi=hi, span=span)
show(rows)
