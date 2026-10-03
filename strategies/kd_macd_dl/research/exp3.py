from harness import show, baseline_rows
from wf import run
from mapping import eval_out
base = dict(H=20, label="pos", hidden=(16, 8), ens=3, epochs=10, stride=1)
rows = baseline_rows()
for H in (20, 60):
    out = run({**base, "H": H, "label": "xs"}, verbose=False)
    for lo, hi in [(0.2, 0.4), (0.2, 0.6), (0.4, 0.6)]:
        for span in (5, 20):
            rows += eval_out(out, f"xsH{H} q{lo}-{hi} s{span}", lo=lo, hi=hi, span=span)
show(rows)
