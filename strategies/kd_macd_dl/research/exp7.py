import itertools, numpy as np, pandas as pd
from harness import show, baseline_rows, data, evaluate
from wf import run
from mapping import rule_pos
base = dict(H=20, label="xs", hidden=(16, 8), ens=3, epochs=10, stride=1)
d = data()
BR = {r: {c: rule_pos(df, r) for c, df in d.items()} for r in ("either_x2", "either_x3", "both_x2", "both_x3")}
rows = baseline_rows()
for r in BR:
    rows += evaluate(BR[r], f"{r} base")
def veto(out, r, span, half):
    pos = {}
    for c, df in d.items():
        b = BR[r][c]
        p0 = out["preds"][c]
        p = p0.ewm(span=span, adjust=False).mean().where(p0.notna())
        q = out["thr"][c]
        v = (p >= q["q40"]).astype(float)
        if half:
            v = np.maximum(v, 0.5 * ((p >= q["q20"]) & (p < q["q40"])).astype(float))
        pos[c] = np.maximum(b, v).where(p.notna(), b)
    return pos
for H, model in itertools.product((5, 20, 60), ("mlp", "lr")):
    out = run({**base, "H": H, "model": model}, verbose=False)
    for r in BR:
        for span in (10, 20):
            rows += evaluate(veto(out, r, span, True), f"{model} H{H} s{span} {r}")
df = show(rows)
df.to_pickle("exp7.pkl")
