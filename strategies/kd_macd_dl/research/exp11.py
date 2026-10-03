import numpy as np, pandas as pd
from harness import show, baseline_rows, data, evaluate
from wf import run
from mapping import rule_pos
d = data()
base = dict(H=60, label="xs", hidden=(16, 8), ens=3, epochs=10, stride=1)
out = run(base, verbose=False)
outlr = run({**base, "model": "lr"}, verbose=False)
rows = baseline_rows()
rng = np.random.default_rng(1)
for r in ("both_x3", "either_x3", "both_x5", "both_x2"):
    for tag, o in (("mlp", out), ("lr", outlr), ("rand", None)):
        F1, F3 = {}, {}
        for c, df in d.items():
            b = rule_pos(df, r)
            if o is None:
                z = pd.Series(rng.standard_normal(len(df)), index=df.index).ewm(span=60, adjust=False).mean()
                p = z; q20, q40 = z.quantile(0.2), z.quantile(0.4); valid = pd.Series(True, index=df.index)
                valid[: 0] = False
            else:
                p0 = o["preds"][c]; p = p0.ewm(span=10, adjust=False).mean().where(p0.notna()); valid = p.notna()
                q20, q40 = o["thr"][c]["q20"], o["thr"][c]["q40"]
            strong = (p >= q40); weak = (p < q20)
            F1[c] = pd.Series(np.where(b.astype(bool) | strong, 1.0, 0.5), index=df.index).where(valid, 0.5 + 0.5 * b)
            F3[c] = pd.Series(np.where(b.astype(bool) | strong, 1.0, np.where(weak, 0.0, 0.5)), index=df.index).where(valid, b)
        if tag == "mlp":
            rows += evaluate({c: 0.5 + 0.5 * rule_pos(df, r) for c, df in d.items()}, f"floor {r}")
        rows += evaluate(F1, f"F1 floor+veto {r} {tag}")
        rows += evaluate(F3, f"F3 3lvl {r} {tag}")
show(rows)
