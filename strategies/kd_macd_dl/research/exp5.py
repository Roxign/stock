import numpy as np, pandas as pd
from harness import show, baseline_rows, data, evaluate
from wf import run
from base_rules import state_machine
from stocklab.indicators import macd, tw_kd
base = dict(H=60, label="xs", hidden=(16, 8), ens=3, epochs=10, stride=1)
out = run(base, verbose=False)
d = data()
def rule(df, s):
    kd = tw_kd(df, 9 * s, 3 * s, 3 * s); m = macd(df["close"], 12 * s, 26 * s, 9 * s)
    k, dd, dif, osc = kd["k"], kd["d"], m["dif"], m["osc"]
    return {"either": pd.Series(state_machine((k > dd) | (osc > 0), (k < dd) & (osc < 0) & (dif < 0)), index=df.index),
            "both": pd.Series(state_machine((k > dd) & (osc > 0), (k < dd) & (osc < 0)), index=df.index)}
rows = baseline_rows()
for s in (2, 3, 5):
    for rn in ("either", "both"):
        pos_b, pos_c, pos_h = {}, {}, {}
        for c, df in d.items():
            b = rule(df, s)[rn]
            p = out["preds"][c].ewm(span=10, adjust=False).mean().where(out["preds"][c].notna())
            q = out["thr"][c]
            strong = (p >= q["q40"]).astype(float)
            mid = ((p >= q["q20"]) & (p < q["q40"])).astype(float)
            veto = np.maximum(b, strong)  # stay long if DL says relatively strong
            pos_b[c] = b
            pos_c[c] = veto.where(p.notna(), b)
            pos_h[c] = np.maximum(b, np.maximum(strong, 0.5 * mid)).where(p.notna(), b)
        rows += evaluate(pos_b, f"{rn}_x{s} base")
        rows += evaluate(pos_c, f"{rn}_x{s} +DLveto")
        rows += evaluate(pos_h, f"{rn}_x{s} +DLvetoHalf")
show(rows)
