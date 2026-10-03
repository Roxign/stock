import numpy as np, pandas as pd
from harness import baseline_rows, data, evaluate, show
from base_rules import state_machine
from stocklab.indicators import macd, tw_kd

def rules(df):
    out = {}
    for s in (1, 2, 3, 5):
        kd = tw_kd(df, 9*s, 3*s, 3*s); m = macd(df["close"], 12*s, 26*s, 9*s)
        k, d, dif, osc = kd["k"], kd["d"], m["dif"], m["osc"]
        out[f"both_x{s}"] = state_machine((k > d) & (osc > 0), (k < d) & (osc < 0))
        out[f"osc_x{s}"] = (osc > 0).astype(float).to_numpy()
        out[f"dif_x{s}"] = (dif > 0).astype(float).to_numpy()
        # long unless both bearish & dif<0
        out[f"either_x{s}"] = state_machine((k > d) | (osc > 0), (k < d) & (osc < 0) & (dif < 0))
    return {n: pd.Series(v, index=df.index) for n, v in out.items()}

d = data()
allr = {c: rules(df) for c, df in d.items()}
rows = baseline_rows(("is",))
for name in next(iter(allr.values())):
    rows += evaluate({c: allr[c][name] for c in d}, name, ("is",))
show(rows)
