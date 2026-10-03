import numpy as np
import pandas as pd

from harness import data
from stocklab.indicators import atr, ema, macd, tw_kdj as tw_kd


def feats(df):
    c = df["close"]
    a = atr(df)
    out = {}
    kd = tw_kd(df)
    out["k"] = kd["k"]
    out["kd"] = kd["k"] - kd["d"]
    m = macd(c)
    out["dif_atr"] = m["dif"] / a
    out["osc_atr"] = m["osc"] / a
    out["dif_px"] = m["dif"] / c * 100
    out["dosc"] = (m["osc"] - m["osc"].shift(3)) / a
    kdw = tw_kd(df, 45, 15, 15)  # ~weekly
    out["kw"] = kdw["k"]
    out["kdw"] = kdw["k"] - kdw["d"]
    mw = macd(c, 60, 130, 45)
    out["difw_atr"] = mw["dif"] / a
    out["oscw_atr"] = mw["osc"] / a
    o = df["open"]
    for h in (5, 20, 60):
        out[f"f{h}"] = o.shift(-1 - h) / o.shift(-1) - 1
    return pd.DataFrame(out)


d = data()
F = pd.concat({c: feats(df) for c, df in d.items()}, names=["code", "date"])
F = F[(F.index.get_level_values("date") >= "2010-06-01") & (F.index.get_level_values("date") <= "2020-09-30")].dropna()
print(len(F))
for col in ["k", "kd", "dif_atr", "osc_atr", "dif_px", "dosc", "kw", "kdw", "difw_atr", "oscw_atr"]:
    q = pd.qcut(F[col], 10, labels=False, duplicates="drop")
    g = F.groupby(q)[["f5", "f20", "f60"]].mean() * 100
    up = F.groupby(q)["f20"].apply(lambda s: (s > 0).mean() * 100)
    print(f"\n{col}  (mean fwd return % by decile, and P(f20>0))")
    print(pd.concat([g, up.rename("p20up")], axis=1).round(2).T.to_string())
