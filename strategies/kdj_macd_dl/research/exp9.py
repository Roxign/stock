import numpy as np, pandas as pd
from harness import show, baseline_rows, data, evaluate
from wf import run, auc_report
from mapping import rule_pos
d = data()
base = dict(H=20, label="xs", hidden=(16, 8), ens=3, epochs=10, stride=1)
rows = baseline_rows()
# (2) drawdown-risk model
for H, thr in ((20, 0.10), (60, 0.15)):
    cfg = {**base, "H": H, "label": "dd", "thr": thr}
    out = run(cfg, verbose=False)
    auc, q = auc_report(out, cfg)
    print("dd", H, thr, "AUC", round(auc, 3), q)
    for cut in ("q60", "q40"):
        pos = {}
        for c, df in d.items():
            p0 = out["preds"][c]; p = p0.ewm(span=10, adjust=False).mean().where(p0.notna())
            risky = p >= out["thr"][c][cut]  # thr are quantiles of P(dd); high = risky
            pos[c] = (1.0 - 0.5 * risky.astype(float)).where(p.notna(), rule_pos(df, "either_x5"))
        rows += evaluate(pos, f"ddH{H} half-when>{cut}")
# (1) vote / floor designs with the xs H60 model
out = run({**base, "H": 60}, verbose=False)
for r in ("both_x3", "either_x3"):
    vote, floor = {}, {}
    for c, df in d.items():
        b = rule_pos(df, r); p0 = out["preds"][c]; p = p0.ewm(span=10, adjust=False).mean().where(p0.notna())
        strong = (p >= out["thr"][c]["q40"]).astype(float)
        vote[c] = (0.5 * b + 0.5 * strong).where(p.notna(), b)
        floor[c] = (0.5 + 0.5 * b)
    rows += evaluate(vote, f"vote {r}+DL")
    rows += evaluate(floor, f"floor {r} (0.5/1)")
# (3) random control for veto (either_x3, matched veto rate ~ P(p>=q40) and persistence via smoothing)
rng = np.random.default_rng(0)
for trial in range(3):
    pos = {}
    for c, df in d.items():
        b = rule_pos(df, "either_x3")
        z = pd.Series(rng.standard_normal(len(df)), index=df.index).ewm(span=60, adjust=False).mean()
        strong = (z >= z.quantile(0.4)).astype(float)  # ~60% of bars "strong", similar persistence
        pos[c] = np.maximum(b, strong)
    rows += evaluate(pos, f"random veto either_x3 #{trial}")
show(rows)
