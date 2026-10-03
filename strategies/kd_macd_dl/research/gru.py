"""Tiny GRU on 20-day sequences of 8 KD/MACD features, same xs H60 labels / purge as production. IS AUC only."""
import sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from sklearn.metrics import roc_auc_score
from harness import data
from strategies.kd_macd_dl import core
torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
T, STRIDE = 20, 3
d = data()
stocks, _ = core._prepare(d)
for s in stocks:
    ind = s["ind"]; a = ind["atr"].to_numpy()
    s["F"] = np.column_stack([ind["k"] / 100 - .5, ind["d"] / 100 - .5, ind["dif"] / a, ind["osc"] / a,
                              ind["kw"] / 100 - .5, ind["dw"] / 100 - .5, ind["difw"] / a, ind["oscw"] / a]).astype(np.float32)
class GRU(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.g = torch.nn.GRU(8, 12, batch_first=True); self.o = torch.nn.Linear(12, 1)
    def forward(self, x):
        _, h = self.g(x); return self.o(h[-1]).squeeze(-1)
print("GRU params", sum(p.numel() for p in GRU().parameters()))
def seqs(s, idx):
    return np.stack([s["F"][i - T + 1: i + 1] for i in idx])
P, Y, YR = [], [], []
t0 = time.time()
for Yr in range(2012, 2021):
    cut = np.datetime64(f"{Yr}-01-01"); nxt = np.datetime64(f"{Yr+1}-01-01")
    Xs, ys = [], []
    for s in stocks:
        n = len(s["y"]); i = np.arange(n)
        m = (i >= core.WARM) & ~np.isnan(s["y"]) & (s["known"] < cut) & (i % STRIDE == 0)
        idx = i[m]
        if len(idx): Xs.append(seqs(s, idx)); ys.append(s["y"][idx])
    X = np.concatenate(Xs); y = np.concatenate(ys)
    mu = X.reshape(-1, 8).mean(0); sd = X.reshape(-1, 8).std(0)
    Z = np.clip((X - mu) / sd, -5, 5).astype(np.float32)
    torch.manual_seed(Yr); g = torch.Generator().manual_seed(Yr)
    net = GRU(); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4); lf = torch.nn.BCEWithLogitsLoss()
    Zt, yt = torch.from_numpy(Z), torch.from_numpy(y.astype(np.float32))
    for ep in range(10):
        perm = torch.randperm(len(yt), generator=g)
        for b in range(0, len(yt), 256):
            ii = perm[b:b + 256]; opt.zero_grad(); l = lf(net(Zt[ii]), yt[ii]); l.backward(); opt.step()
    with torch.no_grad():
        for s in stocks:
            dates = s["index"].values; i = np.arange(len(dates))
            m = (dates >= cut) & (dates < nxt) & ~np.isnan(s["y"]) & (i >= T)
            idx = i[m]
            if len(idx):
                Zs = np.clip((seqs(s, idx) - mu) / sd, -5, 5).astype(np.float32)
                P.append(torch.sigmoid(net(torch.from_numpy(Zs))).numpy()); Y.append(s["y"][idx]); YR.append(np.full(len(idx), Yr))
    print(Yr, len(y), f"{time.time()-t0:.0f}s", flush=True)
P, Y, YR = map(np.concatenate, (P, Y, YR))
print("GRU pooled AUC", round(roc_auc_score(Y, P), 4), {int(yr): round(roc_auc_score(Y[YR == yr], P[YR == yr]), 3) for yr in np.unique(YR)})
