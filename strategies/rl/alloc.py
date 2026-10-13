"""Cross-sectional allocator: one account, the policy splits it across all stocks (plus cash) and rebalances every k
bars. Timing a single stock can only step aside; an allocator can always be fully invested and still differ from
buy-and-hold by WHICH stocks it holds, so "just hold" is not a trivial optimum.

Trained by direct reinforcement: weights = softmax(stock scores, cash score); the reward of a block is the exact
buy-and-hold return of the chosen weights over the block, minus fees on the turnover from the drifted weights.

    python -m strategies.rl.alloc

Benchmarks over the same blocks: equal weight (rebalanced each block, same fees), simple factor portfolios
(momentum, short-term reversal, low volatility; top 20), and 0050.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
import torch

from stocklab.etf import load_etf
from stocklab.trials import log_trial

from .common import AGENT, DEV_YEARS, OOS_YEARS, load_panel, save, train_rows
from .direct import normalised
from .panel import GROUPS, MKT_FEATS

torch.set_num_threads(8)
OUT = __import__("pathlib").Path(__file__).parent / "results" / "alloc.json"


def blocks(P, rows, k):
    """Block starts and per-stock block returns G[b, n] (exact compounding of R; unusable bars count as 0)."""
    use = P.warm & P.ok
    starts = rows[::k]
    lr = np.log1p(np.where(use, P.R, 0.0).astype(np.float64))
    c = np.vstack([np.zeros((1, lr.shape[1])), np.cumsum(lr, axis=0)])
    ends = np.minimum(starts + k, len(P.R))
    G = np.exp(c[ends] - c[starts]) - 1
    return starts, G, use[starts]


def port(P, W, G):
    """Net block returns for weights W[b, n] (cash = 1 - sum), with fees on turnover from the drifted weights."""
    r = np.zeros(len(W))
    prev = np.zeros(W.shape[1])
    for b in range(len(W)):
        tc = (P.buy * np.maximum(W[b] - prev, 0) + P.sell * np.maximum(prev - W[b], 0)).sum()
        gross = (W[b] * G[b]).sum()
        r[b] = gross - tc
        prev = W[b] * (1 + G[b]) / (1 + gross)
    return r


def stats(r, k):
    lw = np.log1p(r)
    per = 252 / k
    eq = np.exp(np.cumsum(lw))
    return {"cagr": float(math.exp(lw.mean() * per) - 1), "sharpe": float(r.mean() / r.std() * math.sqrt(per)),
            "mdd": float((eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min())}


class Policy(torch.nn.Module):
    def __init__(self, f, fm, h=32, cash=True):
        super().__init__()
        self.s = torch.nn.Sequential(torch.nn.Linear(f, h), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(h, 1))
        self.c = torch.nn.Sequential(torch.nn.Linear(fm, 16), torch.nn.GELU(), torch.nn.Linear(16, 1)) if cash else None
        self.temp = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, xm, mask):
        s = self.s(x).squeeze(-1) * torch.exp(self.temp)
        s = torch.where(mask, s, torch.full_like(s, -1e9))
        if self.c is not None:
            s = torch.cat([s, self.c(xm)], -1)
        w = torch.softmax(s, -1)
        return w[..., : mask.shape[-1]]


def fit(P, Xn, cols_m, year, k, cfg, seed=0):
    tr = train_rows(P, year, k)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = Policy(Xn.shape[2], len(cols_m), cash=cfg.get("cash", True))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    buy, sell = torch.tensor(P.buy, dtype=torch.float32), torch.tensor(P.sell, dtype=torch.float32)
    span = 26 * k
    for step in range(cfg.get("steps", 400)):
        s = rng.integers(0, len(tr) - span)
        starts, G, mask = blocks(P, tr[s : s + span], k)
        mask = mask & cfg["universe"][None, :]
        x = torch.from_numpy(Xn[starts])
        xm = torch.from_numpy(Xn[starts][:, 0, :][:, cols_m])
        W = net(x, xm, torch.from_numpy(mask))
        Gt = torch.from_numpy(G.astype(np.float32))
        gross = (W * Gt).sum(1)
        drift = torch.cat([torch.zeros_like(W[:1]), (W * (1 + Gt) / (1 + gross[:, None]))[:-1]], 0).detach()
        tc = (buy * (W - drift).clamp(min=0) + sell * (drift - W).clamp(min=0)).sum(1)
        r = gross - tc
        loss = -torch.log1p(r).mean() if cfg.get("obj", "log") == "log" else -(r.mean() / (r.std() + 1e-6))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def run_learned(P, k, cfg, years=DEV_YEARS):
    cols = P.cols([f for g in cfg["groups"] for f in GROUPS[g]])
    cols_m = [cols.index(c) for c in P.cols(MKT_FEATS) if c in cols] or list(range(min(4, len(cols))))
    rows = np.flatnonzero(np.isin(P.year, years))
    starts, G, mask = blocks(P, rows, k)
    mask = mask & cfg["universe"][None, :]
    W = np.zeros(G.shape)
    for year in years:
        Xn = normalised(P, cols, year)
        net = fit(P, Xn, cols_m, year, k, cfg)
        b = np.flatnonzero(P.year[starts] == year)
        net.eval()
        with torch.no_grad():
            W[b] = net(torch.from_numpy(Xn[starts[b]]), torch.from_numpy(Xn[starts[b]][:, 0, :][:, cols_m]), torch.from_numpy(mask[b])).numpy()
    return port(P, W, G), W


def factor(P, k, key, top=20, years=DEV_YEARS, universe=None, low=False):
    rows = np.flatnonzero(np.isin(P.year, years))
    starts, G, mask = blocks(P, rows, k)
    if universe is not None:
        mask = mask & universe[None, :]
    W = np.zeros(G.shape)
    for b, t in enumerate(starts):
        if not mask[b].any():
            continue
        v = P.X[t, :, P.fi[key]].astype(float) if key else np.zeros(G.shape[1])
        v = np.where(mask[b], v * (-1 if low else 1), -np.inf)
        if key is None:
            W[b, mask[b]] = 1 / mask[b].sum()
        else:
            idx = np.argsort(-v)[: min(top, mask[b].sum())]
            W[b, idx] = 1 / len(idx)
    return port(P, W, G)


def etf_0050(P, k, years=DEV_YEARS):
    df = load_etf("0050")
    o = df["open"].reindex(P.dates).ffill().to_numpy()
    R = np.r_[o[2:] / o[1:-1] - 1, 0, 0]
    rows = np.flatnonzero(np.isin(P.year, years))
    starts = rows[::k]
    lr = np.log1p(np.nan_to_num(R))
    c = np.r_[0, np.cumsum(lr)]
    g = np.exp(c[np.minimum(starts + k, len(R))] - c[starts]) - 1
    g[0] -= 0.001425
    return g


def main(period="dev"):
    """period="oos" runs the rules fixed after the development run once on 2021-."""
    P = load_panel()
    years = DEV_YEARS if period == "dev" else OOS_YEARS
    allu = np.ones(P.R.shape[1], bool)
    extra = ~P.is50
    res = {}
    learned = (("RL 配置：全部特徵", {"groups": ("tech", "xs", "chips", "mkt"), "universe": allu}),
               ("RL 配置：不留現金", {"groups": ("tech", "xs", "chips", "mkt"), "universe": allu, "cash": False}),
               ("RL 配置：只用技術面（不留現金）", {"groups": ("tech", "xs"), "universe": allu, "cash": False}),
               ("RL 配置：只用 55 檔代表股", {"groups": ("tech", "xs", "chips", "mkt"), "universe": extra}),
               ("RL 配置：Sharpe 目標", {"groups": ("tech", "xs", "chips", "mkt"), "universe": allu, "obj": "sharpe"}))
    if period == "oos":
        learned = tuple(x for x in learned if x[0] in ("RL 配置：Sharpe 目標", "RL 配置：只用 55 檔代表股"))
    for k in (5, 20):
        print(f"--- 每 {k} 天調整一次（{period}）---", flush=True)
        rows = {"等權（每期再平衡）": factor(P, k, None, years=years), "0050 買進持有": etf_0050(P, k, years),
                "動能：近一年最強 20 檔": factor(P, k, "r250", years=years), "動能：近 60 日最強 20 檔": factor(P, k, "r60", years=years),
                "反轉：近 5 日最弱 20 檔": factor(P, k, "r5", low=True, years=years), "低波動 20 檔": factor(P, k, "vol60", low=True, years=years),
                "動能（只用 55 檔代表股）": factor(P, k, "r250", universe=extra, years=years),
                "等權（只用 55 檔代表股）": factor(P, k, None, universe=extra, years=years)}
        for name, cfg in learned:
            t0 = time.time()
            r, W = run_learned(P, k, cfg, years)
            rows[name] = r
            print(f"  ({name} 訓練 {time.time() - t0:.0f}s，平均現金 {1 - W.sum(1).mean():.0%}，平均持有檔數（權重>1%）{(W > 0.01).sum(1).mean():.0f})", flush=True)
        res[k] = {}
        for name, r in rows.items():
            s = stats(r, k)
            res[k][name] = s
            log_trial(AGENT, f"alloc_k{k}_{name}", {"family": "alloc", "k": k, "name": name}, s, period=period)
            print(f"  {name:<24} CAGR {s['cagr']:6.1%}  Sharpe {s['sharpe']:5.2f}  MDD {s['mdd']:6.1%}", flush=True)
    old = {}
    if OUT.exists():
        import json

        old = json.loads(OUT.read_text(encoding="utf-8"))
    save(OUT, old | {period: res})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
