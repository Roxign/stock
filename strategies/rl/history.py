"""Let the model see the recent history of every feature, not just today's values.

Three ways, each trained walk-forward on 2013-2020 like the rest of the study:
  lag    every one of the 65 features at lags 0, 1, 2, 5, 10, 20, 40, 60 days (520 inputs), ridge regression
  tcn    a causal dilated convolution over the last 64 or 128 days of all 65 features
  gru    a recurrent network reading all 65 features one day at a time; its memory is warmed up on the 250 days
         before each prediction (gradients reach back about 250 days)
Two objectives: "sup" predicts the forward h-day return (to measure IC against the snapshot models: 0.065 / 0.066 /
0.119 / 0.163 for 1 / 5 / 20 / 60 days), "log" / "budget50" trade directly (direct reinforcement, as direct.py).

    python -m strategies.rl.history lag
    python -m strategies.rl.history tcn64_log gru_sup20 ...
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from .bottleneck import _ic
from .common import DEV_YEARS, OOS_YEARS, evaluate, forward, line, load_panel, save, train_rows
from .direct import POS_DIR, Cfg, _cost, _fees, _loss, normalised
from .panel import GROUPS

torch.set_num_threads(4)
OUT = __import__("pathlib").Path(__file__).parent / "results" / "history.json"
ALL = [f for g in ("tech", "xs", "chips", "mkt") for f in GROUPS[g]]
LAGS = (0, 1, 2, 5, 10, 20, 40, 60)


# ------------------------------------------------------------------ lagged features + ridge (linear baseline)

def lag_ic(P, horizons=(1, 5, 20, 60), years=DEV_YEARS):
    cols = P.cols(ALL)
    T, N = P.R.shape
    res = {}
    for h in horizons:
        y_all = forward(P, h)
        for lags in ((0,), LAGS):
            preds, ys, rt, cn = [], [], [], []
            for year in years:
                Xn = normalised(P, cols, year)
                tr = train_rows(P, year, h)
                tr = tr[tr >= max(LAGS)]
                m = P.warm[tr] & np.isfinite(y_all[tr])
                ti, ni = np.nonzero(m)
                keep = (tr[ti] % 3) == 0
                ti, ni = tr[ti[keep]], ni[keep]
                X = np.concatenate([Xn[ti - l, ni] for l in lags], axis=1)
                y = y_all[ti, ni]
                Xa = np.column_stack([X, np.ones(len(X), np.float32)])
                A = (Xa.T @ Xa).astype(np.float64)
                w = np.linalg.solve(A + np.diag(np.r_[np.ones(X.shape[1]), 0]) * len(X) * 1e-2, (Xa.T @ y.astype(np.float32)).astype(np.float64))
                te = np.flatnonzero(P.year == year)
                mt = P.warm[te] & np.isfinite(y_all[te])
                tj, nj = np.nonzero(mt)
                tj = te[tj]
                Z = np.concatenate([Xn[tj - l, nj] for l in lags], axis=1).astype(np.float64)
                preds.append(Z @ w[:-1] + w[-1]), ys.append(y_all[tj, nj]), rt.append(tj), cn.append(nj)
            ic = _ic(np.concatenate(preds), np.concatenate(ys), np.concatenate(rt), np.concatenate(cn), T, N)
            key = f"h{h}_{'today' if lags == (0,) else 'lags60'}"
            res[key] = ic
            print(f"  預測 {h:>2} 天・{'只看今天' if lags == (0,) else '看過去 60 天（8 個時間點）'}：擇時 IC {ic['ts_ic']:+.3f}  橫斷面 IC {ic['xs_ic']:+.3f}", flush=True)
    return res


# ------------------------------------------------------------------ sequence networks

class TCN(torch.nn.Module):
    def __init__(self, f, span, h=32):
        super().__init__()
        dil, rf, layers, d = [], 1, [], f
        k = 3
        while rf + (k - 1) * 2 ** len(dil) <= span:   # receptive field = span - 1 days back plus today
            dl = 2 ** len(dil)
            dil.append(dl)
            rf += (k - 1) * dl
        self.rf = rf
        self.convs = torch.nn.ModuleList()
        for dl in dil:
            self.convs.append(torch.nn.Conv1d(d, h, k, dilation=dl))
            d = h
        self.drop = torch.nn.Dropout(0.1)
        self.head = torch.nn.Conv1d(h, 1, 1)

    def forward(self, x):  # x [L, N, F] -> [L, N]
        z = x.permute(1, 2, 0)
        for c in self.convs:
            z = torch.nn.functional.gelu(c(torch.nn.functional.pad(z, ((c.kernel_size[0] - 1) * c.dilation[0], 0))))
        return self.head(self.drop(z)).squeeze(1).T


class GRU(torch.nn.Module):
    def __init__(self, f, h=32):
        super().__init__()
        self.inp = torch.nn.Linear(f, h)
        self.gru = torch.nn.GRU(h, h)
        self.drop = torch.nn.Dropout(0.1)
        self.head = torch.nn.Linear(h, 1)
        self.rf = 250

    def forward(self, x):
        z, _ = self.gru(torch.tanh(self.inp(x)))
        return self.head(self.drop(z)).squeeze(-1)


def make(kind, f):
    if kind.startswith("tcn"):
        return TCN(f, int(kind[3:]))
    return GRU(f)


def fit_year(P, Xn, year, kind, objective, h, steps=500, win=192, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make(kind, Xn.shape[2])
    ctx = net.rf if kind.startswith("tcn") else 60
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    use = P.warm & P.ok
    tr = train_rows(P, year, h if objective == "sup" else 1)
    tr = tr[tr >= ctx]
    fit, val = tr[:-250], tr[-250:]
    cfg = Cfg(objective, budget=0.5 if objective == "budget50" else None)
    buy, sell, sshort = _fees(P, 1.0)
    if objective == "sup":
        y_all = forward(P, h).astype(np.float32)
        ysd = float(np.nanstd(y_all[fit][use[fit]]))

    def loss_on(rows, train):
        net.train(train)
        full = np.arange(rows[0] - ctx, rows[-1] + 1)
        out = net(torch.from_numpy(Xn[full]))[ctx:]
        mask = torch.from_numpy(use[rows])
        if objective == "sup":
            y = torch.from_numpy(np.nan_to_num(y_all[rows] / ysd))
            mask = mask & torch.from_numpy(np.isfinite(y_all[rows]))
            return (((out - y) ** 2) * mask).sum() / mask.sum()
        p = torch.where(mask, torch.sigmoid(out), torch.zeros_like(out))
        ret = p * torch.from_numpy(P.R[rows]) - _cost(p, buy, sell, sshort)
        return _loss(ret, mask, p, cfg)

    best, state, bad = np.inf, None, 0
    for step in range(steps):
        s = rng.integers(0, len(fit) - win)
        loss = loss_on(fit[s : s + win], True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 25 == 24:
            with torch.no_grad():
                v = float(loss_on(val, False))
            if v < best - 1e-5:
                best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 6:
                    break
    if state:
        net.load_state_dict(state)
    te = np.flatnonzero(P.year == year)
    lead = 250 if kind == "gru" else ctx
    full = np.arange(max(te[0] - lead, 0), te[-1] + 1)
    net.eval()
    with torch.no_grad():
        out = net(torch.from_numpy(Xn[full])).numpy()[-len(te):]
    return te, out


def run_net(P, name, years=DEV_YEARS):
    kind, objective = name.split("_")[0], name.split("_")[1]
    h = int(objective[3:]) if objective.startswith("sup") else 1
    objective = "sup" if objective.startswith("sup") else objective
    cols = P.cols(ALL)
    out = np.full(P.R.shape, np.nan, np.float32)
    t0 = time.time()
    for year in years:
        Xn = normalised(P, cols, year)
        te, o = fit_year(P, Xn, year, kind, objective, h)
        out[te] = o
    period = "dev" if years == DEV_YEARS else "oos"
    if objective == "sup":
        y = forward(P, h)
        rows = np.flatnonzero(np.isin(P.year, years))
        m = P.warm[rows] & np.isfinite(y[rows]) & np.isfinite(out[rows])
        ti, ni = np.nonzero(m)
        ic = _ic(out[rows][m], y[rows][m], rows[ti], ni, *P.R.shape)
        print(f"{name:<18} 預測 {h} 天：擇時 IC {ic['ts_ic']:+.3f}  橫斷面 IC {ic['xs_ic']:+.3f}  ({time.time() - t0:.0f}s)", flush=True)
        return {"ic": ic}
    pos = 1 / (1 + np.exp(-np.nan_to_num(out, nan=-50.0)))
    m = evaluate(P, pos, years, name=f"history_{name}", config={"family": "history", "name": name}, log=True, period=period)
    print(line(name, m), f"({time.time() - t0:.0f}s)", flush=True)
    print(line("  └ 別檔訊號 B2", evaluate(P, np.roll(pos, 1, axis=1), years)), flush=True)
    np.save(POS_DIR / f"history_{name}_{period}.npy", pos)
    return {k: v for k, v in m.items() if not isinstance(v, dict)} | {"ew_sharpe": m["ew"]["sharpe"], "ew_bh_sharpe": m["ew_bh"]["sharpe"]}


def main(names):
    """Results go to results/history_<first name>.json, so several runs can go in parallel."""
    global OUT
    OUT = OUT.with_name(f"history_{names[0]}.json")
    P = load_panel()
    old = {}
    if OUT.exists():
        import json

        old = json.loads(OUT.read_text(encoding="utf-8"))
    for name in names:
        if name == "lag":
            old["lag"] = lag_ic(P)
        elif name.endswith("_oos"):
            old[name] = run_net(P, name[:-4], OOS_YEARS)
        else:
            old[name] = run_net(P, name)
        save(OUT, old)


if __name__ == "__main__":
    main(sys.argv[1:])
