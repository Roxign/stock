"""Direct reinforcement (Moody & Saffell 2001): the policy network maps today's features to a position and is trained
by gradient ascent on the backtest itself. Here the market does not react to our orders, so the reward of every
position is known in closed form and differentiable (p * R minus the fees of |change in p|) -- this is policy
gradient with the exact gradient instead of a sampled one.

    python -m strategies.rl.direct                 # every config in CONFIGS on the development years
    python -m strategies.rl.direct log budget50    # a subset

Objectives (the "push it past buy-and-hold" ideas):
  log       maximise mean log(1 + net return) per stock-day (Kelly); relative-to-B&H rewards give the same optimum
  sharpe    maximise the Sharpe ratio of the equal-weight portfolio of the per-stock positions
  budgetXX  log objective with the average position forced to XX% -> the model must choose WHICH days to hold
  detrend   returns minus the stock's own average drift in the training years: only timing can earn
  mn        market-neutral: returns minus the day's cross-sectional mean, positions in [-1, 1] (stock selection)
  ls        long/short in [-1, 1] on raw returns (shorts pay the 融券 fee)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from .common import DEV_YEARS, OOS_YEARS, evaluate, line, load_panel, train_rows
from .panel import GROUPS

torch.set_num_threads(8)
POS_DIR = __import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "rl" / "pos"


@dataclass
class Cfg:
    name: str
    objective: str = "log"
    groups: tuple = ("tech", "xs", "chips", "mkt")
    side: str = "long"           # long: [0, 1]; ls: [-1, 1]
    budget: float | None = None  # forced mean |position|
    smooth: float = 0.0          # EMA factor applied to the raw position (0 = none); makes the policy trade less
    hidden: tuple = (32, 16)
    steps: int = 600
    window: int = 256
    lr: float = 1e-3
    wd: float = 1e-3
    dropout: float = 0.1
    seeds: tuple = (0,)
    cost_mult: float = 1.0       # >1 trains with inflated fees (a regulariser against churning)
    patience: int = 6            # validation checks (every 25 steps) without improvement before stopping
    extra: dict = field(default_factory=dict)


CONFIGS = [
    Cfg("log"),
    Cfg("log_smooth", smooth=0.8),
    Cfg("sharpe", objective="sharpe"),
    Cfg("budget50", budget=0.5),
    Cfg("budget80", budget=0.8),
    Cfg("detrend", objective="detrend"),
    Cfg("ls", side="ls"),
    Cfg("mn", objective="mn", side="ls"),
    Cfg("log_tech", groups=("tech",)),
    Cfg("log_mkt", groups=("mkt",)),
    Cfg("budget50_cost3", budget=0.5, cost_mult=3.0),
    Cfg("log_nostop", steps=2000, patience=10_000),   # is early stopping what keeps it away from buy-and-hold?
    Cfg("log_ens5", seeds=(0, 1, 2, 3, 4)),           # does averaging five nets help?
]


class Net(torch.nn.Module):
    def __init__(self, f, hidden, dropout):
        super().__init__()
        layers, d = [], f
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Dropout(dropout)]
            d = h
        layers.append(torch.nn.Linear(d, 1))
        self.f = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x).squeeze(-1)


def _ema(raw, a):
    """EMA along time (dim 0) of [L, N] with factor a, as a causal convolution (kernel truncated at 40 bars)."""
    if a <= 0:
        return raw
    k = torch.tensor([(1 - a) * a ** i for i in range(40)], dtype=raw.dtype)
    k = k / k.sum()
    x = raw.T.unsqueeze(1)  # [N, 1, L]
    x = torch.nn.functional.pad(x, (39, 0), mode="replicate")
    return torch.nn.functional.conv1d(x, k.flip(0).view(1, 1, -1)).squeeze(1).T


def _fees(P, mult):
    return (torch.tensor(P.buy * mult, dtype=torch.float32), torch.tensor(P.sell * mult, dtype=torch.float32),
            torch.tensor((P.sell + P.short) * mult, dtype=torch.float32))


def _cost(p, buy, sell, sshort):
    prev = torch.cat([p[:1].detach(), p[:-1]], 0)
    L, Lp, S, Sp = p.clamp(min=0), prev.clamp(min=0), (-p).clamp(min=0), (-prev).clamp(min=0)
    return buy * (L - Lp).clamp(min=0) + sell * (Lp - L).clamp(min=0) + sshort * (S - Sp).clamp(min=0) + buy * (Sp - S).clamp(min=0)


def _target(P, rows, cfg, drift):
    R = P.R[rows].astype(np.float32)
    if cfg.objective == "detrend":
        R = R - drift[None, :]
    elif cfg.objective == "mn":
        m = P.warm[rows]
        R = R - (np.where(m, R, 0).sum(1, keepdims=True) / np.maximum(m.sum(1, keepdims=True), 1))
    return R


def _loss(ret, mask, p, cfg):
    m = mask.float()
    if cfg.objective == "sharpe":
        ew = (ret * m).sum(1) / m.sum(1).clamp(min=1)
        loss = -(ew.mean() / (ew.std() + 1e-6)) * 15.87
    else:
        loss = -(torch.log1p(ret.clamp(min=-0.5)) * m).sum() / m.sum() * 252
    if cfg.budget is not None:
        e = (p.abs() * m).sum() / m.sum()
        loss = loss + 50.0 * (e - cfg.budget) ** 2
    return loss


def fit_year(P, year, cfg, seed, Xn, cols):
    tr = train_rows(P, year, 1)
    val = tr[-250:]
    fit = tr[:-250]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = Net(len(cols), cfg.hidden, cfg.dropout)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    buy, sell, sshort = _fees(P, cfg.cost_mult)
    use = P.warm & P.ok
    drift = np.array([P.R[fit][use[fit][:, n], n].mean() if use[fit][:, n].any() else 0.0 for n in range(P.R.shape[1])], dtype=np.float32)
    act = torch.sigmoid if cfg.side == "long" else torch.tanh

    def run(rows, train):
        x = torch.from_numpy(Xn[rows])
        R = torch.from_numpy(_target(P, rows, cfg, drift))
        mask = torch.from_numpy(use[rows])
        net.train(train)
        p = _ema(act(net(x)), cfg.smooth)
        p = torch.where(mask, p, torch.zeros_like(p))
        ret = p * R - _cost(p, buy, sell, sshort)
        return _loss(ret, mask, p, cfg)

    best, best_state, bad = np.inf, None, 0
    for step in range(cfg.steps):
        s = rng.integers(0, len(fit) - cfg.window)
        loss = run(fit[s : s + cfg.window], True)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 24:
            with torch.no_grad():
                v = float(run(val, False))
            if v < best - 1e-4:
                best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if best_state is not None and cfg.patience < 1000:  # no-stop runs keep the final weights
        net.load_state_dict(best_state)
    te = np.flatnonzero(P.year == year)
    rows = np.arange(max(te[0] - 60, 0), te[-1] + 1)
    net.eval()
    with torch.no_grad():
        p = _ema(act(net(torch.from_numpy(Xn[rows]))), cfg.smooth).numpy()
    return te, p[-len(te):]


def normalised(P, cols, upto_year):
    rows = train_rows(P, upto_year, 1)
    m = P.warm[rows]
    x = P.X[rows][m][:, cols]
    mu, sd = x.mean(0), x.std(0) + 1e-6
    return np.clip((P.X[:, :, cols] - mu) / sd, -5, 5).astype(np.float32)


def run(P, cfg, years=DEV_YEARS, verbose=True):
    names = [f for g in cfg.groups for f in GROUPS[g]]
    cols = P.cols(names)
    pos = np.zeros(P.R.shape, dtype=np.float32)
    t0 = time.time()
    for year in years:
        Xn = normalised(P, cols, year)  # statistics of the training years only
        acc = None
        for seed in cfg.seeds:
            te, p = fit_year(P, year, cfg, seed, Xn, cols)
            acc = p if acc is None else acc + p
        pos[te] = acc / len(cfg.seeds)
    m = evaluate(P, pos, years, name=f"direct_{cfg.name}", config=cfg.__dict__ | {"family": "direct"}, log=not cfg.name.startswith("planted"),
                 period="dev" if years == DEV_YEARS else "oos")
    if verbose:
        print(line(cfg.name, m), f"({time.time() - t0:.0f}s)", flush=True)
    POS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(POS_DIR / f"direct_{cfg.name}_{'dev' if years == DEV_YEARS else 'oos'}.npy", pos)
    return m, pos


def control_other_stock(P, pos, years=DEV_YEARS):
    """B2: trade each stock with the positions the model chose for a different stock that day."""
    return evaluate(P, np.roll(pos, 1, axis=1), years)


if __name__ == "__main__":
    P = load_panel()
    pick = sys.argv[1:]
    for cfg in CONFIGS:
        if pick and cfg.name not in pick:
            continue
        m, pos = run(P, cfg)
        print(line("  └ 別檔訊號 B2", control_other_stock(P, pos)), flush=True)
