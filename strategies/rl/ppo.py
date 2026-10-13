"""PPO (Schulman et al. 2017): a trial-and-error agent that only sees the rewards of the actions it took.

Environment: one stock, one bar at a time. Observation = the panel features of that bar plus the agent's own state
(position, days held, open return). Action = target position from a small discrete set, filled at the next open.
Reward = log(1 + net return of the bar) after fees (x100), optionally relative to buy-and-hold or with the stock's
average drift removed. Many episodes (random stock, random start inside the training years) run in parallel.

    python -m strategies.rl.ppo                # every config on the development years 2013-2020
    python -m strategies.rl.ppo rel ls         # a subset

Walk-forward: the 2013 model trains from scratch on 2009-2012; each later year's model starts from the previous
year's weights and trains further on the data now available (still nothing from the test year).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

from .common import DEV_YEARS, evaluate, line, load_panel, train_rows
from .direct import POS_DIR, normalised
from .panel import GROUPS

torch.set_num_threads(8)


@dataclass
class Cfg:
    name: str
    actions: tuple = (0.0, 0.5, 1.0)
    reward: str = "abs"          # abs | rel (minus buy-and-hold) | detrend (minus the stock's training drift)
    groups: tuple = ("tech", "xs", "chips", "mkt")
    ent: float = 0.01            # entropy bonus: the exploration "push"
    gamma: float = 0.9
    lam: float = 0.9
    envs: int = 256
    horizon: int = 128
    iters_first: int = 60
    iters_next: int = 20
    lr: float = 3e-4
    hidden: int = 64
    seed: int = 0


CONFIGS = [
    Cfg("abs"),
    Cfg("rel", reward="rel"),
    Cfg("rel_ent05", reward="rel", ent=0.05),
    Cfg("detrend", reward="detrend"),
    Cfg("ls_rel", reward="rel", actions=(-1.0, 0.0, 1.0)),
    Cfg("rel_mkt", reward="rel", groups=("mkt",)),
    Cfg("rel_g99", reward="rel", gamma=0.99, lam=0.95),
]


class AC(torch.nn.Module):
    def __init__(self, f, h, a):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(f, h), torch.nn.Tanh(), torch.nn.Linear(h, h), torch.nn.Tanh())
        self.pi, self.v = torch.nn.Linear(h, a), torch.nn.Linear(h, 1)

    def forward(self, x):
        z = self.body(x)
        return self.pi(z), self.v(z).squeeze(-1)


def _cost(P, n, a, prev):
    L, Lp, S, Sp = np.maximum(a, 0), np.maximum(prev, 0), np.maximum(-a, 0), np.maximum(-prev, 0)
    return (P.buy[n] * np.maximum(L - Lp, 0) + P.sell[n] * np.maximum(Lp - L, 0)
            + (P.sell[n] + P.short[n]) * np.maximum(S - Sp, 0) + P.buy[n] * np.maximum(Sp - S, 0))


class Env:
    """Vectorised episodes over the usable bars of each stock inside the training rows."""

    def __init__(self, P, Xn, rows, cfg, rng):
        self.P, self.Xn, self.cfg, self.rng = P, Xn, cfg, rng
        use = P.warm & P.ok
        self.paths = []
        for n in range(P.R.shape[1]):
            r = rows[use[rows, n]]
            if len(r) > cfg.horizon + 5:
                self.paths.append((n, r))
        self.drift = np.zeros(P.R.shape[1], np.float32)
        for n, r in self.paths:
            self.drift[n] = P.R[r, n].mean()
        self.A = np.array(cfg.actions, np.float32)
        self.reset(np.arange(cfg.envs))

    def reset(self, idx):
        if not hasattr(self, "n"):
            E = self.cfg.envs
            self.n, self.rows, self.k = np.zeros(E, int), [None] * E, np.zeros(E, int)
            self.pos, self.held, self.entry = np.zeros(E, np.float32), np.zeros(E, np.float32), np.zeros(E, np.float32)
        for e in idx:
            n, r = self.paths[self.rng.integers(len(self.paths))]
            s = self.rng.integers(0, len(r) - self.cfg.horizon - 1)
            self.n[e], self.rows[e], self.k[e] = n, r[s : s + self.cfg.horizon + 1], 0
            self.pos[e], self.held[e], self.entry[e] = 0.0, 0.0, 0.0

    def obs(self):
        t = np.array([self.rows[e][self.k[e]] for e in range(len(self.n))])
        x = self.Xn[t, self.n]
        return np.column_stack([x, self.pos, np.minimum(self.held, 120) / 60, np.clip(self.entry, -0.5, 0.5) * 5]).astype(np.float32), t

    def step(self, a_idx, t):
        P, cfg = self.P, self.cfg
        a = self.A[a_idx]
        R = P.R[t, self.n]
        net = a * R - _cost(P, self.n, a, self.pos)
        r = np.log1p(np.maximum(net, -0.5))
        if cfg.reward == "rel":
            r = r - np.log1p(R)
        elif cfg.reward == "detrend":
            r = np.log1p(np.maximum(a * (R - self.drift[self.n]) - _cost(P, self.n, a, self.pos), -0.5))
        same = (np.sign(a) == np.sign(self.pos)) & (a != 0)
        self.held = np.where(same, self.held + 1, np.where(a != 0, 1, 0)).astype(np.float32)
        self.entry = np.where(a != 0, np.where(same, (1 + self.entry) * (1 + np.sign(a) * R) - 1, np.sign(a) * R), 0).astype(np.float32)
        self.pos = a
        self.k += 1
        done = self.k >= self.cfg.horizon
        return (r * 100).astype(np.float32), done


def train(P, Xn, rows, cfg, net=None, iters=60, rng=None):
    f = Xn.shape[2] + 3
    net = net or AC(f, cfg.hidden, len(cfg.actions))
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    env = Env(P, Xn, rows, cfg, rng)
    E, H = cfg.envs, cfg.horizon
    for it in range(iters):
        O = np.zeros((H, E, f), np.float32)
        A = np.zeros((H, E), int)
        LP = np.zeros((H, E), np.float32)
        V = np.zeros((H + 1, E), np.float32)
        Rw = np.zeros((H, E), np.float32)
        env.reset(np.arange(E))
        for h in range(H):
            o, t = env.obs()
            with torch.no_grad():
                logits, v = net(torch.from_numpy(o))
                d = torch.distributions.Categorical(logits=logits)
                a = d.sample()
            O[h], A[h], LP[h], V[h] = o, a.numpy(), d.log_prob(a).numpy(), v.numpy()
            Rw[h], _ = env.step(a.numpy(), t)
        o, _ = env.obs()
        with torch.no_grad():
            V[H] = net(torch.from_numpy(o))[1].numpy()
        adv = np.zeros((H, E), np.float32)
        g = np.zeros(E, np.float32)
        for h in reversed(range(H)):
            delta = Rw[h] + cfg.gamma * V[h + 1] - V[h]
            g = delta + cfg.gamma * cfg.lam * g
            adv[h] = g
        ret = adv + V[:H]
        o_t, a_t, lp_t = torch.from_numpy(O.reshape(-1, f)), torch.from_numpy(A.reshape(-1)), torch.from_numpy(LP.reshape(-1))
        adv_t, ret_t = torch.from_numpy(adv.reshape(-1)), torch.from_numpy(ret.reshape(-1))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        N = len(a_t)
        for _ in range(4):
            perm = torch.randperm(N)
            for s in range(0, N, 4096):
                b = perm[s : s + 4096]
                logits, v = net(o_t[b])
                d = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(d.log_prob(a_t[b]) - lp_t[b])
                pg = -torch.min(ratio * adv_t[b], ratio.clamp(0.8, 1.2) * adv_t[b]).mean()
                loss = pg + 0.5 * ((v - ret_t[b]) ** 2).mean() - cfg.ent * d.entropy().mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
    return net


def act_year(P, Xn, net, cfg, year, state):
    """Greedy actions through the year for every stock at once, continuing from `state` (pos, held, entry)."""
    use = P.warm & P.ok
    te = np.flatnonzero(P.year == year)
    A = np.array(cfg.actions, np.float32)
    pos, held, entry = state
    out = np.zeros((len(te), P.R.shape[1]), np.float32)
    for k, t in enumerate(te):
        o = np.column_stack([Xn[t], pos, np.minimum(held, 120) / 60, np.clip(entry, -0.5, 0.5) * 5]).astype(np.float32)
        with torch.no_grad():
            a = A[net(torch.from_numpy(o))[0].argmax(1).numpy()]
        a = np.where(use[t], a, pos)
        R = np.where(use[t], P.R[t], 0)
        same = (np.sign(a) == np.sign(pos)) & (a != 0)
        held = np.where(same, held + 1, np.where(a != 0, 1, 0))
        entry = np.where(a != 0, np.where(same, (1 + entry) * (1 + np.sign(a) * R) - 1, np.sign(a) * R), 0)
        pos = a
        out[k] = a
    return te, out, (pos, held, entry)


def run(P, cfg, years=DEV_YEARS):
    cols = P.cols([f for g in cfg.groups for f in GROUPS[g]])
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    pos = np.zeros(P.R.shape, np.float32)
    N = P.R.shape[1]
    state = (np.zeros(N, np.float32), np.zeros(N, np.float32), np.zeros(N, np.float32))
    net, t0 = None, time.time()
    for k, year in enumerate(years):
        Xn = normalised(P, cols, year)
        rows = train_rows(P, year, 1)
        net = train(P, Xn, rows, cfg, net, cfg.iters_first if k == 0 else cfg.iters_next, rng)
        te, p, state = act_year(P, Xn, net, cfg, year, state)
        pos[te] = p
    m = evaluate(P, pos, years, name=f"ppo_{cfg.name}", config=cfg.__dict__ | {"family": "ppo"}, log=not cfg.name.startswith("planted"),
                 period="dev" if years == DEV_YEARS else "oos")
    print(line(f"PPO {cfg.name}", m), f"({time.time() - t0:.0f}s)", flush=True)
    print(line("  └ 別檔訊號 B2", evaluate(P, np.roll(pos, 1, axis=1), years)), flush=True)
    POS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(POS_DIR / f"ppo_{cfg.name}_{'dev' if years == DEV_YEARS else 'oos'}.npy", pos)
    return m, pos


if __name__ == "__main__":
    P = load_panel()
    pick = sys.argv[1:]
    for cfg in CONFIGS:
        if not pick or cfg.name in pick:
            run(P, cfg)
