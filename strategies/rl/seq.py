"""Maybe the hand-made features are the bottleneck: let a 1-D convolutional network read the last 64 bars of raw
series (close-to-close, overnight gap, intraday move, volume ratio, market return) and learn its own features,
trained by direct reinforcement exactly like direct.py.

    python -m strategies.rl.seq
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .common import DEV_YEARS, evaluate, line, load_panel, train_rows
from .direct import POS_DIR, Cfg, _cost, _fees, _loss, normalised
from .panel import GROUPS

torch.set_num_threads(8)
LAG = 64


def channels(P):
    o, c = P.open.astype(np.float64), P.close.astype(np.float64)
    pc = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
    ch = [np.log(c / pc), np.log(o / pc), np.log(c / o), P.X[:, :, P.fi["vratio"]], P.X[:, :, P.fi["m_r1"]]]
    S = np.stack([np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) for x in ch], 0)  # [C, T, N]
    return S.astype(np.float32)


class SeqNet(torch.nn.Module):
    def __init__(self, c, extra=0):
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv1d(c, 16, 5), torch.nn.GELU(), torch.nn.Conv1d(16, 16, 5, dilation=2), torch.nn.GELU(),
                                        torch.nn.Conv1d(16, 16, 5, dilation=4), torch.nn.GELU(), torch.nn.AdaptiveAvgPool1d(1))
        self.head = torch.nn.Sequential(torch.nn.Linear(16 + extra, 16), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(16, 1))

    def forward(self, seq, x=None):  # seq [B, C, LAG]
        z = self.conv(seq).squeeze(-1)
        if x is not None:
            z = torch.cat([z, x], -1)
        return self.head(z).squeeze(-1)


def windows(S, rows):
    """[len(rows), N, C, LAG] lagged windows ending at each row."""
    lo = rows[0] - LAG + 1
    seg = torch.from_numpy(S[:, max(lo, 0) : rows[-1] + 1])  # [C, L+LAG-1, N]
    if lo < 0:
        seg = torch.nn.functional.pad(seg, (0, 0, -lo, 0))
    w = seg.unfold(1, LAG, 1)  # [C, L, N, LAG]
    return w.permute(1, 2, 0, 3)


def run(P, cfg, use_feats=False, years=DEV_YEARS):
    S0 = channels(P)
    use = P.warm & P.ok
    cols = P.cols([f for g in ("tech", "xs", "chips", "mkt") for f in GROUPS[g]])
    pos = np.zeros(P.R.shape, np.float32)
    buy, sell, sshort = _fees(P, cfg.cost_mult)
    t0 = time.time()
    for year in years:
        tr = train_rows(P, year, 1)
        m = use[tr]
        sd = np.array([S0[c][tr][m].std() + 1e-6 for c in range(len(S0))], np.float32)
        S = np.clip(S0 / sd[:, None, None], -6, 6).astype(np.float32)
        Xn = normalised(P, cols, year) if use_feats else None
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        net = SeqNet(len(S), len(cols) if use_feats else 0)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        fit, val = tr[:-250], tr[-250:]

        def loss_on(rows, train):
            net.train(train)
            w = windows(S, rows)
            L, N = w.shape[:2]
            x = torch.from_numpy(Xn[rows]).reshape(L * N, -1) if use_feats else None
            p = torch.sigmoid(net(w.reshape(L * N, len(S), LAG), x)).reshape(L, N)
            mask = torch.from_numpy(use[rows])
            p = torch.where(mask, p, torch.zeros_like(p))
            ret = p * torch.from_numpy(P.R[rows]) - _cost(p, buy, sell, sshort)
            return _loss(ret, mask, p, cfg)

        best, state, bad = np.inf, None, 0
        for step in range(cfg.steps):
            s = rng.integers(LAG, len(fit) - 128)
            loss = loss_on(fit[s : s + 128], True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 25 == 24:
                with torch.no_grad():
                    v = float(loss_on(val, False))
                if v < best - 1e-4:
                    best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= 6:
                        break
        if state:
            net.load_state_dict(state)
        te = np.flatnonzero(P.year == year)
        net.eval()
        with torch.no_grad():
            for a in range(0, len(te), 64):
                rows = te[a : a + 64]
                w = windows(S, rows)
                L, N = w.shape[:2]
                x = torch.from_numpy(Xn[rows]).reshape(L * N, -1) if use_feats else None
                pos[rows] = torch.sigmoid(net(w.reshape(L * N, len(S), LAG), x)).reshape(L, N).numpy()
    name = f"seq_{cfg.name}{'_feats' if use_feats else ''}"
    m = evaluate(P, pos, years, name=name, config=cfg.__dict__ | {"family": "seq", "use_feats": use_feats}, log=True)
    print(line(name, m), f"({time.time() - t0:.0f}s)", flush=True)
    print(line("  └ 別檔訊號 B2", evaluate(P, np.roll(pos, 1, axis=1), years)), flush=True)
    np.save(POS_DIR / f"{name}_dev.npy", pos)


if __name__ == "__main__":
    P = load_panel()
    run(P, Cfg("log"))
    run(P, Cfg("budget50", budget=0.5))
    run(P, Cfg("budget50", budget=0.5), use_feats=True)
