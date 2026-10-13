"""Shared evaluation for the RL study: a vectorised version of stocklab.backtest (next-open fills, fees, tax, 融券 fee)
plus walk-forward splits and controls.

A position p[t, n] in [-1, 1] is chosen at bar t's close and earns p * R[t] (R = open[t+2] / open[t+1] - 1, see
panel.py); changing it costs the commission / tax of the traded fraction. Stock-days that can't trade (`ok` false)
carry the previous position. Every window starts flat, buy-and-hold included (it pays the buy fee on day one).

Development uses walk-forward test years 2013-2020 (models trained only on bars before each test year, purged by
the label horizon). 2021- is the out-of-sample period, evaluated once per frozen configuration.
"""

from __future__ import annotations

import json
import math

import numpy as np

from stocklab.trials import log_trial

from .panel import Panel

DEV_YEARS = tuple(range(2013, 2021))
OOS_YEARS = tuple(range(2021, 2027))
AGENT = "rl"


def costs(P, p, prev):
    L, Lp, S, Sp = np.maximum(p, 0), np.maximum(prev, 0), np.maximum(-p, 0), np.maximum(-prev, 0)
    return (P.buy * np.maximum(L - Lp, 0) + P.sell * np.maximum(Lp - L, 0)
            + (P.sell + P.short) * np.maximum(S - Sp, 0) + P.buy * np.maximum(Sp - S, 0))


def window(P, years):
    """Rows of the calendar in the given years and the per-stock usable mask there."""
    rows = np.flatnonzero(np.isin(P.year, years))
    return rows, P.warm[rows]


def net_returns(P, pos, rows, use):
    """Daily net returns [len(rows), N] of positions pos[rows] (NaN outside `use`), positions carried where unusable."""
    p = np.where(use, pos[rows], np.nan).astype(np.float64)
    # carry the last position through unusable bars; start flat
    for k in range(1, len(rows)):
        m = np.isnan(p[k])
        p[k, m] = p[k - 1, m]
    p = np.nan_to_num(p)
    prev = np.vstack([np.zeros((1, p.shape[1])), p[:-1]])
    r = p * P.R[rows] - costs(P, p, prev)
    listed = np.cumsum(use, axis=0) > 0
    return np.where(listed, r, np.nan), p


def _stats(r):
    r = r[np.isfinite(r)]
    if len(r) < 60:
        return None
    lw = np.log1p(r)
    eq = np.exp(np.cumsum(lw))
    sd = r.std()
    return {"cagr": float(math.exp(lw.sum() * 252 / len(r)) - 1), "sharpe": float(r.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0,
            "mdd": float((eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min()), "days": len(r)}


def evaluate(P, pos, years=DEV_YEARS, cols=None, name=None, config=None, log=False, period="dev"):
    """Median per-stock metrics, the equal-weight portfolio of the per-stock strategies, timing skill versus a
    constant position with the same average exposure (B1), and the excess over buy-and-hold."""
    rows, use = window(P, years)
    if cols is not None:
        mask = np.zeros(use.shape[1], bool)
        mask[cols] = True
        use = use & mask
    r, p = net_returns(P, pos, rows, use)
    bh, _ = net_returns(P, np.ones_like(pos), rows, use)
    live = np.isfinite(r)
    cnt = np.maximum(live.sum(axis=0), 1)
    expo = np.where(live, np.abs(p), 0).sum(axis=0) / cnt
    sign = np.where(np.where(live, p, 0).sum(axis=0) < 0, -1.0, 1.0)
    b1_all, _ = net_returns(P, np.broadcast_to(sign * expo, pos.shape), rows, use)
    dp = np.abs(np.diff(np.vstack([np.zeros((1, p.shape[1])), np.where(live, p, 0)]), axis=0))
    turn = dp.sum(axis=0) / cnt * 252
    per, per_bh, per_b1, exposure, turnover = [], [], [], [], []
    for n in range(r.shape[1]):
        s = _stats(r[:, n])
        if not s:
            continue
        per.append(s), per_bh.append(_stats(bh[:, n])), per_b1.append(_stats(b1_all[:, n]))
        exposure.append(float(expo[n])), turnover.append(float(turn[n]))
    med = lambda L, k: float(np.median([x[k] for x in L]))
    ew = lambda x: _stats(np.nanmean(x, axis=1)) if np.isfinite(x).any() else None
    diff = np.nanmean(r - bh, axis=1)
    diff = diff[np.isfinite(diff)]
    t_ex = float(diff.mean() / diff.std() * math.sqrt(len(diff))) if diff.std() > 0 else 0.0
    d1 = np.nanmean(r - b1_all, axis=1)
    d1 = d1[np.isfinite(d1)]
    t_b1 = float(d1.mean() / d1.std() * math.sqrt(len(d1))) if d1.std() > 0 else 0.0
    out = {
        "n": len(per), "cagr": med(per, "cagr"), "sharpe": med(per, "sharpe"), "mdd": med(per, "mdd"),
        "bh_cagr": med(per_bh, "cagr"), "bh_sharpe": med(per_bh, "sharpe"), "bh_mdd": med(per_bh, "mdd"),
        "b1_cagr": med(per_b1, "cagr"), "b1_sharpe": med(per_b1, "sharpe"),
        "exposure": float(np.median(exposure)), "turnover": float(np.median(turnover)),
        "win_sharpe_bh": float(np.mean([a["sharpe"] > b["sharpe"] for a, b in zip(per, per_bh)])),
        # a constant position has buy-and-hold's Sharpe, so timing skill versus B1 is judged on CAGR
        "win_cagr_b1": float(np.mean([a["cagr"] > b["cagr"] for a, b in zip(per, per_b1)])),
        "ew": ew(r), "ew_bh": ew(bh), "ew_b1": ew(b1_all), "t_excess_bh": t_ex, "t_timing_b1": t_b1,
    }
    if log and name:
        log_trial(AGENT, name, config or {}, {k: v for k, v in out.items() if not isinstance(v, dict)}
                  | {"ew_sharpe": out["ew"]["sharpe"], "ew_cagr": out["ew"]["cagr"]}, period=period)
    return out


def line(name, m):
    e, eb = m["ew"], m["ew_bh"]
    return (f"{name:<34} 中位 CAGR {m['cagr']:6.1%} Sh {m['sharpe']:5.2f} MDD {m['mdd']:6.1%} | 持股 {m['exposure']:4.0%} 換手 {m['turnover']:5.1f}"
            f" | 勝B&H {m['win_sharpe_bh']:4.0%} CAGR勝B1 {m['win_cagr_b1']:4.0%} | 等權 Sh {e['sharpe']:4.2f} (B&H {eb['sharpe']:4.2f}, B1 {m['ew_b1']['sharpe']:4.2f})"
            f" t(超額) {m['t_excess_bh']:5.2f} t(擇時) {m['t_timing_b1']:5.2f}")


def train_rows(P, year, horizon=1, start_year=2009):
    """Calendar rows usable for training a model applied in `year`: labels (horizon bars of R) end before Jan 1."""
    first = np.searchsorted(P.year, year)
    last = first - horizon - 2
    lo = np.searchsorted(P.year, start_year)
    return np.arange(lo, max(lo, last))


def forward(P, h):
    """Forward h-bar log return of the position return series: sum_{k<h} log(1 + R[t+k])."""
    lr = np.log1p(P.R.astype(np.float64)) * P.ok
    c = np.vstack([np.zeros((1, lr.shape[1])), np.cumsum(lr, axis=0)])
    T = lr.shape[0]
    out = np.full_like(lr, np.nan)
    out[: T - h + 1] = c[h:] - c[: T - h + 1]
    return out


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def load_panel():
    return Panel()
