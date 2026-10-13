"""Where is the bottleneck? Diagnostics run before (and independent of) any RL agent.

    python -m strategies.rl.bottleneck            # all diagnostics on the development years 2013-2020
    python -m strategies.rl.bottleneck ic         # only the information map (slowest part, ~3 min)

D1  ceiling: perfect foresight (DP over long/flat with fees), and perfect foresight limited to one decision per H bars
D2  required skill: a noisy oracle whose signal correlates rho with the forward H-bar return; the rho at which timing
    starts to beat buy-and-hold is the skill any learner (RL included) must reach
D3  achieved skill: walk-forward out-of-sample IC of supervised models per feature group and horizon
D4  where the return is earned: overnight gap vs. the tradable open-to-open return, and whether overnight
    information (US markets) is still usable at the next open
D5  statistical power: the information ratio needed to tell a real edge from luck in the available years
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

from .common import DEV_YEARS, evaluate, forward, load_panel, save, train_rows, window
from .panel import GROUPS

OUT = __import__("pathlib").Path(__file__).parent / "results" / "bottleneck.json"


def oracle_dp(P, years=DEV_YEARS):
    """Best long/flat path with perfect foresight and real fees (log-wealth DP per stock)."""
    rows, use = window(P, years)
    R = np.where(use, P.R[rows], 0.0).astype(np.float64)
    lr = np.log1p(R)
    b, s = np.log1p(-P.buy), np.log1p(-P.sell)
    v0, v1 = np.zeros(R.shape[1]), np.full(R.shape[1], -np.inf)
    arg = np.zeros(R.shape, bool)
    for k in range(len(rows)):
        n0 = np.maximum(v0, v1 + s)          # flat after: from flat, or sell
        n1 = np.maximum(v1, v0 + b) + lr[k]  # long after: from long, or buy
        arg[k] = (v0 + b > v1)
        v0, v1 = n0, n1
    best = np.maximum(v0, v1 + s)
    days = use.sum(axis=0)
    cagr = np.exp(best * 252 / np.maximum(days, 1)) - 1
    return float(np.median(cagr[days > 250]))


def oracle_blocks(P, h, years=DEV_YEARS, rho=1.0, keep=0.5, seeds=10):
    """Decide once per h bars (non-overlapping blocks). Signal = rho * z(block return) + noise; hold the block when
    the signal is in the top `keep` share of this stock's signals (keep=1 -> always long). Returns median metrics."""
    rows, use = window(P, years)
    fw = forward(P, h)[rows]
    T, N = fw.shape
    starts = np.arange(0, T, h)
    out = []
    for seed in range(seeds if rho < 1 else 1):
        rng = np.random.default_rng(seed)
        pos = np.zeros((len(P.R), N))
        y = fw[starts]
        z = (y - np.nanmean(y, axis=0)) / np.nanstd(y, axis=0)
        sig = rho * np.nan_to_num(z) + math.sqrt(max(0.0, 1 - rho ** 2)) * rng.standard_normal(z.shape)
        thr = np.nanquantile(np.where(np.isfinite(z), sig, np.nan), 1 - keep, axis=0)
        hold = (sig >= thr).astype(float)
        blk = np.repeat(hold, h, axis=0)[:T]
        pos[rows] = blk
        out.append(evaluate(P, pos, years))
    return {k: float(np.mean([m[k] for m in out])) for k in ("cagr", "sharpe", "mdd", "exposure", "bh_cagr", "bh_sharpe", "turnover")} \
        | {"ew_sharpe": float(np.mean([m["ew"]["sharpe"] for m in out])), "ew_bh_sharpe": out[0]["ew_bh"]["sharpe"]}


def required_skill(P, horizons=(1, 5, 20, 60), rhos=(0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5), keeps=(0.5, 0.8)):
    res = {}
    for h in horizons:
        for keep in keeps:
            curve = []
            for rho in rhos:
                m = oracle_blocks(P, h, rho=rho, keep=keep)
                curve.append({"rho": rho} | m)
                print(f"  H={h:>2} keep={keep:.0%} rho={rho:4.2f}: CAGR {m['cagr']:6.1%} Sharpe {m['sharpe']:5.2f} (B&H {m['bh_cagr']:6.1%} / {m['bh_sharpe']:4.2f})")
            res[f"h{h}_keep{int(keep * 100)}"] = curve
    return res


def _std(Xtr, Xte):
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    return np.clip((Xtr - mu) / sd, -5, 5), np.clip((Xte - mu) / sd, -5, 5)


def _ridge(Xtr, ytr, Xte, lam=10.0):
    A = Xtr.T @ Xtr + lam * len(Xtr) * 1e-3 * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ (ytr - ytr.mean()))
    return Xte @ w


def _ic(pred, y, rows_t, cols_n, T, N):
    """timing IC (per-stock correlation over time, averaged), pooled IC, cross-sectional rank IC (per day)."""
    Pm = np.full((T, N), np.nan)
    Ym = np.full((T, N), np.nan)
    Pm[rows_t, cols_n], Ym[rows_t, cols_n] = pred, y
    ts = []
    for n in range(N):
        m = np.isfinite(Pm[:, n]) & np.isfinite(Ym[:, n])
        if m.sum() > 200 and Pm[m, n].std() > 0:
            ts.append(np.corrcoef(Pm[m, n], Ym[m, n])[0, 1])
    xs = []
    for t in range(T):
        m = np.isfinite(Pm[t]) & np.isfinite(Ym[t])
        if m.sum() >= 10:
            a, b = Pm[t, m].argsort().argsort(), Ym[t, m].argsort().argsort()
            xs.append(np.corrcoef(a, b)[0, 1])
    return {"ts_ic": float(np.nanmean(ts)), "pooled_ic": float(np.corrcoef(pred, y)[0, 1]), "xs_ic": float(np.nanmean(xs))}


def info_map(P, horizons=(1, 5, 20, 60), groups=("tech", "xs", "mkt", "chips", "all"), models=("ridge", "gbm")):
    from sklearn.ensemble import HistGradientBoostingRegressor

    res = {}
    T, N = P.R.shape
    for h in horizons:
        y_all = forward(P, h)
        for g in groups:
            names = [f for k in GROUPS for f in GROUPS[k]] if g == "all" else GROUPS[g]
            cols = P.cols(names)
            for model in models:
                if model == "gbm" and g not in ("all", "tech"):
                    continue
                t0 = time.time()
                preds, ys, rt, cn = [], [], [], []
                for year in DEV_YEARS:
                    tr = train_rows(P, year, h)
                    te = np.flatnonzero(P.year == year)
                    mtr = P.warm[tr] & np.isfinite(y_all[tr])
                    mte = P.warm[te] & np.isfinite(y_all[te])
                    if g == "chips":
                        mtr &= P.is50[None, :] & (P.X[tr][:, :, P.fi["fnet20"]] != 0)
                        mte &= P.is50[None, :]
                    ti, ni = np.nonzero(mtr)
                    Xtr = P.X[tr[ti], ni][:, cols].astype(np.float64)
                    ytr = y_all[tr[ti], ni]
                    if model == "gbm":  # every 3rd day keeps the fit fast; overlapping labels make neighbours redundant
                        keep = (tr[ti] % 3) == 0
                        Xtr, ytr = Xtr[keep], ytr[keep]
                    tj, nj = np.nonzero(mte)
                    Xte = P.X[te[tj], nj][:, cols].astype(np.float64)
                    if model == "ridge":
                        a, b = _std(Xtr, Xte)
                        p = _ridge(a, ytr, b)
                    else:
                        gb = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                                                           l2_regularization=1.0, random_state=0)
                        p = gb.fit(Xtr, ytr).predict(Xte)
                    preds.append(p), ys.append(y_all[te[tj], nj]), rt.append(te[tj]), cn.append(nj)
                ic = _ic(np.concatenate(preds), np.concatenate(ys), np.concatenate(rt), np.concatenate(cn), T, N)
                res[f"h{h}_{g}_{model}"] = ic
                print(f"  H={h:>2} {g:<5} {model:<5}: timing IC {ic['ts_ic']:+.3f}  pooled {ic['pooled_ic']:+.3f}  cross-sectional {ic['xs_ic']:+.3f}  ({time.time() - t0:.0f}s)")
    return res


def overnight(P, years=DEV_YEARS):
    """Share of buy-and-hold return earned overnight (close -> next open) and whether US-session information that
    arrives overnight still predicts the tradable return after the next open."""
    rows, use = window(P, years)
    on = np.where(use, P.R_on[rows], np.nan)
    cc = np.where(use, P.R_cc[rows], np.nan)
    R = np.where(use, P.R[rows], np.nan)
    lon, lcc = np.nanmean(np.log1p(on)) * 252, np.nanmean(np.log1p(cc)) * 252
    out = {"overnight_log_ret_ann": float(lon), "close_close_log_ret_ann": float(lcc), "intraday_log_ret_ann": float(lcc - lon)}
    for f in ("sox1", "nas1", "tsm1", "m_r1", "r1", "gap"):
        x = P.X[rows][:, :, P.fi[f]]
        m = use & np.isfinite(on) & np.isfinite(R)
        out[f"corr_{f}_vs_gap_next"] = float(np.corrcoef(x[m], on[m])[0, 1])
        out[f"corr_{f}_vs_R"] = float(np.corrcoef(x[m], R[m])[0, 1])
    return out


def zero_cost_skill(P, rhos=(0.0, 0.02, 0.05, 0.1, 0.15)):
    """The same noisy oracle with every fee set to zero: how much of the required skill is there only to pay fees."""
    import copy

    Q = copy.copy(P)
    Q.buy, Q.sell, Q.short = P.buy * 0, P.sell * 0, P.short * 0
    res = {}
    for h in (1, 5, 20):
        res[h] = []
        for rho in rhos:
            m = oracle_blocks(Q, h, rho=rho, keep=0.8)
            res[h].append({"rho": rho} | m)
            print(f"  零成本 H={h:>2} rho={rho:4.2f}: CAGR {m['cagr']:6.1%} Sharpe {m['sharpe']:5.2f} (B&H {m['bh_cagr']:6.1%} / {m['bh_sharpe']:4.2f})")
    return res


def survivorship(P):
    """Buy-and-hold of today's 50 constituents vs. the 55 other stocks (former constituents, sector peers) vs. 0050."""
    from stocklab.etf import load_etf

    one = np.ones_like(P.R)
    out = {}
    for label, years in (("dev", DEV_YEARS), ("oos", tuple(range(2021, 2027)))):
        a = evaluate(P, one, years, cols=np.flatnonzero(P.is50))
        b = evaluate(P, one, years, cols=np.flatnonzero(~P.is50))
        df = load_etf("0050")
        o = df["open"].reindex(P.dates).ffill().to_numpy()
        R = np.r_[o[2:] / o[1:-1] - 1, 0, 0]
        rows = np.flatnonzero(np.isin(P.year, years))
        r = np.nan_to_num(R[rows])
        e = math.exp(np.log1p(r).sum() * 252 / len(r)) - 1
        out[label] = {"50": {"cagr": a["cagr"], "sharpe": a["sharpe"], "ew_cagr": a["ew"]["cagr"]},
                      "extra55": {"cagr": b["cagr"], "sharpe": b["sharpe"], "ew_cagr": b["ew"]["cagr"]},
                      "etf0050": {"cagr": e, "sharpe": float(r.mean() / r.std() * math.sqrt(252))}}
        print(f"  {label}: 今天的 50 檔 中位 CAGR {a['cagr']:.1%}（等權 {a['ew']['cagr']:.1%}）；其他 55 檔 {b['cagr']:.1%}（等權 {b['ew']['cagr']:.1%}）；0050 {e:.1%}")
    return out


def yearly(P):
    """Equal-weight buy-and-hold return per calendar year (which years are bear markets)."""
    out = {}
    for y in range(2009, 2027):
        rows, use = window(P, (y,))
        r = np.where(use, P.R[rows], np.nan)
        ew = np.nanmean(r, axis=1)
        out[y] = float(np.exp(np.log1p(np.nan_to_num(ew)).sum()) - 1)
    print("  等權買進持有逐年：" + "、".join(f"{y} {v:+.0%}" for y, v in out.items()))
    return out


def power(years=(8, 5.8, 18)):
    """Information ratio (annualised excess return / tracking error vs buy-and-hold) needed for t = 2."""
    return {f"{y}y": 2 / math.sqrt(y) for y in years}


def main(which=None):
    P = load_panel()
    res = {}
    if which in (None, "ceiling"):
        print("D1 完美預知上限（只做多、含費用）")
        res["oracle_dp_cagr"] = oracle_dp(P)
        print(f"  每天都知道明天：中位 CAGR {res['oracle_dp_cagr']:.0%}")
        res["oracle_blocks"] = {}
        for h in (1, 5, 20, 60, 120):
            m = oracle_blocks(P, h, keep=0.5)
            res["oracle_blocks"][h] = m
            print(f"  每 {h:>3} 天決定一次、完全知道這段漲跌（只持有較好的一半）：CAGR {m['cagr']:6.1%} Sharpe {m['sharpe']:4.2f}")
    if which in (None, "skill"):
        print("D2 需要多準：雜訊預言者")
        res["required"] = required_skill(P)
    if which in (None, "overnight"):
        print("D4 報酬在哪裡賺到")
        res["overnight"] = overnight(P)
        for k, v in res["overnight"].items():
            print(f"  {k}: {v:+.3f}")
    if which in (None, "ic"):
        print("D3 實際能預測多少（逐年前推、樣本外）")
        res["ic"] = info_map(P)
    if which in (None, "more"):
        print("D6 零成本時需要多準")
        res["zero_cost"] = zero_cost_skill(P)
        print("D7 倖存者偏差")
        res["survivorship"] = survivorship(P)
        res["yearly_ew"] = yearly(P)
    res["power_ir_needed"] = power()
    old = {}
    if OUT.exists():
        import json

        old = json.loads(OUT.read_text(encoding="utf-8"))
    save(OUT, old | res)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
