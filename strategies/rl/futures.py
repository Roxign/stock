"""The bottleneck analysis says the daily market signal exists (out-of-sample timing IC about 0.05) but stock fees and
tax are about 30x what it can pay for. Index futures (台指期 / 小台 / 微台) cost roughly 0.01% per side all-in
(futures transaction tax 0.002%, commission, one tick of slippage), so test the same signal on the TAIEX traded with
futures-like costs versus ETF-like costs.

Convention: decided at the cash close of day t, filled at the open of t+1, held to the open of t+2. The traded
return is 0050's own open-to-open return, a stand-in for the index future (the future opens at 08:45, 0050 in the
09:00 opening auction). NOT the TAIEX "open": the index's first value of the day still uses the previous close for
stocks that have not traded yet, so it lags overnight news. Measured on TAIEX opens the same model shows a daily IC of
0.21 after 2021; on 0050's tradable opens it is 0.07. Any backtest filled at TAIEX opens is therefore invalid.

    python -m strategies.rl.futures
"""

from __future__ import annotations

import math

import numpy as np

from stocklab import external as ext
from stocklab.etf import load_etf

from stocklab.trials import log_trial

from .common import AGENT, DEV_YEARS, OOS_YEARS, load_panel, save, train_rows
from .panel import GROUPS, MKT_FEATS, STOCK_FEATS, _stock_frame

OUT = __import__("pathlib").Path(__file__).parent / "results" / "futures.json"
COSTS = {"期貨級成本 0.01%/邊": (0.0001, 0.0001), "ETF 級成本": (0.001425, 0.002425)}


def data(P):
    df = ext.load_ohlcv("taiex").reindex(P.dates)
    f = _stock_frame(df)
    X = np.column_stack([f[n].to_numpy() for n in STOCK_FEATS] + [P.X[:, 0, P.fi[n]] for n in MKT_FEATS]).astype(np.float64)
    o = load_etf("0050")["open"].reindex(P.dates).ffill().to_numpy()
    R = np.r_[o[2:] / o[1:-1] - 1, np.nan, np.nan]
    return X, R


def predict(P, X, R, h, years):
    lr = np.log1p(np.nan_to_num(R))
    c = np.r_[0, np.cumsum(lr)]
    y = np.full(len(R), np.nan)
    y[: len(R) - h + 1] = c[h:] - c[: len(R) - h + 1]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    pred = np.full(len(R), np.nan)
    for year in years:
        tr = train_rows(P, year, h)
        tr = tr[ok[tr] & (P.year[tr] >= 2009)]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z = np.column_stack([np.clip((X[tr] - mu) / sd, -5, 5), np.ones(len(tr))])
        w = np.linalg.solve(Z.T @ Z + np.diag(np.r_[np.ones(X.shape[1]), 0]) * len(tr) * 1e-2, Z.T @ y[tr])
        te = np.flatnonzero(P.year == year)
        Xt = np.nan_to_num(X[te])
        pred[te] = np.clip((Xt - mu) / sd, -5, 5) @ w[:-1] + w[-1]
    return pred


def run_policy(pos, R, rows, buy, sell):
    p = pos[rows]
    prev = np.r_[0.0, p[:-1]]
    L, Lp, S, Sp = np.maximum(p, 0), np.maximum(prev, 0), np.maximum(-p, 0), np.maximum(-prev, 0)
    cost = buy * (np.maximum(L - Lp, 0) + np.maximum(Sp - S, 0)) + sell * (np.maximum(Lp - L, 0) + np.maximum(S - Sp, 0))
    return p * np.nan_to_num(R[rows]) - cost


def stats(x):
    lw = np.log1p(x)
    eq = np.exp(np.cumsum(lw))
    return {"cagr": math.exp(lw.mean() * 252) - 1, "sharpe": float(x.mean() / x.std() * math.sqrt(252)),
            "mdd": float((eq / np.maximum.accumulate(np.r_[1, eq])[1:] - 1).min())}


def main():
    P = load_panel()
    X, R = data(P)
    res = {}
    for h in (1, 5):
        pred = predict(P, X, R, h, DEV_YEARS + OOS_YEARS)
        sgn = np.sign(np.nan_to_num(pred))
        pols = {"多空（依預測正負）": sgn, "做多或空手": (sgn > 0).astype(float), "加減碼 0.5 / 1.5 倍": 1 + 0.5 * sgn}
        for per, yrs in (("dev", DEV_YEARS), ("oos", OOS_YEARS)):
            rows = np.flatnonzero(np.isin(P.year, yrs))
            m = np.isfinite(pred[rows]) & np.isfinite(R[rows])
            ic = float(np.corrcoef(pred[rows][m], R[rows][m])[0, 1]) if h == 1 else float("nan")
            hit = float(np.mean(np.sign(pred[rows][m]) == np.sign(R[rows][m])))
            bh = stats(run_policy(np.ones(len(R)), R, rows, 0.0001, 0.0001))
            print(f"[{per}] 預測 {h} 天：日 IC {ic:+.3f}，方向命中率 {hit:.1%}；0050 買進持有 {bh['cagr']:+.1%}/{bh['sharpe']:.2f}/MDD {bh['mdd']:.0%}", flush=True)
            for cname, (b, s) in COSTS.items():
                for pname, pos in pols.items():
                    st = stats(run_policy(pos, R, rows, b, s))
                    turn = float(np.abs(np.diff(pos[rows])).sum() / len(rows) * 252)
                    res[f"h{h}_{per}_{cname}_{pname}"] = st | {"turnover": turn, "ic": ic, "hit": hit, "bh": bh}
                    log_trial(AGENT, f"futures_h{h}_{cname}_{pname}", {"family": "futures", "h": h, "cost": cname, "policy": pname},
                              {"sharpe": st["sharpe"], "cagr": st["cagr"], "mdd": st["mdd"], "bh_sharpe": bh["sharpe"]}, period=per)
                    print(f"   {cname:<14} {pname:<14} {st['cagr']:+6.1%} / Sharpe {st['sharpe']:5.2f} / MDD {st['mdd']:5.0%}  年換手 {turn:4.0f} 倍", flush=True)
    save(OUT, res)


if __name__ == "__main__":
    main()
