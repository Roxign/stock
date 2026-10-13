"""Market-level timing, the one place where the information map found an IC near the required level, with the two
structural limits removed one at a time:

  1. too few independent episodes: Taiwan alone has a handful of bull/bear cycles since 2008, so the model is also
     trained on six other markets (S&P 500, Nasdaq, Nikkei, KOSPI, Hang Seng, Shanghai), pooled, walk-forward;
  2. the 100% cap: a signal that says "better than usual" cannot be used when the position is already full, so the
     exposure may go up to 150% (futures / margin), paying financing on the borrowed part.

Every index uses the same convention: decided at the close of day t, earning close t+1 -> close t+2 (one day of
delay, a conservative stand-in for next-open fills on the Taipei calendar), ETF-like fees. The final check trades
0050 itself at the next open.

    python -m strategies.rl.market
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from stocklab import external as ext
from stocklab.etf import load_etf
from stocklab.indicators import macd, tw_kdj

from stocklab.trials import log_trial

from .common import AGENT, DEV_YEARS, OOS_YEARS, save

OUT = __import__("pathlib").Path(__file__).parent / "results" / "market.json"
MARKETS = ["taiex", "spx", "nasdaq", "n225", "kospi", "hsi", "sse"]
LABEL = {"taiex": "台股加權", "spx": "S&P 500", "nasdaq": "那斯達克", "n225": "日經", "kospi": "韓國", "hsi": "恒生", "sse": "上證"}
BUY, SELL = 0.001425, 0.002425
FEATS = ["r5", "r20", "r60", "r120", "r250", "vol20", "vol60", "vratio", "ma20", "ma60", "ma240", "dd250", "pos250", "k", "d",
         "dif", "osc", "g_breadth", "vix", "vix20", "us10y20", "dxy20"]


def features(df, glob):
    c = df["close"]
    lr = np.log(c).diff()
    kd, m = tw_kdj(df), macd(c)
    f = pd.DataFrame(index=df.index)
    for n in (5, 20, 60, 120, 250):
        f[f"r{n}"] = np.log(c / c.shift(n))
    f["vol20"], f["vol60"] = lr.rolling(20).std() * 16, lr.rolling(60).std() * 16
    f["vratio"] = f["vol20"] / f["vol60"]
    for n in (20, 60, 240):
        f[f"ma{n}"] = c / c.rolling(n).mean() - 1
    f["dd250"] = c / c.rolling(250, min_periods=1).max() - 1
    hi, lo = c.rolling(250).max(), c.rolling(250).min()
    f["pos250"] = (c - lo) / (hi - lo)
    f["k"], f["d"] = kd["k"] / 100, kd["d"] / 100
    f["dif"], f["osc"] = m["dif"] / c * 100, m["osc"] / c * 100
    return f.join(glob)


def load():
    cal = ext.calendar()
    data = {k: ext.load_ohlcv(k).reindex(cal).ffill() for k in MARKETS}
    g = ext.load_market(["vix", "us10y", "dxy"], index=cal).ffill()
    above = pd.DataFrame({k: d["close"] > d["close"].rolling(200).mean() for k, d in data.items()}).mean(axis=1)
    glob = pd.DataFrame({"g_breadth": above, "vix": g["vix"] / 100, "vix20": g["vix"].diff(20) / 100,
                         "us10y20": g["us10y"].diff(20) / 10, "dxy20": np.log(g["dxy"] / g["dxy"].shift(20))}, index=cal)
    F = {k: features(d, glob) for k, d in data.items()}
    C = pd.DataFrame({k: d["close"] for k, d in data.items()})
    return cal, F, C


def fit_predict(cal, F, C, h, train_on, years):
    """Walk-forward ridge on the forward h-day return (close t+1 -> close t+1+h); z-scored prediction per market."""
    lc = np.log(C)
    y = (lc.shift(-1 - h) - lc.shift(-1))
    Z = {k: pd.Series(np.nan, index=cal) for k in MARKETS}
    for year in years:
        start = pd.Timestamp(f"{year}-01-01")
        cut = cal[cal < start][-(h + 2)]
        Xs, ys = [], []
        for k in train_on:
            X = F[k].loc[(F[k].index >= "2009-06-01") & (F[k].index <= cut)]
            yy = y[k].loc[X.index]
            ok = X.notna().all(axis=1) & yy.notna()
            Xs.append(X[ok].to_numpy()), ys.append(yy[ok].to_numpy())
        X, yv = np.vstack(Xs), np.concatenate(ys)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xa = np.column_stack([np.clip((X - mu) / sd, -5, 5), np.ones(len(X))])
        w = np.linalg.solve(Xa.T @ Xa + np.diag(np.r_[np.ones(X.shape[1]), 0]) * len(X) * 1e-2, Xa.T @ yv)
        ptr = Xa @ w
        pm, ps = ptr.mean(), ptr.std() + 1e-12
        for k in MARKETS:
            idx = F[k].index[(F[k].index.year == year)]
            Xt = F[k].loc[idx].to_numpy()
            p = np.clip((Xt - mu) / sd, -5, 5) @ w[:-1] + w[-1]
            Z[k].loc[idx] = (p - pm) / ps
    return pd.DataFrame(Z)


def policy(z, k, lmax, band=0.1, smooth=5):
    """Exposure = clip(1 + k * smoothed z, 0, lmax), traded only when it moves by more than `band`."""
    zs = z.ewm(span=smooth, adjust=False).mean().to_numpy()
    out = np.zeros(len(zs))
    cur = 0.0
    for i, v in enumerate(zs):
        if np.isfinite(v):
            want = float(np.clip(1 + k * v, 0, lmax))
            if abs(want - cur) > band:
                cur = want
        out[i] = cur
    return out


def backtest(r, pos, fin):
    """Daily net returns: pos[t] earns r[t] (already one bar ahead), fees on changes, financing on exposure > 1."""
    prev = np.r_[0.0, pos[:-1]]
    d = pos - prev
    cost = BUY * np.maximum(d, 0) + SELL * np.maximum(-d, 0)
    return pos * r - cost - np.maximum(pos - 1, 0) * fin / 252


def stats(x):
    x = x[np.isfinite(x)]
    lw = np.log1p(x)
    eq = np.exp(np.cumsum(lw))
    return {"cagr": math.exp(lw.mean() * 252) - 1, "sharpe": x.mean() / x.std() * math.sqrt(252) if x.std() > 0 else 0.0,
            "mdd": float((eq / np.maximum.accumulate(np.r_[1, eq])[1:] - 1).min())}


def main():
    cal, F, C = load()
    R = C.shift(-2) / C.shift(-1) - 1   # decision at t earns close t+1 -> close t+2
    res = {}
    years_all = DEV_YEARS + OOS_YEARS
    for h in (20, 60):
        for train_on, tl in ((MARKETS, "7 個市場一起學"), (["taiex"], "只用台股")):
            Z = fit_predict(cal, F, C, h, train_on, years_all)
            for lmax in (1.0, 1.5):
                for fin in ((0.03, 0.065) if lmax > 1 else (0.0,)):
                    key = f"h{h}_{'pool' if len(train_on) > 1 else 'tw'}_L{lmax}_fin{fin}"
                    res[key] = {}
                    for per, yrs in (("dev", DEV_YEARS), ("oos", OOS_YEARS)):
                        rows = np.isin(cal.year, yrs)
                        line = []
                        for mk in MARKETS:
                            pos = policy(Z[mk], 0.5, lmax)
                            r = np.nan_to_num(R[mk].to_numpy())
                            net = backtest(r, pos, fin)[rows]
                            bh = backtest(r, np.ones(len(r)), 0)[rows]
                            e = pos[rows].mean()
                            b1 = backtest(r, np.full(len(r), e), fin)[rows]
                            s, sb, s1 = stats(net), stats(bh), stats(b1)
                            res[key].setdefault(per, {})[mk] = {"strat": s, "bh": sb, "b1": s1, "exposure": float(e)}
                            line.append(f"{LABEL[mk]} {s['cagr']:+.1%}/{s['sharpe']:.2f}（B&H {sb['cagr']:+.1%}/{sb['sharpe']:.2f}，同平均曝險 {s1['cagr']:+.1%}）")
                        print(f"[{per}] 預測 {h} 天・{tl}・最高曝險 {lmax:.0%}・融資成本 {fin:.1%}\n   " + "\n   ".join(line), flush=True)
            # 0050 itself at the next open, from the Taiwan-index signal
            df = load_etf("0050")
            o = df["open"].reindex(cal).ffill().to_numpy()
            r50 = np.nan_to_num(np.r_[o[2:] / o[1:-1] - 1, 0, 0])
            for lmax in (1.0, 1.5):
                pos = policy(Z["taiex"], 0.5, lmax)
                for per, yrs in (("dev", DEV_YEARS), ("oos", OOS_YEARS)):
                    rows = np.isin(cal.year, yrs)
                    s = stats(backtest(r50, pos, 0.03)[rows])
                    sb = stats(backtest(r50, np.ones(len(r50)), 0)[rows])
                    e = pos[rows].mean()
                    s1 = stats(backtest(r50, np.full(len(r50), e), 0.03)[rows])
                    name = f"0050_h{h}_{'pool' if len(train_on) > 1 else 'tw'}_L{lmax}_{per}"
                    res[name] = {"strat": s, "bh": sb, "b1": s1, "exposure": float(e)}
                    log_trial(AGENT, f"market_{name}", {"family": "market", "h": h, "pool": len(train_on) > 1, "lmax": lmax, "fin": 0.03},
                              {"sharpe": s["sharpe"], "cagr": s["cagr"], "mdd": s["mdd"], "exposure": float(e), "bh_sharpe": sb["sharpe"]}, period=per)
                    print(f"   0050 隔天開盤成交 [{per}] 最高 {lmax:.0%}：{s['cagr']:+.1%}/{s['sharpe']:.2f}/MDD {s['mdd']:.0%}"
                          f"（B&H {sb['cagr']:+.1%}/{sb['sharpe']:.2f}/MDD {sb['mdd']:.0%}，同平均曝險 {e:.0%} {s1['cagr']:+.1%}/{s1['sharpe']:.2f}）", flush=True)
    save(OUT, res)


if __name__ == "__main__":
    main()
