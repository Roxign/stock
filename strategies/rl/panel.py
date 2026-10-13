"""Daily panel for the reinforcement-learning study: every stock on one TWSE calendar, features causal.

Convention (same as the backtester): the position chosen with information up to bar t's close is filled at bar t+1's
open, so it earns R[t] = open[t+2] / open[t+1] - 1 (each stock's own next two bars; a halt is spanned by the bar
before it). Features at t only use data published before the t+1 open (stocklab.external's `avail` rule).

    python -m strategies.rl.panel        # builds data/rl/panel.npz (about 30 s)

Arrays are [T, N] or [T, N, F] over the calendar `dates` and the stocks `codes`; `ok` marks bars where the stock
trades at t, t+1 and t+2 (only those can carry a position decision).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stocklab import external as ext
from stocklab.data import ROOT, load
from stocklab.indicators import macd, sma, tw_kdj
from stocklab.shortrules import fees
from stocklab.universe import CODES, EXTRA_CODES

OUT = ROOT / "data" / "rl" / "panel.npz"
START = "2008-01-01"

STOCK_FEATS = [
    "k", "d", "j", "k_d", "dk", "dif", "sig", "osc", "dosc",
    "ma5", "ma20", "ma60", "ma240", "r1", "r5", "r20", "r60", "r120", "r250",
    "vol20", "vol60", "pos60", "pos250", "dd250", "vratio", "sat_hi", "sat_lo", "gap",
]
XS_FEATS = ["xs_r5", "xs_r20", "xs_r60", "xs_r250", "xs_vol60", "xs_ma60"]
CHIP_FEATS = ["has_chips", "fnet5", "fnet20", "tnet20", "fratio20", "margin20", "pbr_pct", "dy", "rev_yoy"]
MKT_FEATS = [
    "m_r1", "m_r5", "m_r20", "m_r60", "m_vol20", "m_ma60", "m_ma240", "m_dd250", "breadth20", "breadth60",
    "sox1", "sox5", "nas1", "tsm1", "tsm5", "vix", "vix5", "us10y20", "twd20", "fnet_mkt5", "margin_mkt20", "pc_oi",
]
FEATS = STOCK_FEATS + XS_FEATS + CHIP_FEATS + MKT_FEATS
GROUPS = {"tech": STOCK_FEATS, "xs": XS_FEATS, "chips": CHIP_FEATS, "mkt": MKT_FEATS}


def _stock_frame(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    kd, m = tw_kdj(df), macd(c)
    lr = np.log(c).diff()
    out = pd.DataFrame(index=df.index)
    out["k"], out["d"], out["j"] = kd["k"] / 100, kd["d"] / 100, kd["j"] / 100
    out["k_d"] = (kd["k"] - kd["d"]) / 100
    out["dk"] = kd["k"].diff() / 100
    out["dif"], out["sig"], out["osc"] = m["dif"] / c * 100, m["macd"] / c * 100, m["osc"] / c * 100
    out["dosc"] = m["osc"].diff() / c * 100
    for n in (5, 20, 60, 240):
        out[f"ma{n}"] = c / sma(c, n) - 1
    for n in (1, 5, 20, 60, 120, 250):
        out[f"r{n}"] = np.log(c / c.shift(n))
    out["vol20"], out["vol60"] = lr.rolling(20).std(), lr.rolling(60).std()
    for n in (60, 250):
        hi, lo = c.rolling(n).max(), c.rolling(n).min()
        out[f"pos{n}"] = (c - lo) / (hi - lo).replace(0, np.nan)
    out["dd250"] = c / c.rolling(250, min_periods=1).max() - 1
    out["vratio"] = np.log((v + 1) / (v.rolling(20).mean() + 1))
    hi_run = (kd["k"] > 80).astype(int)
    lo_run = (kd["k"] < 20).astype(int)
    out["sat_hi"] = hi_run.groupby((hi_run == 0).cumsum()).cumsum() / 10
    out["sat_lo"] = lo_run.groupby((lo_run == 0).cumsum()).cumsum() / 10
    out["gap"] = o / c.shift() - 1
    # the trade return of a decision at t: in at t+1's open, out at t+2's open (this stock's own bars)
    out["R"] = o.shift(-2) / o.shift(-1) - 1
    # overnight and intraday pieces of R, for the diagnostics
    out["R_on"] = o.shift(-1) / c - 1
    out["R_cc"] = c.shift(-1) / c - 1
    return out


def _chips(code: str, index, end) -> pd.DataFrame:
    f = ext.load_stock_fields(code, ["foreign_net", "trust_net", "foreign_ratio", "margin_balance", "shares_issued",
                                     "pbr", "dividend_yield", "revenue_yoy"], index=index, end=end)
    vol = load(code)["volume"].reindex(index)
    out = pd.DataFrame(index=index)
    fnet, tnet = f["foreign_net"].fillna(0), f["trust_net"].fillna(0)
    out["has_chips"] = 1.0
    out["fnet5"] = fnet.rolling(5).sum() / vol.rolling(5).sum()
    out["fnet20"] = fnet.rolling(20).sum() / vol.rolling(20).sum()
    out["tnet20"] = tnet.rolling(20).sum() / vol.rolling(20).sum()
    out["fratio20"] = f["foreign_ratio"].diff(20) / 100
    out["margin20"] = f["margin_balance"].diff(20) / f["shares_issued"]
    out["pbr_pct"] = f["pbr"].rolling(500, min_periods=120).rank(pct=True)
    out["dy"] = f["dividend_yield"] / 100
    out["rev_yoy"] = f["revenue_yoy"].clip(-1, 3)
    return out


def _market(cal, end, close) -> pd.DataFrame:
    m = ext.load_market(["taiex_tr", "sox", "nasdaq", "tsm", "vix", "us10y", "usdtwd", "mkt_foreign_net",
                         "taiex_turnover", "mkt_margin_value", "pc_ratio_oi"], index=cal, end=end)
    tr = m["taiex_tr"].ffill()
    lr = np.log(tr).diff()
    out = pd.DataFrame(index=cal)
    for n in (1, 5, 20, 60):
        out[f"m_r{n}"] = np.log(tr / tr.shift(n))
    out["m_vol20"] = lr.rolling(20).std()
    out["m_ma60"], out["m_ma240"] = tr / tr.rolling(60).mean() - 1, tr / tr.rolling(240).mean() - 1
    out["m_dd250"] = tr / tr.rolling(250, min_periods=1).max() - 1
    out["breadth20"] = (close > close.rolling(20, min_periods=20).mean()).where(close.notna()).mean(axis=1)
    out["breadth60"] = (close > close.rolling(60, min_periods=60).mean()).where(close.notna()).mean(axis=1)
    us = lambda k, n: np.log(m[k].ffill() / m[k].ffill().shift(n))
    out["sox1"], out["sox5"], out["nas1"] = us("sox", 1), us("sox", 5), us("nasdaq", 1)
    out["tsm1"], out["tsm5"] = us("tsm", 1), us("tsm", 5)
    out["vix"] = m["vix"].ffill() / 100
    out["vix5"] = m["vix"].ffill().diff(5) / 100
    out["us10y20"] = m["us10y"].ffill().diff(20) / 10
    out["twd20"] = us("usdtwd", 20)
    out["fnet_mkt5"] = m["mkt_foreign_net"].fillna(0).rolling(5).sum() / m["taiex_turnover"].fillna(0).rolling(5).sum()
    out["margin_mkt20"] = np.log(m["mkt_margin_value"].ffill() / m["mkt_margin_value"].ffill().shift(20))
    out["pc_oi"] = m["pc_ratio_oi"].ffill() / 100 - 1
    return out


def build(verbose=True):
    codes = list(CODES) + list(EXTRA_CODES)
    raw = {c: load(c) for c in codes}
    end = max(df.index[-1] for df in raw.values())
    cal = ext.calendar(end)
    cal = cal[cal >= pd.Timestamp(START)]
    frames = {}
    for c, df in raw.items():
        f = _stock_frame(df)
        if c in CODES:
            f = f.join(_chips(c, df.index, end))
        frames[c] = f.reindex(cal)
    T, N = len(cal), len(codes)
    get = lambda col: pd.DataFrame({c: frames[c][col] for c in codes}, index=cal)
    close = pd.DataFrame({c: raw[c]["close"].reindex(cal) for c in codes})
    xs = {}
    for name, col in (("xs_r5", "r5"), ("xs_r20", "r20"), ("xs_r60", "r60"), ("xs_r250", "r250"), ("xs_vol60", "vol60"), ("xs_ma60", "ma60")):
        xs[name] = get(col).rank(axis=1, pct=True)
    mkt = _market(cal, end, close)
    X = np.zeros((T, N, len(FEATS)), dtype=np.float32)
    for j, name in enumerate(FEATS):
        if name in STOCK_FEATS:
            a = get(name).to_numpy()
        elif name in XS_FEATS:
            a = xs[name].to_numpy()
        elif name in CHIP_FEATS:
            a = np.column_stack([frames[c][name].to_numpy() if name in frames[c] else np.full(T, np.nan) for c in codes])
        else:
            a = np.repeat(mkt[name].to_numpy()[:, None], N, axis=1)
        X[:, :, j] = a
    R = get("R").to_numpy()
    valid = close.notna().to_numpy()
    ok = valid & np.isfinite(R)
    nan_feat = ~np.isfinite(X)
    X[nan_feat] = 0.0
    # a stock-day can be used once its slow features exist (about a year of history)
    warm = np.isfinite(get("r250").to_numpy()) & np.isfinite(get("ma240").to_numpy())
    f = [fees(c) for c in codes]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X=X, R=np.nan_to_num(R).astype(np.float32), R_on=np.nan_to_num(get("R_on").to_numpy()).astype(np.float32),
                        R_cc=np.nan_to_num(get("R_cc").to_numpy()).astype(np.float32), ok=ok, warm=warm & ok,
                        open=raw_frame(raw, cal, "open"), close=close.to_numpy().astype(np.float32),
                        dates=np.array(cal.strftime("%Y-%m-%d"), dtype="U10"), codes=np.array(codes), feats=np.array(FEATS),
                        is50=np.array([c in CODES for c in codes]), buy=np.array([x["buy"] for x in f]),
                        sell=np.array([x["sell"] for x in f]), short=np.array([x["short"] for x in f]))
    if verbose:
        print(f"panel {T} days x {N} stocks x {len(FEATS)} features -> {OUT}  ({cal[0].date()} .. {cal[-1].date()})")
        miss = nan_feat[warm & ok].mean(axis=0)
        print("missing share after warm-up:", ", ".join(f"{n} {m:.0%}" for n, m in zip(FEATS, miss) if m > 0.02))


def raw_frame(raw, cal, col):
    return pd.DataFrame({c: df[col].reindex(cal) for c, df in raw.items()}).to_numpy().astype(np.float32)


class Panel:
    """Loaded panel with convenience views."""

    def __init__(self, path=OUT):
        z = np.load(path, allow_pickle=False)
        for k in z.files:
            setattr(self, k, z[k])
        self.dates = pd.DatetimeIndex(self.dates)
        self.year = self.dates.year.to_numpy()
        self.fi = {n: i for i, n in enumerate(self.feats)}

    def cols(self, names):
        return [self.fi[n] for n in names]


if __name__ == "__main__":
    build()
