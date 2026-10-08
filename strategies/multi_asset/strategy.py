"""Multi-asset allocation across the 0050 ETF, gold (00635U), US Treasuries 20Y (00679B) and oil (00642U).

research/diagnosis.md: timing a single stock against cash loses mostly by holding less, and idle cash earns nothing.
These portfolios keep the money working in whichever assets are trending, and hold the investable 0050 instead of
the survivorship-biased 50 stocks. Decisions at each month's last close, filled at the next open; an ETF joins once
it is listed. weights() feeds the single-account portfolio scoreboard; positions() marks when each ETF is held.
"""

import numpy as np
import pandas as pd

from strategies.trend.strategy import trend_state

ASSETS = ("0050", "00635U", "00679B", "00642U")
VOL_N = 60


def _panel(data):
    codes = [c for c in ASSETS if c in data]
    close = pd.DataFrame({c: data[c]["close"] for c in codes}).sort_index()
    return codes, close


def _month_end(idx):
    """Last bar of each month (the final bar of the data counts too)."""
    m = idx.year.to_numpy() * 12 + idx.month.to_numpy()
    return pd.Series(np.r_[m[1:] != m[:-1], True], index=idx)


def _monthly(weights_at, idx):
    """Weights decided on month-end bars, held until the next decision."""
    w = weights_at.copy()
    w.loc[~_month_end(idx).to_numpy()] = np.nan
    return w.ffill().fillna(0.0)


def _inv_vol(close):
    return 1.0 / (np.log(close).diff().rolling(VOL_N).std() * np.sqrt(252))


def _trend(close):
    return pd.DataFrame({c: trend_state(close[c].dropna()).reindex(close.index) for c in close.columns})


def risk_parity(data):
    codes, close = _panel(data)
    iv = _inv_vol(close)
    w = iv.div(iv.sum(axis=1), axis=0)
    return _monthly(w, close.index)


def trend_ivol(data):
    codes, close = _panel(data)
    iv = _inv_vol(close) * (_trend(close) == 1)
    tot = iv.sum(axis=1)
    w = iv.div(tot.where(tot > 0), axis=0).fillna(0.0)
    return _monthly(w, close.index)


def riskoff_switch(data):
    codes, close = _panel(data)
    tr = _trend(close)
    w = pd.DataFrame(0.0, index=close.index, columns=codes)
    risk_on = tr["0050"] == 1
    w.loc[risk_on, "0050"] = 1.0
    for c in ("00679B", "00635U"):
        if c in codes:
            w.loc[~risk_on & (tr[c] == 1), c] = 0.5
    return _monthly(w, close.index)


def _as_weights(fn):
    def weights(data):
        w = fn(data)
        return {c: w[c].reindex(data[c].index).ffill().fillna(0.0) for c in data if c in w}
    return weights


def _as_positions(fn):
    def positions(data):
        w = fn(data)
        return {c: (w[c].reindex(data[c].index).ffill().fillna(0.0) > 0).astype(float) if c in w else
                pd.Series(0.0, index=data[c].index) for c in data}
    return positions


_COMMON = """
**資產**：0050（台股大盤，可投資、沒有倖存者偏差）、黃金 00635U（2015/4 上市）、美債 20 年 00679B（2017/1 上市）、石油 00642U（2015/9 上市）；尚未上市的資產不納入。
**時點**：每月最後一個交易日收盤後決定，次月第一個交易日開盤調整；其餘時間不交易。
**評估**：這是投資組合策略，請看總覽的「投資組合」計分板，和 0050 買進持有、0050 定期定額比較；個股頁面只顯示「是否持有該 ETF」。
**MultiCharts**：需要多商品投資組合（Portfolio Trader）；目前僅提供 Python 回測。
"""

STRATEGIES = [
    {
        "id": "ma_risk_parity",
        "label": "多資產風險平價（不擇時）",
        "family": "多空與跨資產",
        "universe": "etfs",
        "description": "**規則**：每月依近 60 日波動度的倒數分配資金到所有已上市的 ETF（波動越小的配越多），一直持有、不判斷趨勢。用來衡量「單純分散」的效果。\n" + _COMMON,
        "multicharts": None,
        "positions": _as_positions(risk_parity),
        "weights": _as_weights(risk_parity),
    },
    {
        "id": "ma_trend_ivol",
        "label": "多資產趨勢＋風險平價",
        "family": "多空與跨資產",
        "universe": "etfs",
        "description": "**規則**：每月只持有趨勢向上的資產（126 日均線 ±0.5 個月波動度遲滯帶），並依波動度倒數分配；全部向下時持有現金。\n跨資產趨勢追蹤是文獻中最穩健的擇時證據（Moskowitz, Ooi, Pedersen 2012；Hurst, Ooi, Pedersen 2017；Faber 2007）。\n" + _COMMON,
        "multicharts": None,
        "positions": _as_positions(trend_ivol),
        "weights": _as_weights(trend_ivol),
    },
    {
        "id": "ma_riskoff_switch",
        "label": "股債金切換",
        "family": "多空與跨資產",
        "universe": "etfs",
        "description": "**規則**：0050 趨勢向上時全部持有 0050；向下時改持有美債與黃金各 50%（各自趨勢向上才持有，否則該半部位持有現金）。\n概念同「雙動能」：股市轉弱時，資金不閒置，改放在通常與股市反向的資產。\n" + _COMMON,
        "multicharts": None,
        "positions": _as_positions(riskoff_switch),
        "weights": _as_weights(riskoff_switch),
    },
]
