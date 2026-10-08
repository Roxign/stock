"""Long/short strategies under Taiwan's 融券 rules (the engine enforces suspensions, forced covers, the 平盤以下 rule
and margin calls; see stocklab/shortrules.py). research/diagnosis.md motivates the design: out-of-market days only fall
in bear regimes, so shorting is gated by a market downtrend or used on assets with two-sided trends (ETFs).
"""

import numpy as np
import pandas as pd

from stocklab.indicators import macd, tw_kdj
from strategies.trend.strategy import ew_index, trend_state

STOP = 0.10
SHORT_SIZE = 0.5


def _turn_up(x):
    return (x > x.shift(1)) & (x.shift(1) <= x.shift(2))


def _turn_dn(x):
    return (x < x.shift(1)) & (x.shift(1) >= x.shift(2))


def rebound_signals(df):
    """The user's rule: buy when K and the MACD histogram turn up on the same bar while DIF and the signal line are
    below zero; sell when both turn down while DIF and the signal line are above zero."""
    kd = tw_kdj(df)
    m = macd(df["close"])
    buy = _turn_up(kd["k"]) & _turn_up(m["osc"]) & (m["dif"] < 0) & (m["macd"] < 0)
    sell = _turn_dn(kd["k"]) & _turn_dn(m["osc"]) & (m["dif"] > 0) & (m["macd"] > 0)
    return buy.to_numpy(), sell.to_numpy()


def rebound_ls_one(df, stop=STOP):
    """+1 after a buy signal, -1 after a sell signal; a close beyond the stop (measured from the fill at the next
    open) goes flat until the next signal."""
    buy, sell = rebound_signals(df)
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    out = np.zeros(len(df))
    pos, entry, pending = 0.0, None, False
    for i in range(len(df)):
        if pending:  # filled at this bar's open
            entry, pending = o[i], False
        if pos and entry is not None and ((pos > 0 and c[i] < entry * (1 - stop)) or (pos < 0 and c[i] > entry * (1 + stop))):
            pos, entry = 0.0, None
        if buy[i] and pos <= 0:
            pos, entry, pending = 1.0, None, True
        elif sell[i] and pos >= 0:
            pos, entry, pending = -1.0, None, True
        out[i] = pos
    return pd.Series(out, index=df.index)


def rebound_ls(data):
    return {code: rebound_ls_one(df) for code, df in data.items()}


def trend_regime_ls(data):
    """Long while the stock's own trend is up; short (half size) only when both the stock and the equal-weight market
    of the 50 are in a downtrend; otherwise flat."""
    mkt_up = trend_state(ew_index(data))
    out = {}
    for code, df in data.items():
        up = trend_state(df["close"])
        m = mkt_up.reindex(df.index).ffill().fillna(1.0)
        out[code] = pd.Series(np.where(up == 1, 1.0, np.where(m == 0, -SHORT_SIZE, 0.0)), index=df.index)
    return out


def etf_trend_long(data):
    return {code: trend_state(df["close"]).astype(float) for code, df in data.items()}


def etf_trend_ls(data):
    return {code: 2.0 * trend_state(df["close"]) - 1.0 for code, df in data.items()}


_RULES = """
**放空規則（回測引擎自動套用，見 research/short_rules.md）**：只有融券標的才能放空（新上市約 6 個月後）；停止融券賣出期間（除權息、股東會前）不得新增空單，並在停券第一天開盤強制回補；平盤以下限制期間（2013/9/23 前的一般股票、2008/9/22–2009/1/2、2015/8/24–9/18 全面禁止，以及前一日跌停、2020/3/20–6/9 與 2022/10/21–2023/2/23 前一日跌 3.5% 以上）開盤價低於前一日收盤時不得放空；擔保維持率低於 130% 時下一個開盤強制回補；放空成本 = 手續費 0.1425% + 證交稅（股票 0.3%、ETF 0.1%、債券 ETF 免稅）+ 融券手續費 0.08%，回補時再付手續費。
**MultiCharts**：可用 SellShort / BuyToCover 下單，但 MultiCharts 不會自動處理停券與強制回補，實盤需自行依證交所「停資停券預告」處理。
"""

_REBOUND = """
**規則（你指定的 KDJ+MACD 反彈邏輯，多空版本）**
- 做多：K 值與 MACD 柱狀體（OSC）同一天由下往上轉折，且 DIF 與 MACD 線都在零軸之下。
- 放空：K 值與 OSC 同一天由上往下轉折，且 DIF 與 MACD 線都在零軸之上（原本只賣出，這裡改成反手放空）。
- 停損：多單收盤跌破進場價 10%、空單收盤漲破進場價 10%，隔日開盤出場並空手等下一個訊號。
- 部位只有 +1、0、−1（實際能否放空依台灣規則）。
""" + _RULES

STRATEGIES = [
    {
        "id": "ls_rebound_stocks",
        "label": "KDJ+MACD 反彈・多空（個股）",
        "family": "多空與跨資產",
        "universe": "stocks",
        "description": _REBOUND,
        "multicharts": "multicharts/ls_rebound.txt",
        "positions": rebound_ls,
    },
    {
        "id": "ls_trend_regime",
        "label": "趨勢＋空頭才放空（個股）",
        "family": "多空與跨資產",
        "universe": "stocks",
        "description": """
**規則**
- 個股趨勢（126 日均線 ±0.5 個月波動度的遲滯帶，與「均線趨勢減碼」相同）向上 → 做多 100%。
- 個股趨勢向下 **且** 50 檔等權指數的趨勢也向下（確認空頭）→ 放空 50%。
- 個股向下但大盤向上 → 空手（不在多頭市場放空個股）。

**為什麼這樣設計**：診斷報告發現，策略空手的日子在一般時期股價仍上漲（放空會賠），只有空頭年才下跌；因此只在大盤確認空頭時放空，而且只放空一半部位。
""" + _RULES,
        "multicharts": None,
        "positions": trend_regime_ls,
    },
    {
        "id": "ls_rebound_etfs",
        "label": "KDJ+MACD 反彈・多空（ETF）",
        "family": "多空與跨資產",
        "universe": "etfs",
        "description": _REBOUND,
        "multicharts": "multicharts/ls_rebound.txt",
        "positions": rebound_ls,
    },
    {
        "id": "etf_trend_long",
        "label": "趨勢追蹤・只做多（ETF）",
        "family": "多空與跨資產",
        "universe": "etfs",
        "description": """
**規則**：對 0050、黃金（00635U）、石油（00642U）、美債 20 年（00679B）各自判斷趨勢（126 日均線 ±0.5 個月波動度遲滯帶）：向上持有 100%，向下空手。
文獻上跨資產的時間序列動能最穩健（Moskowitz, Ooi, Pedersen 2012；Hurst, Ooi, Pedersen 2017）。參數沿用股票趨勢策略，沒有針對 ETF 調整。
""",
        "multicharts": None,
        "positions": etf_trend_long,
    },
    {
        "id": "etf_trend_ls",
        "label": "趨勢追蹤・多空（ETF）",
        "family": "多空與跨資產",
        "universe": "etfs",
        "description": """
**規則**：同「趨勢追蹤・只做多（ETF）」，但趨勢向下時改為放空 100%（黃金、石油、美債的下跌趨勢也能獲利）。
""" + _RULES,
        "multicharts": None,
        "positions": etf_trend_ls,
    },
]
