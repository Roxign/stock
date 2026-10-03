"""均值回歸 / 擺盪指標策略 (mean reversion / oscillator family).

All signals use only data up to bar t's close; the engine fills at bar t+1's open.
Indicators are written to match MultiCharts built-ins so the PowerLanguage versions in multicharts/ agree:
  RSI      Wilder smoothing, RSI = 100 * avgUp / (avgUp + avgDown) (= MC's 50 * (1 + NetChgAvg / TotChgAvg)),
           50 when there has been no movement at all.
  Average  simple moving average of closes.
Each rule is a small 0/1 state machine evaluated bar by bar, mirrored line for line in the .txt files.
See RESEARCH.md for the experiments behind the parameter choices.
"""

import numpy as np
import pandas as pd

FAMILY = "均值回歸"


# ---------------------------------------------------------------- indicators
def _rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    tot = up + dn
    return (100 * up / tot.where(tot > 0)).fillna(50.0)


def _sma(close, n):
    return close.rolling(n).mean()


def _hold(entry, exit_, max_hold=None):
    """Flat -> long when `entry` is true at a bar's close; long -> flat when `exit_` is true (bars after the entry bar)
    or once max_hold bars have passed since the entry signal. Mirrors the PowerLanguage state machines."""
    e = np.asarray(entry, dtype=bool)
    x = np.asarray(exit_, dtype=bool)
    out = np.zeros(len(e))
    pos, held = 0, 0
    for i in range(len(e)):
        if pos:
            held += 1
            if x[i] or (max_hold is not None and held >= max_hold):
                pos = 0
        elif e[i]:
            pos, held = 1, 0
        out[i] = pos
    return out


# ---------------------------------------------------------------- rules
def combo_positions(df, rsi_len=14, os_level=25, ob_level=70, max_hold=60, drop_days=5, drop_pct=0.10, exit_ma=10):
    """A: RSI(14) < 25 -> hold until RSI(14) > 70 or 60 bars.  B: 5-day drop > 10% -> hold until close > SMA(10).
    Long when either sub-system is long."""
    c = df["close"]
    r = _rsi(c, rsi_len)
    a = _hold(r < os_level, r > ob_level, max_hold)
    b = _hold(c / c.shift(drop_days) - 1 < -drop_pct, c > _sma(c, exit_ma))
    return pd.Series(np.maximum(a, b), index=df.index)


def swing_positions(df, fast_len=2, buy_level=10, slow_len=14, sell_level=75):
    """Buy when RSI(2) < 10 (short pullback); sell when RSI(14) > 75 (overheated); wait for the next pullback."""
    c = df["close"]
    return pd.Series(_hold(_rsi(c, fast_len) < buy_level, _rsi(c, slow_len) > sell_level), index=df.index)


def classic_positions(df, rsi_len=2, buy_level=5, trend_len=200, exit_len=5):
    """Connors RSI(2): close > SMA(200) and RSI(2) < 5 -> buy; exit when close > SMA(5)."""
    c = df["close"]
    entry = (_rsi(c, rsi_len) < buy_level) & (c > _sma(c, trend_len))
    return pd.Series(_hold(entry, c > _sma(c, exit_len)), index=df.index)


def _each(fn):
    return lambda data: {code: fn(df) for code, df in data.items()}


# ---------------------------------------------------------------- descriptions
COMMON_NOTES = """
**共同注意事項**

- 訊號以收盤價計算，於**次一根 K 棒開盤**成交；成本以手續費 0.1425%（買賣各一次）＋ 證交稅 0.3%（賣出）計，來回約 0.585%。
- 僅做多、全進全出（0 或 1 部位）。
- 回測使用的是**還原權息**價格；若在 MultiCharts 使用未還原資料，除權息缺口可能觸發假訊號。
- 股票池為「目前」的 0050 成分股，屬於事後挑選的贏家，對「逢低買進」類策略特別有利（存活者偏差），實際績效可能較差。
"""

COMBO_DESC = """
### 超跌反彈組合（RSI14 深度超賣 ＋ 五日急跌）

由兩個獨立的「買超跌、等反彈」子系統組成，**任一個持有中就持有股票**：

| 子系統 | 進場（收盤判斷） | 出場（收盤判斷，次日開盤賣） |
|---|---|---|
| A 中期超賣 | RSI(14) < 25 | RSI(14) > 70，或進場訊號後滿 60 根 K 棒 |
| B 短期急跌 | 收盤價較 5 日前下跌超過 10% | 收盤價站上 10 日均線 |

**為什麼這樣設計**

- 經典短線均值回歸（Connors RSI(2)、累積 RSI、IBS 等）在台股個股上，扣成本前每筆平均只賺約 0.5–0.95%，被來回 0.585% 的成本吃掉大半。樣本內，零成本時 IBS 策略的 Sharpe 有 0.70，含成本後變 −0.12。
- 只有**跌幅夠深**的情況，事後反彈才明顯大於成本：RSI(14) < 25 之後 5 / 20 個交易日的平均超額報酬約 +1.3% / +3.6%，五日跌超過 10% 之後約 +2.2% / +3.6%（以次日開盤進場計算，相對於所有交易日的平均）。
- 反彈要好幾週才走完，所以 A 用「RSI 回到 70 或約一季（60 根）」出場；B 是短線急跌，站回 10 日均線就跑，把資金時間留給 A。
- 停損：測試過 8%–25% 的停損，全都讓績效變差或沒有改善（與 Kaminski & Lo 的結論一致：均值回歸過程中停損會砍在反彈前），所以不設停損。

**樣本內（2010–2020）50 檔中位數**：CAGR 4.7%、MDD −25.9%、Sharpe 0.47、持有時間約 16%，Sharpe 勝過買進持有的股票占 39%（買進持有：CAGR 9.6%、MDD −46%、Sharpe 0.51）。
附近參數的 Sharpe：RSI 門檻 20 / 30 → 0.48 / 0.37；出場 65 / 75 → 0.44 / 0.51；時間停損 50 / 70 根 → 0.38 / 0.43；急跌門檻 8% / 12% → 0.45 / 0.48。參數取中間值而非最佳值（出場 75 較好，但選 70 置中）。

**樣本外（2021–2026/10，只跑一次）**：CAGR 6.3%、MDD −26.8%、Sharpe 0.49，與樣本內相近；但同期買進持有 CAGR 35.7%、Sharpe 1.10，只有 14% 的股票 Sharpe 勝過買進持有。主要獲利來自 2025 年 4 月的全市場急跌反彈。

**弱點**

- 大部分時間空手（約 84%），多頭年份遠遠落後買進持有；樣本內的報酬有相當比例來自 2011、2015、2020 等**全市場恐慌後的反彈**。個股自己的利空急跌，事後報酬和平常差不多。
- 2010–2015 盤整期表現較好（Sharpe 勝 B&H 57% 的股票），2016–2020 多頭期則大幅落後。
""" + COMMON_NOTES

SWING_DESC = """
### 拉回買進・過熱賣出（RSI2 / RSI14 波段）

- **買進**：RSI(2) < 10（短線拉回）。
- **賣出**：RSI(14) > 75（中期過熱）。賣出後等下一次 RSI(2) < 10 再買回。

**定位**：持股時間約 90%，本質上是「買進持有，但在過熱時先下車、拉回再上車」。用來回答「賣在過熱、買在拉回，能不能勝過一直抱著？」。

**樣本內（2010–2020）50 檔中位數**：CAGR 8.9%、MDD −43.7%、Sharpe 0.52、持有時間 91%，每檔約 14 筆交易；Sharpe 勝過買進持有的股票占 41%。買進門檻 5–15、賣出門檻 75–80 的結果都相近（Sharpe 0.50–0.53）；賣出門檻 70 則明顯較差（0.46），因為太早下車。
加上 200 日均線趨勢濾網反而變差，因此不使用。

**樣本外（2021–2026/10，只跑一次）**：CAGR 24.2%、MDD −42.1%、Sharpe 1.00、持有時間 87%（買進持有：35.7%、−45.5%、1.10）；Sharpe 勝過買進持有的股票占 30%，勝過定期定額 CAGR 的占 70%。2022 年空頭時中位數 −10.8%，比買進持有的 −8.3% 還差。

**弱點**

- 與買進持有幾乎同進同出，回撤也幾乎一樣深（約 −44%）；優勢很小，主要差異來自過熱後的少數幾次出場。
- 在強勢多頭（例如 2021 年以後）中，過熱可能持續很久，提早賣出會錯過漲幅。
""" + COMMON_NOTES

CLASSIC_DESC = """
### Connors RSI(2) 經典版（對照組）

Larry Connors《Short Term Trading Strategies That Work》中最有名的規則，**原封不動不調參**，用來對照：

- **進場**：收盤價在 200 日均線之上，且 RSI(2) < 5。
- **出場**：收盤價站上 5 日均線。

**樣本內（2010–2020）50 檔中位數**：CAGR 0.5%、MDD −15.8%、Sharpe 0.13、平均每筆淨報酬 +0.36%、持有約 3.6 天、勝率 61%。
**樣本外（2021–2026/10）**：CAGR 0.7%、MDD −19.6%、Sharpe 0.15，結論不變。

**為什麼放這個**：它在美股 ETF 上很有名，但在台股個股上，**每筆交易的平均優勢（扣成本前約 0.95%）大半被 0.585% 的來回成本吃掉**。零成本時 Sharpe 有 0.44。若手續費打 2.8 折，Sharpe 約 0.23；再假設證交稅只有 0.1%（ETF 稅率），約 0.33，仍然不如買進持有。
次日開盤進場**不是**問題：以訊號當天收盤進場與次日開盤進場的報酬幾乎一樣。
""" + COMMON_NOTES

STRATEGIES = [
    {
        "id": "mean_reversion_combo",
        "label": "超跌反彈組合",
        "family": FAMILY,
        "description": COMBO_DESC.strip(),
        "multicharts": "multicharts/mean_reversion_combo.txt",
        "positions": _each(combo_positions),
    },
    {
        "id": "mean_reversion_swing",
        "label": "拉回買進・過熱賣出",
        "family": FAMILY,
        "description": SWING_DESC.strip(),
        "multicharts": "multicharts/mean_reversion_swing.txt",
        "positions": _each(swing_positions),
    },
    {
        "id": "mean_reversion_rsi2_classic",
        "label": "Connors RSI(2) 經典對照",
        "family": FAMILY,
        "description": CLASSIC_DESC.strip(),
        "multicharts": "multicharts/mean_reversion_rsi2_classic.txt",
        "positions": _each(classic_positions),
    },
]
