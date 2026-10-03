"""趨勢追蹤 (trend following) family.

Three published variants, one parameter set for all 50 stocks, all decided on the daily close and
filled by the engine at the next bar's open:

  trend_donchian     close breaks the highest close of the previous 120 bars -> 1; close breaks the
                     lowest close of the previous 60 bars -> 0 (turtle-style channel breakout).
  trend_sma_half     trend state from close vs SMA(126) with a volatility-scaled hysteresis band;
                     exposure 1 in an uptrend, 0.5 in a downtrend (never fully out).
  trend_dual_market  0.5 * stock trend state + 0.5 * market trend state (same rule applied to an
                     equal-weighted index of all stocks, built only from returns up to each date).

See RESEARCH.md for the experiments behind these choices.
"""

import numpy as np
import pandas as pd

from stocklab.indicators import sma

# ---- parameters (chosen in-sample 2010-2020, see RESEARCH.md) ----
DC_ENTRY = 120   # Donchian entry channel: highest close of the previous 120 bars (~half a year)
DC_EXIT = 60     # Donchian exit channel: lowest close of the previous 60 bars (~one quarter)
MA_N = 126       # trend moving average (~6 months)
BAND_K = 0.5     # hysteresis band, in units of one-month volatility
VOL_N = 60       # bars used to estimate daily volatility
MONTH = 21       # bars per month (scales daily vol to monthly vol)


def _hold(enter, exit_):
    """Flat -> long when `enter`; long -> flat when `exit_`. NaN comparisons are False, so state is kept."""
    out = np.zeros(len(enter))
    cur = 0.0
    for i in range(len(enter)):
        if cur == 0.0 and enter[i]:
            cur = 1.0
        elif cur == 1.0 and exit_[i]:
            cur = 0.0
        out[i] = cur
    return out


def trend_state(close, n=MA_N, k=BAND_K, vol_n=VOL_N):
    """1 while in an uptrend, else 0.

    z = (close / SMA(n) - 1) / (std of daily log returns over vol_n bars * sqrt(21)).
    Enter when z > +k, leave when z < -k: the band is k one-month standard deviations wide on each side,
    so a quiet stock needs a smaller move than a volatile one to flip the state.
    """
    m = sma(close, n)
    sig = np.log(close).diff().rolling(vol_n).std() * np.sqrt(MONTH)
    z = (close / m - 1) / sig.where(sig > 0)
    return pd.Series(_hold((z > k).to_numpy(), (z < -k).to_numpy()), index=close.index)


def ew_index(data):
    """Equal-weighted index of every stock that trades on a date (mean of that day's close-to-close returns)."""
    rets = pd.DataFrame({c: df["close"].pct_change() for c, df in data.items()})
    return (1 + rets.mean(axis=1).fillna(0.0)).cumprod()


def donchian(data, n_in=DC_ENTRY, n_out=DC_EXIT):
    out = {}
    for code, df in data.items():
        cl = df["close"]
        hi = cl.rolling(n_in).max().shift(1)
        lo = cl.rolling(n_out).min().shift(1)
        out[code] = pd.Series(_hold((cl > hi).to_numpy(), (cl < lo).to_numpy()), index=df.index)
    return out


def sma_half(data):
    return {code: 0.5 + 0.5 * trend_state(df["close"]) for code, df in data.items()}


def dual_market(data):
    mkt = trend_state(ew_index(data))
    out = {}
    for code, df in data.items():
        m = mkt.reindex(df.index).fillna(0.0)
        out[code] = 0.5 * trend_state(df["close"]) + 0.5 * m
    return out


_COMMON = """
**共通細節**
- 訊號一律用當日收盤價計算，隔日開盤成交（回測引擎規則），手續費 0.1425%、賣出另加證交稅 0.3%。
- 參數只用樣本內（2010–2020）挑選，50 檔股票共用同一組參數，未對個股調參。
- 研究過程、被淘汰的想法與樣本外結果請見 `strategies/trend/RESEARCH.md`。
"""

DESC_DONCHIAN = """
**唐奇安通道突破（海龜式，只做多）**

**規則**
- 空手時：今日收盤價 **高於前 120 個交易日的最高收盤價**（約半年新高）→ 明日開盤全部買進。
- 持有時：今日收盤價 **低於前 60 個交易日的最低收盤價**（約一季新低）→ 明日開盤全部賣出。
- 部位只有 0 或 1。

**參數與選擇理由**
- 進場 120 日／出場 60 日。海龜原始規則是 20/10 與 55/20，但在這 50 檔台股上短通道來回被洗、手續費與證交稅吃掉大部分報酬（20/10 樣本內 CAGR 中位數只有 0.8%）。
- 掃描進場 60–250 日 × 出場 20–120 日，100–120／60–100 是一塊穩定高原（Sharpe 0.38–0.43），取中間、好記的 120/60，而不是格點上最好的那一格。

**樣本內（2010–2020）中位數**：CAGR 5.1%、最大回撤 −38%、Sharpe 0.42、持有率約 45%（買進持有：9.6%、−46%、0.51）。

**樣本外（2021–2026/10）中位數**：CAGR 22.4%、最大回撤 −41%、Sharpe 0.80（買進持有：35.7%、−46%、1.10；定期定額 25.3%）。樣本外三個趨勢版本中表現最差：多頭中常在高點附近才進場、回檔一季才出場，回撤沒有明顯縮小。

**特性與注意事項**
- 一半以上的時間空手，回撤明顯較小，但報酬也較低；在 V 型反彈（2012、2019、2020）會大幅落後買進持有。
- 勝率約五成，獲利來自少數大波段；單檔結果差異很大。
- 50 檔是「現在」的 0050 成分股（事後挑出的贏家），這種倖存者偏差對「空手避險」類策略不利。
""" + _COMMON

DESC_SMA_HALF = """
**均線趨勢減碼（滿倉／半倉）**

**規則**
1. 計算 126 日簡單均線 MA，以及近 60 日「日對數報酬」標準差 × √21（約一個月的波動度 σ）。
2. 偏離度 z = (收盤價 ÷ MA − 1) ÷ σ。
3. z > +0.5 → 判定為「上升趨勢」；z < −0.5 → 判定為「下降趨勢」；介於中間則維持原判定（遲滯帶，避免在均線附近來回進出）。
4. 上升趨勢持有 **100%**，下降趨勢只持有 **50%**，永遠不完全空手。

**參數與選擇理由**
- 126 日 ≈ 半年，介於 Faber 的 10 個月均線與 MOP 時間序列動能的 6–12 個月之間；樣本內 6 個月比 10–12 個月好。
- 遲滯帶以「波動度」為單位，讓低波動的金融股與高波動的電子股可以共用同一組參數。
- 鄰近參數（均線 105–168 日、帶寬 0.25–1.0σ）Sharpe 都在 0.45–0.52，結果不依賴單一參數。
- 研究發現：趨勢訊號對這 50 檔股票幾乎沒有「預測報酬」的能力，主要功能是在高波動的下跌階段降低曝險；完全空手會讓報酬被稀釋，保留半倉是風險調整後表現最好的折衷。

**樣本內（2010–2020）中位數**：CAGR 7.4%、最大回撤 −40%、Sharpe 0.52（買進持有：9.6%、−46%、0.51）。

**樣本外（2021–2026/10）中位數**：CAGR 31.1%、最大回撤 −38.5%、Sharpe 1.05（買進持有：35.7%、−45.5%、1.10；定期定額 25.3%）。少賺約 4.6 個百分點／年，換到約 7 個百分點較小的回撤；86% 的股票勝過定期定額。

**特性與注意事項**
- 這是「降低回撤」的工具，不是「打敗大盤」的工具：多頭年份會落後買進持有。
- 因為永遠至少持有 50%，「完整交易」只會有 1 筆；實際的加減碼請看「部位調整次數」，中位數每年約 2 次。
- 實際下單需整張或零股，回測採用可分割股數。
""" + _COMMON

DESC_DUAL = """
**個股＋大盤雙趨勢（0／50／100%）**

**規則**
- 個股趨勢：與「均線趨勢減碼」相同（126 日均線、±0.5 個月波動度遲滯帶）。
- 大盤趨勢：同一套規則套用在「50 檔等權重指數」上（每天取當天有交易的股票的平均日報酬累乘，只用到當天以前的資料）。
- 部位 = 0.5 ×（個股是否上升趨勢）+ 0.5 ×（大盤是否上升趨勢）：兩者皆多 → 100%，一多一空 → 50%，兩者皆空 → 0%。

**參數與選擇理由**
- 研究「大盤濾網有沒有用」（同一組 126 日／0.5σ 規則）：只用大盤趨勢決定進出（所有股票同進同出）樣本內 Sharpe 0.47、只用個股趨勢 0.43、兩者各半 0.44（回撤 −41% 比只看個股的 −45% 小）；「兩者都多才持有」0.39 最差（任一個假訊號就出場）。結論：大盤濾網只有小幅幫助，主要是降低回撤。
- 曾發現「每月月初檢查一次」的版本樣本內 Sharpe 高達 0.54，但把檢查日改到月中或月底就掉到 0.32–0.49，屬於「再平衡時點運氣」，因此改用每日檢查＋遲滯帶的版本。

**樣本內（2010–2020）中位數**：CAGR 5.8%、最大回撤 −41%、Sharpe 0.44、有持股的天數約 80%（平均曝險約 66%）、每年約 4 次加減碼（買進持有：9.6%、−46%、0.51）。

**樣本外（2021–2026/10）中位數**：CAGR 29.7%、最大回撤 −38.2%、Sharpe 1.05（買進持有：35.7%、−45.5%、1.10；定期定額 25.3%）。

**特性與注意事項**
- 大盤指數是由目前 50 檔成分股等權重組成，與加權指數或 0050（台積電權重極高）不同；MultiCharts 版本只能用加權指數或 0050 當作 data2 近似。
- 50 檔共用同一個大盤訊號，大盤轉空時所有股票會同時減碼，屬於系統性的避險，而非個股選擇。
""" + _COMMON

STRATEGIES = [
    {
        "id": "trend_donchian",
        "label": "唐奇安通道 120/60",
        "family": "趨勢追蹤",
        "description": DESC_DONCHIAN.strip(),
        "multicharts": "multicharts/trend_donchian.txt",
        "positions": donchian,
    },
    {
        "id": "trend_sma_half",
        "label": "均線趨勢減碼",
        "family": "趨勢追蹤",
        "description": DESC_SMA_HALF.strip(),
        "multicharts": "multicharts/trend_sma_half.txt",
        "positions": sma_half,
    },
    {
        "id": "trend_dual_market",
        "label": "個股＋大盤雙趨勢",
        "family": "趨勢追蹤",
        "description": DESC_DUAL.strip(),
        "multicharts": "multicharts/trend_dual_market.txt",
        "positions": dual_market,
    },
]
