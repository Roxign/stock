"""KD + MACD rule-based strategies (family: KD+MACD 規則). See RESEARCH.md for the study behind them.

Both variants are long-only with 0/1 exposure, use the Taiwan-style KD (9,3,3) and MACD (12,26,9) from
stocklab.indicators, and decide each bar with data up to that bar's close only (the engine fills at the next open).
"""

import numpy as np
import pandas as pd

from stocklab.indicators import macd, tw_kd

KD_N = 9
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
OVERSOLD, OVERBOUGHT = 20, 80


def _cross_up(a, b):
    return np.r_[False, (a[1:] > b[1:]) & (a[:-1] <= b[:-1])]


def _cross_dn(a, b):
    return np.r_[False, (a[1:] < b[1:]) & (a[:-1] >= b[:-1])]


def _latch(entry, exit_):
    """0/1 state: switches on at an entry bar and off at an exit bar (the two never coincide for these rules)."""
    out = np.zeros(len(entry))
    on = False
    for i in range(len(entry)):
        if not on:
            on = bool(entry[i])
        elif exit_[i]:
            on = False
        out[i] = 1.0 if on else 0.0
    return out


def _signals(df):
    kd = tw_kd(df, KD_N)
    m = macd(df["close"], MACD_FAST, MACD_SLOW, MACD_SIG)
    k, d = kd["k"].to_numpy(), kd["d"].to_numpy()
    return {
        "k": k,
        "golden": _cross_up(k, d),  # K crosses above D
        "death": _cross_dn(k, d),  # K crosses below D
        "signal": m["macd"].to_numpy(),  # MACD signal line (EMA9 of DIF)
        "osc": m["osc"].to_numpy(),  # histogram = DIF - signal
    }


def trend_bounce(data):
    """Long while the MACD signal line is above zero, OR while a KD oversold-bounce trade is open
    (opened by a golden cross with K < 20, closed by a death cross with K > 80)."""
    out = {}
    for code, df in data.items():
        s = _signals(df)
        trend = (s["signal"] > 0).astype(float)
        bounce = _latch(s["golden"] & (s["k"] < OVERSOLD), s["death"] & (s["k"] > OVERBOUGHT))
        out[code] = pd.Series(np.maximum(trend, bounce), index=df.index)
    return out


def classic_zone(data):
    """Textbook combo: buy on a KD golden cross below 20 while the MACD histogram rises;
    sell on a KD death cross above 80 while the histogram falls."""
    out = {}
    for code, df in data.items():
        s = _signals(df)
        osc = s["osc"]
        rising = np.r_[False, osc[1:] > osc[:-1]]
        falling = np.r_[False, osc[1:] < osc[:-1]]
        entry = s["golden"] & (s["k"] < OVERSOLD) & rising
        exit_ = s["death"] & (s["k"] > OVERBOUGHT) & falling
        out[code] = pd.Series(_latch(entry, exit_), index=df.index)
    return out


DESC_TREND_BOUNCE = """\
**一句話**：MACD 訊號線在零軸之上就抱著（順勢）；跌到 KD 超賣區出現黃金交叉時也進場搶反彈（逆勢），兩者任一成立就持有。

**規則**（日線，收盤後判斷，隔天開盤成交，只做多，部位 0% 或 100%）
1. 趨勢腿：MACD 訊號線（DIF 的 9 日 EMA）> 0 → 想持有。
2. 反彈腿：K < 20 時 KD 黃金交叉 → 開啟反彈狀態；之後 K > 80 時 KD 死亡交叉 → 關閉。
3. 趨勢腿或反彈腿任一成立就持有，兩者都不成立就全部賣出。

**參數**：KD 9,3,3（台灣慣用、1/3 平滑）、MACD 12,26,9、超賣 20 / 超買 80。全部是教科書預設值，沒有針對個股或期間調整，50 檔共用同一組。

**為什麼這樣設計**（樣本內 2010–2020 的研究結果）
- 事件研究顯示，MACD 訊號線跌破零軸後 20 日平均落後股票本身漂移約 0.8%，是有效的「出場」訊號；但單用它（持有率約 58%）會錯過很多上漲，Sharpe 只有 0.42。
- KD 在超賣區（K < 20）之後幾天的報酬明顯高於平均（短期反轉），KD 高檔死亡交叉之後反而續漲 —— 所以 KD 用來「逆勢補位」，而不是追高殺低。
- 兩者合併後持有率約 85%，樣本內 Sharpe 0.51（與買進持有相同）、CAGR 中位數 8.9%（買進持有 9.6%），47% 的股票 Sharpe 勝過買進持有、76% 的股票勝過定期定額。鄰近參數（MACD 10~14 / 22~30、KD 7~12 日、超買 70~90）結果都在 Sharpe 0.49~0.53 之間。

**樣本外 2021–2026**：CAGR 中位數 29.0%（買進持有 35.7%、定期定額 25.3%），Sharpe 0.98（買進持有 1.10），MDD 中位數 −45.6%（與買進持有相同）；只有 14% 的股票報酬勝過買進持有，60% 勝過定期定額。2022 空頭年小幅領先，其餘多頭年份落後。

**注意**
- 這是「接近買進持有、偶爾避開空頭」的策略，不是報酬增強器；在大多頭年份會落後。
- 超賣門檻 20 是局部最佳：改成 15 或 25 時樣本內 Sharpe 落在 0.42~0.52（超買 80 時分別為 0.50 / 0.47），對這個門檻較敏感。
- 回測使用還原權值日線、買賣手續費 0.1425%、賣出證交稅 0.3%，未計滑價。
"""

DESC_CLASSIC = """\
**一句話**：台灣最常見的教科書組合 —— KD 低檔黃金交叉且 MACD 柱狀體翻揚就買，KD 高檔死亡交叉且柱狀體轉弱就賣。放在這裡當作**對照組**。

**規則**（日線，收盤後判斷，隔天開盤成交，只做多，部位 0% 或 100%）
- 進場：K < 20 時 KD 黃金交叉，且 MACD 柱狀體（OSC = DIF − 訊號線）比前一天高（綠柱縮短 / 紅柱變長）。
- 出場：K > 80 時 KD 死亡交叉，且 MACD 柱狀體比前一天低。

**參數**：KD 9,3,3、MACD 12,26,9、超賣 20 / 超買 80（教科書預設值，未調整）。

**結果**
- 樣本內 2010–2020：CAGR 中位數 4.4%（買進持有 9.6%）、Sharpe 0.35（0.51）、MDD −40.8%（−46.1%）、持有率約 39%、勝率 67%，只有 14% 的股票報酬勝過買進持有。
- 樣本外 2021–2026：CAGR 中位數 9.4%（買進持有 35.7%）、Sharpe 0.61（1.10）、MDD −34.0%、勝率 80%，只有 2% 的股票勝過買進持有。
- 若拿掉 20 / 80 區間限制（任何 KD 交叉 + 柱狀體方向都交易），樣本內每檔約交易 200 次，CAGR 中位數 −8.3%：交易成本與短期反轉把訊號完全吃掉。

**為什麼保留它**：讓你在每一檔股票上直接看到「勝率高 ≠ 賺得多」—— 這套規則勝率很高，但持有時間短、常在主升段之前就下車，長期報酬遠低於買進持有。MACD 柱狀體確認在研究中沒有改善純 KD 低買高賣（Sharpe 0.37 → 0.35）。
"""

STRATEGIES = [
    {
        "id": "kd_macd_rule_trend_bounce",
        "label": "MACD趨勢＋KD超賣反彈",
        "family": "KD+MACD 規則",
        "description": DESC_TREND_BOUNCE,
        "multicharts": "multicharts/kd_macd_rule_trend_bounce.txt",
        "positions": trend_bounce,
    },
    {
        "id": "kd_macd_rule_classic_zone",
        "label": "KD低檔金叉＋MACD柱確認（傳統對照）",
        "family": "KD+MACD 規則",
        "description": DESC_CLASSIC,
        "multicharts": "multicharts/kd_macd_rule_classic_zone.txt",
        "positions": classic_zone,
    },
]
