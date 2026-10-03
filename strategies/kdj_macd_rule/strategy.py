"""KDJ + MACD rule-based strategies (family: KDJ+MACD 規則). See RESEARCH.md for the study behind them.

Both variants are long-only with 0/1 exposure, use the Taiwan-style KDJ (9,3,3; J = 3K - 2D) and MACD (12,26,9)
from stocklab.indicators, and decide each bar with data up to that bar's close only (the engine fills at the next open).

Cross-validation hooks (stocklab/cv.py "retune" mode): each entry has `param_grid` (<= 40 combos around the published
values, covering the neighbourhood explored in RESEARCH.md A4.4-A4.5) and `build(params) -> positions fn`;
build(published defaults) reproduces `positions` exactly.
"""

import itertools
from functools import partial

import numpy as np
import pandas as pd

from stocklab.indicators import macd as macd_lines, tw_kdj

KDJ_N = 9
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
J_LOW, D_LOW = 0, 20  # 三線超賣: J < 0 and D < 20 (which forces K < 13.3)
J_HIGH, J_DAYS, D_HIGH = 100, 3, 80  # 持續超買: J > 100 for 3 bars in a row while D > 80

TB_DEFAULTS = {"kdj_n": KDJ_N, "macd": (MACD_FAST, MACD_SLOW, MACD_SIG), "j_low": J_LOW, "d_low": D_LOW,
               "j_high": J_HIGH, "j_days": J_DAYS, "d_high": D_HIGH}
CZ_DEFAULTS = {"kdj_n": KDJ_N, "macd": (MACD_FAST, MACD_SLOW, MACD_SIG), "j_low": J_LOW, "j_high": J_HIGH}


def _latch(entry, exit_):
    """0/1 state: switches on at an entry bar and off at a later exit bar (exit is not checked on the entry bar)."""
    out = np.zeros(len(entry))
    on = False
    for i in range(len(entry)):
        if not on:
            on = bool(entry[i])
        elif exit_[i]:
            on = False
        out[i] = 1.0 if on else 0.0
    return out


def _streak(mask):
    """Number of consecutive True values ending at each bar."""
    out = np.zeros(len(mask), dtype=int)
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        out[i] = run
    return out


def _signals(df, kdj_n=KDJ_N, macd_p=(MACD_FAST, MACD_SLOW, MACD_SIG)):
    kdj = tw_kdj(df, kdj_n)
    m = macd_lines(df["close"], *macd_p)
    return {
        "k": kdj["k"].to_numpy(),
        "d": kdj["d"].to_numpy(),
        "j": kdj["j"].to_numpy(),
        "signal": m["macd"].to_numpy(),  # MACD signal line (EMA9 of DIF)
        "osc": m["osc"].to_numpy(),  # histogram = DIF - signal
    }


def trend_bounce(data, kdj_n=KDJ_N, macd=(MACD_FAST, MACD_SLOW, MACD_SIG), j_low=J_LOW, d_low=D_LOW,
                 j_high=J_HIGH, j_days=J_DAYS, d_high=D_HIGH):
    """Long while the MACD signal line is above zero, OR while a KDJ bounce trade is open.
    The bounce trade opens only while the signal line is <= 0, on J < 0 with D < 20, and closes after
    J has stayed above 100 for 3 bars in a row with D > 80 (defaults; other values are for cross-validation)."""
    out = {}
    for code, df in data.items():
        s = _signals(df, kdj_n, macd)
        trend = s["signal"] > 0
        entry = ~trend & (s["j"] < j_low) & (s["d"] < d_low)
        exit_ = (_streak(s["j"] > j_high) >= j_days) & (s["d"] > d_high)
        bounce = _latch(entry, exit_)
        out[code] = pd.Series(np.maximum(trend.astype(float), bounce), index=df.index)
    return out


def classic_zone(data, kdj_n=KDJ_N, macd=(MACD_FAST, MACD_SLOW, MACD_SIG), j_low=J_LOW, j_high=J_HIGH):
    """Textbook KDJ + MACD histogram combo: buy when J < 0 while the histogram rises;
    sell when J > 100 while the histogram falls (defaults; other values are for cross-validation)."""
    out = {}
    for code, df in data.items():
        s = _signals(df, kdj_n, macd)
        osc = s["osc"]
        rising = np.r_[False, osc[1:] > osc[:-1]]
        falling = np.r_[False, osc[1:] < osc[:-1]]
        entry = (s["j"] < j_low) & rising
        exit_ = (s["j"] > j_high) & falling
        out[code] = pd.Series(_latch(entry, exit_), index=df.index)
    return out


# ---- cross-validation grids (stocklab/cv.py "retune") ----
# trend_bounce, 38 combos: MACD speed x bounce entry x bounce exit (36), plus KDJ 7 / 12 days at the defaults.
#   MACD (10,22,9) / (12,26,9) / (14,30,9); entry (J, D) < (-5, 15) strict / (0, 20) published / (5, 30) loose;
#   exit (J > 100 days, D >) = (1, 80) quick / (3, 80) published / (5, 85) patient / (3, 70) low D.
TB_GRID = [
    {**TB_DEFAULTS, "macd": m, "j_low": jl, "d_low": dl, "j_days": jd, "d_high": dh}
    for m, (jl, dl), (jd, dh) in itertools.product(
        [(10, 22, 9), (12, 26, 9), (14, 30, 9)],
        [(-5, 15), (0, 20), (5, 30)],
        [(1, 80), (3, 80), (5, 85), (3, 70)],
    )
] + [{**TB_DEFAULTS, "kdj_n": n} for n in (7, 12)]

# classic_zone, 38 combos: J buy level x J sell level x KDJ days (36), plus fast / slow MACD at the defaults.
CZ_GRID = [
    {**CZ_DEFAULTS, "kdj_n": n, "j_low": jl, "j_high": jh}
    for jl, jh, n in itertools.product([-10, -5, 0, 5], [90, 100, 110], [7, 9, 12])
] + [{**CZ_DEFAULTS, "macd": m} for m in ((10, 22, 9), (14, 30, 9))]


def build_trend_bounce(params):
    return partial(trend_bounce, **{**TB_DEFAULTS, **params})


def build_classic_zone(params):
    return partial(classic_zone, **{**CZ_DEFAULTS, **params})


DESC_TREND_BOUNCE = """\
**一句話**：MACD 訊號線在零軸之上就抱著（順勢）；趨勢不在時，等 KDJ 跌到「三線超賣」（J < 0 且 D < 20）才進場搶反彈，直到 J 連續 3 天 > 100 且 D > 80（持續超買）才結束反彈單。兩者任一成立就持有。

**規則**（日線，收盤後判斷，隔天開盤成交，只做多，部位 0% 或 100%）
1. 趨勢腿：MACD 訊號線（DIF 的 9 日 EMA）> 0 → 想持有。
2. 反彈腿：趨勢腿不成立（訊號線 ≤ 0）時，若 **J < 0 且 D < 20** → 開啟反彈狀態；之後 **J 連續 3 天 > 100 且 D > 80** → 關閉。
3. 趨勢腿或反彈腿任一成立就持有，兩者都不成立就全部賣出。

**參數**：KDJ 9,3,3（台灣慣用、1/3 平滑，J = 3K − 2D）、MACD 12,26,9、J 超賣 0 / D 超賣 20、J 超買 100 連續 3 天 / D 超買 80。全部是常見口訣值，50 檔共用同一組，沒有分股或分期間調整。

**為什麼這樣設計**（樣本內 2010–2020 研究）
- J 值的確比 K 值多一點資訊：J < 0 的日子隔天超額報酬年化 +34%，比 K < 20 的 +27% 強；J < 0 連續 3 天且 MACD 偏空時，之後 5 天平均超額 +1.09%（K < 20 黃金交叉只有 +0.34%）。所以反彈腿改用 J 在「跌勢中」提早進場，而不是等 KD 黃金交叉。
- 但 J > 100 只是短線過熱：J 一碰 100 就賣，樣本內 Sharpe 從 0.49 掉到 0.41，因為 J 破 100 之後的弱勢只維持 1～5 天，20 天後平均已經沒有落後。出場因此改成「J 連續 3 天 > 100 且 D > 80」這種強勢鈍化後才下車（等於只在反彈真正走完一段時賣）。
- 結果：樣本內 CAGR 中位數 9.0%（買進持有 9.6%）、Sharpe 0.49（0.49）、MDD −45.0%（−46.1%）、持有率約 86%；47% 的股票報酬勝過買進持有、53% 的股票 Sharpe 勝過買進持有。
- 與上一輪 KD 版（K < 20 金叉 → K > 80 死叉）相比：逐檔 Sharpe 差的中位數 +0.03，但 90% 信賴區間包含 0 —— **J 讓結果略好一點，但差距在雜訊範圍內**。
- 鄰近參數（J 超賣 −5～10、D 超賣 15～30、超買連續 1～5 天、D 超買 70～85、KDJ 7/9/12 日、MACD 10~14/22~30）樣本內 Sharpe 約 0.43~0.53，沒有明顯的尖峰。
- 也測試了台灣常看的均線（5/10/20/60/120/240 日）：站上季線/年線才做、多頭排列、突破月線/季線進出場都沒有改善，因此沒有加入（詳見研究筆記）。

**樣本外 2021–2026**（參數凍結後只跑一次）：CAGR 中位數 29.3%（買進持有 35.7%、定期定額 25.3%），Sharpe 1.04（買進持有 1.10），MDD 中位數 −44.1%（買進持有 −45.5%）；18% 的股票報酬勝過買進持有、78% 勝過定期定額。比上一輪 KD 版（29.0% / Sharpe 0.98）好一點（74% 的股票 Sharpe 較高），主要是因為反彈單抱得比較久、在大多頭裡持股比較高，而不是擇時變準。2022 空頭年與買進持有打平，其餘年份小幅落後。

**交叉驗證（8 折，每折約 2 年）**：固定參數下只有 3/8 折的 Sharpe 中位數勝過買進持有（2014–15、2016–17、2020–21），41% 的「股票×折」勝過；最大回撤 7/8 折較小，但只少 0～5 個百分點。在 2010–11、2018–19、2022–23 這些弱勢折並沒有比較好（Sharpe −0.06 vs 0.05、0.64 vs 0.65、0.47 vs 0.51）：趨勢腿一關，反彈腿幾乎馬上在跌勢中進場，又要等持續超買才出場，所以下跌期間持股率仍有 91～97%。每折改用「其他 7 折最好的參數」（38 組）時，7/8 折選到幾乎不出場（持股約 97%）的版本，結果更接近買進持有但只勝 2/8 折。**它不是空頭避險工具，在各種市況下都只是接近買進持有。**

**注意**
- 這是「接近買進持有、偶爾避開空頭」的策略，不是報酬增強器；在大多頭年份會落後。
- J 版反彈腿進場比 KD 版早，**急跌時會更早接刀**：2020 年 2~3 月疫情急跌期間，個股中位數虧 17.7%（KD 版 10.3%，買進持有 26.0%）。
- 回測使用還原權值日線、買賣手續費 0.1425%、賣出證交稅 0.3%，未計滑價。
"""

DESC_CLASSIC = """\
**一句話**：網路上最常見的 KDJ 口訣「J 值負值買、J 值破 100 賣」再加上 MACD 柱狀體確認。放在這裡當作**對照組**。

**規則**（日線，收盤後判斷，隔天開盤成交，只做多，部位 0% 或 100%）
- 進場：J < 0，且 MACD 柱狀體（OSC = DIF − 訊號線）比前一天高（綠柱縮短 / 紅柱變長）。
- 出場：J > 100，且 MACD 柱狀體比前一天低。

**參數**：KDJ 9,3,3、MACD 12,26,9、J 超賣 0 / 超買 100（口訣值，未調整）。

**結果**
- 樣本內 2010–2020：CAGR 中位數 3.2%（買進持有 9.6%）、Sharpe 0.31（0.49）、MDD −42.0%（−46.1%）、持有率約 40%、勝率 75%，只有 14% 的股票報酬勝過買進持有。
- 樣本外 2021–2026：CAGR 中位數 11.1%（買進持有 35.7%）、Sharpe 0.72（1.10）、MDD −40.1%、持有率約 50%、勝率 100%（中位數），只有 10% 的股票勝過買進持有。
- 若拿掉 MACD 柱狀體確認（純「J < 0 買、J > 100 賣」），樣本內每檔約交易 54 次，CAGR 中位數只剩 2.7%：J 太靈敏，一反彈就碰到 100，賣在起漲初段。

**交叉驗證（8 折，每折約 2 年）**：固定參數下 8 折的 Sharpe 中位數全部輸給買進持有，連 2010–11、2014–15、2018–19、2022–23 這些弱勢折也輸（例如 2018–19 為 0.31 vs 0.65）；最大回撤 8 折都較小，只是因為持股率只有 37～57%。每折改用「其他 7 折最好的參數」（38 組）時，每折都選到賣出門檻 J > 110 —— 這時柱狀體幾乎都還在上升，賣出條件幾乎不會成立，等於「買了就不賣」的買進持有。交叉驗證的答案很直接：這套口訣的賣出規則本身就在扣分。

**為什麼保留它**：讓你在每一檔股票上直接看到「勝率高 ≠ 賺得多」—— 這套規則勝率很高，但持有時間短、常在主升段之前就下車，長期報酬遠低於買進持有。它是上一輪 KD 對照組（K < 20 金叉 + 柱狀體）的 KDJ 版本，結果也差不多（樣本內 Sharpe 0.34 → 0.31）。**不建議實際使用。**
"""

STRATEGIES = [
    {
        "id": "kdj_macd_rule_trend_bounce",
        "label": "MACD趨勢＋KDJ三線超賣反彈",
        "family": "KDJ+MACD 規則",
        "description": DESC_TREND_BOUNCE,
        "multicharts": "multicharts/kdj_macd_rule_trend_bounce.txt",
        "positions": trend_bounce,
        "param_grid": TB_GRID,
        "build": build_trend_bounce,
    },
    {
        "id": "kdj_macd_rule_classic_zone",
        "label": "KDJ口訣＋MACD柱確認（傳統對照）",
        "family": "KDJ+MACD 規則",
        "description": DESC_CLASSIC,
        "multicharts": "multicharts/kdj_macd_rule_classic_zone.txt",
        "positions": classic_zone,
        "param_grid": CZ_GRID,
        "build": build_classic_zone,
    },
]
