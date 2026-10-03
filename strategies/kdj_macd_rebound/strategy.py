"""KDJ + MACD rebound strategies (family: KDJ+MACD 反彈), implementing the user's rule:
buy when KDJ and MACD have both started to turn up while MACD is below the zero axis; sell when KD and MACD both turn
down while MACD is above the zero axis; every buy carries a stop-loss. See RESEARCH.md for the study behind them.

Long-only, 0/1 exposure, Taiwan KDJ (9,3,3; J = 3K - 2D) and MACD (12,26,9) from stocklab.indicators. Every bar is
decided with data up to its close; the engine fills at the next open. The stop is close-based: the entry price is the
open of the bar after the buy signal, and from that bar's close on, a close below the stop exits at the next open
(MultiCharts: if Close < EntryPrice * (1 - s) then Sell next bar at market).
"""

import itertools
import math

import numpy as np
import pandas as pd

from stocklab.indicators import atr, macd, tw_kdj

BASE = {
    "buy_kdj": "k",      # KDJ 開始反彈: k = K 勾頭向上, j = J 勾頭向上, kx = KD 黃金交叉
    "buy_macd": "osc",   # MACD 開始反彈: osc = 柱狀體 (DIF - MACD) 勾頭向上, dif = DIF 勾頭向上
    "win": 1,            # 兩者的「開始反彈」需在幾根K棒內先後出現（1 = 同一根）
    "zone": "both",      # MACD 為負/正: both = DIF 與 MACD 線都 < 0 / > 0, dif = 只看 DIF
    "sell_kdj": "k",     # KD 往下: k = K 勾頭向下, j = J 勾頭向下, kx = KD 死亡交叉
    "sell_macd": "osc",  # MACD 往下: osc = 柱狀體勾頭向下, x = DIF 跌破 MACD 線（死亡交叉）
    "sell_win": 1,
    "stop": "pct",       # pct = 收盤 < 進場價 x (1 - stop_val); atr = 收盤 < 進場價 - stop_val x ATR14（訊號當日）
    "stop_val": 0.10,
}
TURN = dict(BASE)
DEADCROSS = {**BASE, "sell_macd": "x"}


def _prev(x, n):
    out = np.full(len(x), np.nan)
    out[n:] = x[:-n]
    return out


def _turn(x, up=True):
    """Starts to turn: x[t] > x[t-1] after x[t-1] <= x[t-2] (mirror for down)."""
    x1, x2 = _prev(x, 1), _prev(x, 2)
    with np.errstate(invalid="ignore"):
        ev = (x > x1) & (x1 <= x2) if up else (x < x1) & (x1 >= x2)
    return np.nan_to_num(ev).astype(bool)


def _cross(a, b, up=True):
    out = np.zeros(len(a), dtype=bool)
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1]) if up else (a[1:] < b[1:]) & (a[:-1] >= b[:-1])
    return out


def _moving(x, up=True):
    out = np.zeros(len(x), dtype=bool)
    out[1:] = x[1:] > x[:-1] if up else x[1:] < x[:-1]
    return out


def _recent(ev, n):
    if n <= 1:
        return ev
    return pd.Series(ev.astype(float)).rolling(n, min_periods=1).max().to_numpy() > 0


def _kdj_side(s, kind, up):
    """(event, still-moving) for the KDJ leg."""
    if kind == "kx":
        return _cross(s["k"], s["d"], up), (s["k"] > s["d"]) if up else (s["k"] < s["d"])
    x = s[kind]
    return _turn(x, up), _moving(x, up)


def _macd_side(s, kind, up):
    """(event, still-moving) for the MACD leg."""
    if kind == "x":
        return _cross(s["dif"], s["sig"], up), (s["osc"] > 0) if up else (s["osc"] < 0)
    x = s[kind]
    return _turn(x, up), _moving(x, up)


def _both(a, b, n):
    """Both legs started within the last n bars, the later one today, and both still moving that way today."""
    (ea, ma), (eb, mb) = a, b
    return ((ea & _recent(eb, n)) | (eb & _recent(ea, n))) & ma & mb


def _zone(s, kind, neg):
    if kind == "both":
        return (s["dif"] < 0) & (s["sig"] < 0) if neg else (s["dif"] > 0) & (s["sig"] > 0)
    return s["dif"] < 0 if neg else s["dif"] > 0


def _indicators(df):
    k = tw_kdj(df)
    m = macd(df["close"])
    return {"k": k["k"].to_numpy(), "d": k["d"].to_numpy(), "j": k["j"].to_numpy(),
            "dif": m["dif"].to_numpy(), "sig": m["macd"].to_numpy(), "osc": m["osc"].to_numpy()}


def signals(df, p):
    """Raw buy / sell signal arrays for one stock (decided at each bar's close)."""
    s = _indicators(df)
    buy = _both(_kdj_side(s, p["buy_kdj"], True), _macd_side(s, p["buy_macd"], True), p["win"]) & _zone(s, p["zone"], True)
    sell = _both(_kdj_side(s, p["sell_kdj"], False), _macd_side(s, p["sell_macd"], False), p["sell_win"]) \
        & _zone(s, p["zone"], False)
    return buy, sell


def _state(df, buy, sell, p):
    """0/1 target with the close-based stop. Entry price = open of the bar after the buy signal, so the stop is
    checked from that bar's close onward."""
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    a = atr(df).to_numpy() if p["stop"] == "atr" else None
    out = np.zeros(len(c))
    on, sig, stop_px = False, -1, -math.inf
    for t in range(len(c)):
        if not on:
            if buy[t]:
                on, sig, stop_px = True, t, -math.inf
        else:
            if t == sig + 1:
                stop_px = o[t] * (1 - p["stop_val"]) if p["stop"] == "pct" else o[t] - p["stop_val"] * a[sig]
            if sell[t] or c[t] < stop_px:
                on = False
        out[t] = 1.0 if on else 0.0
    return out


def build(params):
    p = {**BASE, **params}

    def positions(data):
        out = {}
        for code, df in data.items():
            buy, sell = signals(df, p)
            out[code] = pd.Series(_state(df, buy, sell, p), index=df.index)
        return out

    return positions


def _grid(base, axes):
    keys = list(axes)
    return [{**base, **dict(zip(keys, vals))} for vals in itertools.product(*axes.values())]


# 36 combos each: the readings of the user's phrases explored in RESEARCH.md around each published variant.
GRID_TURN = _grid(TURN, {"buy_kdj": ["k", "j", "kx"], "buy_macd": ["osc", "dif"], "zone": ["both", "dif"],
                         "stop_val": [0.07, 0.10, 0.15]})
GRID_DEADCROSS = _grid(DEADCROSS, {"sell_kdj": ["k", "j", "kx"], "sell_win": [1, 3], "zone": ["both", "dif"],
                                   "stop_val": [0.07, 0.10, 0.15]})

DESC_TURN = """\
**一句話**：把「KDJ 與 MACD 皆開始反彈、且 MACD 為負時買進；KD 與 MACD 往下、且 MACD 為正時賣出；買進有停損」直接寫成程式的**忠實版本**。MACD 兩條線都在零軸下時，K 值與 MACD 柱狀體**同一天一起勾頭向上**就買；兩條線都在零軸上時，K 值與柱狀體**同一天一起勾頭向下**就賣；每筆買進都有 10% 收盤停損。

**規則**（日線，收盤後判斷，隔天開盤成交，只做多，部位 0% 或 100%）
1. 買進（同一根 K 棒同時成立）
   - KDJ 開始反彈：K 值勾頭向上（今天 K > 昨天 K，且昨天 K ≤ 前天 K）。
   - MACD 開始反彈：柱狀體 OSC = DIF − MACD 勾頭向上（綠柱開始縮短：今天 OSC > 昨天，且昨天 ≤ 前天）。
   - MACD 為負：DIF < 0 **且** MACD 線（DIF 的 9 日 EMA）< 0，兩條線都在零軸下。
2. 賣出（同一根 K 棒同時成立）：K 值勾頭向下、柱狀體勾頭向下，且 DIF > 0、MACD 線 > 0。
3. 停損：進場價 = 買進訊號隔天的開盤價。從進場那天的收盤開始，只要**收盤 < 進場價 × 0.9**，隔天開盤賣出。MultiCharts 寫法是 `if Close < EntryPrice * 0.9 then Sell next bar at market`，**不要用 SetStopLoss**（盤中觸價出場，和回測不同）。
4. 賣出或停損後，要等下一個買進訊號才會再進場。

**參數**：KDJ 9,3,3（台灣慣用 1/3 平滑，J = 3K − 2D）、MACD 12,26,9、停損 10%。50 檔共用同一組，沒有分股調整。

**每個詞怎麼變成程式**（只用樣本內 2010–2020 比較，詳見研究筆記）
- 「KDJ 開始反彈」：K 勾頭、J 勾頭、KD 黃金交叉三種讀法樣本內 Sharpe 中位數都只有 0.07～0.09，沒有差別；選 K 勾頭，因為它就是「開始」反彈的第一天（J 太靈敏，黃金交叉是事後確認）。
- 「MACD 開始反彈」：柱狀體勾頭（0.09）比 DIF 勾頭（0.00）好；MACD 黃金交叉在零軸下又要與 K 勾頭同時出現的機會太少（持股率只有 6%）。
- 「皆」：同一天 vs 3 天內、5 天內先後出現，Sharpe 0.09～0.12，差在雜訊內，選最直接的同一天。
- 「MACD 為負／為正」：只看 DIF 為 0.09、只看 MACD 線為 0.17、兩線都在零軸下為 0.18；台灣教學把「快慢線都在 0 軸之下」當成空頭區，所以採兩線同時判斷，賣出端對稱為兩線都 > 0。若拿掉「MACD 為正才賣」，Sharpe 掉到 −0.13，所以這個條件很重要。
- 停損：5%～20% 與 2～4 倍 ATR 都測過，彼此差距在雜訊內；10% 是好記的整數。**停損本身在樣本內略為扣分**（與不停損相比，逐檔 Sharpe 差中位數 −0.06），但它限制了單筆虧損，是使用者指定的條件。

**結果**
- 樣本內 2010–2020：CAGR 中位數 1.5%（買進持有 8.8%、定期定額 5.8%）、Sharpe 0.18（0.48）、MDD −39.6%（−46.1%）、持股率 37%、每檔約 33 筆交易、勝率 60%；只有 8% 的股票報酬勝過買進持有。
- **樣本外 2021–2026/10/2**（設計凍結後只跑一次）：CAGR 5.3%（買進持有 35.7%、定期定額 25.3%）、Sharpe 0.47（1.10）、MDD −34.1%（−45.5%）、持股率 33%；**沒有任何一檔**報酬勝過買進持有。
- **交叉驗證（8 折，每折約 2 年、各自重新開始）**：Sharpe 中位數 −0.04 / 0.60 / −0.38 / 0.37 / 0.30 / 0.70 / 0.23 / 0.69（買進持有 0.05 / 0.70 / 0.29 / 0.80 / 0.65 / 0.86 / 0.51 / 1.37），**0/8 折勝過**；最大回撤 8/8 折較小（因為只持有 3～5 成）；只有 2010–11 歐債空頭折 CAGR 較好（−2.1% vs −4.0%）；「股票×折」只有 28% 的 Sharpe 勝過。每折改用其他 7 折最好的參數（36 組）時，每折都選到 15% 停損，結果幾乎一樣（0/8 折）。
- **對照組**（同樣持股率但沒有擇時）：樣本內「固定持股」Sharpe 0.43、「別檔訊號」0.30、「波動目標」0.46，**都比策略的 0.18 好**；樣本外分別為 0.98 / 0.47 / 0.93，策略 0.47 只和「拿別檔股票的訊號來交易」一樣。

**為什麼不好**
- 事件研究：買進訊號後 5 天、20 天的平均超額報酬只有 +0.01%、+0.15%（t 值接近 0），**等於沒有預測力**；賣出訊號後 5 天股價反而平均再多漲 0.08%。每檔每年約 3 次來回、每次成本約 0.585%，成本直接變成虧損。
- 零軸下的反彈在台灣教學中本來就被視為「空頭反彈」；這套規則買在空頭反彈，卻在多頭的第一次小回檔（柱狀體一縮短）就賣出，剛好錯過主升段：2019、2023、2025 年每年落後買進持有 20% 以上。
- 只在明顯空頭年有幫助：2011 年（+9.1 個百分點）、2022 年（+4.5）勝過買進持有。但 2011、2015 的急跌段持股率反而升到 56～59%（平常 37%），因為跌勢中零軸下的反彈訊號特別多，買進後又常被停損；停損出場平均每筆 −12.6%（收盤確認、隔天開盤才賣，常有跳空）。

**注意**：這是照口訣忠實實作的版本，用來回答「這套規則在 0050 成分股上有沒有用」。回測顯示它在各種市況都落後買進持有，**不建議實際使用**。回測使用還原權值日線、手續費 0.1425%、證交稅 0.3%，未計滑價。
"""

DESC_DEADCROSS = """\
**一句話**：買進規則與 `kdj_macd_rebound_turn` 完全相同（零軸下 K 值與柱狀體同天勾頭向上）；只把賣出條件的「MACD 往下」改讀成 **MACD 死亡交叉**（DIF 由上往下跌破 MACD 線），而不是柱狀體一縮短就賣，所以抱得比較久。同樣有 10% 收盤停損。

**規則**（日線，收盤後判斷，隔天開盤成交，只做多，部位 0% 或 100%）
1. 買進（同一根 K 棒）：K 值勾頭向上、柱狀體 OSC 勾頭向上，且 DIF < 0、MACD 線 < 0。
2. 賣出（同一根 K 棒）：K 值勾頭向下（今天 K < 昨天 K，且昨天 K ≥ 前天 K），**且 MACD 死亡交叉**（今天 DIF < MACD 線，昨天 DIF ≥ MACD 線），且 DIF > 0、MACD 線 > 0。
3. 停損：進場價 = 買進訊號隔天開盤價；從進場那天收盤起，收盤 < 進場價 × 0.9 → 隔天開盤賣出（MultiCharts：`if Close < EntryPrice * 0.9 then Sell next bar at market`，不要用 SetStopLoss）。

**參數**：KDJ 9,3,3、MACD 12,26,9、停損 10%，50 檔共用。

**為什麼有這個版本**
- 樣本內研究發現，使用者規則裡**影響最大的是賣出條件的讀法**：柱狀體在多頭裡每年勾頭向下十幾次（每檔每年約 13 次賣出訊號），一縮短就賣會賣在主升段的第一次小回檔；改成 MACD 死亡交叉後，每年只剩約 1.4 次賣出訊號。樣本內 Sharpe 從 0.18 提高到 0.42，持股率從 37% 提高到 76%。
- 「MACD 往下」本來就可以讀成「柱狀體縮短」或「MACD 死叉」兩種意思，兩者都符合口語；這個版本用來呈現「同一套口訣、不同讀法」的差距。

**結果**
- 樣本內 2010–2020：CAGR 中位數 6.4%（買進持有 8.8%、定期定額 5.8%）、Sharpe 0.42（0.48）、MDD −46.4%（−46.1%）、持股率 76%、每檔約 18 筆交易、勝率 50%；18% 的股票報酬勝過買進持有、27% 的股票 Sharpe 勝過。
- **樣本外 2021–2026/10/2**（設計凍結後只跑一次）：CAGR 18.4%（買進持有 35.7%、定期定額 25.3%）、Sharpe 0.80（1.10）、MDD −42.5%（−45.5%）、持股率 73%；12% 的股票報酬勝過買進持有、48% 勝過定期定額。
- 50 檔放在同一個帳戶（可持現金）：樣本外 CAGR 34.4%、MDD −23.7%、Sharpe 1.79，看起來勝過 0050 買進持有（29.3% / −33.8% / 1.28），但**同樣 50 檔等權買進持有**是 48.5% / −30.7% / 1.79 —— 優勢來自「50 檔等權」而不是擇時，而且成分股是事後名單（倖存者偏差）。
- **交叉驗證（8 折）**：Sharpe 中位數 −0.11 / 0.60 / −0.01 / 0.69 / 0.49 / 0.84 / 0.33 / 1.07（買進持有 0.05 / 0.70 / 0.29 / 0.80 / 0.65 / 0.86 / 0.51 / 1.37），**0/8 折勝過**；回撤 7/8 折較小但只少 0.5～5.6 個百分點；38% 的「股票×折」Sharpe 勝過。每折改用其他 7 折最好的參數（36 組）時，都選到「只看 DIF 正負」＋「KD 死叉」的版本（持股率約 80～87%），勝 2/8 折（2012–13、2020–21），只是更接近買進持有。
- **對照組**：樣本內「固定持股」0.46、「別檔訊號」0.48、「波動目標」0.44，策略 0.42 一個都沒贏；樣本外 1.07 / 0.78 / 1.01 vs 0.80。**比較好的成績幾乎全部來自「持有比較多」，不是擇時。**

**注意**
- 約一半的出場是停損（樣本內 48%，平均每筆 −12.7%），訊號出場平均 +20.8%；賣出訊號很少，多數交易靠停損或一直抱著。
- 在 2010–11、2014–15、2022–23 這些弱勢折都沒有比較好（Sharpe −0.11 / −0.01 / 0.33 vs 0.05 / 0.29 / 0.51）：空頭裡零軸下的反彈訊號很多，常常買進後又被停損。
- 回測使用還原權值日線、手續費 0.1425%、證交稅 0.3%，未計滑價。不建議實際使用。
"""

STRATEGIES = [
    {
        "id": "kdj_macd_rebound_turn",
        "label": "KDJ與MACD零軸下同步翻揚（10%停損）",
        "family": "KDJ+MACD 反彈",
        "description": DESC_TURN,
        "multicharts": "multicharts/kdj_macd_rebound_turn.txt",
        "positions": build(TURN),
        "param_grid": GRID_TURN,
        "build": build,
    },
    {
        "id": "kdj_macd_rebound_deadcross",
        "label": "同步翻揚買、MACD死叉才賣（10%停損）",
        "family": "KDJ+MACD 反彈",
        "description": DESC_DEADCROSS,
        "multicharts": "multicharts/kdj_macd_rebound_deadcross.txt",
        "positions": build(DEADCROSS),
        "param_grid": GRID_DEADCROSS,
        "build": build,
    },
]
