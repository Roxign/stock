"""KDJ + MACD (+ Taiwan MA lines) + small deep-learning (MLP) strategies. See RESEARCH.md for the experiments."""

import numpy as np
import pandas as pd

from . import core

N_IN = len(core.feature_names())               # 48
PARAMS_PER_NET = core.n_params(N_IN)
PARAMS_TOTAL = PARAMS_PER_NET * len(core.SEEDS)


def _ok(sig):
    return sig["ps"].notna() & sig["q_lo"].notna()


def floor_from(signals):
    out = {}
    for code, sig in signals.items():
        base = sig["base"]
        strong = sig["ps"] >= sig["q_hi"]
        pos = np.where((base == 1) | strong, 1.0, 0.5)
        fallback = 0.5 + 0.5 * base  # before the first model (2010-2011): rule only
        out[code] = pd.Series(np.where(_ok(sig), pos, fallback), index=sig.index)
    return out


def dl_floor(data):
    """0.5 core position; 1.0 when the weekly KDJ+MACD rule is long OR the MLP rates the stock relatively strong."""
    return floor_from(core.cached_walk_forward(data)[0])


def cv_floor(data, folds):
    """Purged k-fold CV of dl_floor (evaluate.py --cv): each fold traded with a model trained without it. Not tradable."""
    return floor_from(core.cached_purged_cv(data, folds)[0])


_RULE = (
    "**基礎 KDJ+MACD 規則**（週線等效參數 KDJ(45,15,15)、MACD(60,130,45)，J = 3K − 2D）：\n"
    "週線等效 J 由 0 以下回到 0 以上（離開超賣區）→ 持有；J 由 100 以上跌回 100 以下（離開超買區）→ 出場，"
    "但若當時 MACD 柱狀體 OSC > 0 且仍在放大（漲勢仍在加速）則續抱。\n"
)

_COMMON = f"""
**深度學習模型（兩個 DL 版本共用同一個模型）**

- **輸入特徵（{N_IN} 個）**：
  - KDJ：日 KDJ(9,3,3) 的 K、D、J（最近 5 根）；週線等效 KDJ(45,15,15) 的 K、D、J（今天與 5 根前）。K、D、J 縮放成 x/100 − 0.5。
  - MACD：日 MACD(12,26,9) 的 DIF、OSC（最近 5 根）與週線等效 MACD(60,130,45)（今天與 5 根前），全部除以 ATR(14)，跨股票可比、不受還原權值影響。
  - 均線（台灣常用的週 5、雙週 10、月 20、季 60、半年 120、年 240 日線）：(收盤 − 均線)/ATR、20/60/120/240 日線的 5 日變化/ATR、
    均線排列分數（相鄰兩條均線短在長之上 +1、之下 −1，五組平均）與「完全多頭排列」「完全空頭排列」旗標。
  - 標準化用的平均數/標準差只取自當年訓練集，截斷在 ±5 個標準差。
- **架構**：多層感知器 MLP {N_IN} → 16 → 8 → 1（ReLU、sigmoid），每個網路 {PARAMS_PER_NET:,} 個參數；3 個不同種子的網路取平均，合計 **{PARAMS_TOTAL:,} 個參數**（遠低於 1 萬）。
- **標籤（橫斷面相對強弱）**：未來 60 個交易日（次日開盤到 61 日後開盤）的報酬是否勝過同一天 50 檔成分股的中位數。
- **前推式訓練**：每年 1 月重新訓練（2012–2026 共 15 個模型），50 檔合併、擴張視窗；只用「標籤區間在當年 1/1 前已全部結束」的樣本（purge），
  訓練完全確定性（固定種子、固定順序、`torch` 確定性運算）。第一個模型在 2012 年啟用；**2010–2011 沒有模型時照基礎規則操作**（底倉版為 0.5 + 0.5×規則）。
- **輸出轉部位**：3 個機率平均 → 10 日 EMA → 與「該模型在訓練集上預測值的第 20 / 40 百分位」比較：≥ 第 40 百分位 = 相對強，< 第 20 百分位 = 相對弱。只輸出 0 / 0.5 / 1。

{_RULE}
**注意事項（請務必讀）**
- **DL 訊號很弱，而且樣本外變得更弱**：前推式 AUC 樣本內 0.534、樣本外只有 0.511（0.5 = 亂猜；2021、2022、2024 三年低於 0.5）。
  同樣特徵的邏輯斯迴歸樣本外 0.521、不含均線的 KDJ+MACD 網路 0.523，都比發佈的模型好。
- **J 線沒有帶來可測量的資訊**：J = 3K − 2D 是 K、D 的線性組合，網路第一層本來就能自己算出來；有/無 J 的 AUC 差距在 ±0.002 以內（等於不同隨機種子的差距）。
  使用 J 是依照使用者指定的設計，不是因為它提高了預測力。
- **均線特徵的樣本內小幅進步在樣本外消失**：以 ATR 為單位的均線特徵讓樣本內 AUC 從 0.532 升到 0.534，樣本外反而從 0.523 降到 0.511；以百分比表示的均線特徵連樣本內都變差。
- **DL 對績效的貢獻無法和「多持有一點」區分**：樣本外，把 MLP 訊號換成「別檔股票的訊號」（統計性質相同、與本檔無關），底倉版 CAGR 30.1–31.2%、Sharpe 1.06–1.07，和真訊號（31.4%、1.06）一樣。
- 成分股是「今天」的 0050 成分股（存活者偏差），使所有擇時策略在回測中都顯得吃虧，也讓相對強弱標籤偏樂觀。
- 研究（只用樣本內）約測了 60 組基礎規則設定、57 次前推式模型訓練、22 組策略回測，有資料探勘偏誤的風險；詳見 RESEARCH.md。
- MultiCharts 版本內嵌的是最新一年（2026，訓練到 2025 年底）的凍結模型，要更新模型需重新執行 `gen_multicharts.py`。
"""

_RESULTS = """
**績效（50 檔中位數；樣本內 49 檔）**

| | 樣本內 CAGR | MDD | Sharpe | 樣本外 CAGR | MDD | Sharpe | 樣本外持股比 |
|---|---|---|---|---|---|---|---|
| 買進持有 | 9.6% | −46.1% | 0.49 | 35.7% | −45.5% | 1.10 | 100% |
| 定期定額 | 5.8% | −35.8% | 0.49 | 25.3% | −35.4% | 1.04 | – |
| 半倉底倉 floor | 9.0% | −43.3% | 0.52 | 31.4% | −43.3% | 1.06 | 90.8% |
| 三段部位 gate | 9.5% | −43.5% | 0.49 | 29.5% | −42.5% | 1.04 | 87.1% |
| 基礎規則 base | 5.5% | −40.2% | 0.40 | 11.2% | −34.6% | 0.66 | 39.8% |
"""

STRATEGIES = [
    {
        "id": "kdj_macd_dl_floor",
        "label": "KDJ+MACD+MLP 半倉底倉",
        "family": "KDJ+MACD+深度學習",
        "description": (
            "**規則**：永遠至少持有 0.5 部位（底倉）。當週線等效 KDJ+MACD 規則為持有，**或** MLP 判斷該股相對強（機率 ≥ 訓練集第 40 百分位）時加到 1.0；"
            "兩者都不支持時減回 0.5。不會完全出場，所以『完整交易』只有 1 筆；加減碼次數請看『部位調整次數』（樣本外中位數 26 次，約每年 4–5 次）。\n\n"
            "**想法**：在長期上漲的台股權值股中，完全空手的機會成本很高；保留一半底倉，只在『技術面不支持且相對弱勢』時減碼，用來降低回撤。\n"
            "結果：樣本內 Sharpe 0.52（買進持有 0.49），樣本外 CAGR 31.4% 低於買進持有 35.7%、MDD 少 2 個百分點、Sharpe 1.06 對 1.10。\n\n"
            "**交叉驗證（8 折，purged）**：每一折改由「其他 7 折 + 2008–09 年」訓練的同一個模型交易（剔除標籤區間與該折重疊的樣本，折後再空出 60 根 K 棒），部位對應不變。"
            "Sharpe 只在 2022–23 勝過買進持有（0.53 對 0.51）、2010–11 平手，其他 6 折落後，多頭折落後最多；MDD 5/8 折較小，但同持股比的固定曝險與別檔訊號控制組回撤一樣小。"
            "對別檔訊號控制組只勝 4/8 折、對沒有模型的「規則 + 底倉」2/8 折——看不出模型在任何市場狀態有穩定加值。"
            "這是穩健性檢查（前面幾折的模型用到未來資料訓練），不是可交易的績效。\n"
            + _RESULTS + _COMMON
        ),
        "multicharts": "multicharts/kdj_macd_dl_floor.txt",
        "positions": dl_floor,
        "cv_positions": cv_floor,
    },
]
