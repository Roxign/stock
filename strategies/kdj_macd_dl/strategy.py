"""KD + MACD + small deep-learning (MLP) strategies. See RESEARCH.md for the experiments behind these choices."""

import numpy as np
import pandas as pd

from . import core

N_IN = 28
PARAMS_PER_NET = core.n_params(N_IN)          # 28*16+16 + 16*8+8 + 8*1+1 = 609
PARAMS_TOTAL = PARAMS_PER_NET * len(core.SEEDS)  # 3-net ensemble = 1,827


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


def gate_from(signals):
    out = {}
    for code, sig in signals.items():
        base = sig["base"]
        strong = sig["ps"] >= sig["q_hi"]
        weak = sig["ps"] < sig["q_lo"]
        pos = np.where((base == 1) | strong, 1.0, np.where(weak, 0.0, 0.5))
        out[code] = pd.Series(np.where(_ok(sig), pos, base), index=sig.index)  # fallback: rule only
    return out


def dl_floor(data):
    """0.5 core position; 1.0 when weekly KD+MACD is bullish OR the MLP rates the stock relatively strong."""
    return floor_from(core.cached_walk_forward(data)[0])


def dl_gate(data):
    """1.0 when weekly KD+MACD bullish or MLP strong; 0 when KD+MACD bearish AND MLP weak; else 0.5."""
    return gate_from(core.cached_walk_forward(data)[0])


def base_rule(data):
    """The KD+MACD rule both DL variants build on, without any model (for the DL-vs-rule comparison)."""
    return {code: pd.Series(core.base_rule(core.indicator_frame(df)), index=df.index) for code, df in data.items()}


_COMMON = f"""
**深度學習模型（兩個 DL 版本共用同一個模型）**

- **輸入特徵（28 個，全部來自 KD 與 MACD）**：日 KD(9,3,3) 的 K、D（最近 5 根）；日 MACD(12,26,9) 的 DIF、OSC 除以 ATR(14)（最近 5 根）；
  「週線等效」KD(45,15,15) 的 K、D 與 MACD(60,130,45) 的 DIF、OSC/ATR（今天與 5 根前）。KD 縮放到 −0.5~0.5，MACD 除以 ATR 所以跨股票可比、不受價格還原影響。
  標準化用的平均數/標準差只取自當年訓練集。
- **架構**：多層感知器 MLP 28 → 16 → 8 → 1（ReLU、sigmoid），每個網路 {PARAMS_PER_NET} 個參數；3 個不同種子的網路取平均，合計 **{PARAMS_TOTAL:,} 個參數**（遠低於 1 萬）。
- **標籤（橫斷面相對強弱）**：預測「未來 60 個交易日（次日開盤到 61 日後開盤）的報酬，是否勝過同一天 50 檔成分股的中位數」。
  研究發現用「絕對報酬是否為正」當標籤時，KD/MACD 模型在樣本內逐年 AUC 在 0.45~0.56 間亂跳、毫無穩定性；改成相對強弱後整體 AUC 約 0.53，是唯一穩定的訊號。
- **前推式訓練（walk-forward）**：每年 1 月重新訓練一次，只使用「標籤區間在當年 1/1 前已全部結束」的樣本（purge），50 檔股票合併訓練、擴張視窗（2010 起）。
  第一個模型在 2012 年啟用（用 2010-08~2011 的資料）；**2010–2011 沒有模型時，直接照基礎 KD+MACD 規則操作**（半倉底倉版為 0.5 + 0.5×規則）。
- **輸出轉成部位**：3 個網路的機率平均後取 10 日 EMA 平滑，與「該模型在訓練集上預測值的第 20 / 40 百分位」比較：
  ≥ 第 40 百分位 =「相對強」，< 第 20 百分位 =「相對弱」。只輸出 0 / 0.5 / 1，避免連續輸出每天微調付手續費。

**基礎 KD+MACD 規則**（週線等效參數）：K>D 且 OSC>0 時轉多；K<D 且 OSC<0 時轉空；其餘時間維持原狀態。

**注意事項（請務必讀）**
- **DL 的訊號很弱**：前推式 AUC 樣本內 0.531、樣本外 0.518（0.5 = 亂猜），2012–2026 共 15 年中有 13 年 > 0.5，訊號是真的但很小。
- **DL 對績效的貢獻無法和「多持有一點」區分**：消融實驗中，把 MLP 換成邏輯斯迴歸、或把「別檔股票的 MLP 訊號」套到這檔（訊號統計性質相同但與本檔 KD/MACD 無關），
  夏普值與 MLP 版本相差不超過 0.05（樣本內與樣本外皆然）；沒有任何模型的「規則 + 半倉底倉」也和半倉底倉版差不多。相對基礎規則的進步，主要來自部位更常保持在場內，而不是 DL 的預測能力；詳見 RESEARCH.md。
- 成分股是「今天」的 0050 成分股，等於事先挑好了過去 15 年的贏家（存活者偏差），使所有擇時策略在回測中都顯得吃虧，也讓相對強弱標籤偏樂觀。
- 研究過程（只用樣本內）約跑了 60 組前推式模型設定、250 組回測；最後設定以「對鄰近設定穩健」為準，但仍有資料探勘偏誤的風險。
- MultiCharts 版本內嵌的是最新一年（訓練到前一年底）的凍結模型，要更新模型需重新執行 `gen_multicharts.py`。
"""

STRATEGIES = [
    {
        "id": "kd_macd_dl_floor",
        "label": "KD+MACD+MLP 半倉底倉",
        "family": "KD+MACD+深度學習",
        "description": (
            "**規則**：永遠至少持有 0.5 部位（底倉）。當週線等效 KD+MACD 規則為多頭，**或** MLP 判斷該股相對強（機率 ≥ 訓練集第 40 百分位）時加到 1.0；"
            "兩者都不支持時減回 0.5。不會完全出場，所以『完整交易』只有 1 筆；加減碼次數請看『部位調整次數』。\n\n"
            "**想法**：在長期上漲的台股權值股中，完全空手的機會成本很高；保留一半底倉，只在『技術面轉空且相對弱勢』時減碼，用來降低回撤。\n"
            + _COMMON
        ),
        "multicharts": "multicharts/kd_macd_dl_floor.txt",
        "positions": dl_floor,
    },
    {
        "id": "kd_macd_dl_gate",
        "label": "KD+MACD+MLP 三段部位",
        "family": "KD+MACD+深度學習",
        "description": (
            "**規則**：週線等效 KD+MACD 規則為多頭，**或** MLP 判斷相對強（≥ 第 40 百分位）→ 部位 1.0；"
            "KD+MACD 為空頭 **且** MLP 判斷相對弱（< 第 20 百分位）→ 部位 0（空手）；其他情況 → 0.5。\n\n"
            "**想法**：MLP 扮演『否決賣訊』的角色——KD+MACD 轉空時，只有相對弱勢的股票才真正出場，相對強勢的股票續抱。\n"
            + _COMMON
        ),
        "multicharts": "multicharts/kd_macd_dl_gate.txt",
        "positions": dl_gate,
    },
    {
        "id": "kd_macd_dl_base",
        "label": "週KD+MACD 基礎規則（無 DL）",
        "family": "KD+MACD+深度學習",
        "description": (
            "上面兩個深度學習版本所依附的基礎規則，**沒有任何模型**，用來對照 DL 到底有沒有加分。\n\n"
            "**規則**：使用『週線等效』參數（日線資料上把 KD 9,3,3 與 MACD 12,26,9 的參數都乘以 5）：KD(45,15,15)、MACD(60,130,45)。"
            "K>D 且 OSC(DIF−MACD)>0 時隔日開盤全數買進；K<D 且 OSC<0 時隔日開盤全數賣出；其餘時間維持原狀態。\n\n"
            "**為什麼選這組**：樣本內測試過日線（×1）到 ×5 的 KD/MACD 組合，日線版本交易過於頻繁（每年約 10 次），"
            "手續費加交易稅讓年化報酬變成負的；放慢到週線等效後交易次數降到約每年 2 次。"
            "參數倍數不是用報酬最佳化挑的，而是取『週線』這個常見、好解釋的設定。\n\n"
            "**注意**：單獨使用時在樣本內明顯輸給買進持有（台股權值股長期上漲，空手成本高），它的用途是當作 DL 版本的比較基準。"
        ),
        "multicharts": "multicharts/kd_macd_dl_base.txt",
        "positions": base_rule,
    },
]
