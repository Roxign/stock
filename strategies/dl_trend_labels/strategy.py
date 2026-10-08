"""dl_trend_labels: trend-scanning / triple-barrier labels + small causal TCN -> buy/sell state machine.

One published strategy (family 深度學習 2.0). See core.py for the pipeline and RESEARCH.md for the experiments.
"""

from . import core

SID = "dl_trend_labels_tcn"
N_PARAMS_NET = core.n_params(core.MODEL)          # 4,340 per net
N_PARAMS = N_PARAMS_NET * len(core.SEEDS)         # 13,020 in total

_MEMO: dict = {}


def walk_forward(data):
    """Walk-forward predictions memoised on the exact input data (positions + export share one training run)."""
    key = core.fingerprint(data)
    if key not in _MEMO:
        while len(_MEMO) >= 4:                    # keep a few (the lookahead check alternates full / truncated data)
            _MEMO.pop(next(iter(_MEMO)))
        _MEMO[key] = core.walk_forward(data)
    return _MEMO[key]


def positions(data):
    """1.0 after the smoothed P(up barrier first) rises above the median of the model's recent predictions; 0.5
    after the smoothed P(down barrier first) rises above their 80th percentile; at least 20 bars between switches.
    1.0 (buy-and-hold) before the first model (2010-2011) and in a new listing's first 130 bars."""
    return core.positions_from(walk_forward(data), data, **core.DEFAULT_MAP)


def cv_positions(data, folds):
    """Purged k-fold CV (evaluate.py --cv): each fold traded by a model trained on all other folds. Not tradable."""
    return core.cv_positions_from(core.purged_cv(data, folds), data, **core.DEFAULT_MAP)


DESCRIPTION = """
**想法**（文獻回顧 E5）：不預測價格，直接學「買／賣訊號」。標籤用 López de Prado 的**趨勢掃描**（往後 20、30…120 天各做一次
對時間的 OLS，取 |t 值| 最大者的正負號為標籤、|t| 為樣本權重）與**三重障礙**（往後 80 天，±1.5·σ20·√80，先碰上緣 / 都沒碰 /
先碰下緣），小型因果 TCN 同時學兩種標籤，再用狀態機把機率轉成 50% / 100% 部位。

**輸入**（只用到當天收盤為止的資料）：最近 64 天的序列，每天 22 個通道——以波動度標準化的日報酬、隔夜報酬、日內報酬
（KMID）、K 棒形狀 KLEN / KUP / KLOW、量 z 分數、KDJ 的 K/D/J、MACD 的 DIF / OSC、距 5/20/60/120/240 日均線、波動度水準、
以這天收盤為基準的相對價格，以及 **50 檔成分股合成的等權重大盤** 3 個通道（大盤報酬、隔夜報酬、距 60 日線，多股票輸入）；
另有 39 個同源的表格摘要特徵（動能、回撤、量、大盤動能…，也就是 GBDT 對照組的輸入）。

**架構與參數**：1×1 卷積 → 4 層殘差因果卷積（16 通道、kernel 3、膨脹 1/2/4/8，感受野 31 天）→ 最後一天＋平均池化，
並接一層表格特徵分支，輸出 1 個趨勢掃描 logit＋3 類三重障礙 logit。**每個網路 4,340 個參數，3 個種子的集成共 13,020 個**。

**訓練**：每年 1 月 1 日重新訓練（第一個模型 2012 年，用 2008–2011 的資料；2010–2011 沒有模型，部位 = 1.0）。
只用「標籤已完全知道」的樣本：標籤最長看到 t+120 根 K 棒，加 5 根 embargo，樣本要到 t+125 根才可用。
損失 = 加權 BCE（趨勢掃描）+ CE（三重障礙）；AdamW、最多 12 個 epoch，以訓練窗最後 12 個月（再 purge 一次）提前停止；
3 個種子各用不同的 1/3 天數；標準化統計量只來自訓練資料。在 2 個 CPU 執行緒上 50 檔全部算完約 4.5 分鐘。

**部位**：P(先碰上緣) 與 P(先碰下緣) 先做 5 日 EMA；部位 0.5 時若 P(上) > θ_in 就加到 1.0，部位 1.0 時若 P(下) > θ_out
就降到 0.5；兩次切換至少隔 20 根 K 棒。θ_in、θ_out 是該年模型在驗證期預測的第 50、80 百分位（不看標籤，只用訓練窗內的資料）。

**結果（49–50 檔中位數）**：樣本內 2010–2020 CAGR 7.6%、Sharpe 0.45、MDD −44.4%（買進持有 8.8% / 0.48 / −46.1%）；
樣本外 2021–今 30.5% / 1.05 / −40.0%（買進持有 35.7% / 1.10 / −45.5%）。
**對照組**：相同平均曝險的固定部位（B1）兩段都比較好（8.3% / 0.47、32.5% / 1.09）；別檔訊號（B2）樣本內 6.6% / 0.42、
樣本外 31.3% / 1.04，樣本外與本策略無法區分。相同輸入、標籤與狀態機的邏輯斯迴歸（樣本內 Sharpe 0.38、樣本外 1.06）與
HistGradientBoosting（0.40、1.06）樣本內較差、樣本外反而略好；把模型換成「距 120 日線」這一個趨勢特徵、其餘狀態機不變，得到 0.44 / 1.06，
幾乎一樣——模型學到的主要是「低波動＋順勢」。
**交叉驗證（8 折，purged + 125 根 embargo，每折由沒看過該折的模型交易）**：各折 Sharpe 0.07 / 0.51 / 0.19 / 0.71 / 0.71 /
0.71 / 0.44 / 1.29，只有 2/8 折勝過買進持有（0.05 / 0.70 / 0.29 / 0.80 / 0.65 / 0.86 / 0.51 / 1.37），但 8/8 折回撤較小。
**50 檔同一帳戶**：樣本內 12.9% / MDD −26.1% / Sharpe 0.93（等權買進持有 14.2% / −28.8% / 0.89）；
樣本外 42.9% / −21.2% / 1.85（48.5% / −30.7% / 1.79）。

**注意**：(1) 沒有打敗買進持有，也沒有打敗固定曝險對照組；較小的回撤主要來自「平均只持有約 89%」。
(2) 走勢判斷幾乎是整體市場的時機（每天跨股票的 IC ≈ −0.02），各年 AUC 從 0.38 到 0.68 大幅波動。
(3) MultiCharts 版需要每天用 Python 匯出訊號（export_signals.py）當 data2 讀入。詳見 RESEARCH.md。
"""

STRATEGIES = [
    {
        "id": SID,
        "label": "趨勢標籤 TCN 買賣訊號",
        "family": "深度學習 2.0",
        "description": DESCRIPTION.strip(),
        "multicharts": f"multicharts/{SID}.txt",
        "positions": positions,
        "cv_positions": cv_positions,
    }
]
