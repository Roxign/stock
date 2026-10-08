"""dl_revenue_flow: E2 of research/dl_literature.md -- a low-turnover model on Taiwan-specific information
(月營收, 三大法人, 外資持股, 融資融券/借券, 本益比/股價淨值比/殖利率 percentiles, price and market context).

One published strategy (family 深度學習 2.0): HistGradientBoosting (the E2 main model) on all 45 inputs, absolute
20-bar forward return label, one decision per month on the revenue release day. See core.py for the pipeline and
RESEARCH.md for the experiments (MLP with stock embedding, ridge, rules, label variants, cross-sectional inputs).
"""

from . import core

SID = "dl_revenue_flow_gbdt"
KIND = "gbdt"
FLOOR = 0.5          # exposure when the forecast is <= 0 (1.0 when > 0)
TOP_K, BUFFER = 20, 10
# fitted "parameters" of the latest (2026) model: 6,384 split thresholds + 6,840 leaf values over 456 trees, 3 seeds
# together (HistGradientBoosting has no weights in the neural-network sense)
N_PARAMS = 13_224

_MEMO: dict = {}


def forecasts(data):
    """Walk-forward forecasts (one model per calendar year from 2012, expanding window), memoised on the exact input
    data so positions(), weights() and export_signals share one training run."""
    key = core.fingerprint(data)
    if key not in _MEMO:
        _MEMO.clear()
        stocks = core.cached_prepare(data)
        pred, _ = core.walk_forward(stocks, KIND, core.ALL, core.SEEDS)
        _MEMO[key] = (stocks, pred)
    return _MEMO[key]


def positions(data):
    """Per stock: 1.0 when the latest revenue-day forecast of the next-20-bar return is > 0, else 0.5; held until the
    next revenue release day (one decision per month). 1.0 (buy-and-hold) before the first model (2010-2011)."""
    stocks, pred = forecasts(data)
    return core.abs_positions(stocks, pred, floor=FLOOR, thr=0.0, fallback=1.0)


def weights(data):
    """Single account: on each revenue release day hold the 20 stocks with the highest forecast (a held stock stays
    while its rank is <= 30), 1/20 of equity each. Equal weight over the listed stocks before 2012."""
    stocks, pred = forecasts(data)
    return core.topk_weights(stocks, pred, TOP_K, BUFFER)


def cv_positions(data, folds):
    """Purged k-fold CV (evaluate.py --cv): each fold traded by a model trained on all other folds, purging training
    samples whose label window (+5-bar embargo) overlaps the fold and the 25 bars after it. Not tradable."""
    stocks = core.cached_prepare(data)
    pred, _, _ = core.purged_cv(stocks, folds, KIND, core.ALL, core.SEEDS)
    return core.abs_positions(stocks, pred, floor=FLOOR, thr=0.0, fallback=1.0)


DESCRIPTION = """
**月營收＋籌碼的低換手模型（研究計畫 E2）。** 每月只在「營收可用日」（次月 10 日起第一個交易日再加 1 個交易日）決策一次，
預測未來 20 個交易日的**絕對**報酬，持有到下一次營收公布。

**輸入（41 個，皆為 t+1 開盤前已公布）：** 月營收（單月／3 個月／12 個月年增、加速度、季節調整月增、是否創 12 個月／歷史新高、
營收月份新舊）；外資／投信／自營商淨買超 ÷ 成交量（5／20／60 日，2012-05 起）；外資持股比變化；融資餘額變化與 z 分數、券資比、
借券餘額變化與 z 分數；本益比／股價淨值比／殖利率的自身 5 年百分位；20–250 日報酬、波動、距 52 週高點、量比；
大盤（加權報酬指數、外資大盤買賣超、VIX、SOX、大盤融資、50 檔等權報酬）；營收年增、60 日報酬、外資買超的當日橫斷面排名。

**標籤／輸出：** t+1 開盤到 t+21 開盤的對數報酬（迴歸）。

**模型：** HistGradientBoosting（15 葉、學習率 0.03、每葉至少 400 筆），50 檔合併訓練；3 個種子（各用不同的 1/3 交易日）平均。
2026 年模型共 456 棵樹、13,224 個門檻與葉值（不是神經網路權重）。對照的小型 MLP（43 維＋4 維股票嵌入 → 32 → 16 → 1，
每個種子 2,281 個參數）與脊迴歸都比較差，見 RESEARCH.md。

**訓練：** 每年 1 月重訓（2012 起、擴張視窗），只用標籤結束＋5 日 embargo 早於該年的樣本，最後 12 個月當驗證尾段決定樹的數量；
標準化與截尾只用訓練資料。2010–2011 沒有模型 → 買進持有／等權。

**部位：** 逐檔：預測 > 0 → 100%，否則 50%。投組版（`weights`）：持有預測前 20 名（已持有者排名 30 內續抱），各占總資金 1/20。

**結果（2010–2020 樣本內／2021–2026/10 樣本外）：**
- 逐檔 Sharpe 中位數 0.46／1.08，買進持有 0.48／1.10、固定曝險對照 B1 0.47／1.10、別檔訊號 B2 0.45／1.05、波動目標 B3 0.44／1.08
  → **逐檔擇時沒有價值**（平均曝險 95–97%，幾乎等於買進持有）。
- 投組版 CAGR／Sharpe 16.1%／0.93 → 58.9%／1.91，50 檔等權 14.2%／0.89 → 48.5%／1.79，0050 買進持有 10.4%／0.71 → 29.3%／1.28，
  0050 定期定額 7.9%／0.81 → 22.0%／1.37；樣本外 MDD −33.2%（等權 −30.7%），年換手約 2 倍。
- **但交叉驗證不支持投組版的優勢：** 8 折 purged CV 平均 Sharpe 比等權低 0.15，只贏 3/8 折；單一種子之間樣本外 Sharpe 標準差 0.20。
  樣本外的超額報酬集中在 2024 年後的 AI 硬體股（奇鋐、創意、緯穎⋯）。
- 逐檔 8 折 CV：Sharpe 只在 1–2/8 折贏買進持有。

**注意：** 模型最常用的是大盤與動能特徵，月營收／法人資料的增量很小（樣本內橫斷面 IC ≈ 0）；股票池是今天的 0050 成分股（存活者偏差）；
本資料夾約 170 組設定都記在試驗登記表。MultiCharts 需用 `export_signals.py` 匯出的訊號檔當 data2。
"""

STRATEGIES = [
    {
        "id": SID,
        "label": "月營收＋籌碼 GBDT（月頻）",
        "family": "深度學習 2.0",
        "description": DESCRIPTION.strip(),
        "multicharts": f"multicharts/{SID}.txt",
        "positions": positions,
        "weights": weights,
        "cv_positions": cv_positions,
    }
]
