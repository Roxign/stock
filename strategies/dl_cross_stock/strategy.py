"""Published strategy of the E4 cross-stock attention experiment (see RESEARCH.md and core.py).

Frozen design (chosen on in-sample walk-forward 2012-2020 and in-sample purged folds f1-f5 only):
  model    core.CrossStockNet(kind="attn", skip=True, dropout 0.3), AdamW wd 0.05, lr 1e-3, batch 32 dates,
           <= 30 epochs, early stopping (patience 5) on a purged 500-day validation tail, ListNet aux weight 0.5,
           3 seeds averaged, retrained every 1 January on all earlier data (first model 2012, trained on 2008-2011)
  positions  EMA(20) of the predicted vol-normalised 20-day return -> want = z / (2 * sigma * sqrt(20)) clipped to
           [0, 1] -> 25 % steps with a 0.5 no-trade band; before the first model (2008-2011) fully invested
  weights  first bar of each month: the 10 stocks with the best EMA(20) prediction (> 0), 10 % each, held while still
           in the top 20; empty slots are cash
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import core

KIND = "attn"
TRAIN_CFG = {"model": {"skip": True, "drop": 0.3}, "wd": 0.05, "val_days": 500}
TOP_K = 10
CV_WARMUP = 60          # bars before each CV fold predicted by that fold's model only to warm up the EMA / band

_PRED: dict = {}


def predictions(data):
    """Walk-forward (yearly retrained) 3-seed ensemble predictions on the panel of `data`; memoised per data."""
    P = core.build_panel(data)
    if P["key"] not in _PRED:
        preds, _ = core.walk_forward(data, KIND, cfg=TRAIN_CFG)
        _PRED.clear()
        _PRED[P["key"]] = core.ensemble(preds)
    return P, _PRED[P["key"]]


def positions(data):
    P, pred = predictions(data)
    return core.positions_from(P, pred, data)


def weights(data):
    P, pred = predictions(data)
    return core.weights_from(P, pred, data, k=TOP_K)


def cv_fold_predictions(data, folds, kind=KIND, cfg=TRAIN_CFG, seeds=core.SEEDS):
    """Purged blocked k-fold. For each fold a model (3 seeds) is trained on every date whose label window ends before
    the fold or that starts more than H bars after it (embargo), never on the fold itself; it predicts the fold plus
    CV_WARMUP bars before it (only used to warm up the EMA and the no-trade band, so a fold's positions depend on no
    other fold's model). Returns {fold: (lo_row, f0_row, f1_row, pred (T, N) with NaN outside lo..f1)}."""
    P = core.build_panel(data)
    dates = P["dates"]
    T = len(dates)
    ye = core.label_end(P, np.arange(T))
    has = (P["AV"] & ~np.isnan(P["Y"])).sum(axis=1) >= 5
    rows = np.arange(T)
    out = {}
    for i, (f, (a, b)) in enumerate(folds.items()):
        f0 = int(dates.searchsorted(pd.Timestamp(a)))
        f1 = T if b is None else int(dates.searchsorted(pd.Timestamp(b), side="right"))
        if f0 >= T or f1 <= f0:
            continue
        ok = has & ((ye < f0) | (rows > f1 - 1 + core.H)) & ((rows < f0) | (rows >= f1))
        tr, va = core.split_rows(P, rows[ok], (core.DEFAULT_CFG | cfg)["val_days"])
        lo = 0 if i == 0 else max(f0 - CV_WARMUP, 0)
        ps = []
        for s in seeds:
            fitted = core.fit(P, kind, tr, va, s, cfg)
            ps.append(core.predict(P, fitted, np.arange(lo, f1)))
        pred = np.full((T, core.N), np.nan, np.float32)
        pred[lo:f1] = core.ensemble(dict(enumerate(ps)))
        out[f] = (lo, f0, f1, pred)
    return P, out


def cv_positions(data, folds):
    P, fp = cv_fold_predictions(data, folds)
    dates = P["dates"]
    parts = {code: [] for code in data}
    for f, (lo, f0, f1, pred) in fp.items():
        pos = core.positions_from(P, pred, data)
        d0, d1 = dates[0 if lo == 0 else f0], dates[f1 - 1]
        for code in data:
            parts[code].append(pos[code].loc[d0:d1])
    out = {}
    for code, df in data.items():
        s = pd.concat(parts[code]) if parts[code] else pd.Series(dtype=float)
        s = s[~s.index.duplicated()]
        out[code] = s.reindex(df.index).fillna(core.POS_CFG["fallback"])
    return out


DESCRIPTION = r"""
**跨股注意力模型（E4，深度學習 2.0）**：一個模型同時看 50 檔成分股，同一天的 50 檔彼此「互相參考」，預測每一檔未來 20 個交易日的**絕對**（不是相對排名）波動標準化報酬，再換算成每檔的持股比例；組合版本則每月挑預測最好的 10 檔。

**輸入（每天、每檔，全部在下一個台股開盤前可得）**
- 20 日視窗（每天 4 個）：隔夜跳空、日內報酬、高低振幅（都除以 60 日 EWMA 波動）、量比（對 60 日均量取 log）。
- 21 個快照特徵：5/20/60/120/250 日波動標準化動能、三組 MACD（8/24、16/48、32/96）、20/60 日累積隔夜與日內報酬、年化波動、波動比、距 52 週高點、5 日量比、外資持股比 20/60 日變化、融資餘額占股本 20 日變化、月營收年增率（法定公布期限後才使用）、股價淨值比 3 年 z 分數。
- 18 個市場特徵（閘門與市場 token）：50 檔等權報酬 5/20/60 日與波動、站上 60 日線比例、橫斷面離散度、費半 1/5/20 日、台積電 ADR 1/5 日、VIX 與其 20 日變化、美元/台幣 20 日、美債 10 年 20 日變化、外資買賣超占成交值、成交值變化、選擇權 P/C 比。
- 股票代號嵌入（8 維）與產業嵌入（7 類，4 維）。

**標籤 / 輸出**：y = ln(第 t+21 天開盤 / 第 t+1 天開盤) ÷ (σ_t·√20)（σ 為 60 日 EWMA 日波動），截斷在 ±3；損失 = MSE ＋ 0.5 × ListNet 排名輔助損失。

**架構（每個種子 15,317 個可訓練參數，3 個種子共 45,951 個）**：市場向量經 sigmoid 閘門逐欄縮放每檔輸入（MASTER 簡化版）→ 共用的兩層 MLP 編碼器（20 日視窗攤平 + 快照 + 嵌入 → 32 → 32）＋市場 token → **一層 4 頭跨股自注意力**（同一天的 50 檔加上市場 token，沒有資料的股票被遮罩；殘差乘上從 0 開始學習的係數，ReZero）→ 前饋層 → 共用輸出頭，另有一條從原始輸入到輸出的線性旁路。

**訓練**：每年 1 月 1 日用之前所有資料重新訓練（擴張視窗；第一個模型 2012 年，用 2008–2011 資料）；標籤視窗與測試期重疊的樣本全部剔除（purge）並加 5 根 K 棒 embargo；最後 500 個交易日（同樣 purge）當驗證集做提前停止；標準化統計量只用訓練資料；AdamW（lr 1e-3、weight decay 0.05、dropout 0.3），3 個種子取平均。只用 CPU，50 檔全部部位約 4–5 分鐘。

**部位對應**：預測值取 20 日 EMA 平滑 → 均異數式目標 z ÷ (2·σ·√20)，夾在 0～1 → 以 25% 為一級、0.5 的不交易帶（目標與現有部位相差至少 0.5 才調整）。2008–2011 年還沒有模型時全額持有。**組合版（weights）**：每月第一個交易日持有預測最好（且 > 0）的 10 檔、每檔 10%，已持有的股票只要還在前 20 名就保留，空位放現金。

**結果（50 檔中位數；完整表格見 RESEARCH.md）**

| 期間 | 策略 Sharpe / CAGR / MDD | 買進持有 | B1 固定曝險 | B2 別檔訊號 | B3 波動目標 |
|---|---|---|---|---|---|
| 2012–2020 walk-forward | 0.66 / 8.7% / -31% | 0.59 / 11.0% / -44% | 0.56 | 0.61 | 0.52 |
| 樣本外 2021–2026/10 | **0.64 / 9.0% / -30%** | **1.10 / 35.7% / -46%** | 1.00 | 0.71 | 0.91 |

- 同一組輸入的對照（2012–2020 每檔 Sharpe）：**嶺迴歸 0.66、關掉注意力 0.61、打亂股票身分 0.66**、扁平單層網路 0.44、HistGradientBoosting 0.52、GBDT＋人工同業特徵 0.52——跨股注意力本身沒有可量測的增益，可預測的部分幾乎是線性的市場擇時，而且在樣本外消失（市場 IC 0.16 → -0.01）。
- 8 折 purged 交叉驗證：Sharpe 只贏買進持有 4/8 折、報酬 3/8 折、回撤較小 7/8 折。
- 單一帳戶組合（前 10 名各 10%）：樣本外 Sharpe 1.69、CAGR 23.2%、MDD -14.1%，對比 50 檔等權買進持有 1.79 / 48.5% / -30.7%、0050 買進持有 1.28 / 29.3% / -33.8%：回撤只有一半，但報酬少很多。橫斷面排名能力（rank IC 約 0.05）是樣本外唯一留下來的東西。

**注意事項**：**沒有打敗控制組，也遠輸買進持有**。模型在 2013–2014 與 2024–2026 的多頭幾乎空手（平均持股 0–34%），2022 空頭反而持有 68%。50 檔的橫斷面太小，注意力學不到穩定的跨股關係。需要外部資料（美股、匯率、籌碼、月營收），MultiCharts 版本以 Python 匯出的每日目標曝險當 data2 交易（multicharts/ 檔頭有步驟）。已通過截斷前視檢查（0 根 K 棒不同）。
"""

STRATEGIES = [
    {
        "id": "dl_cross_stock_attn",
        "label": "跨股注意力（50 檔共同模型）",
        "family": "深度學習 2.0",
        "description": DESCRIPTION,
        "multicharts": "multicharts/dl_cross_stock_attn.txt",
        "positions": positions,
        "weights": weights,
        "cv_positions": cv_positions,
    }
]
