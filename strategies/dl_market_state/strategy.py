"""dl_market_state: one market-level model sets a common exposure g_t for all 50 stocks (E3 in
research/dl_literature.md). Research notes: RESEARCH.md; model code: core.py; in-sample experiments: experiments.py;
final (frozen) evaluation tables: final_eval.py.
"""

from . import core

SID = "dl_market_state_mlp"
KIND = "mlp"
# frozen after the in-sample study (RESEARCH.md section 5): tiny utility MLP, 39 inputs -> 16 tanh -> 1 sigmoid,
# 5 seeds averaged, initialised at the unconditional optimum, H = 20-day label, floor 50 %, EMA 10, 0.25 grid
CFG = dict(hidden=(16,), gamma=3.0, H=20, prior_init=True)
FSET = "all_fut"
SMOOTH, FLOOR = 10, 0.5
N_INPUTS = len(core.feature_names(FSET))
N_PARAMS = core.n_params(N_INPUTS, CFG["hidden"]) * 5  # 5 seed nets per yearly model


def common_exposure(data):
    F, aux = core.cached_frame(data)
    raw, _, _ = core.walk_forward(F, aux, KIND, CFG, FSET)
    return core.exposure_path(raw, smooth=SMOOTH, floor=FLOOR)


def positions(data):
    return core.to_positions(common_exposure(data), data)


def cv_positions(data, folds):
    F, aux = core.cached_frame(data)
    paths = core.purged_cv(F, aux, KIND, CFG, folds, FSET)
    g = core.cv_exposure(paths, folds, F.index, smooth=SMOOTH, floor=FLOOR)
    return core.to_positions(g, data)


DESCRIPTION = """**大盤狀態 MLP：一個模型決定 50 檔共用的曝險 g_t（E3）。誠實結論：沒有勝過買進持有，也沒有勝過常數曝險對照；2020 年後模型一直輸出 100%，樣本外與買進持有完全相同。**

**輸入（39 個大盤層級特徵，t 收盤後計算，只用 t+1 開盤前已公布的資料）：**
- 價格（14）：50 檔等權指數 5/20/60/120 日報酬（波動標準化）、20 日波動、波動比、距 240 日高點回撤、站上 60/120/240 日線的家數比例（廣度）、橫斷面離散度、60 日平均相關性、大盤成交值變化、加權報酬指數相對等權指數。
- 籌碼（7）：外資／投信／自營商淨買超（除以成交值）、融資餘額變化、融資／成交值與融券／融資比（250 日 z 分數）。
- 選擇權（2）：put/call 未平倉比與成交量比（250 日 z 分數）。
- 全球（12）：SOX、S&P 500、VIX、美債 10 年與利差、美元指數、新台幣、日經、KOSPI（美股用前一晚收盤）。
- 外資台指期／選擇權未平倉（4）：**2018-06 起才有**，之前缺值＝訓練平均，另加「有資料」旗標。

**標籤／輸出：** 直接輸出曝險 g ∈ [0,1]（sigmoid），以二次效用損失 −mean(g·R − 1.5·g²·R²) 訓練，R＝t+1 開盤起 20 日的等權大盤報酬。

**架構與參數：** MLP 39 → 16（tanh）→ 1，每個 657 個參數，5 個種子取平均，合計 **3,285 個可訓練參數**。最後一層初始化在訓練集的無條件最佳曝險（沒學到東西＝常數曝險）。

**訓練：** 每年 1 月 1 日用當時已完全實現的標籤重訓（擴張窗，2008-07 起；第一個模型 2012 年，2010–2011 固定 100%）；標準化只用訓練資料；早停用訓練窗最後 252 根（之前 25 根剔除）；AdamW、輸入雜訊、weight decay。

**部位映射：** EMA(10) 平滑 → 下限 50%（0.5 + 0.5·g）→ 0.25 一格量化並加遲滯帶，所以只會是 50%／75%／100%；所有股票同一個曝險，下一根開盤成交。

**結果（個股 Sharpe 中位數）：** 樣本內 2010–2020：0.43（CAGR 7.5%、MDD −46.9%、平均曝險 89%）vs 買進持有 0.48；B1 常數曝險 0.47、B3 波動目標 0.43。樣本外 2021–：1.10／35.7%／−45.5%，**與買進持有完全相同**（曝險恆為 100%）。三組種子 IS 0.422 ± 0.004。同輸入的 logistic／ridge／GBDT：IS 0.44／0.42／0.40，OOS 1.01／0.99／0.99。8 折 purged CV 平均 0.640 vs 買進持有 0.654（只在 2014–15、2016–17 小勝）。投組（50 檔一個帳戶）IS 11.7%／−28.0%／Sharpe 0.85 vs 50 檔等權買進持有 14.2%／−28.8%／0.89；OOS 兩者相同（48.5%、Sharpe 1.79）。

**注意：** 有效樣本極少（2008–2020 約 160 個不重疊的 20 日樣本、5～10 次大回撤），回撤預測 AUC 約 0.5；樣本內試過 100 多種設定（research/trials.jsonl）。模型學到的最佳答案就是「一直持有」。若想降低回撤，`trend_sma_half` 或大盤波動目標更簡單有效。MultiCharts 版需每天執行 `export_signals.py` 匯出目標曝險當 data2。詳見 RESEARCH.md。"""

STRATEGIES = [
    {
        "id": SID,
        "label": "市場狀態 MLP（共同曝險）",
        "family": "深度學習 2.0",
        "description": DESCRIPTION,
        "multicharts": f"multicharts/{SID}.txt",
        "positions": positions,
        "cv_positions": cv_positions,
    }
]
