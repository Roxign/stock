"""dl_position: E1 of research/dl_literature.md -- a small MLP that outputs each stock's POSITION directly, trained on
a cost-aware Sharpe objective (Deep Momentum Network style, long-only). One published strategy (family 深度學習 2.0).
See core.py for the pipeline and RESEARCH.md for the experiments.
"""

from . import core

SID = "dl_position_sharpe_net"
CFG = dict(core.DEFAULT)                                      # frozen network / training configuration
MAP = dict(col="wn", floor=0.0, span=10, step=0.25)           # frozen position mapping
N_FEATURES = len(core.feature_names(CFG["features"]))         # 32
N_PARAMS_NET = core.n_params(N_FEATURES, CFG["hidden"])       # 1,601 per net
N_PARAMS = N_PARAMS_NET * len(CFG["seeds"])                   # 8,005 for the 5-seed ensemble

_MEMO: dict = {}


def walk_forward(data):
    """Yearly walk-forward outputs memoised on the exact input data (positions, export and analysis share one run)."""
    key = core.fingerprint(data)
    if key not in _MEMO:
        while len(_MEMO) >= 4:                # the lookahead check alternates full / truncated data
            _MEMO.pop(next(iter(_MEMO)))
        _MEMO[key] = core.walk_forward(data, CFG)
    return _MEMO[key]


def _map(raw, col=MAP["col"], floor=MAP["floor"], span=MAP["span"], step=MAP["step"]):
    return {c: core.to_positions(core.target(df, col, floor), floor=floor, span=span, step=step) for c, df in raw.items()}


def positions(data):
    """min(1, network position / its mean on the model's training rows) -> 10-day EMA -> quarter steps with a
    0.1875 no-trade band. 1.0 (buy-and-hold) before the first model (2010-2011) and while features are missing."""
    raw, _ = walk_forward(data)
    return _map(raw)


def raw_positions(data):
    """The network's own position w (no normalisation), same smoothing / band. Reported in RESEARCH.md, not published."""
    raw, _ = walk_forward(data)
    return _map(raw, col="w")


def cv_positions(data, folds):
    """Purged k-fold CV (evaluate.py --cv): each fold traded by a model trained on all other data, purged and
    embargoed around the fold. Evaluation only, not tradable."""
    by_fold, _ = core.purged_cv(data, folds, CFG)
    return core.cv_to_positions(by_fold, **MAP)


DESCRIPTION = """
**想法**（文獻回顧 E1，Deep Momentum Network 的只做多版本）：不預測價格，讓網路**直接輸出每檔的部位** w ∈ [0, 1]，
訓練目標就是「扣掉真實手續費與證交稅之後的 Sharpe」，看它能不能自己學會何時降低曝險。
**輸入**（32 維，皆只用到 t 日收盤與 t+1 開盤前已公布的資料）：個股 17 維——以波動度標準化的 1/5/20/60/120/250 日報酬、
三組 MACD（8/24、16/48、32/96，除以價格×波動）、20/60 日累計隔夜與日內報酬、20 日實現波動、20/60 日波動比、
距 52 週高點、5/60 日量比；**50 檔等權重大盤** 7 維（5/20/60/250 日報酬、波動、距高點、MACD，多股票輸入）；
外部市場 8 維——加權指數 20 日、費半 1/20 日、那斯達克 60 日、台積電 ADR 1 日報酬、VIX 水準與相對 60 日均值、美元/台幣 20 日
（美股用台北時間 t+1 清晨已收盤的 t 日收盤）。
**輸出與損失**：w = sigmoid(z)。以同一檔連續 63 天為一段，批次 64 段一起算報酬
R = w_t·(O_{t+2}/O_{t+1} − 1) − 0.1425%·(Δw)⁺ − 0.4425%·(Δw)⁻ − 0.1%·|Δw|，損失 = −年化 Sharpe(R)。
**架構與參數**：MLP 32→32→16→1（ReLU、dropout 0.1），**每個網路 1,601 個參數，5 個種子集成共 8,005 個**。
**訓練**：每年 1 月 1 日用當時已知的資料重新訓練（擴張視窗；第一個模型 2012 年，用 2009–2010 訓練、2011 驗證，2008 年只當特徵暖機；
2010–2011 沒有模型，部位 = 1.0）；標籤 (t+2 開盤) 在 1 月 1 日前已知的樣本才用；以訓練窗最後 12 個月（前面再 purge 5 根）
提前停止；標準化統計量只來自訓練資料；AdamW、最多 40 epoch。50 檔全部算完約 1.5–2 分鐘（2 個 CPU 執行緒）。
**部位**：網路自己的 w 平均只有 0.44（太保守，見下），所以實際部位用**相對訊號**
min(1, w ÷ 該年模型在訓練資料上的平均 w)：網路比平常更保守時才減碼。再做 10 日 EMA（部分調整），
量化成 0 / 25 / 50 / 75 / 100%，且與目前部位相差 ≥ 18.75% 才換（無交易帶）。
**結果（49–50 檔中位數）**：樣本內 2010–2020 CAGR 9.6%、Sharpe 0.51、MDD −44.4%、平均曝險 98%（買進持有 8.8% / 0.48 / −46.1%）；
樣本外 2021–今 35.5% / 1.10 / −45.1%、曝險 99.7%（買進持有 35.7% / 1.10 / −45.5%）——**樣本外就是買進持有**。
5 個種子各自的 Sharpe：樣本內 0.501 ± 0.005、樣本外 1.094 ± 0.008。
**對照組**：樣本內勝過固定曝險 B1（8.7% / 0.47）與波動目標 B3（7.9% / 0.44），但與「別檔訊號」B2（9.5% / 0.50）幾乎一樣
——網路學到的是**整個市場的時機**（每天 w 的變異有 87–92% 來自 50 檔的共同成分），而且是**逆勢**的：大盤／那斯達克 60 日
漲多時降低曝險、VIX 升高與台幣走弱時提高。樣本外 B1 / B2 / B3 都是 35% / 1.09–1.10，沒有差別。
相同輸入的 Ridge 與 HistGradientBoosting（預測 20 日波動標準化報酬再依百分位映射）樣本內 Sharpe 0.50 / 0.45、
樣本外 0.89 / 1.04；trend_sma_half 0.46 / 1.05；kdj_macd_dl_floor 0.48 / 1.04。
**未正規化的網路部位 w**（未發布）：樣本內 6.3% / 0.42 / −40.2%（曝險 63%），樣本外 19.7% / 1.04 / −24.1%（48%），
不勝 B1（0.45 / 1.01）也不勝 B2（0.41 / 1.07）。
**交叉驗證（8 折）**：逐年 walk-forward 各折 Sharpe 0.05 / 0.87 / 0.29 / 0.80 / 0.66 / 0.86 / 0.51 / 1.38；
purged CV（每折由沒看過該折、前後 purge＋embargo 的模型交易）0.05 / 0.70 / 0.29 / 0.80 / 0.65 / 0.86 / 0.51 / 1.41，
與買進持有（0.05 / 0.70 / 0.29 / 0.80 / 0.65 / 0.86 / 0.51 / 1.37）幾乎完全相同；唯一明顯的優勢（2012–13 的 0.87）只來自第一個模型。
**50 檔同一帳戶**：樣本內 17.0% / MDD −26.9% / Sharpe 1.09（等權買進持有 14.2% / −28.8% / 0.89；但把訊號換成別檔的 B2 也有 15.4% / 0.97，
差距主要來自減碼後再加碼時順便把該檔調回 1/50 權重的再平衡效果）；樣本外 47.8% / −26.2% / 1.88（等權買進持有 48.5% / −30.7% / 1.79）。
**注意**：(1) 沒有打敗買進持有；樣本內的小幅優勢來自 2012 年一個模型，CV 與樣本外都看不到。
(2) 文獻擔心的「收斂成波動擇時動能」沒有發生，但它收斂成另一種東西：對整體市場的逆勢擇時，換到別檔股票也一樣有效（B2），不是個股訊號。
(3) 共試了約 95 組設定（research/trials.jsonl），請以 Deflated Sharpe 的角度看待樣本內數字。
(4) MultiCharts 版需要每天用 Python 匯出訊號（export_signals.py）當 data2 讀入。詳見 RESEARCH.md。
"""

STRATEGIES = [
    {
        "id": SID,
        "label": "成本感知部位網路（Sharpe 目標）",
        "family": "深度學習 2.0",
        "description": DESCRIPTION.strip(),
        "multicharts": f"multicharts/{SID}.txt",
        "positions": positions,
        "cv_positions": cv_positions,
    }
]
