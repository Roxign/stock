# 深度學習交易策略：文獻研究與下一輪實驗設計

範圍：0050 現有 50 檔成分股、日資料（2010-01 → 2026-10）、t 日收盤後決策、t+1 開盤成交、單邊成本 0.1425%＋賣出 0.3% 證交稅（來回約 0.585%）、每檔部位 ∈ [0,1]、逐檔評估後取中位數、樣本內 2010–2020／樣本外 2021–。
背景：`strategies/kdj_macd_dl` 的小型 MLP 以「未來 60 日是否贏過橫斷面中位數」為標籤，前推 AUC 0.531（樣本內）/ 0.518（樣本外），而且改善和「多持有一點」無法區分；目前沒有策略在樣本外贏過買進持有（CAGR 35.7%、Sharpe 1.10、MDD −45.5%）。

引用的文獻都是本次實際查閱摘要/內文或確認存在的來源。標「（推論）」的是根據文獻與本 repo 結果推出來的看法，不是文獻原文的結論。

---

## 摘要：最重要的 10 點結論

1. **成本是第一道關卡。** 文獻裡漂亮的 DL 交易結果多半在每筆 2–10 bp 的成本下成立：Deep Momentum Network 的優勢在 2–3 bp 以上就消失（Lim 等 2019）；Spatio-Temporal Momentum 在 46 檔美股上，Sharpe 由 0 bp 時的 2.67 掉到 10 bp 時的 1.27（Tan 等 2023）；Fischer & Krauss（2018）的 LSTM 在 2010 年後扣成本約為零。我們單邊約 14–44 bp，高出一個數量級 → **成本與換手必須寫進訓練目標；預測/持有期至少數週；部位要加無交易帶**。
2. **「訊號存在」≠「策略贏」。** FinLab 在台股用 LightGBM 預測「下月報酬贏過市場中位數」，2019–2026 樣本外 Rank IC 穩定在 0.027，CPCV 15 折全部為正，但 CAGR 26.0% 輸 0050 的 29.9%，MDD −45.8% 對 −34.0%（月換手約 164%）。M6 競賽也發現預測準確度和投資績效關聯很弱。這和我們「AUC 0.518 但贏不了買進持有」完全一致。
3. **（推論）逐檔、只做多的評估方式，幾乎無法把「相對強弱」變現。** 依 Grinold 基本定律 IR ≈ IC·√breadth，AUC 0.52–0.53 約相當於 IC 0.04–0.07（第 5.6 節）。在 50 檔多空組合中（每年約 200 次獨立押注）這足以產生 0.5 以上的 IR；但逐檔擇時每檔每年只有約 4 次獨立的 60 日押注，而且相對訊號和「該檔絕對報酬」的相關性被大盤成分稀釋。→ 下一輪的標籤/輸出應該是**絕對（或風險調整後）報酬，或直接輸出部位**；相對排名只當輔助任務，或另外設一個「投資組合層級」的計分板。
4. **價量特徵比較能預測「波動」，而不是「方向」。** 本 repo 的回撤模型 AUC 0.576，但拿來減碼反而降低 Sharpe；波動目標（vol targeting）在樣本內能提高風險資產的 Sharpe（Harvey 等 2018；Moreira & Muir 2017），但 103 個策略的即時（樣本外）檢驗顯示沒有系統性改善（Cederburg 等 2020）；Nagel（2025）證明「高複雜度」的市場擇時模型實際上等於「依波動調整的動能」。→ **每個 DL 模型都要和「波動目標規則」與「趨勢規則」這兩個簡單複製品比較**。
5. **KD/MACD 以外，台股有證據的新輸入是「籌碼」與「月營收」。** 外資淨買賣與當日報酬強烈正相關、價格衝擊延續數日（Richards 2005）；台股機構交易獲利、散戶虧損，外資拿走約一半的機構利潤（Barber 等 2009）；外資大量買進的大型股之後有異常報酬（Lu 等 2012）；外資在台指期與台指選擇權的部位帶有資訊（Lin 2011；Lee & Wang 2016；Chuang 等 2019）；放空流量能預測報酬（Lin 等 2023）；散戶淨買負向預測（Zheng 2013）；月營收年增率有動能（Hung 等 2025），營收創新高後收盤買進持有一個月扣成本仍顯著為正（Lai 等 2025）。這些資料 FinMind 幾乎都有，多數從 2001–2005 年開始。
6. **免費的新特徵：隔夜／日內報酬拆解。** 美股的動能利潤主要發生在隔夜、反轉發生在日內（Lou, Polk & Skouras 2019）；台股則是日內動能為正、隔夜動能為負，效果持續約一年（Ho 等 2023）。只要 OHLC 就能計算，而且我們在開盤成交，剛好站在這兩段報酬的交界。
7. **模型大小不是瓶頸。** 淺層網路勝過深層網路（Gu, Kelly & Xiu 2020）；Qlib 基準在人工特徵（Alpha158）上，XGBoost/LightGBM 的 IC（0.050/0.045）高於 MLP、GRU、LSTM、Transformer（0.026–0.038）；單一隱藏層的網路就是最好的（Tan 等 2023）；2.2k 參數的 LSTM 在韓股和 XGBoost 打平（Kang & Kim 2025）。→ ≤ 5 萬參數綽綽有餘；**梯度提升樹（GBDT）是必要的對照組**。
8. **多股票輸入：先做「共用模型＋市場/產業脈絡特徵」，再考慮跨股注意力。** 用所有股票合併訓練的「通用模型」勝過逐檔模型（Sirignano & Cont 2019，高頻資料）；跨股注意力／圖網路的進步主要出現在 300–800 檔中國股票的排名任務（HIST、MASTER、Qlib Alpha360），系統性回顧也指出圖網路的增益不穩定。50 檔的橫斷面很小，預期增益有限。
9. **輸出：直接學部位，而不是預測價格。** 在相同輸入下，Lim 等（2019）的排名是：Sharpe 損失 > 報酬損失 > 迴歸 > 分類。直接輸出部位就是使用者要的「買賣訊號」，而且可以把成本放進損失函數。（推論）我們的交易不影響價格，強化學習在這裡等同「可微分回測＋直接最佳化」，只是變異更大；DRL 交易文獻也有嚴重的可重現性問題（Millea 2021）。
10. **評估：多種子、統計所有試驗次數、曝險對照組。** Qlib 回報的 DL 模型在 20 個隨機種子之間，資訊比率的標準差達 0.2–0.5；多個 agent 平行實驗等於多重檢定，所有 agent 的試驗數都要一起算進 Deflated Sharpe／PBO；存活者偏差讓「多持有」自動得分，所以每個策略都要附「相同平均曝險的固定部位」與「相同持續性的隨機訊號」兩個對照組。

---

## 1. 輸入特徵

### 1.1 證據總表

| 特徵族 | 文獻 | 發現 | 對我們的意義／限制 | 資料（起始） |
|---|---|---|---|---|
| 動能、短期反轉、波動、流動性 | Gu, Kelly, Xiu 2020 | 所有方法都認同主導變數是動能（含產業動能、短期反轉）、流動性、波動；用 NN 預測做 S&P 500 擇時，Sharpe 0.77 對買進持有 0.51 | 月頻、全美股；我們已有大部分 | Yahoo OHLCV（已有） |
| 台股動能 | Chui, Titman, Wei 2010；Bui 等 2023 | 台灣是少數個股動能不明顯的市場；但以 86 個異常為輸入的 NN/PLS，多空月報酬 1.2–1.5%，前 20 名重要變數中有 5 個和動能有關 | 動能在台股要用非線性方式使用；多空結果 ≠ 逐檔多頭 | 已有 |
| 隔夜 vs 日內 | Lou, Polk, Skouras 2019；Ho 等 2023 | 美股：動能利潤在隔夜，反轉在日內。台股：日內動能正、隔夜動能負，持續約一年，較少崩盤 | **零成本新特徵**：累計隔夜（C→O）與日內（O→C）報酬，5/20/60/250 日 | 已有 |
| 52 週高點距離 | George & Hwang 2004 | 接近 52 週高點的股票表現較好，可解釋大部分動能 | 只要收盤價 | 已有 |
| 成交量 | Gervais 等 2001；Llorente 等 2002 | 異常高量後一個月股價上漲（能見度效果）；量與報酬自相關的關係取決於資訊交易比例 | 用量／60 日均量，以及它和報酬的交互作用 | 已有 |
| K 棒形狀與量價公式 | Qlib Alpha158；Kakushadze 2016 | Alpha158：KMID、KLEN、KUP、KLOW、ROC、MA、STD、BETA、RSQR、RSV、IMAX/IMIN、CORR、CNTP、VMA 等，窗格 5–60。101 alphas 平均持有 0.6–6.4 天 | 現成特徵清單；101 alphas 的換手對我們太高 | 已有 |
| 產業／關聯股 | Moskowitz & Grinblatt 1999；Hou 2007；Cohen & Frazzini 2008 | 產業動能解釋大部分個股動能；大型股領先小型股主要發生在同產業內；客戶報酬領先供應商 | 50 檔中半導體／電子供應鏈很多：台積電、同產業平均的落後報酬是最便宜的「多股票」特徵 | 已有＋人工產業分類 |
| 外資買賣、持股 | Richards 2005；Barber 等 2009；Lu, Fang, Nieh 2012；JMFM 2019 | 外資淨買與當日報酬強烈正相關，衝擊延續數日，並在美股上漲的隔日買超；機構買進的股票一個月後比賣出的多約 80 bp；外資大量買進的大型股之後有異常報酬；外資在熱絡市場主導價格發現，在冷清市場是追隨者 | 最有希望的新資訊；效果多在數日內 → 注意換手，並讓模型依市場狀態調整 | FinMind `TaiwanStockInstitutionalInvestorsBuySell`（2005–；FinLab 版本 2012-05 起）、`TaiwanStockShareholding`（2004–） |
| 機構羊群 | Lee, Lin, Xia 2025 | 機構羊群在下跌期正向預測、多頭期負向預測 | 正負號隨市場狀態改變 → 適合非線性模型 | 同上 |
| 散戶、融資融券、放空 | Zheng 2013；Short sale and stock returns (TWSE)；Lin, Ho, Ko 2023 | 散戶淨買負向預測短期報酬；高度被放空的股票之後一個月負異常報酬；高券資比＋高融資時高估可持續一年；短期與長期放空流量都能預測報酬 | 融資增減、券資比、借券賣出變化 | `TaiwanStockMarginPurchaseShortSale`（2001–）、`TaiwanStockSecuritiesLending`（2001–）、`TaiwanDailyShortSaleBalances`（2005-07–） |
| 月營收 | Hung, Lu, Yang 2025；Lai 等 2025 | 營收年增率有動能，而且獨立於季報資訊；營收創新高：隔日開盤短線過度反應，但收盤買進持有一個月扣成本仍顯著為正 | 月頻事件、低換手，最符合我們的成本結構 | `TaiwanStockMonthRevenue`（2002–）；**歷史公布日沒有記錄** → 用「次月 10 日」規則保守對齊 |
| 評價 | FinLab 機器學習實測 | 特徵重要度前段：股價／240 日線、量比、PB、營收年增 | 50 檔的評價差異小 → 用「自身歷史百分位」比橫斷面排名合適 | `TaiwanStockPER`（2005-10–） |
| 期貨／選擇權部位 | Lin 2011；Lee & Wang 2016；Chuang 等 2019 | 外資在台指期的新倉交易對現貨指數最具資訊；台指選擇權中只有外資的交易有預測力（價外、短天期最明顯），買權 O/S 比優於 put/call 比；外資在期貨的優勢主要來自資訊 | 大盤層級的擇時特徵 | FinMind 期貨/選擇權法人只從 2018-06 起 → 需向期交所下載更早的歷史（起始年未驗證） |
| 美股、匯率、VIX | Richards 2005；Fung, Lam, Lam 2010 | 外資在美股上漲後隔日買超；亞洲指數期貨對美股極端報酬過度反應、日內反轉，扣成本仍可獲利 | （推論）在「t 日收盤後決策、t+1 開盤成交」的框架下，美股 t 日的資訊已經反映在 t+1 開盤價；只能安全使用美國 t−1 日以前的資料，當作狀態變數 | Yahoo `^SOX ^IXIC TSM TWD=X ^VIX ^TWII` |

### 1.2 實務重點
- **時間對齊比特徵選擇更重要。** 三大法人、融資融券、外資持股、PER 都在收盤後公布（FinMind 標示 15:00–21:00 更新），可用於「t 日收盤後決策」；月營收保守起見視為「次月 11 日收盤後」才可用；季報以法定期限（5/15、8/14、11/14、3/31）之後才可用。**目前的 `--lookahead` 截斷檢查只會截斷價格資料，外部資料的載入器也必須一起截斷**，否則檢查抓不到洩漏。
- **正規化：** 使用波動標準化的報酬（Lim 等 2019、Tan 等 2023 的做法：r/σ√k）、各檔自身的滾動 z 分數或百分位，以及當日橫斷面排名；統計量只能來自訓練窗。
- **大型股的現實：** Avramov 等（2023）發現深度學習的獲利主要來自難以套利的股票（微型股、財務困難股）與高波動期間，排除後明顯減弱，再扣成本更差。我們的 50 檔正好是最容易套利的那一群 → 對效果大小要保守。

---

## 2. 輸出與標籤

| 輸出型態 | 文獻 | 證據 | 對我們的意義 |
|---|---|---|---|
| 固定期間漲跌分類／報酬迴歸 | Krauss 等 2017；Fischer & Krauss 2018 | 以「贏過橫斷面中位數」為標籤的 DNN/GBT/RF/LSTM 在 S&P 500 有效，但利潤逐年下降，2010 年後扣成本約為零 | 本 repo：絕對方向 AUC ≈ 0.50；相對 AUC 0.53 但逐檔無法變現 |
| 排名（pairwise/listwise） | Poh 等 2021；Feng 等 2019 | 學習排序讓橫斷面動能策略的 Sharpe 約提升三倍；RSR 以圖網路做股票排名 | 只有在「組合層級」評估才有用；逐檔評估下只能當輔助損失 |
| 三重障礙＋meta-labeling | López de Prado 2018；Joubert 2022 | 過濾假訊號、決定部位大小 | 本 repo 已驗證：AUC 0.54 的過濾器降低 MDD，但砍掉持有率，Sharpe 反而下降 |
| 趨勢掃描（trend scanning） | López de Prado 2020 | 對每個 t，在多個前視期間做 OLS，取 t 值最大者的正負號為標籤、|t| 為樣本權重 | 自然的「買／賣訊號」標籤，前視期間由資料決定 |
| 轉折點／連續趨勢標籤 | Chang, Fan, Liu 2009；Wu 等 2020 | PLR 把價格切成線段標出轉折點，再用 NN 學習；連續趨勢標籤以回檔比例 ω 劃分漲跌段 | 證據多為分類準確度，很少有扣成本的交易測試 |
| 三重障礙＋原始 OHLCV | Kang & Kim 2025 | 韓股全市場、100 日原始 OHLCV、2,235 參數的 LSTM、29 日 ±9% 障礙：F1 0.431，與 XGBoost 0.432 相同 | 小模型讀原始序列可行；但沒有交易成本與部位測試 |
| **直接輸出部位（Sharpe/效用損失）** | Lim 等 2019；Wood 等 2021；Tan 等 2023；Zhang 等 2020b | 網路輸出 tanh 部位，直接最大化 Sharpe；可在損失中加入 c·|Δw|；Sharpe 損失 > 報酬損失 > 迴歸 > 分類 | 最貼近我們的目標函數；多頭版改用 sigmoid；必須加換手懲罰 |
| 強化學習 | Zhang, Zohren, Roberts 2020a；Millea 2021 | DRL 在 50 檔期貨上扣成本仍有正報酬，並證明線性效用下與投資組合理論等價；批判性回顧指出方法分歧、多數無程式碼、難以重現 | （推論）交易不影響價格時，直接部位網路就是不需探索的 RL，不建議另做 RL |

**把預測轉成部位：** Campbell & Thompson（2008）指出，加上簡單限制（例如預測不得為負）後，很小的樣本外 R² 對均值—變異投資人仍有經濟價值 → 用平滑映射 w = clip(μ̂/(γσ̂²), 0, 1)，不要用 0/1 開關。Gârleanu & Pedersen（2013）：有成本時最佳策略是「瞄準未來目標、每次只往目標移動一部分」→ w_t = w_{t−1} + κ(ŵ_t − w_{t−1})，或設無交易帶。

---

## 3. 多股票架構

| 做法 | 文獻 | 設定與發現 | 對我們的意義 |
|---|---|---|---|
| 通用（合併）模型 | Sirignano & Cont 2019；Gu 等 2020 | 用數百到上千檔美股的高頻委託簿資料訓練的通用模型勝過逐檔模型，也能轉移到沒看過的股票 | 本 repo 已經合併 50 檔訓練；可再加 4–8 維股票嵌入 |
| 扁平化橫斷面輸入 | Tan 等 2023 | 46 檔美股，每檔 14 個特徵 × 5 天全部攤平輸入單層網路，同時輸出所有股票部位；Sharpe 損失＋L1 換手懲罰 | 最直接的「多股票輸入、多部位輸出」；但每天只有一筆樣本（樣本內約 2,700 筆），容易過度配適 |
| 跨股注意力 | Li 等 2024（MASTER）；Kelly 等 2025（AIPM） | MASTER：158 個 alpha＋63 個市場特徵、回看 8 天，交替做股內（時間）與股間注意力，並用市場資訊做特徵閘門；AIPM 把 transformer 放進隨機折現因子，跨資產共享資訊，大幅降低定價誤差 | 排列等變、共享權重 → 比扁平化省樣本；50 檔的注意力模式可能不穩 |
| 圖神經網路 | Feng 等 2019；Xu 等 2021（HIST）；GNN 系統性回顧 2024 | Qlib Alpha360（原始價量）：HIST IC 0.052、GRU 0.049、GATs 0.048、LightGBM 0.040；回顧指出靜態圖、關係變動、可解釋性是主要問題 | 產業圖可以人工定義；但增益通常小，而且只在排名任務上被證明 |
| 市場狀態條件化 | MASTER；Lin 等 2021（TRA） | TRA 用路由器把樣本分配給多個預測器，IC 0.053 → 0.059 | 和「依市場狀態改變正負號」的籌碼證據相符 |

**何時多股票模型勝過逐檔模型（綜合判斷）：** 每檔資料不足時（合併訓練等於擴大樣本）、跨股關係穩定時（產業、供應鏈）、以及評估本身是橫斷面的時候。反例：Abe & Nakayama（2018）在日股發現較深的網路較好，和 Gu 等的結論相反，顯示這類結果和資料集高度相關。（推論）我們的情況是 50 檔、逐檔評估，跨股資訊的主要價值很可能只是「大盤與產業狀態」，用手工特徵就能便宜地加入；跨股注意力的增量要用消融實驗證明。

---

## 4. 小模型架構（≤ 5 萬參數、CPU 數分鐘）

- **基準一定要有：** 邏輯斯／脊迴歸、`sklearn.ensemble.HistGradientBoosting*`（requirements 已有 scikit-learn，不用另裝 LightGBM，省硬碟）。在人工特徵上，GBDT 通常不輸 DL（Qlib Alpha158；Grinsztajn 等 2022 的表格資料基準）。
- **證據：** 淺層勝深層（Gu 等 2020，NN 採 32-16-8、多種子平均）；在原始序列上 GRU/LSTM 才勝過 GBDT（Qlib Alpha360）；線性模型在長期預測上常勝過 Transformer（Zeng 等 2023）；TCN 是序列建模的合理起點（Bai 等 2018）；DoubleEnsemble（樣本重新加權＋特徵選擇的集成）是 Qlib Alpha158 上資訊比率最高的模型（1.34）。
- **建議尺寸：** MLP 24→32→16→1（約 1.3k）；GRU 隱藏 16（約 2k）；TCN 3 層 × 16 通道（約 3–5k）；單層跨股注意力 d=32（約 15k）。
- **正則化：** 在 purge 後的驗證集上提前停止、weight decay、dropout 0.1–0.3、輸入雜訊、5–10 個種子平均；標籤做 winsorize 或橫斷面 z 分數（Qlib 做法），訓練時丟掉 5% 極端標籤（MASTER 做法）。
- **「複雜度的美德」要小心：** Kelly, Malamud, Zhou（2024）主張參數多於樣本的模型能提高擇時效果；Nagel（2025）指出在小訓練窗下，這類模型其實是「依波動調整的動能」，換成有反轉的人工資料仍會做出同樣的策略 → 一定要和簡單規則的複製品比較。

---

## 5. 評估陷阱

1. **重疊標籤的洩漏：** 標籤期間 H 的樣本，訓練與測試之間要 purge ≥ H，並加 embargo；交叉驗證可用 CPCV（López de Prado 2018，第 7、12 章）。本 repo 已做 purge；新加的外部資料也要遵守。
2. **多重檢定：** Harvey, Liu, Zhu（2016）建議新因子 t > 3；White 的 Reality Check（Sullivan, Timmermann, White 1999）；Deflated Sharpe（Bailey & López de Prado 2014）與 PBO（Bailey 等）。**多個 agent 平行做，等於試驗數相加**——請在共用登記表記錄每一組設定，最後用總試驗數計算 DSR。Arnott, Harvey, Markowitz（2019）的回測規範：事先寫下假設、記錄所有嘗試、不要回頭改。
3. **非定態與衰退：** Krauss 等（2017）、Fischer & Krauss（2018）的利潤隨時間下降；Welch & Goyal（2008）：大盤報酬預測變數多數在樣本外失效。
4. **成本與換手：** 回報每年換手與「打平成本所需的邊際」；Qlib 基準的成本是買 0.05%／賣 0.15%（來回 0.20%），大約只有我們 0.585% 的三分之一，換手和 IR 不能直接比較。
5. **存活者偏差：** 股票池是今天的 0050，等於事先挑好贏家 → 「出場」看起來特別貴，「多持有」自動得分。對照組（第 6 節）就是為了把這部分扣掉；有能力的話，加入過去被剔除的成分股做穩健性檢查。
6. **AUC ≈ 0.52 能不能賺錢（推論）：** 若預測分數與報酬近似二元常態，AUC ≈ Φ(1.13·ρ)，所以 AUC 0.531 ↔ IC ≈ 0.07，0.518 ↔ IC ≈ 0.04。Grinold（1989）IR ≈ IC·√breadth：50 檔、60 日期間的多空組合 breadth 至多約 200/年 → 扣成本前 IR 約 0.5–1；但逐檔、只做多的擇時 breadth 約 4/年，而且相對訊號被大盤成分稀釋 → 實際上接近 0，和本 repo「MLP ≈ 別檔訊號」的結果一致。結論：要嘛換成絕對報酬標籤／直接部位，要嘛新增一個「50 檔投組（允許持現金）」的計分板，讓相對訊號有地方變現。
7. **種子變異：** Qlib Alpha158 上 MLP 的 IR 為 1.14 ± 0.23、GRU 0.52 ± 0.25、Transformer 0.40 ± 0.26 → 一律報告 ≥ 5 個種子的平均與標準差；IC 高 ≠ 組合績效好（MLP 的 IC 低於 LightGBM，但 IR 較高）。

---

## 6. 實驗設計（依優先順序）

### 6.0 所有實驗共用的規範
- **前推式訓練：** 每年 1 月重訓、擴張視窗、第一個模型 2012 年啟用；purge = 標籤期間 + 5 日 embargo；提前停止用訓練窗最後 12 個月（兩端都 purge）。標準化統計量只取訓練集。5 個種子平均。設定在樣本內凍結後，樣本外只跑一次。
- **必須贏過的對照組（樣本內與樣本外都要列）：** 
  - B0 買進持有、定期定額；
  - B1 **相同平均曝險的固定部位**：w ≡ 該策略在該檔的平均曝險，其餘持現金；
  - B2 **相同持續性的隨機訊號**：把模型輸出錯配到別檔（本 repo 的「別檔訊號」）；
  - B3 簡單規則複製品：波動目標 w = min(1, σ*/σ̂₂₀)（σ* 選到平均曝險相同），以及 `trend_sma_half`；
  - B4 同樣輸入與標籤的邏輯斯／脊迴歸，以及 HistGradientBoosting。
- **回報：** 中位數 CAGR／Sharpe／MDD、平均曝險、年換手、Sharpe 勝 B&H 的比例、逐年 AUC/IC、各種子標準差，以及（建議）50 檔等權投組的 Sharpe。

### E1（優先 1）成本感知的直接部位網路（DMN 多頭版）
- **假設：** 直接最佳化「扣成本後的逐檔效用」，比先預測再映射更能利用微弱訊號，也能學到何時降低曝險。
- **輸入（約 24 維，只需現有資料＋yfinance）：** 波動標準化報酬（1/5/20/60/120/250 日）、三組 MACD（8/24、16/48、32/96，除以價格波動）、累計隔夜與日內報酬（20/60 日）、實現波動（20/60 日）與比值、52 週高點距離、量比；市場脈絡：50 檔等權指數的同樣特徵、^TWII、^SOX、TSM、TWD=X、^VIX（只用美國 t−1 日以前）。
- **輸出／損失：** w_t = f + (1−f)·sigmoid(z_t)，f ∈ {0, 0.5} 兩個版本。以 63 日序列為一個樣本，損失 = −(年化 Sharpe) 或 −[mean(R) − ½γ·var(R)]，其中 R_t = w_t·(O_{t+2}/O_{t+1} − 1) − 0.1425%·(Δw)⁺ − 0.4425%·(Δw)⁻，再加 α·|Δw| 換手懲罰（Tan 等 2023）。推論時加部分調整 κ ∈ {0.1, 0.2} 或無交易帶。
- **架構：** MLP 24→32→16→1（約 1.3k 參數）或 GRU(16)（約 2k），可選 4 維股票嵌入；5 個種子合計 < 1 萬。
- **主要風險：** 收斂成「永遠滿倉」（這本身就是答案）或純波動擇時（必須和 B3 比較）；Sharpe 損失對小批次很吵，要用大批次（多檔、多段一起算）。難度：低到中。

### E2（優先 2）月營收＋籌碼的低換手事件模型
- **假設：** 月營收與法人/融資/借券資料帶有價格以外的新資訊，而且月頻決策的換手低到成本能承受。
- **輸入（約 25 維，FinMind）：** 月營收年增、月增、3 個月平均年增、年增加速度、是否創 12 個月／歷史新高；PE、PB、殖利率的自身 5 年百分位；外資／投信／自營商淨買占成交量（1/5/20/60 日）、外資持股比 20/60 日變化；融資餘額變化、券資比、借券賣出餘額變化；價格脈絡（20/60/120 日報酬、波動）。
- **標籤：** 下次營收公布前（約 20 日）或 40 日的**絕對**報酬（t+1 開盤 → t+1+H 開盤），用 Huber 迴歸；輔助任務為橫斷面排名。
- **架構：** 主模型 HistGradientBoosting；對照 MLP 25→32→16→1（約 1.5k）＋股票嵌入。
- **決策頻率與部位：** 每月營收可用日（次月 11 日收盤後）更新一次，法人特徵可每週更新；w = 0.5 + 0.5·clip(μ̂/(γσ̂²), 0, 1)，最短持有 20 日。
- **主要風險：** 文獻多是多空、全市場（含中小型股）的結果，對 50 檔大型股的「絕對」擇時價值不確定（Avramov 等 2023）；FinLab 的台股實測 Rank IC 只有 0.027；需要下載約 300 次 API（在 600 次/小時的限制內）並仔細做時間對齊。難度：中（以資料工程為主）。

### E3（優先 3）大盤狀態曝險模型（市場層級籌碼＋全球脈絡）
- **假設：** 本 repo 已發現 DL 的改善多半來自「大盤狀態＋多持有」，把這部分做成明確的大盤模型，並加入外資與衍生性商品部位。
- **輸入（市場層級約 30 維）：** 加權指數／50 檔等權指數的多期報酬與波動；50 檔的廣度（站上 60/120/240 日線比例）、橫斷面離散度、平均相關性；三大法人合計淨買（`TaiwanStockTotalInstitutionalInvestors`，2004–）；全市場融資餘額變化（`TaiwanStockTotalMarginPurchaseShortSale`，2001–）；台指選擇權 put/call 與買權 O/S（由 `TaiwanOptionDaily` 計算，2001-12–）；外資台指期淨未平倉（期交所下載）；SOX、NASDAQ、TSM、TWD、VIX、美國 10 年期殖利率（t−1）。
- **標籤／輸出：** 等權指數未來 20/60 日絕對報酬與最大回撤（多任務），或直接輸出共同曝險 g_t ∈ [0.3, 1] 並用效用損失訓練。
- **架構：** 以正則化邏輯斯／脊迴歸為主（< 100 參數），MLP 30→16→1（約 500）與淺層 GBDT 對照。依 Campbell & Thompson 限制預測不得極端，並向無條件平均收縮。
- **部位：** w_i = g_t（或 0.5 + 0.5·g_t），可再乘個股趨勢；遲滯帶。對照 `trend_dual_market`。
- **主要風險：** 有效樣本極少（樣本內只有約 5–10 次大回撤），過度配適風險最高；Welch & Goyal 的警告；外資資訊多在 1–數日內反映，太快的訊號會被成本吃掉。難度：模型低、資料中。

### E4（優先 4）跨股注意力模型（多股票輸入、絕對報酬輸出）
- **假設：** 其他股票（台積電、同產業、領先股）的狀態能幫忙預測某檔的絕對報酬，而且學到的交互作用比手工特徵多。
- **輸入：** 每檔用 E1（若有則加 E2）的特徵、回看 20 日；市場 token（E3 的市場特徵）；8 維股票嵌入＋人工產業嵌入（約 6 類）。
- **架構（約 1.5 萬參數）：** 共享的 GRU(32) 時間編碼器 → 市場閘門（MASTER 簡化版）→ 單層 4 頭股間自注意力（d=32）→ 前饋層 → 每檔輸出未來 20 日波動標準化的絕對報酬（主任務）與 ListNet 排名（輔助任務，Poh 等 2021）。缺資料的股票（如 7769 早期）用遮罩。
- **必要消融：** 關掉注意力（等於合併通用模型）、打亂股票身分、Tan 等式的扁平化單層網路（約 2 萬參數）、GBDT＋手工同業特徵（產業平均落後報酬、台積電落後報酬）。
- **部位：** 同 E1 的均值—變異映射＋部分調整。
- **主要風險：** 文獻證據來自 300–800 檔的排名任務；50 檔、每天一組，注意力很可能只學到大盤平均（E3 用更便宜的方法就能做到）。難度：高。

### E5（優先 5）趨勢掃描／三重障礙標籤＋小型序列模型（買賣訊號）
- **假設：** 讓資料決定「趨勢段」的標籤，比固定 H 日漲跌更接近可交易的買賣點；小模型直接讀原始序列可以學到指標以外的形態。
- **輸入：** 60–100 日原始序列：以最新收盤標準化的 OHLC、量 z 分數、隔夜／日內報酬、Alpha158 的 K 棒特徵（KMID、KLEN、KUP、KLOW）。
- **標籤：** (a) 趨勢掃描：前視 10–60 日中 t 值最大的趨勢正負號，|t| 當樣本權重；(b) 三重障礙：±2·σ₂₀·√H、H = 40，三類（漲停利／到期／跌停損）。purge = 最長前視期間。
- **架構：** 因果 TCN（3 層、16 通道、膨脹 1/2/4，約 3–5k）或 GRU(16)（約 1.5k）；變體可換成 Jiang, Kelly, Xiu（2023）式的 20 日 K 線圖 CNN（論文在美股多空 Sharpe 很高，且能轉移到其他市場，但屬多空、未扣台股成本）。
- **部位：** 狀態機——P(上升) > θ_in 進到 1，P(下降) > θ_out 退到底倉 0.5；θ 取訓練集分位數；最短持有 10 日。
- **對照：** 同標籤的 GBDT（Alpha158 式特徵）、`trend_sma_half`、`kdj_macd_dl_floor`。
- **主要風險：** 趨勢標籤主要能從「最近的趨勢」預測，模型容易退化成另一個趨勢跟隨者（等於現有 trend 策略）；標籤前視期間長，purge 寫錯就會嚴重洩漏。難度：中。

**組合建議：** E1 是「輸出端」的框架，E2/E3 是「資訊集」。第一輪先各自獨立驗證；若 E2 或 E3 的特徵在 B4（GBDT）上有增量，再把它們接到 E1 的直接部位網路，或用 E3 的大盤預測 × E2/E4 的個股預測組成 μ̂ = β̂·μ̂_mkt + α̂_i 再映射成部位。不建議另做強化學習。

---

## 參考文獻

**輸入特徵**
- Gu, S., Kelly, B., Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *RFS* 33(5). https://www.nber.org/system/files/working_papers/w25398/w25398.pdf
- Jiang, J., Kelly, B., Xiu, D. (2023). (Re-)Imag(in)ing Price Trends. *JF* 78(6). https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13268
- Lou, D., Polk, C., Skouras, S. (2019). A Tug of War: Overnight versus Intraday Expected Returns. *JFE* 134(1). https://personal.lse.ac.uk/polk/research/TugOfWar.pdf
- Ho, H.-W., Hsiao, Y.-J., Lo, W.-C., Yang, N.-T. (2023). Momentum investing and a tale of intraday and overnight returns: Evidence from Taiwan. *PBFJ* 82. https://ideas.repec.org/a/eee/pacfin/v82y2023ics0927538x23002226.html
- Chui, A., Titman, S., Wei, K.C.J. (2010). Individualism and Momentum around the World. *JF*.（工作論文版）http://www.fin.ntu.edu.tw/~conference/conference2004/proceedings/proceeding/6/6-4(A63).pdf
- Bui, D.G., Kong, D.-R., Lin, C.-Y., Lin, T.-C. (2023). Momentum in Machine Learning: Evidence from the Taiwan Stock Market. *PBFJ* 82. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4595220
- George, T., Hwang, C.-Y. (2004). The 52-Week High and Momentum Investing. *JF* 59. https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2004.00695.x
- Gervais, S., Kaniel, R., Mingelgrin, D. (2001). The High-Volume Return Premium. *JF* 56. https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00349
- Llorente, G., Michaely, R., Saar, G., Wang, J. (2002). Dynamic Volume-Return Relation of Individual Stocks. *RFS* 15(4). https://academic.oup.com/rfs/article-abstract/15/4/1005/1567663
- Kakushadze, Z. (2016). 101 Formulaic Alphas. https://arxiv.org/abs/1601.00991
- Microsoft Qlib：Alpha158 特徵定義 https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py
- Moskowitz, T., Grinblatt, M. (1999). Do Industries Explain Momentum? *JF* 54. https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00146
- Hou, K. (2007). Industry Information Diffusion and the Lead-Lag Effect in Stock Returns. *RFS* 20(4). https://www.ssrn.com/abstract=1151155
- Cohen, L., Frazzini, A. (2008). Economic Links and Predictable Returns. *JF* 63. https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2008.01379.x
- Richards, A. (2005). Big Fish in Small Ponds: The Trading Behavior and Price Impact of Foreign Investors in Asian Emerging Equity Markets. *JFQA* 40(1). https://ideas.repec.org/a/cup/jfinqa/v40y2005i01p1-27_00.html
- Barber, B., Lee, Y.-T., Liu, Y.-J., Odean, T. (2009). Just How Much Do Individual Investors Lose by Trading? *RFS* 22(2). https://faculty.haas.berkeley.edu/odean/papers%20current%20versions/justhowmuchdoindividualinvestorslose_rfs_2009.pdf
- Lu, Y.-C., Fang, H., Nieh, C.-C. (2012). The price impact of foreign institutional herding on large-size stocks in the Taiwan stock market. *RQFA* 39(2). https://ideas.repec.org/a/kap/rqfnac/v39y2012i2p189-208.html
- Foreign investors' trading behavior and market conditions: Evidence from Taiwan (2019). *J. Multinational Financial Management* 52–53. https://ideas.repec.org/a/eee/mulfin/v52-53y2019is1042444x19300490.html
- Lee, N.R.-L., Lin, C.-L., Xia, H. (2025). Market states and institutional herds: evidence from the Taiwan stock market. *Cogent Economics & Finance* 13(1). https://www.tandfonline.com/doi/abs/10.1080/23322039.2025.2571399
- Zheng, Z. (2013). The Impact of Individual Investor Trading on Stock Returns. *Emerging Markets Finance and Trade* 49(sup3). https://www.tandfonline.com/doi/abs/10.2753/REE1540-496X4904S305
- Short sale and stock returns: Evidence from the Taiwan Stock Exchange (TWSE 1991–2004). https://www.sciencedirect.com/science/article/abs/pii/S1062976908000811
- Lin, C., Ho, H.-W., Ko, K.-C. (2023). Shorting flows and return predictability in Taiwan. *PBFJ* 77. https://ideas.repec.org/a/eee/pacfin/v77y2023ics0927538x22001111.html
- Hung, W., Lu, C.C., Yang, J.J. (2025). Market reaction to monthly revenue momentum. *RQFA*. https://scholars.ncu.edu.tw/en/publications/market-reaction-to-monthly-revenue-momentum/
- Lai, C.-C., Tsai, W.-H., Lin, Y.-N., Lin, A.Y. (2025). Trading on Record-Breaking Monthly Revenue Announcements. SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5345252
- Lin (2011). Information Content for Investor Groups in TAIEX Futures Trading. *Asia-Pacific J. Financial Studies*. https://onlinelibrary.wiley.com/doi/10.1111/j.2041-6156.2011.01045.x
- Lee, Y.-H., Wang, D.K. (2016). Information content of investor trading behavior: Evidence from Taiwan index options market. *PBFJ*. https://www.sciencedirect.com/science/article/abs/pii/S0927538X16300415
- Chuang, Y.-W., Lin, Y.-F., Weng, P.-S. (2019). Why and how do foreign institutional investors outperform domestic investors in futures trading: Evidence from Taiwan. *J. Futures Markets* 39(3). https://onlinelibrary.wiley.com/doi/abs/10.1002/fut.21975
- Fung, A.K.-W., Lam, K., Lam, K.-M. (2010). Do the prices of stock index futures in Asia overreact to U.S. market returns? *J. Empirical Finance* 17(3). https://ideas.repec.org/a/eee/empfin/v17y2010i3p428-440.html
- FinMind 資料集文件：籌碼面 https://finmind.github.io/tutor/TaiwanMarket/Chip/ ；基本面 https://finmind.github.io/tutor/TaiwanMarket/Fundamental/ ；技術面 https://finmind.github.io/tutor/TaiwanMarket/Technical/ ；衍生性商品 https://finmind.github.io/tutor/TaiwanMarket/Derivative/
- FinLab：三大法人資料說明 https://finlab.finance/data/institutional-flow

**輸出與標籤**
- Lim, B., Zohren, S., Roberts, S. (2019). Enhancing Time Series Momentum Strategies Using Deep Neural Networks. https://arxiv.org/abs/1904.04912
- Wood, K., Giegerich, S., Roberts, S., Zohren, S. (2021). Trading with the Momentum Transformer. https://arxiv.org/abs/2112.08534
- Tan, W.L., Roberts, S., Zohren, S. (2023). Spatio-Temporal Momentum: Jointly Learning Time-Series and Cross-Sectional Strategies. *JFDS*. https://arxiv.org/abs/2302.10175
- Poh, D., Lim, B., Zohren, S., Roberts, S. (2021). Building Cross-Sectional Systematic Strategies by Learning to Rank. *JFDS* 3(2). https://arxiv.org/abs/2012.07149
- Zhang, Z., Zohren, S., Roberts, S. (2020a). Deep Reinforcement Learning for Trading. *JFDS* 2(2). https://www.pm-research.com/content/iijjfds/2/2/25
- Zhang, Z., Zohren, S., Roberts, S. (2020b). Deep Learning for Portfolio Optimization. https://arxiv.org/abs/2005.13665
- Millea, A. (2021). Deep Reinforcement Learning for Trading—A Critical Survey. *Data* 6(11). https://www.mdpi.com/2306-5729/6/11/119
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley（目錄）https://toc.library.ethz.ch/objects/pdf03/e01_978-1-119-48208-6_01.pdf
- López de Prado, M. (2020). *Machine Learning for Asset Managers*（趨勢掃描，§5.4）；實作說明 https://mlfinpy.readthedocs.io/en/latest/Labelling.html
- Joubert, J. (2022). Meta-Labeling: Theory and Framework. *JFDS* 4(3). https://www.pm-research.com/content/iijjfds/4/3/31
- Chang, P.C., Fan, C.Y., Liu, C.H. (2009). Integrating a Piecewise Linear Representation Method and a Neural Network Model for Stock Trading Points Prediction. *IEEE TSMC-C* 39(1). https://www.researchgate.net/publication/220508486_Integrating_a_Piecewise_Linear_Representation_Method_and_a_Neural_Network_Model_for_Stock_Trading_Points_Prediction
- Wu, D. 等 (2020). A Labeling Method for Financial Time Series Prediction Based on Trends. *Entropy* 22(10). https://www.mdpi.com/1099-4300/22/10/1162
- Kang, S., Kim, J.-K. (2025). Stock Price Prediction Using Triple Barrier Labeling and Raw OHLCV Data: Evidence from Korean Markets. https://arxiv.org/abs/2504.02249
- Krauss, C., Do, X.A., Huck, N. (2017). Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500. *EJOR* 259. https://www.econstor.eu/bitstream/10419/130166/1/856307327.pdf
- Fischer, T., Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *EJOR* 270. https://doi.org/10.1016/j.ejor.2017.11.054
- Campbell, J., Thompson, S. (2008). Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average? *RFS* 21(4). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1212066
- Gârleanu, N., Pedersen, L.H. (2013). Dynamic Trading with Predictable Returns and Transaction Costs. *JF* 68. https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12080

**多股票架構**
- Sirignano, J., Cont, R. (2019). Universal features of price formation in financial markets: perspectives from deep learning. *Quantitative Finance* 19(9). https://arxiv.org/abs/1803.06917
- Feng, F., He, X., Wang, X., Luo, C., Liu, Y., Chua, T.-S. (2019). Temporal Relational Ranking for Stock Prediction. *ACM TOIS* 37(2). https://dl.acm.org/doi/10.1145/3309547
- Xu, W. 等 (2021). HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information. https://arxiv.org/abs/2110.13716
- Li, T. 等 (2024). MASTER: Market-Guided Stock Transformer for Stock Price Forecasting. *AAAI* 38. https://arxiv.org/abs/2312.15235 ；程式 https://github.com/SJTU-DMTai/MASTER
- Kelly, B., Kuznetsov, B., Malamud, S., Xu, T.A. (2025). Artificial Intelligence Asset Pricing Models. NBER w33351. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5089371
- Lin, H. 等 (2021). Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport. *KDD*. https://arxiv.org/abs/2106.12950
- A Systematic Review on Graph Neural Network-based Methods for Stock Market Forecasting (2024). *ACM Computing Surveys* 57(2). https://dl.acm.org/doi/10.1145/3696411
- Abe, M., Nakayama, H. (2018). Deep Learning for Forecasting Stock Returns in the Cross-Section. *PAKDD*. https://arxiv.org/abs/1801.01777
- Microsoft Qlib 基準結果（Alpha158/Alpha360、20 個種子）https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md

**小模型架構**
- Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS*. https://neurips.cc/virtual/2022/poster/55627
- Zeng, A., Chen, M., Zhang, L., Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? *AAAI*. https://github.com/cure-lab/LTSF-Linear
- Bai, S., Kolter, J.Z., Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. https://arxiv.org/abs/1803.01271
- Zhang, C. 等 (2020). DoubleEnsemble: A New Ensemble Method Based on Sample Reweighting and Feature Selection for Financial Data Analysis. *ICDM*. https://arxiv.org/abs/2010.01265
- Kelly, B., Malamud, S., Zhou, K. (2024). The Virtue of Complexity in Return Prediction. *JF* 79. https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13298
- Nagel, S. (2025). Seemingly Virtuous Complexity in Return Prediction. NBER w34104. https://www.nber.org/papers/w34104

**評估**
- Bailey, D., López de Prado, M. (2014). The Deflated Sharpe Ratio. *JPM* 40(5). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Bailey, D., Borwein, J., López de Prado, M., Zhu, Q. The Probability of Backtest Overfitting. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Harvey, C., Liu, Y., Zhu, H. (2016). … and the Cross-Section of Expected Returns. *RFS* 29(1). https://academic.oup.com/rfs/article-abstract/29/1/5/1843824
- Sullivan, R., Timmermann, A., White, H. (1999). Data-Snooping, Technical Trading Rule Performance, and the Bootstrap. *JF* 54. https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00163
- Arnott, R., Harvey, C., Markowitz, H. (2019). A Backtesting Protocol in the Era of Machine Learning. *JFDS* 1(1). https://people.duke.edu/~charvey/Research/Published_Papers/G138_A_backtesting_protocol.pdf
- Avramov, D., Cheng, S., Metzker, L. (2023). Machine Learning vs. Economic Restrictions: Evidence from Stock Return Predictability. *Management Science* 69(5). https://pubsonline.informs.org/doi/10.1287/mnsc.2022.4449
- Welch, I., Goyal, A. (2008). A Comprehensive Look at the Empirical Performance of Equity Premium Prediction. *RFS* 21(4). https://ideas.repec.org/a/oup/rfinst/v21y2008i4p1455-1508.html
- Grinold, R. (1989). The Fundamental Law of Active Management. *JPM* 15(3). https://www.pm-research.com/content/iijpormgmt/15/3/30
- Makridakis, S. 等 (2023). The M6 forecasting competition: Bridging the gap between forecasting and investment decisions. https://arxiv.org/abs/2310.13357
- Moreira, A., Muir, T. (2017). Volatility-Managed Portfolios. *JF* 72(4). https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513
- Harvey, C. 等 (2018). The Impact of Volatility Targeting. *JPM* 45(1). https://www.pm-research.com/content/iijpormgmt/45/1/14
- Cederburg, S., O'Doherty, M., Wang, F., Yan, X. (2020). On the performance of volatility-managed portfolios. *JFE* 138(1). https://econpapers.repec.org/RePEc:eee:jfinec:v:138:y:2020:i:1:p:95-117
- Moskowitz, T., Ooi, Y.H., Pedersen, L.H. (2012). Time Series Momentum. *JFE* 104(2). https://ideas.repec.org/a/eee/jfinec/v104y2012i2p228-250.html
- Daniel, K., Moskowitz, T. (2016). Momentum Crashes. *JFE* 122(2). https://www.nber.org/papers/w20439
- FinLab（2026）：機器學習預測股價沒用？改用 AI 產生交易訊號，台股 8 年樣本外實測。https://finlab.finance/blog/machine-learning-predict-stock-price
