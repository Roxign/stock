# 外部資料來源調查與共用載入器（給深度學習策略用）

> 調查日期：2026-10-03。所有來源都是**免費、免註冊、免 token**，實際打過 API 驗證。
> 程式：`stocklab/external.py`；測試：`tests/test_external.py`；快取：`data/external/`（已 gitignore，約 22 MB）。

## 一、結論與建議

1. **FinMind 開放 API（免 token）就夠用**：單一請求就能拿到一檔股票 2008 起（`stocklab.data.START`）的完整日資料，比逐日打證交所 / 櫃買 / 期交所 API 便宜幾千倍，所以官方逐日 API 完全不需要用。未註冊約 300 次/小時（實測兩次都在約 300 次後被擋），全部 50 檔 × 6 種資料 ≈ 300 次，額度內約 6–10 分鐘下載完；超過額度時程式會自動等待（最久約 1 小時）。
2. **Yahoo（yfinance）** 提供國際指數、ADR、利率、原物料，品質好；但 **新台幣匯率 TWD=X 與 0050.TW 在 Yahoo 上是壞的**（見第五節），改用 FinMind 的台銀匯率與加權報酬指數。
3. **期交所官網** 只用來抓台指選擇權 Put/Call Ratio（2007 起、每次限一個月），三大法人期貨部位官網只給最近 3 年，改用 FinMind（2018-06-05 起）。
4. **給 DL agent 的建議輸入**（依預期價值排序，詳見第七節）：
   - 隔夜國際：`sox`、`nasdaq`、`tsm`（ADR）、`vix`、`us10y`、`usdtwd` 的報酬 / 變化。
   - 個股籌碼：`foreign_net`、`trust_net`（除以成交量或 `shares_issued` 標準化）、`foreign_ratio` 的 5/20 日變化、`margin_balance`、`sbl_balance`（除以 `shares_issued`）。
   - 評價與基本面：`pbr`、`dividend_yield`、`per`、`revenue_yoy`。
   - 大盤：`taiex_tr` 報酬、`taiex_turnover`、`mkt_foreign_net`、`pc_ratio_oi`；`fut_foreign_net_oi` 只有 2018-06 起，若要用請只在 2019 年後訓練或加缺值遮罩。
   - 多股輸入：`load_price_panel` 取 50 檔收盤價做截面特徵（相對強弱排名、類股平均報酬，類股用 `industry()`）。
5. **策略裡每次呼叫 loader 都要帶 `end=ext.data_end(data)`**，否則 `stocklab.runner.check_lookahead` 截斷 `data` 時外部資料不會跟著截斷。用一個玩具策略（sox、外資買賣超、營收年增）跑過 `check_lookahead`：三個截斷點 0 根 K 棒不同。

## 二、對齊規則（防偷看未來，最重要）

本專案的約定：**第 t 根 K 棒產生的目標部位，在 t+1 根開盤（台北 09:00）成交**，所以 t 的特徵可以用「t+1 開盤前已公布」的任何資料。

`external.py` 給每一筆原始資料一個 **可用日 avail**：在台北時間 avail 這一天結束前（最晚隔天 09:00 前）一定已公布。載入器對每個台股交易日 t 回傳 **avail ≤ t 的最新一筆**（只往前填、不往後看；超過 14 個日曆天沒有新資料則為 NaN，月營收為 75 天）。因為 t+1 開盤一定在 t 之後的日曆日，所以 t 拿到的每個值都在成交前公布。

這個規則**只依賴 t 本身**，不依賴「下一個交易日是哪天」，所以把資料截斷在任何一天重算，之前的值完全不變（符合本專案的截斷檢查）。代價是遇到台股休市時略為保守：例如台股春節休市期間的美股收盤，要到開市後第一根 K 棒才會被使用。

| 序列 | 公布時間（台北） | avail |
|---|---|---|
| 台股收盤資料 d（加權指數、個股價量） | d 日 13:30 | d |
| 證交所/期交所盤後統計 d（三大法人、外資持股、本益比、期權法人、P/C ratio） | d 日約 15:00–17:00 | d |
| 融資融券、借券賣出餘額 d | d 日約 21:00–21:30 | d |
| 亞股收盤 d（日經、KOSPI、恒生、上證） | d 日 14:00–16:10 | d |
| 美股 / CME 收盤 d（紐約 16:00–17:00） | **d+1 日 04:00–06:00**（早於 09:00 開盤） | d |
| Yahoo 外匯日線 d（倫敦約 23:00 收） | d+1 日 06:00–07:00 | d |
| 台銀美元牌告匯率 d | d 日營業時間內 | d |
| 月營收（m 月） | 法定期限 m+1 月 10 日（遇假日順延） | 10 日當天或之後第一個台股交易日，**再加 1 個交易日**保險 |

注意：在預設模式下，台股 t 日的美股欄位是**美國同一日曆日 t 的收盤**（台北 t+1 清晨才出來，但早於 t+1 開盤成交），這是合法的，而且對隔日開盤後的走勢很有資訊量。

**`at_close=True` 模式**（`load_market` / `load_stock_fields` / `load_stock_panel` / `load_ohlcv` 都有）：只用台股 t 日 13:30 收盤當下已知的資料——台股收盤價用當天，其他所有序列（美股、亞股、法人籌碼、月營收）一律只用 t 之前日期的資料。要在 MultiCharts 收盤時就算出訊號、或想保守一點時用這個。注意 `strategies/_example/strategy.py` 的契約文字寫的是「用到 t 收盤為止的資料」；若以這句為準，請一律用 `at_close=True`（預設模式是依「t+1 開盤前已公布」的規則）。

**流量 vs 存量**：當日流量（三大法人買賣超、成交金額、借券賣出量，見 `external.FLOWS`）只出現在第一根拿到該筆資料的 K 棒，之後沒有新資料的 K 棒為 NaN（不會把昨天的買賣超複製到今天）；存量（餘額、持股比例、本益比、營收、價格）則往前填。

**截斷（`end=`）**：所有 loader 都有 `end=` 參數，只回傳 ≤ end 的日期。策略必須用 `end=ext.data_end(data)`（`data` 為策略收到的 dict），讓 `stocklab.runner` 的截斷檢查在截斷 `data` 時也同步截斷外部資料。因為 t 的值只依賴 t 以前的資料，截斷後的結果與完整結果在 end 以前完全相同（測試有驗證）。

月營收：FinMind 的 `create_time` 只有 2026 年以後才有值，而且是 FinMind 抓取時間（例如 2026 年 2 月營收顯示 4/21），不是實際公告時間，所以不用它，一律用法定期限規則。FinMind 存的是最新（可能更正過）的營收數字，極少數更正會有輕微的事後資訊，影響可忽略。

## 三、已取得的資料與覆蓋率

| 欄位 | 來源 | 頻率 | 單位 | 起 | 迄 | tw_gap% | nan% | 公布 → avail |
|---|---|---|---|---|---|---|---|---|
| `n225` | Yahoo ^N225 | 日 | price | 2008-01-04 | 2026-10-02 | 6.34 | 0.00 | 亞股收盤 d → d |
| `kospi` | Yahoo ^KS11 | 日 | price | 2008-01-02 | 2026-10-02 | 4.02 | 0.00 | 亞股收盤 d → d |
| `hsi` | Yahoo ^HSI | 日 | price | 2008-01-02 | 2026-10-02 | 3.78 | 0.00 | 亞股收盤 d → d |
| `sse` | Yahoo 000001.SS | 日 | price | 2008-01-02 | 2026-09-30 | 3.78 | 0.00 | 亞股收盤 d → d |
| `spx` | Yahoo ^GSPC | 日 | price | 2008-01-02 | 2026-10-02 | 3.63 | 0.00 | 美股收盤 d ＝台北 d+1 04–06 時 → d |
| `nasdaq` | Yahoo ^IXIC | 日 | price | 2008-01-02 | 2026-10-02 | 3.63 | 0.00 | 美股收盤 d ＝台北 d+1 04–06 時 → d |
| `sox` | Yahoo ^SOX | 日 | price | 2008-01-02 | 2026-10-02 | 3.63 | 0.00 | 美股收盤 d ＝台北 d+1 04–06 時 → d |
| `vix` | Yahoo ^VIX | 日 | price | 2008-01-02 | 2026-10-02 | 3.58 | 0.00 | 美股收盤 d ＝台北 d+1 04–06 時 → d |
| `tsm` | Yahoo TSM | 日 | price | 2008-01-02 | 2026-10-02 | 3.63 | 0.00 | 美股收盤 d ＝台北 d+1 04–06 時 → d |
| `nvda` | Yahoo NVDA | 日 | price | 2008-01-02 | 2026-10-02 | 3.63 | 0.00 | 美股收盤 d ＝台北 d+1 04–06 時 → d |
| `ewt` | Yahoo EWT | 日 | price | 2008-01-02 | 2026-10-02 | 3.63 | 0.00 | 美股收盤 d ＝台北 d+1 04–06 時 → d |
| `us10y` | Yahoo ^TNX | 日 | price | 2008-01-02 | 2026-10-02 | 3.67 | 0.00 | 美股收盤 d ＝台北 d+1 04–06 時 → d |
| `us3m` | Yahoo ^IRX | 日 | price | 2008-01-02 | 2026-10-02 | 3.67 | 0.00 | 美股收盤 d ＝台北 d+1 04–06 時 → d |
| `dxy` | Yahoo DX-Y.NYB | 日 | price | 2008-01-02 | 2026-10-02 | 3.58 | 0.00 | 美股收盤 d ＝台北 d+1 04–06 時 → d |
| `wti` | Yahoo CL=F | 日 | price | 2008-01-02 | 2026-10-02 | 3.60 | 0.00 | 美股收盤 d ＝台北 d+1 04–06 時 → d |
| `gold` | Yahoo GC=F | 日 | price | 2008-01-02 | 2026-10-02 | 3.60 | 0.00 | 美股收盤 d ＝台北 d+1 04–06 時 → d |
| `copper` | Yahoo HG=F | 日 | price | 2008-01-02 | 2026-10-02 | 3.60 | 0.00 | 美股收盤 d ＝台北 d+1 04–06 時 → d |
| `usdjpy` | Yahoo JPY=X | 日 | price | 2008-01-01 | 2026-10-02 | 0.93 | 0.02 | 倫敦收盤 d ＝台北 d+1 06–07 時 → d |
| `taiex` | FinMind TaiwanStockPrice | 日 | index | 2008-01-02 | 2026-10-02 | 0.00 | 0.00 | 台股收盤 d 13:30 → d |
| `taiex_turnover` | FinMind TaiwanStockPrice | 日 | NTD，流量 | 2008-01-02 | 2026-10-02 | 0.00 | 0.00 | 台股收盤 d 13:30 → d |
| `taiex_tr` | FinMind TaiwanStockTotalReturnIndex | 日 | index | 2008-01-02 | 2026-10-02 | 0.00 | 0.00 | 台股收盤 d 13:30 → d |
| `usdtwd` | FinMind TaiwanExchangeRate | 日 | TWD per USD | 2008-01-02 | 2026-10-02 | 0.02 | 0.00 | 台銀牌告 d → d |
| `mkt_foreign_net` | FinMind TaiwanStockTotalInstitutionalInvestors | 日 | NTD，流量 | 2008-01-02 | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `mkt_trust_net` | FinMind TaiwanStockTotalInstitutionalInvestors | 日 | NTD，流量 | 2008-01-02 | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `mkt_dealer_net` | FinMind TaiwanStockTotalInstitutionalInvestors | 日 | NTD，流量 | 2008-01-02 | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `mkt_margin_lots` | FinMind TaiwanStockTotalMarginPurchaseShortSale | 日 | lots (1000 sh) | 2008-01-02 | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `mkt_short_lots` | FinMind TaiwanStockTotalMarginPurchaseShortSale | 日 | lots (1000 sh) | 2008-01-02 | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `mkt_margin_value` | FinMind TaiwanStockTotalMarginPurchaseShortSale | 日 | NTD | 2008-01-02 | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `fut_foreign_net_oi` | FinMind TaiwanFuturesInstitutionalInvestors | 日 | contracts | 2018-06-05 | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `fut_trust_net_oi` | FinMind TaiwanFuturesInstitutionalInvestors | 日 | contracts | 2018-06-05 | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `fut_dealer_net_oi` | FinMind TaiwanFuturesInstitutionalInvestors | 日 | contracts | 2018-06-05 | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `opt_foreign_call_net_oi` | FinMind TaiwanOptionInstitutionalInvestors | 日 | contracts | 2018-06-05 | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `opt_foreign_put_net_oi` | FinMind TaiwanOptionInstitutionalInvestors | 日 | contracts | 2018-06-05 | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `pc_ratio_vol` | TAIFEX pcRatio | 日 | % | 2008-01-02 | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `pc_ratio_oi` | TAIFEX pcRatio | 日 | % | 2008-01-02 | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `foreign_net [50/50]` | FinMind TaiwanStockInstitutionalInvestorsBuySell | 日 | shares，流量 | 2012-05-02（最晚 2024-11-01） | 2026-10-02 | 0.03 | 0.03 | d 盤後（15:00–21:30）→ d |
| `trust_net [50/50]` | FinMind TaiwanStockInstitutionalInvestorsBuySell | 日 | shares，流量 | 2012-05-02（最晚 2024-11-01） | 2026-10-02 | 0.03 | 0.03 | d 盤後（15:00–21:30）→ d |
| `dealer_net [50/50]` | FinMind TaiwanStockInstitutionalInvestorsBuySell | 日 | shares，流量 | 2012-05-02（最晚 2024-11-01） | 2026-10-02 | 0.03 | 0.03 | d 盤後（15:00–21:30）→ d |
| `inst_net [50/50]` | FinMind TaiwanStockInstitutionalInvestorsBuySell | 日 | shares，流量 | 2012-05-02（最晚 2024-11-01） | 2026-10-02 | 0.03 | 0.03 | d 盤後（15:00–21:30）→ d |
| `foreign_ratio [50/50]` | FinMind TaiwanStockShareholding | 日 | % | 2008-01-02（最晚 2025-11-26） | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `shares_issued [50/50]` | FinMind TaiwanStockShareholding | 日 | shares | 2008-01-02（最晚 2025-11-26） | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `margin_balance [50/50]` | FinMind TaiwanStockMarginPurchaseShortSale | 日 | shares | 2008-01-02（最晚 2026-06-02） | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `short_balance [50/50]` | FinMind TaiwanStockMarginPurchaseShortSale | 日 | shares | 2008-01-02（最晚 2026-06-02） | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `sbl_balance [50/50]` | FinMind TaiwanDailyShortSaleBalances | 日 | shares | 2008-01-02（最晚 2026-02-26） | 2026-10-02 | 0.04 | 0.00 | d 盤後（15:00–21:30）→ d |
| `sbl_sell [50/50]` | FinMind TaiwanDailyShortSaleBalances | 日 | shares，流量 | 2008-01-02（最晚 2026-02-26） | 2026-10-02 | 0.04 | 0.04 | d 盤後（15:00–21:30）→ d |
| `per [50/50]` | FinMind TaiwanStockPER | 日 | x | 2008-01-02（最晚 2025-11-27） | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `pbr [50/50]` | FinMind TaiwanStockPER | 日 | x | 2008-01-02（最晚 2025-11-27） | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `dividend_yield [50/50]` | FinMind TaiwanStockPER | 日 | % | 2008-01-02（最晚 2025-11-27） | 2026-10-02 | 0.00 | 0.00 | d 盤後（15:00–21:30）→ d |
| `revenue [50/50]` | FinMind TaiwanStockMonthRevenue | 月 | NTD | 2008-01-03（最晚 2025-10-14） | 2026-09-11 |  | 0.00 | 10 日法定期限後第 1 個交易日 |
| `revenue_yoy [49/50]` | FinMind TaiwanStockMonthRevenue | 月 | ratio | 2008-01-11（最晚 2019-06-11） | 2026-09-11 |  | 0.00 | 10 日法定期限後第 1 個交易日 |
| `revenue_mom [50/50]` | FinMind TaiwanStockMonthRevenue | 月 | ratio | 2008-01-03（最晚 2025-11-11） | 2026-09-11 |  | 0.00 | 10 日法定期限後第 1 個交易日 |

欄位說明：「起/迄」為第一筆/最後一筆的可用日（avail）；`tw_gap%` 為起迄之間台股交易日中當天沒有資料的比例（國外序列包含當地假日，屬正常）；`nan%` 為對齊到台股交易日後仍為 NaN 的比例（流量欄位在沒有新資料的日子本來就是 NaN）。個股欄位的 `[x/50]` 為有資料的股票數，「起」取最早、括號內為最晚開始的那檔（晚上市股票），`gap`/`nan` 取 50 檔中位數。`revenue_yoy` 49/50：7769 鴻勁營收資料不滿一年。月營收的「起」是 2008-01-03，因為 2007 年各月營收在日曆起點（2008-01-02）之前就已公布，統一在第一個可用 K 棒出現。

總結：除了**個股三大法人（2012-05-02 起）**與**期貨/選擇權法人部位（2018-06-05 起）**之外，所有序列都涵蓋 2008-01 → 2026-10-02，台股原生序列幾乎沒有缺漏。快取共約 22 MB。

各序列的單位與說明見 `external.MARKET`、`external.STOCK_FIELDS`（每個欄位都有 `(group, unit, timing, description)`）。

## 四、來源細節

### FinMind（https://api.finmindtrade.com/api/v4/data）
- **免 token 可用**；回傳 JSON，`dataset` + `data_id` + `start_date` + `end_date`，一次給完整歷史，回應 0.3–1.7 秒。
- 速率：官方說明未註冊約 300 次/小時、註冊（免費帳號 token）約 600 次/小時。實測：一小時內約 300 次後回傳 HTTP 402「Requests reach the upper limit」，約一小時後恢復。被限流時，`finmind()` 會自動等 5 分鐘重試（最多 75 分鐘）。
- 如果使用者自行註冊 FinMind 帳號，可把 token 放在環境變數 `FINMIND_TOKEN`，程式會自動帶上（非必要）。
- 使用的資料集：
  - 個股：`TaiwanStockInstitutionalInvestorsBuySell`（三大法人，**2012-05-02 起**，證交所 T86 本身就是從這天開始）、`TaiwanStockShareholding`（外資持股、發行股數）、`TaiwanStockMarginPurchaseShortSale`（融資融券）、`TaiwanDailyShortSaleBalances`（借券賣出餘額）、`TaiwanStockPER`（本益比、殖利率、淨值比）、`TaiwanStockMonthRevenue`（月營收）。
  - 市場：`TaiwanStockPrice`（data_id=TAIEX，加權指數 OHLC + 成交金額，交易日完整）、`TaiwanStockTotalReturnIndex`（報酬指數）、`TaiwanStockTotalInstitutionalInvestors`、`TaiwanStockTotalMarginPurchaseShortSale`、`TaiwanFuturesInstitutionalInvestors`（TX，**2018-06-05 起**）、`TaiwanOptionInstitutionalInvestors`（TXO，2018-06-05 起）、`TaiwanExchangeRate`（USD，台銀牌告）、`TaiwanStockInfo`（產業別）。
- 投資人類別名稱會變：自營商 2014-11-28 以前只有一行 `Dealer`，之後拆成 `Dealer_self`（自行買賣）+ `Dealer_Hedging`（避險）；`Foreign_Dealer_Self`（外資自營商）2017-12-18 起才單列。載入器已合併處理。
- 晚上市股票的資料起點跟上市日一致（例如 6669 緯穎 2019-03、7769 鴻勁 2025-11 上市；6446 藥華藥 2024 年前在櫃買，FinMind 也有）。

### Yahoo Finance（yfinance 1.7）
- 一次批次下載全部代碼。指數與期貨不需還原；個股 / ETF（TSM、NVDA、EWT）是還原權值價，**每次配息後整段歷史水位會變**，請用報酬率不要用價位。
- 下載時會丟掉「台北時間減 8 小時之後」日期的 K 棒，避免把盤中未收完的 K 棒存進快取。
- Yahoo 的台股交易日少了 21 天（2008 起），所以台股交易日曆用 data/raw 的日期與 FinMind 加權指數日期的聯集（`calendar()`，2008-01-02～2026-10-02 共 4605 天）。

### 期交所官網
- `https://www.taifex.com.tw/cht/3/pcRatioDown`（POST，CSV，Big5）：台指選擇權 Put/Call Ratio，**2007 起**都有，但每次查詢區間不得超過約一個月，所以逐月抓（2008-01 起約 226 次，每次間隔 2 秒，約 8 分鐘）；`refresh=True` 時只補抓最後一個月之後。
- `futContractsDateDown`（三大法人期貨部位）：只開放**最近 3 年**，更早的日期回傳 "DateTime error"，因此改用 FinMind。

## 五、失敗、不可用或需要註冊的項目

| 項目 | 狀況 |
|---|---|
| Yahoo `TWD=X`（美元/新台幣） | 2011–2016 有錯價（2011-10-25 報 1.80、2014-12-31 報 3.67）且每天 ±3% 來回跳動（真實日波動約 0.2%），**不可用** → 改用 FinMind 台銀牌告匯率 `usdtwd` |
| Yahoo `0050.TW` | 2014-01-02 單日 -75%、2025 年分割前後有好幾週收盤價不動，**不可用** → 用 `taiex_tr`（加權報酬指數） |
| Yahoo `KRW=X` | 雜訊多（大量單日來回跳動），捨棄；韓國用 `kospi` 即可 |
| Yahoo `^TWOII`（櫃買指數） | Yahoo 查無資料 |
| FinMind `TaiwanStockHoldingSharesPer`（集保股權分散、大戶持股） | 需付費贊助等級（"Your level is free"） |
| FinMind `TaiwanStockGovernmentBankBuySell`（八大行庫） | 需付費贊助等級 |
| FinMind `TaiwanStockMarketValue`、不帶 data_id 的全市場單日查詢 | 需付費贊助等級 |
| 期交所三大法人期貨/選擇權部位 2018-06 以前 | 官網只開放近 3 年，FinMind 從 2018-06-05 開始；目前沒有免費來源 |
| 個股三大法人 2012-05 以前 | 證交所 T86 從 2012-05-02 才開始，任何來源都沒有 |
| 月營收實際公告時間 | FinMind 沒有（`create_time` 只是抓取時間）；公開資訊觀測站新版網站改為動態頁面，未採用，以法定期限規則代替 |

沒有建立任何帳號、沒有使用任何憑證。若使用者想要集保大戶持股或八大行庫資料，可自行考慮 FinMind 付費方案。

## 六、使用方式

```python
from stocklab import external as ext

def positions(data):                                    # 策略收到的 dict[code, DataFrame]
    end = ext.data_end(data)                            # 必用：外部資料截斷在 data 的最後一天
    cal = ext.calendar(end)                             # 台股交易日 DatetimeIndex
    mkt = ext.load_market(end=end)                      # 日期 x 市場/國際欄位（ext.MARKET）
    mkt = ext.load_market(["sox", "vix", "usdtwd"], index=data["2330"].index, end=end)  # 對齊到某檔的 K 棒
    chips = ext.load_stock_fields("2330", end=end)      # 單一股票全部籌碼/評價/營收欄位（ext.STOCK_FIELDS）
    fnet = ext.load_stock_panel("foreign_net", end=end) # 日期 x 50 檔
    close = ext.load_price_panel("close", end=end)      # 日期 x 50 檔（清理後還原價；也可直接從 data 組）
    tsm = ext.load_ohlcv("tsm", end=end)                # 單一 Yahoo 序列 OHLCV；"taiex" 為加權指數
    ind = ext.industry()                                # 股票 -> 產業別（靜態）
    strict = ext.load_market(at_close=True, end=end)    # 只用台股 t 日收盤當下已知的資料
```

- 所有 loader 只讀快取，不會連網；第一次使用前執行 `python -m stocklab.external`（下載缺少的部分並印出覆蓋率），`--refresh` 重新下載全部（期交所只補新的月份）。
- `index=` 可傳任何已排序的日期序列（例如個股自己的 `df.index`）；規則一樣是 avail ≤ t。
- `data/external/manifest.json` 記錄每個快取檔請求的起始日；把 `stocklab.data.START` 改早時，只會補抓缺的那一段（每檔一次請求）。
- `ext.market_raw(col)`、`ext.stock_raw(code, group)` 可取得未對齊的原始序列（index 為 avail 日）。
- 測試：`python tests/test_external.py`（也相容 pytest），逐一驗證特定日期（美國 MLK 假日、春節前後、月營收公布日前後、`at_close` 模式）的對齊值等於正確的較早原始值；對所有市場欄位與 3 檔股票的全部欄位做隨機日期的獨立比對；並檢查 `end=` 截斷後與完整結果一致。

## 七、給 DL agent 的實務建議

- **標準化**：籌碼量（股數）請除以當日成交量或 `shares_issued`；餘額類取 5/20 日變化；評價類（PER/PBR/殖利率）用過去 N 日的滾動分位數或 z-score（只能用過去資料計算統計量，與 `core.py` 的做法一致）。
- **價位不要直接餵**：還原價、指數水位都會漂移，用 log 報酬、與均線距離、或滾動 z-score。
- **缺值**：三大法人 2012-05 以前、期權法人 2018-06 以前、晚上市股票上市前都是 NaN。建議加「是否有值」的遮罩欄位，或只在資料齊全的期間訓練。樣本內期間（2010–2020）前段會缺籌碼資料，需要注意。
- **美股同日收盤**：預設模式下 t 日的 `sox`/`nasdaq` 已包含美國 t 日收盤，對 t+1 開盤後的報酬有預測力，但 t+1 的開盤跳空已反映在成交價裡，不會造成虛假報酬。若策略要在 MultiCharts 盤後立刻下單，請用 `at_close=True` 重新驗證。
- **多股輸入**：`load_price_panel` + `industry()` 可做截面特徵（50 檔報酬排名、類股平均、與台積電的相對強弱、市場寬度=站上 20MA 的比例等），台積電權重 56%，大盤特徵幾乎等同台積電，建議同時看等權平均。
- **重新下載**：`--refresh` 會重抓全部 FinMind（約 300 次請求、受每小時限額影響）；只是要延長到最新日期時再用。
