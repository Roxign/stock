# 台灣50 策略實驗室

**網頁檢視器：<https://roxign.github.io/stock/>**（手機、電腦皆可）
- [今日訊號](https://roxign.github.io/stock/#/signals)：每個策略在最新收盤後的買賣決定（下一個交易日開盤執行）、明日收盤的觸發價（目標價）、匯率與利差風險提示、各策略過去買賣點的準確度
- [個股回測](https://roxign.github.io/stock/#/stock/2330)：選一檔股票，看歷史買賣點、當日訊號、觸發價線與各策略績效

以元大台灣50（0050）的 50 檔成分股、另外 55 檔代表股（開發時沒用過的股票，用來檢驗策略），以及 0050、黃金（00635U）、石油（00642U）、美債 20 年（00679B）ETF 為對象，研究多種交易策略（可做多、也可依台灣融券規則放空），並與「買進持有（什麼都不做）」和「定期定額」比較。策略最終目標平台是 **MultiCharts**：規則型策略附 PowerLanguage 程式碼；回測則以 Python 進行，結果發佈到上面的網頁（原始檔在 `docs/`）。

**先讀這份**：[為什麼策略贏不了買進持有？——問題診斷與改進](research/diagnosis.md)；買賣訊號與通知：[research/signals.md](research/signals.md)；匯率與利差：[research/macro_fx_rates.md](research/macro_fx_rates.md)

- 研究方向（各自在 `strategies/<方向>/`，內含 `RESEARCH.md` 研究筆記與 `multicharts/` 程式碼）
  - `kdj_macd_rule` — KDJ + MACD 規則型
  - `kdj_macd_rebound` — KDJ 與 MACD 同步翻揚買進、轉弱賣出（含停損）
  - `kdj_macd_dl` — KDJ + MACD 結合小型深度學習模型
  - `dl_position`、`dl_revenue_flow`、`dl_market_state`、`dl_cross_stock`、`dl_trend_labels` — 深度學習第二輪（直接輸出部位、月營收＋籌碼、大盤狀態、跨股注意力、趨勢標籤），文獻回顧見 `research/dl_literature.md`
  - `trend` — 趨勢追蹤
  - `mean_reversion` — 均值回歸
  - `long_short` — 多空策略（台灣融券規則，見 `research/short_rules.md`）
  - `multi_asset` — 0050／黃金／美債／石油的多資產配置（投資組合層級，見 `research/etf_data.md`）

## 回測方法

| 項目 | 設定 |
|---|---|
| 資料 | Yahoo Finance 日 K（還原權值），2008-01 起；2008–2009 只用於指標暖機（年線需 240 日）與模型訓練，不列入績效 |
| 成交 | 當日收盤產生訊號，**隔日開盤**成交（不會用到當天還不知道的資訊） |
| 成本 | 買進手續費 0.1425%；賣出手續費 0.1425% + 證交稅 0.3%（未計折扣與滑價） |
| 部位 | 只做多，可用 0～1 的持股比例，允許零股 |
| 本金 | 每段期間都從 100 萬重新開始，並先保留 60 個交易日計算指標 |
| 樣本內 | 2010–2020，用來開發策略與調整參數 |
| 樣本外 | 2021–今，開發時不看，只用來最後驗證 |
| 定期定額 | 本金平均分成每月一份，每月第一個交易日開盤買進；另計 XIRR |

所有策略在 50 檔股票上用**同一組參數**，避免針對個股過度最佳化。每個策略都要通過「截斷資料檢查」：把資料砍到某一天後重算，之前的訊號必須完全不變，以確保沒有偷看未來。

資料清理：台股單日漲跌幅上限 2015/6 以前為 7%、之後為 10%，還原後的資料若出現超過 7.5%／10.5% 的單日變動，視為 Yahoo 未還原的減資、分割或錯誤資料（如 2357 華碩 2010 年、2603 長榮 2022 年、2012/1/2 多檔的異常 K 棒）並往前還原；在後來才上市的股票開頭成群出現時，視為上市前（興櫃）資料而剔除（6669、6446、7769）。

**交叉驗證（8 折）**：只看「2021 至今」這段大多頭，買進持有幾乎無法被超越，策略在空頭或盤整時的優點會被掩蓋。因此把 2010 至今切成 8 段（2010–11、2012–13、2014–15、2016–17、2018–19、2020–21、2022–23、2024–今），每段各用 100 萬本金重新回測，統計策略在幾段勝過買進持有（`evaluate.py --cv`，網頁總覽的「交叉驗證」表）。有三種模式：
- 固定參數：發佈的參數直接用在每一段（2010–2020 的各段和調參資料重疊，只能看「在不同市況的表現」）。
- 重新選參數（規則型）：每一段都改用「在其他 7 段平均表現最好的參數」來測，等於每段都是沒看過的資料。
- 重新訓練（深度學習）：每一段都用完全沒看過該段資料、並剔除標籤期間重疊樣本（purge／embargo）訓練出的模型。
後兩種會用「未來的段」來幫「過去的段」選參數或訓練，只用來衡量穩健度，實際可交易的版本仍是前推式（walk-forward）的訊號。

**對照組與投資組合計分板**（依 `research/dl_literature.md` 的建議）：
- 對照組：每個策略都要和三個「沒有選股能力」的版本比較——B1 固定曝險（同樣的平均持股比例一次買進）、B2 別檔訊號（把別檔股票的訊號套到這檔）、B3 波動目標（依波動度調整部位、平均持股相同）。贏不了它們，代表策略的好處只是「持股比例」或「大盤擇時」。
- 投資組合計分板：50 檔放在同一個帳戶（可持現金）一起回測，和 0050 買進持有、0050 定期定額、50 檔等權買進持有比較。注意 50 檔是今天的成分股，「50 檔等權買進持有」受倖存者偏差影響，大幅高於 0050。
- 所有實驗設定都記錄在 `research/trials.jsonl`，最後用 Deflated Sharpe 校正多重檢定。

**限制**：成分股是 2026/10/02 的名單，早年回測存在倖存者偏差；2021 年後台股大多頭，買進持有的報酬很難被超越，請同時看 Sharpe 與最大回撤。回測結果不代表未來績效，也不是投資建議。

## 每日訊號與通知

收盤後執行 `daily_update.py`：下載最新資料 → 重算所有策略的部位 → 產生訊號（`stocklab/signals.py`，輸出 `docs/data/signals.json`）→ 重建網頁資料；加 `--push` 會提交並推送到 GitHub Pages。

- **委託**：策略在收盤時決定部位，回測也在隔天開盤成交，所以「今天收盤的決定」就是明天開盤的委託（買進、加碼、賣出、減碼、放空、回補）。
- **觸發價（目標價）**：只用該股自身價格決定的策略，會在 ±10% 漲跌幅內逐一假設明日收盤價重算，找出「明日收盤到多少就會出現買進／賣出（含停損）」；深度學習、跨股票與多資產策略無法事先算出。
- **訊號品質**：每個策略過去買點、賣點之後 5／20 個交易日的超額漲跌、上漲比例、勝率，分樣本內、樣本外、50 檔、代表股、ETF。
- **通知文字**：網頁「今日訊號」頁最下方可複製；Python 端 `stocklab.signals.digest()` 產生相同內容，之後可接 LINE、Email、Telegram 等管道。
- FinMind 未註冊每小時約 300 次請求，每日更新約 400 次，會自動等待；到 FinMind 免費註冊後把 token 設為環境變數 `FINMIND_TOKEN` 可加快。

## 使用方式

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python daily_update.py                                      # 每日：資料 → 部位 → 訊號 → 網頁（--push 推送）
.venv/Scripts/python evaluate.py --family kdj_macd_rule --periods is,oos   # 單一方向的排行（含 50 檔與代表股）
.venv/Scripts/python evaluate.py --family trend --controls --portfolio    # 加上對照組與投資組合計分板
.venv/Scripts/python build_site.py --cached                              # 用已算好的部位重建網頁資料
.venv/Scripts/python -m http.server 8765 -d docs                         # 本機預覽 http://localhost:8765
```

## 在 MultiCharts 使用

1. 開啟 PowerLanguage Editor → File → New → Signal，名稱自訂。
2. 貼上 `strategies/<方向>/multicharts/<策略>.txt`（網頁「策略說明」頁也可直接複製），按 F3 編譯。
3. 在日 K 圖表加入 Signal，於 Format Strategy → Properties 設定手續費 0.1425%、賣出證交稅 0.3% 與滑價。
4. MultiCharts 的資料源若未還原權值，結果會與本專案回測有差異。

## 專案結構

```
stocklab/        共用框架：成分股與代表股、資料下載與清理、指標、回測引擎、對照組、投資組合、試驗紀錄、
                 signals.py（每日訊號、觸發價、訊號品質）、macro.py（匯率與聯準會／日銀利率）
strategies/      各研究方向的策略（_example 為範本）
research/        跨方向的研究：深度學習文獻回顧、資料來源、試驗紀錄
evaluate.py      在終端機印出策略 vs 基準的排行（不改動網頁）
build_site.py    執行全部策略並輸出 docs/data/
daily_update.py  每日更新：資料 → 部位 → 訊號 → 網頁（→ git push）
docs/            GitHub Pages 網頁；engine.js 與 stocklab/backtest.py 邏輯一致
```
