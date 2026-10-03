# 台灣50 策略實驗室

以元大台灣50（0050）的 50 檔成分股為對象，研究多種交易策略，並與「買進持有（什麼都不做）」和「定期定額」比較。策略最終目標平台是 **MultiCharts**：每個策略都附 PowerLanguage 程式碼；回測則以 Python 進行，結果發佈到 GitHub Pages 網頁，可在任何裝置上逐檔檢視。

- 網頁檢視器：`docs/`（GitHub Pages 發佈後網址為 `https://roxign.github.io/stock/`）
- 研究方向（各自在 `strategies/<方向>/`，內含 `RESEARCH.md` 研究筆記與 `multicharts/` 程式碼）
  - `kd_macd_rule` — KD + MACD 規則型
  - `kd_macd_dl` — KD + MACD 結合小型深度學習模型
  - `trend` — 趨勢追蹤
  - `mean_reversion` — 均值回歸

## 回測方法

| 項目 | 設定 |
|---|---|
| 資料 | Yahoo Finance 日 K（還原權值），2010-01 起 |
| 成交 | 當日收盤產生訊號，**隔日開盤**成交（不會用到當天還不知道的資訊） |
| 成本 | 買進手續費 0.1425%；賣出手續費 0.1425% + 證交稅 0.3%（未計折扣與滑價） |
| 部位 | 只做多，可用 0～1 的持股比例，允許零股 |
| 本金 | 每段期間都從 100 萬重新開始，並先保留 60 個交易日計算指標 |
| 樣本內 | 2010–2020，用來開發策略與調整參數 |
| 樣本外 | 2021–今，開發時不看，只用來最後驗證 |
| 定期定額 | 本金平均分成每月一份，每月第一個交易日開盤買進；另計 XIRR |

所有策略在 50 檔股票上用**同一組參數**，避免針對個股過度最佳化。每個策略都要通過「截斷資料檢查」：把資料砍到某一天後重算，之前的訊號必須完全不變，以確保沒有偷看未來。

資料清理：台股單日漲跌幅上限 10%，若還原後的資料出現超過 10.5% 的單日變動，視為 Yahoo 未還原的減資或分割（如 2357 華碩 2010 年、2603 長榮 2022 年）並往前還原；成群出現時視為上市前（興櫃）資料而剔除（6669、6446、7769）。

**限制**：成分股是 2026/10/02 的名單，早年回測存在倖存者偏差；2021 年後台股大多頭，買進持有的報酬很難被超越，請同時看 Sharpe 與最大回撤。回測結果不代表未來績效，也不是投資建議。

## 使用方式

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python evaluate.py --family kd_macd_rule --periods is,oos   # 單一方向的排行
.venv/Scripts/python build_site.py --download                            # 更新股價並重建網頁資料
.venv/Scripts/python -m http.server 8765 -d docs                         # 本機預覽 http://localhost:8765
```

## 在 MultiCharts 使用

1. 開啟 PowerLanguage Editor → File → New → Signal，名稱自訂。
2. 貼上 `strategies/<方向>/multicharts/<策略>.txt`（網頁「策略說明」頁也可直接複製），按 F3 編譯。
3. 在日 K 圖表加入 Signal，於 Format Strategy → Properties 設定手續費 0.1425%、賣出證交稅 0.3% 與滑價。
4. MultiCharts 的資料源若未還原權值，結果會與本專案回測有差異。

## 專案結構

```
stocklab/        共用框架：成分股、資料下載與清理、指標、回測引擎、評估
strategies/      各研究方向的策略（_example 為範本）
evaluate.py      在終端機印出策略 vs 基準的排行（不改動網頁）
build_site.py    執行全部策略並輸出 docs/data/
docs/            GitHub Pages 網頁；engine.js 與 stocklab/backtest.py 邏輯一致
```
