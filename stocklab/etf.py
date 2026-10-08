"""台灣掛牌 ETF（0050、黃金、美債 20 年、石油）的資料層：下載、還原（含息）、ETF 專屬清理、上市前代理序列、metadata。

詳細說明與驗證結果見 research/etf_data.md。

資料來源（皆免帳號、免 token）：
  * 主來源：FinMind TaiwanStockPrice（證交所／櫃買中心每日行情，含週六補行交易日，無假日幽靈 K 棒）、
    TaiwanStockDividendResult（除權息結果表：前收盤價、參考價、配息）、TaiwanStockSplitPrice（分割／反分割參考價）、
    TaiwanStockMarginPurchaseShortSale（融資融券餘額與「註記」欄，含停止融券 X）。
  * 交叉驗證：Yahoo 0050.TW / 00635U.TW / 00642U.TW / 00679B.TWO（價格、配息）。
  * 標的與代理：Yahoo GLD / TLT / USO（還原收盤），匯率用 stocklab.external 的台銀 USD/TWD（usdtwd），
    T-bill 利率用 external 的 us3m（^IRX）。
快取：data/external/etf/（gitignored）。只有 download_etfs() 會連網；載入函式只讀快取。

公開 API
  ETFS / CODES                         ETF 清單與 metadata（dict）
  download_etfs(codes, refresh)        下載全部原始資料到快取
  load_etf(code, start, end, adjusted, extra)   清理後的日 K（open, high, low, close, volume；DatetimeIndex），
                                       預設為總報酬還原（配息再投入、分割調整）價格，與 stocklab.data.load 同格式
  load_all_etfs(codes, **kw)           {code: load_etf(code)}
  load_proxy(code, start, end, full, carry)     上市前的代理序列（美國 ETF × USD/TWD），只含上市前日期（full=False）
  load_extended(code, ...)             明確拼接：上市前用代理（縮放到上市首日價位）+ 上市後真實資料，含 is_proxy 欄
  tracking_stats(code, carry)          代理與 ETF 在重疊期的追蹤統計
  corporate_actions(code)              除息／分割事件與還原因子
  bad_bars(code)                       ETF 專屬異常 K 棒偵測結果（含處置）
  cleaning_report(codes)               每檔清理摘要
  sell_tax(code, dates) / costs(code, date)     證交稅率（債券 ETF 2017-01-01~2026-12-31 停徵）與買賣成本
  load_margin(code)                    融資融券日資料（含註記；註記適用於「次一營業日」）

時間對齊：台股 ETF 收盤 13:30（台北）；美國 ETF 的 d 日收盤在台北 d+1 的 04:00-05:00 公布，
因此台股日 t 對應「t 之前最後一個美國交易日」的收盤（lag 1；實測日報酬相關係數 0.66-0.79，lag 0 只有 0.21-0.24）。
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import external as ext
from .data import START

ETF_DIR = ext.EXT_DIR / "etf"
COMMISSION = 0.001425  # 券商手續費（未折扣），與 stocklab.backtest 相同
ETF_TAX = 0.001  # ETF／受益憑證證交稅 1‰（證券交易稅條例第 2 條第 2 款）
BOND_TAX_EXEMPT = ("2017-01-01", "2026-12-31")  # 證券交易稅條例第 2-1 條第 2 項（民國 106/1/1 ~ 115/12/31）
# 行政院 2026-10-01 通過再延長 10 年至 2036-12-31 的修正草案，尚待立法院三讀；通過後改這裡即可。

# ------------------------------------------------------------------------------------------------ metadata
# 費用：MoneyDJ ETF 基本資料（2026-10 讀取），費用已內含於淨值與市價，回測時不可再扣一次。
# margin：FinMind 融資融券資料中自上市初期起即有融券餘額（實證可融券）；ETF 上市當日即可信用交易，且不受
# 平盤以下不得融券賣出之限制（證交所 ETF 交易規則；金管會 2013-09-23 起開放）。細節由 stocklab/shortrules.py 處理。
ETFS: dict[str, dict] = {
    "0050": {
        "name": "元大台灣50", "asset": "台股大盤", "exchange": "TWSE", "yahoo": "0050.TW",
        "inception": "2003-06-25", "listed": "2003-06-30",
        "index": "臺灣50指數", "replication": "實物（持有成分股）",
        "mgmt_fee": 0.0015, "total_expense": 0.0022, "aum_twd_m": 2_505_277, "aum_date": "2026-10-02",
        "tax": ETF_TAX, "tax_exempt": None, "price_limit": 0.10,
        "currency": "TWD（台股，無外幣曝險）", "distribution": "半年配（約 1 月、7 月除息）",
        "margin": True, "short_below_close": True,
        "splits": "2025-06-18 1 拆 4（2025-06-11~06-17 停止買賣）",
        "underlying": "taiex_tr", "proxy": None,
    },
    "00635U": {
        "name": "期元大S&P黃金", "asset": "黃金", "exchange": "TWSE", "yahoo": "00635U.TW",
        "inception": "2015-04-01", "listed": "2015-04-15",
        "index": "S&P GSCI Gold Excess Return Index", "replication": "COMEX 黃金期貨（期貨信託 ETF）",
        "mgmt_fee": 0.010, "total_expense": 0.0115, "aum_twd_m": 13_665, "aum_date": "2026-10-01",
        "tax": ETF_TAX, "tax_exempt": None, "price_limit": None,
        "currency": "未避險美元（以美元計價期貨，台幣計價受益憑證）", "distribution": "不配息",
        "margin": True, "short_below_close": True, "splits": "無",
        "underlying": "GLD", "proxy": "GLD",
    },
    "00642U": {
        "name": "期元大S&P石油", "asset": "原油", "exchange": "TWSE", "yahoo": "00642U.TW",
        "inception": "2015-08-27", "listed": "2015-09-07",
        "index": "S&P GSCI Crude Oil Enhanced Excess Return Index", "replication": "NYMEX WTI 原油期貨（期貨信託 ETF）",
        "mgmt_fee": 0.010, "total_expense": 0.0115, "aum_twd_m": 1_750, "aum_date": "2026-10-01",
        "tax": ETF_TAX, "tax_exempt": None, "price_limit": None,
        "currency": "未避險美元（以美元計價期貨，台幣計價受益憑證）", "distribution": "不配息",
        "margin": True, "short_below_close": True, "splits": "無",
        "underlying": "USO", "proxy": "USO",
    },
    "00679B": {
        "name": "元大美債20年", "asset": "美國長天期公債", "exchange": "TPEx", "yahoo": "00679B.TWO",
        "inception": "2017-01-11", "listed": "2017-01-17",
        "index": "ICE U.S. Treasury 20+ Year Index", "replication": "實物（持有美國公債）",
        "mgmt_fee": 0.0010, "total_expense": 0.0014, "aum_twd_m": 153_505, "aum_date": "2026-10-01",
        "tax": ETF_TAX, "tax_exempt": BOND_TAX_EXEMPT, "price_limit": None,
        "currency": "未避險美元", "distribution": "季配（2、5、8、11 月除息）",
        "margin": True, "short_below_close": True, "splits": "無",
        "underlying": "TLT", "proxy": "TLT",
    },
}
CODES = list(ETFS)
US_SYMBOLS = ["GLD", "TLT", "USO"]
CARRY_DEFAULT = {"00635U": True, "00642U": True, "00679B": False}  # 期貨 ER 指數：代理扣美國 T-bill 利率

# ------------------------------------------------------------------------------------------------ downloads


def _path(kind, name):
    return ETF_DIR / kind / f"{name}.csv.gz"


def _save(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.drop(columns=[c for c in ("stock_name",) if c in df])
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, compression="gzip", encoding="utf-8", float_format="%.8g")
    tmp.replace(path)


def _yahoo(symbols, path_of, refresh):
    import yfinance as yf

    todo = [s for s in symbols if refresh or not path_of(s).exists()]
    if not todo:
        return
    raw = yf.download([ETFS[s]["yahoo"] if s in ETFS else s for s in todo], start="2003-01-01", auto_adjust=False,
                      actions=True, progress=False, group_by="ticker", threads=True)
    cutoff = ext._cutoff()
    for s in todo:
        sym = ETFS[s]["yahoo"] if s in ETFS else s
        if sym not in raw.columns.get_level_values(0):
            print(f"  Yahoo {sym}: no data", flush=True)
            continue
        d = raw[sym].dropna(subset=["Close"]).copy()
        d.columns = [c.lower().replace(" ", "_") for c in d.columns]
        d.index = pd.DatetimeIndex(d.index).tz_localize(None) if d.index.tz is not None else pd.DatetimeIndex(d.index)
        d = d[d.index < cutoff]
        d.index.name = "date"
        _save(d.reset_index(), path_of(s))


def download_etfs(codes=None, refresh=False):
    """下載（只補缺的，refresh=True 全部重抓）：FinMind 行情／除權息／融資融券（每檔 3 次請求）、分割表（1 次）、
    Yahoo 台股 ETF（交叉驗證）與 GLD/TLT/USO（代理）。FinMind 用 external.finmind() 的節流與重試。"""
    codes = list(codes or CODES)
    fm = {"price": "TaiwanStockPrice", "dividend": "TaiwanStockDividendResult", "margin": "TaiwanStockMarginPurchaseShortSale"}
    for c in codes:
        for kind, ds in fm.items():
            p = _path(kind, c)
            if refresh or not p.exists():
                df = ext.finmind(ds, c, start="2003-01-01")
                _save(df, p)
                print(f"FinMind {ds} {c}: {len(df)} rows", flush=True)
    p = ETF_DIR / "splits.csv.gz"
    if refresh or not p.exists():
        _save(ext.finmind("TaiwanStockSplitPrice", None, start="2003-01-01"), p)
    _yahoo(codes, lambda s: _path("yahoo", s), refresh)
    _yahoo(US_SYMBOLS, lambda s: _path("yahoo", s), refresh)
    _memo.clear()


# ------------------------------------------------------------------------------------------------ raw readers

_memo: dict = {}


def _read(path, **kw):
    if not path.exists():
        raise FileNotFoundError(f"{path} not cached: run `python -m stocklab.etf` first")
    try:
        return pd.read_csv(path, **kw)
    except pd.errors.EmptyDataError:  # FinMind 回傳 0 筆（如不配息的 00635U/00642U 除權息表）
        return pd.DataFrame()


def raw_prices(code, source="finmind") -> pd.DataFrame:
    """未還原日 K。finmind：證交所/櫃買官方行情（主來源）；yahoo：交叉驗證用（含 adj_close、dividends）。"""
    key = ("raw", code, source)
    if key not in _memo:
        if source == "finmind":
            d = _read(_path("price", code), parse_dates=["date"]).set_index("date").sort_index()
            d = d.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume", "Trading_money": "value"})
            d = d[["open", "high", "low", "close", "volume", "value"]].astype(float)
        else:
            d = _read(_path("yahoo", code), parse_dates=["date"]).set_index("date").sort_index()
        d.index = pd.DatetimeIndex(d.index).as_unit("ns")
        _memo[key] = d
    return _memo[key]


def us_etf(symbol) -> pd.DataFrame:
    """Yahoo 美國 ETF（GLD/TLT/USO）日資料，含 adj_close（配息、分割還原），以美國交易日為索引。"""
    return raw_prices(symbol, "yahoo")


def _snap(ratio):
    """分割比例對齊到整數 k 或 1/k（參考價四捨五入到 tick 會有微小誤差）。"""
    for k in range(2, 101):
        for r in (k, 1 / k):
            if abs(ratio / r - 1) < 0.005:
                return r
    return ratio


def corporate_actions(code) -> pd.DataFrame:
    """除息與分割／反分割事件，以「除權息（恢復交易）日」為索引：kind, before（前收盤）, after（參考價）, cash,
    factor（該日之前的價格乘上此因子）, share_ratio（之前的成交量乘上此值）。
    除息 factor = 參考價 / 前收盤 = (P - D) / P；分割 factor = 1 / 分割比例。"""
    rows = []
    dv = _read(_path("dividend", code))
    for _, r in dv.iterrows():
        rows.append({"date": r["date"], "kind": "dividend", "before": r["before_price"], "after": r["after_price"],
                     "cash": r["stock_and_cache_dividend"], "factor": r["after_price"] / r["before_price"], "share_ratio": 1.0})
    sp = _read(ETF_DIR / "splits.csv.gz", dtype={"stock_id": str})
    for _, r in sp[sp["stock_id"] == code].iterrows() if len(sp) else []:
        ratio = _snap(r["before_price"] / r["after_price"])
        rows.append({"date": r["date"], "kind": "split" if ratio > 1 else "reverse_split", "before": r["before_price"],
                     "after": r["after_price"], "cash": 0.0, "factor": 1 / ratio, "share_ratio": ratio})
    out = pd.DataFrame(rows, columns=["date", "kind", "before", "after", "cash", "factor", "share_ratio"])
    out["date"] = pd.to_datetime(out["date"])
    return out.set_index("date").sort_index()


def adjust(raw: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """回溯還原（最新價格 = 實際價格）：事件日之前的 OHLC 乘上累積 factor，成交量乘上累積 share_ratio。
    事件日不在 raw 裡（如停止買賣期間）時套用到該日之前最後一根 K 棒。"""
    df = raw.copy()
    px = ["open", "high", "low", "close"]
    f = np.ones(len(df))
    v = np.ones(len(df))
    for d, a in actions.iterrows():
        before = df.index < d
        f[before] *= a["factor"]
        v[before] *= a["share_ratio"]
    df[px] = df[px].mul(f, axis=0)
    df["volume"] = df["volume"] * v
    df["factor"] = f
    return df


# ------------------------------------------------------------------------------------------------ underlying / proxy


def _on_tw(series: pd.Series, index, lag=1, tol=7) -> pd.Series:
    """把美國日期的序列對到台股日期：lag=1 取「t 之前（不含 t）最後一筆」，lag=0 取「t 當日或之前」。
    超過 tol 個日曆日沒有新值則為 NaN。"""
    s = series.dropna().sort_index()
    idx = pd.DatetimeIndex(index)
    a, t = s.index.values, idx.values
    pos = np.searchsorted(a, t, side="left" if lag else "right") - 1
    ok = pos >= 0
    ok[ok] = (t[ok] - a[pos[ok]]) <= np.timedelta64(tol, "D")
    vals = s.to_numpy()[np.where(ok, pos, 0)].astype(float)
    vals[~ok] = np.nan
    return pd.Series(vals, index=idx)


def _fx(index) -> pd.Series:
    """台銀 USD/TWD（台股日 t 營業時間內的牌告中價）。"""
    return ext.align(ext.market_raw("usdtwd").to_frame(), index)["usdtwd"]


def _carry(index) -> pd.Series:
    """每個台股日區間的美國 3 個月 T-bill 應計報酬（對數），用 t 之前最後一個美國日的 ^IRX 年化殖利率 × 日曆天/365。"""
    idx = pd.DatetimeIndex(index)
    y = _on_tw(ext.market_raw("us3m"), idx).ffill() / 100
    days = pd.Series(idx, index=idx).diff().dt.days.fillna(0)
    return np.log1p(y.shift(1).fillna(y) * days / 365)


def underlying_twd(code, index) -> pd.Series:
    """異常偵測用的標的價位（台幣），對到 index：0050 用加權報酬指數（同日），其他用美國 ETF 還原收盤 × USD/TWD（lag 1）。"""
    und = ETFS[code]["underlying"]
    if und == "taiex_tr":
        return ext.align(ext.market_raw("taiex_tr").to_frame(), index)["taiex_tr"]
    return _on_tw(us_etf(und)["adj_close"], index) * _fx(index)


def _proxy_close(code, index, carry) -> pd.Series:
    lvl = np.log(_on_tw(us_etf(ETFS[code]["proxy"])["adj_close"], index) * _fx(index))
    r = lvl.diff()
    if carry:
        r = r - _carry(index)
    return np.exp(r.fillna(0).cumsum()).where(lvl.notna())


def load_proxy(code, start=START, end=None, full=False, carry=None) -> pd.DataFrame:
    """上市前的代理序列（明確標示，不會自動混入 load_etf）：美國 ETF 還原收盤 × 台銀 USD/TWD，台股日 t 對應 t 之前
    最後一個美國收盤（lag 1）。carry=True（黃金、原油預設）再扣美國 T-bill 利率，近似期貨 Excess Return 指數。
    open = high = low = close（只有收盤價有意義；引擎在 t+1 開盤成交 = 以 t+1 代理收盤成交，偏保守），volume = NaN。
    full=False 只回傳上市日之前的日期；full=True 回傳整段（含與 ETF 重疊期，供比對）。0050 沒有代理。"""
    meta = ETFS[code]
    if meta["proxy"] is None:
        raise ValueError(f"{code} 不需要也沒有代理序列（上市於 {meta['listed']}，早於 START）")
    carry = CARRY_DEFAULT[code] if carry is None else carry
    idx = ext.calendar(end)
    idx = idx[idx >= pd.Timestamp(start)] if start is not None else idx
    c = _proxy_close(code, idx, carry).dropna()
    if not full:
        c = c[c.index < pd.Timestamp(meta["listed"])]
    out = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": np.nan})
    out.index.name = "date"
    return out


def load_extended(code, start=START, end=None, carry=None) -> pd.DataFrame:
    """明確拼接的長歷史：上市前為代理（整段縮放到上市首日收盤＝ETF 還原收盤），上市後為 load_etf()。
    多一欄 is_proxy（1.0 = 代理、0.0 = 真實 ETF）；回測報告必須分開呈現兩段。"""
    real = load_etf(code, start=None, end=end)
    real = real[real.index >= pd.Timestamp(start)] if start is not None else real
    if ETFS[code]["proxy"] is None:
        return real.assign(is_proxy=0.0)
    px = load_proxy(code, start=start, end=end, full=True, carry=carry)
    first = real.index[0]
    if first not in px.index:
        return real.assign(is_proxy=0.0)
    scale = real["close"].iloc[0] / px.loc[first, "close"]
    pre = px[px.index < first].copy()
    pre[["open", "high", "low", "close"]] *= scale
    return pd.concat([pre.assign(is_proxy=1.0), real.assign(is_proxy=0.0)])


# ------------------------------------------------------------------------------------------------ cleaning

SPIKE_SIGMA = 6.0  # 殘差 > 6 倍穩健標準差
SPIKE_MIN = 0.04  # 且 > 4%
STALE_RUN = 3  # 連續 >= 3 根收盤價完全相同


def _robust_sigma(x: pd.Series, window=250) -> pd.Series:
    mad = (x - x.rolling(window, center=True, min_periods=60).median()).abs()
    return 1.4826 * mad.rolling(window, center=True, min_periods=60).median()


def bad_bars(code, raw=None) -> pd.DataFrame:
    """ETF 專屬異常偵測（不用 7%/10% 漲跌幅規則：追蹤國外資產的 ETF 沒有漲跌幅限制）。
    在還原後的價格上找：
      spike  收盤報酬減掉標的（台幣、lag 1）報酬的殘差超過 max(6σ, 4%)，且次日殘差反向回補一半以上（孤立尖峰）；
      stale  連續 >= 3 根收盤價完全相同，且期間標的累積變動 > 3σ；
      jump   單日 |對數報酬| > 35% 且非已知分割日（未處理的分割／反分割）。
    處置（action）：若 Yahoo 該日收盤與 FinMind 不同且與標的一致（Yahoo 殘差正常）→ replace（以 Yahoo 取代收盤）；
    兩來源一致 → keep（真實成交，多為折溢價變動）。回傳 DataFrame（date, kind, ret, und_ret, resid, yahoo_close, action）。"""
    if raw is None:
        raw = _basic(code)[0]
    acts = corporate_actions(code)
    adj = adjust(raw, acts)
    r = np.log(adj["close"]).diff()
    u = np.log(underlying_twd(code, adj.index)).diff()
    ok = u.notna() & r.notna()
    beta = float(np.polyfit(u[ok], r[ok], 1)[0]) if ok.sum() > 100 else 1.0
    e = (r - beta * u.fillna(0)).where(ok)
    sig = _robust_sigma(e).bfill().ffill()
    thr = np.maximum(SPIKE_SIGMA * sig, SPIKE_MIN)
    rows = []
    nxt = e.shift(-1)
    spike = (e.abs() > thr) & (np.sign(nxt) == -np.sign(e)) & (nxt.abs() > 0.5 * e.abs())
    for d in e.index[spike.fillna(False)]:
        rows.append((d, "spike"))
    c = raw["close"]
    run_id = (c != c.shift()).cumsum()
    runs = c.groupby(run_id).agg(["size"]).join(pd.Series(c.index, index=c.index).groupby(run_id).agg(["first", "last"]))
    for _, rr in runs[runs["size"] >= STALE_RUN].iterrows():
        move = u.loc[rr["first"]:rr["last"]].iloc[1:].sum()
        if abs(move) > 3 * sig.loc[rr["last"]] * np.sqrt(rr["size"] - 1):
            rows.append((rr["last"], "stale"))
    split_days = set(acts.index[acts["kind"] != "dividend"])
    for d in r.index[(r.abs() > 0.35).fillna(False)]:
        if d not in split_days:
            rows.append((d, "jump"))
    out = []
    try:
        y = raw_prices(code, "yahoo")["close"]
    except FileNotFoundError:
        y = pd.Series(dtype=float)
    for d, kind in rows:
        yc = y.get(d, np.nan)
        action = "keep"
        if kind == "spike" and np.isfinite(yc) and abs(yc / raw.loc[d, "close"] - 1) > 0.01:
            prev = raw["close"].shift().loc[d]
            y_resid = np.log(yc / prev) - beta * u.loc[d]
            if abs(y_resid) < thr.loc[d] / 2:
                action = "replace"
        out.append({"date": d, "kind": kind, "ret": r.loc[d], "und_ret": u.loc[d], "resid": e.loc[d],
                    "yahoo_close": yc, "close": raw.loc[d, "close"], "action": action})
    return pd.DataFrame(out, columns=["date", "kind", "ret", "und_ret", "resid", "yahoo_close", "close", "action"])


def _basic(code):
    """基本清理：重複日期、非正／缺值價格、成交量 0（無成交 K 棒）、OHLC 不一致。回傳 (raw, 計數 dict)。"""
    d = raw_prices(code).copy()
    rep = {"rows": len(d)}
    dup = d.index.duplicated(keep="last")
    d = d[~dup]
    px = ["open", "high", "low", "close"]
    bad = d[px].isna().any(axis=1) | (d[px] <= 0).any(axis=1) | d["volume"].isna()
    zero = d["volume"] <= 0
    d = d[~bad & ~zero]
    hi = d[px].max(axis=1)
    lo = d[px].min(axis=1)
    fix = (d["high"] < hi) | (d["low"] > lo)
    d["high"], d["low"] = hi, lo
    rep |= {"duplicates": int(dup.sum()), "bad_price": int(bad.sum()), "zero_volume": int((zero & ~bad).sum()),
            "ohlc_fixed": int(fix.sum())}
    return d, rep


def _clean(code):
    if ("clean", code) not in _memo:
        raw, rep = _basic(code)
        flags = bad_bars(code, raw)
        rep["flag_spike"] = int((flags["kind"] == "spike").sum())
        rep["flag_stale"] = int((flags["kind"] == "stale").sum())
        rep["flag_jump"] = int((flags["kind"] == "jump").sum())
        rep["replaced"] = 0
        for _, f in flags[flags["action"] == "replace"].iterrows():
            d, yc = f["date"], f["yahoo_close"]
            raw.loc[d, "close"] = yc
            raw.loc[d, "high"] = max(raw.loc[d, "high"], yc)
            raw.loc[d, "low"] = min(raw.loc[d, "low"], yc)
            rep["replaced"] += 1
        acts = corporate_actions(code)
        rep["dividends"] = int((acts["kind"] == "dividend").sum())
        rep["splits"] = int((acts["kind"] != "dividend").sum())
        _memo[("clean", code)] = (adjust(raw, acts), rep, flags)
    return _memo[("clean", code)]


def load_etf(code, start=START, end=None, adjusted=True, extra=False) -> pd.DataFrame:
    """清理後的日 K（與 stocklab.data.load 同格式：open, high, low, close, volume，DatetimeIndex 名為 date）。
    adjusted=True：總報酬還原（配息再投入、分割調整，最新價 = 實際價；成交量以最新股數為單位）。
    adjusted=False：實際成交價（分割前後價位不連續，0050 於 2025-06-18 1 拆 4）。
    extra=True 另附 raw_open, raw_close（實際價）、factor（還原因子）、cash（當日除息金額）、value（成交金額，元）。
    start 預設 stocklab.data.START（2008-01-01），start=None 取上市以來全部。"""
    df, _, _ = _clean(code)
    raw = raw_prices(code).reindex(df.index)
    out = df[["open", "high", "low", "close", "volume"]].copy()
    if not adjusted:
        out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]].div(df["factor"], axis=0)
        acts = corporate_actions(code)
        ratio = np.ones(len(df))
        for d, a in acts.iterrows():
            ratio[df.index < d] *= a["share_ratio"]
        out["volume"] = out["volume"] / ratio
    if extra:
        acts = corporate_actions(code)
        out["raw_open"] = df["open"] / df["factor"]
        out["raw_close"] = df["close"] / df["factor"]
        out["factor"] = df["factor"]
        out["cash"] = acts.loc[acts["kind"] == "dividend", "cash"].groupby(level=0).sum().reindex(df.index).fillna(0.0)
        out["value"] = raw["value"]
    if start is not None:
        out = out[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out[out.index <= pd.Timestamp(end)]
    out.index.name = "date"
    return out


def load_all_etfs(codes=None, **kw) -> dict[str, pd.DataFrame]:
    return {c: load_etf(c, **kw) for c in (codes or CODES)}


def cleaning_report(codes=None) -> pd.DataFrame:
    rows = []
    for c in codes or CODES:
        df, rep, _ = _clean(c)
        rows.append({"code": c, "first": df.index[0].date(), "last": df.index[-1].date(), **rep})
    return pd.DataFrame(rows).set_index("code")


# ------------------------------------------------------------------------------------------------ costs / margin


def sell_tax(code, dates) -> pd.Series:
    """賣出證交稅率：ETF 1‰；債券 ETF（00679B）在 BOND_TAX_EXEMPT 期間停徵（0）。"""
    idx = pd.DatetimeIndex(pd.to_datetime(dates if not isinstance(dates, (str, pd.Timestamp)) else [dates]))
    tax = pd.Series(ETFS[code]["tax"], index=idx, dtype=float)
    ex = ETFS[code]["tax_exempt"]
    if ex:
        tax[(idx >= pd.Timestamp(ex[0])) & (idx <= pd.Timestamp(ex[1]))] = 0.0
    return tax


def costs(code, date) -> dict:
    """單邊成本（比例）：buy = 手續費；sell = 手續費 + 證交稅（依日期）。融券費用（借券費、融券手續費）不在此。"""
    return {"buy": COMMISSION, "sell": COMMISSION + float(sell_tax(code, date).iloc[0])}


def load_margin(code) -> pd.DataFrame:
    """FinMind 融資融券日資料（股數單位為「張」= 1000 股），含 note（交易所註記：X 停止融券、O 停止融資、
    ! 停止買賣…；註記代表「次一營業日」的狀態）。以交易日為索引。"""
    d = _read(_path("margin", code), parse_dates=["date"], dtype={"Note": str}).set_index("date").sort_index()
    d = d.rename(columns={"Note": "note"})
    d["note"] = d["note"].fillna("").str.strip()
    return d.drop(columns=["stock_id"], errors="ignore")


# ------------------------------------------------------------------------------------------------ tracking


def tracking_stats(code, carry=None, start=None, end=None) -> dict:
    """代理（load_proxy full=True）與 ETF 總報酬（load_etf）在重疊期的追蹤統計（對數報酬）：
    日／週／月相關係數、週 beta、年化追蹤誤差（日、週）、年化報酬（ETF、代理）與年化差距。"""
    carry = CARRY_DEFAULT[code] if carry is None else carry
    etf = load_etf(code, start=None, end=end)["close"]
    px = load_proxy(code, start=None, end=end, full=True, carry=carry)["close"]
    j = pd.concat([np.log(etf), np.log(px)], axis=1, keys=["etf", "proxy"]).dropna()
    if start is not None:
        j = j[j.index >= pd.Timestamp(start)]
    r = j.diff().dropna()
    w = r.resample("W-FRI").sum()
    m = r.resample("ME").sum()
    years = (j.index[-1] - j.index[0]).days / 365.25
    cagr = np.exp((j.iloc[-1] - j.iloc[0]) / years) - 1
    return {
        "code": code, "proxy": ETFS[code]["proxy"] + (" - T-bill" if carry else ""), "from": j.index[0].date(),
        "to": j.index[-1].date(), "days": len(r),
        "corr_d": r.corr().iloc[0, 1], "corr_w": w.corr().iloc[0, 1], "corr_m": m.corr().iloc[0, 1],
        "beta_w": float(np.cov(w["etf"], w["proxy"])[0, 1] / w["proxy"].var()),
        "te_d": float((r["etf"] - r["proxy"]).std() * np.sqrt(252)),
        "te_w": float((w["etf"] - w["proxy"]).std() * np.sqrt(52)),
        "cagr_etf": float(cagr["etf"]), "cagr_proxy": float(cagr["proxy"]),
        "gap": float(cagr["etf"] - cagr["proxy"]),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download Taiwan-listed ETF data into data/external/etf/ and report.")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    download_etfs(refresh=args.refresh)
    pd.set_option("display.width", 220)
    print(cleaning_report().to_string())
    for c in CODES:
        f = bad_bars(c)
        if len(f):
            print(c, "flags:\n", f.round(4).to_string(index=False))
    rows = [tracking_stats(c, carry=k) for c in CODES if ETFS[c]["proxy"] for k in (False, True)]
    print(pd.DataFrame(rows).round(4).to_string(index=False))
