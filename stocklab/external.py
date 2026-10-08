"""External model inputs (global markets, Taiwan chip / valuation / revenue data), aligned without lookahead.

Sources -- all free, no account or token needed:
  * Yahoo Finance via yfinance: Asian / US indices, ADRs, USD/JPY, US rates, commodities.
  * FinMind open API v4 (https://api.finmindtrade.com/api/v4/data) without a token: per-stock 三大法人, 外資持股,
    融資融券, 借券賣出餘額, 本益比/殖利率/股價淨值比, 月營收; market-wide 三大法人 / 融資融券; TAIEX OHLC + 成交金額;
    TAIFEX 三大法人 positions in 台指期 (TX) and 台指選擇權 (TXO); Bank of Taiwan USD/TWD. One request returns a
    stock's whole history.
    Unregistered use is limited to roughly 300 requests / hour; an optional user-registered token can be put in the
    FINMIND_TOKEN environment variable (never required here).
  * TAIFEX official CSV download (www.taifex.com.tw/cht/3/pcRatioDown): 台指選擇權 put/call ratio, one month per call.

Raw downloads (from stocklab.data.START = 2008-01-01 where the source has it) are cached under data/external/
(gitignored) and never re-downloaded unless refresh=True; data/external/manifest.json records the requested start of
each file so earlier history is backfilled only once. Nothing touches the network at import time; loaders only read
the cache. Run ``python -m stocklab.external`` once to download everything and print coverage.

Public API (all return float DataFrames indexed by TW trading dates; all take ``end=``):
  calendar(end)                                    TW trading dates
  load_market(columns, index, at_close, end)       dates x MARKET columns (global + TW market-wide)
  load_ohlcv(name, index, at_close, end)           OHLCV of one Yahoo series or "taiex"
  load_stock_fields(code, fields, index, at_close, end)   dates x STOCK_FIELDS for one stock
  load_stock_panel(field, codes, index, at_close, end)    dates x codes for one STOCK_FIELDS field
  load_price_panel(field, codes, index, end)       dates x codes of cleaned adjusted prices (stocklab.data.load)
  industry(codes)                                  static 產業別 per stock
  data_end(data)                                   last date in a strategy's data dict -> pass as end=

Strategies must cut external data at the last date of the ``data`` they receive, so the runner's truncation check
also cuts it: ``ext.load_market(..., end=ext.data_end(data))``. ``index=`` may be any sorted date index (e.g. one stock's
``df.index``); the default is calendar().

Alignment contract (every loader)
---------------------------------
A strategy's target at Taiwan bar t is filled at the open (09:00 Taipei) of the next TW trading day. Each raw record
gets an *availability date* ``avail``: the Taipei calendar date by whose end -- at the latest before 09:00 of the next
calendar day -- the record is public. For each requested date t the loaders return the most recent record with
``avail <= t`` and at most ``tol`` calendar days old (a forward fill of past values only; NaN before the first record
or when the last one is staler than ``tol``). Since the next open is on a later calendar day than t, every value
returned at t was public before the fill at bar t+1.

The rule depends on t alone, not on which date the next trading day is, so values at t never change when later data
arrives (the project's truncation check holds). The price is a little conservatism around TW holidays: a US close on a
day Taiwan is shut is first used at the next TW bar, although it was technically known one bar earlier.

at_close=True is the stricter "decide at the TW close of t" mode (what a MultiCharts signal computed on the bar close
can see): TW close prices of t, every other series only from dates before t. Per-day flows (FLOWS: net buys, turnover,
new SBL shorts) appear only on the first bar that picks a record up and are NaN on bars without a new record; levels
are carried forward.

  series                                   published (Taipei time)                         avail
  TW close-of-day prices for d (TAIEX, stocks)         13:30 on d                           d
  TWSE / TAIFEX after-close statistics for d            ~15:00 (三大法人, 期權) to ~21:30    d
                                                        (融資融券, 借券) on d
  Asian closes for d (Nikkei, KOSPI, HSI, Shanghai)     14:00-16:10 on d                     d
  US / CME closes for d (16:00-17:00 New York)          04:00-06:00 on d+1 (< 09:00)         d
  Yahoo FX daily bars for d (close ~23:00 London)       06:00-07:00 on d+1 (< 09:00)         d
  Bank of Taiwan USD/TWD board rate for d               during TW business hours of d        d
  monthly revenue for month m                           legal deadline: 10th of m+1          first TW trading day
                                                        (next business day if a holiday)     >= that 10th, plus
                                                                                             REVENUE_EXTRA_LAG days

Caveats: FinMind keeps the latest (possibly restated) monthly revenue; Yahoo adjusted closes (TSM, NVDA, EWT)
change level when new dividends are adjusted -- use returns, not levels.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .data import RAW_DIR, ROOT, START
from .universe import CODES

EXT_DIR = ROOT / "data" / "external"
MANIFEST = EXT_DIR / "manifest.json"  # start date requested for each cached file, so history is backfilled only once
TPE = timezone(timedelta(hours=8))
DAILY_TOL = 14  # calendar days a daily value may be carried forward (covers Lunar New Year closures)
MONTHLY_TOL = 75  # calendar days a monthly revenue value may be carried forward (one missed month)
REVENUE_DEADLINE_DAY = 10
REVENUE_EXTRA_LAG = 1  # safety margin: extra TW trading days after the first trading day >= the 10th

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
TAIFEX_PC_URL = "https://www.taifex.com.tw/cht/3/pcRatioDown"
PAUSE = {"finmind": 1.2, "taifex": 2.0}  # seconds between calls
_last_call: dict[str, float] = {}
_cache: dict = {}  # parsed cache files and the calendar, per process; cleared after downloads

TIMING = {
    "tw": "TW close 13:30 on d -> avail d",
    "tw_eve": "TWSE/TAIFEX statistics for d, published after the close on d (15:00-21:30) -> avail d",
    "asia": "Asian close of d (14:00-16:10 Taipei) -> avail d",
    "us": "US/CME close of d = 04:00-06:00 Taipei on d+1, before the next TW open -> avail d",
    "fx": "Yahoo FX bar of d closes ~23:00 London = 06:00-07:00 Taipei on d+1 -> avail d",
    "bot": "Bank of Taiwan board rate of d, posted during TW business hours of d -> avail d",
    "revenue": "month m: legal deadline the 10th of m+1 -> avail = first TW trading day >= 10th + REVENUE_EXTRA_LAG",
}

# ------------------------------------------------------------------------------------------------ registries

# name: (Yahoo symbol, description, timing key). load_market() exposes the adjusted close under `name`.
YAHOO = {
    "n225": ("^N225", "Nikkei 225", "asia"),
    "kospi": ("^KS11", "KOSPI", "asia"),
    "hsi": ("^HSI", "Hang Seng", "asia"),
    "sse": ("000001.SS", "Shanghai Composite", "asia"),
    "spx": ("^GSPC", "S&P 500", "us"),
    "nasdaq": ("^IXIC", "Nasdaq Composite", "us"),
    "sox": ("^SOX", "PHLX Semiconductor", "us"),
    "vix": ("^VIX", "CBOE VIX", "us"),
    "tsm": ("TSM", "TSMC ADR (adjusted; 1 ADR = 5 shares of 2330)", "us"),
    "nvda": ("NVDA", "NVIDIA (adjusted)", "us"),
    "ewt": ("EWT", "iShares MSCI Taiwan ETF (adjusted)", "us"),
    "us10y": ("^TNX", "US 10-year Treasury yield, %", "us"),
    "us3m": ("^IRX", "US 13-week T-bill yield, %", "us"),
    "dxy": ("DX-Y.NYB", "US dollar index", "us"),
    "wti": ("CL=F", "WTI crude front-month future", "us"),
    "gold": ("GC=F", "Gold front-month future", "us"),
    "copper": ("HG=F", "Copper front-month future", "us"),
    "usdjpy": ("JPY=X", "USD/JPY", "fx"),
}
# Tried and dropped: 0050.TW (Yahoo bars broken: 2014-01-02 -75%, weeks of stale closes; use taiex_tr), TWD=X (bad
# ticks and +-3% day-to-day flip-flops in 2011-2016; usdtwd comes from Bank of Taiwan via FinMind instead), KRW=X
# (noisy), ^TWOII (not on Yahoo).

# cache key: (FinMind dataset, data_id)
FM_MARKET = {
    "taiex": ("TaiwanStockPrice", "TAIEX"),
    "taiex_tr": ("TaiwanStockTotalReturnIndex", "TAIEX"),
    "inst_total": ("TaiwanStockTotalInstitutionalInvestors", None),
    "margin_total": ("TaiwanStockTotalMarginPurchaseShortSale", None),
    "fut_inst_tx": ("TaiwanFuturesInstitutionalInvestors", "TX"),
    "opt_inst_txo": ("TaiwanOptionInstitutionalInvestors", "TXO"),
    "usdtwd": ("TaiwanExchangeRate", "USD"),
    "stock_info": ("TaiwanStockInfo", None),  # static industry labels, see industry()
}
# Where the source's history actually begins (no point requesting earlier); revenue starts a year before START so
# that revenue_yoy exists from START on. None: not a time series.
FM_FIRST = {"inst": "2012-05-02", "fut_inst_tx": "2018-06-05", "opt_inst_txo": "2018-06-05", "stock_info": None,
            "revenue": f"{pd.Timestamp(START).year - 1}-01-01"}
FM_STOCK = {
    "inst": "TaiwanStockInstitutionalInvestorsBuySell",
    "holding": "TaiwanStockShareholding",
    "margin": "TaiwanStockMarginPurchaseShortSale",
    "sbl": "TaiwanDailyShortSaleBalances",
    "per": "TaiwanStockPER",
    "revenue": "TaiwanStockMonthRevenue",
}

# Non-Yahoo market columns: column -> (group, unit, timing key, description)
MARKET_EXTRA = {
    "taiex": ("taiex", "index", "tw", "TAIEX close (FinMind; complete TW calendar)"),
    "taiex_turnover": ("taiex", "NTD", "tw", "TWSE total traded value (成交金額)"),
    "taiex_tr": ("taiex_tr", "index", "tw", "TAIEX total-return index (發行量加權股價報酬指數)"),
    "usdtwd": ("usdtwd", "TWD per USD", "bot", "USD/TWD, Bank of Taiwan spot board rate, mid of buy/sell"),
    "mkt_foreign_net": ("inst_total", "NTD", "tw_eve", "TWSE market 外資 net buy (incl. 外資自營商)"),
    "mkt_trust_net": ("inst_total", "NTD", "tw_eve", "TWSE market 投信 net buy"),
    "mkt_dealer_net": ("inst_total", "NTD", "tw_eve", "TWSE market 自營商 net buy (自行買賣 + 避險)"),
    "mkt_margin_lots": ("margin_total", "lots (1000 sh)", "tw_eve", "market 融資餘額, lots"),
    "mkt_short_lots": ("margin_total", "lots (1000 sh)", "tw_eve", "market 融券餘額, lots"),
    "mkt_margin_value": ("margin_total", "NTD", "tw_eve", "market 融資餘額 value (NTD)"),
    "fut_foreign_net_oi": ("fut_inst_tx", "contracts", "tw_eve", "台指期 TX 外資 net open interest (long - short)"),
    "fut_trust_net_oi": ("fut_inst_tx", "contracts", "tw_eve", "台指期 TX 投信 net open interest"),
    "fut_dealer_net_oi": ("fut_inst_tx", "contracts", "tw_eve", "台指期 TX 自營商 net open interest"),
    "opt_foreign_call_net_oi": ("opt_inst_txo", "contracts", "tw_eve", "台指選 TXO 外資 call net OI (long - short)"),
    "opt_foreign_put_net_oi": ("opt_inst_txo", "contracts", "tw_eve", "台指選 TXO 外資 put net OI (long - short)"),
    "pc_ratio_vol": ("pc_ratio", "%", "tw_eve", "TXO put/call volume ratio x100 (TAIFEX)"),
    "pc_ratio_oi": ("pc_ratio", "%", "tw_eve", "TXO put/call open-interest ratio x100 (TAIFEX)"),
}
MARKET = {n: ("yahoo", "price", t, d) for n, (_, d, t) in YAHOO.items()} | MARKET_EXTRA

# Per-stock fields: field -> (group, unit, timing key, description)
STOCK_FIELDS = {
    "foreign_net": ("inst", "shares", "tw_eve", "外資 net buy (外資及陸資 + 外資自營商); from 2012-05-02"),
    "trust_net": ("inst", "shares", "tw_eve", "投信 net buy; from 2012-05-02"),
    "dealer_net": ("inst", "shares", "tw_eve", "自營商 net buy (自行買賣 + 避險); from 2012-05-02"),
    "inst_net": ("inst", "shares", "tw_eve", "三大法人 total net buy; from 2012-05-02"),
    "foreign_ratio": ("holding", "%", "tw_eve", "外資持股比例 (foreign holding / shares issued)"),
    "shares_issued": ("holding", "shares", "tw_eve", "shares issued (for normalising flows / balances)"),
    "margin_balance": ("margin", "shares", "tw_eve", "融資餘額 (source lots x 1000)"),
    "short_balance": ("margin", "shares", "tw_eve", "融券餘額 (source lots x 1000)"),
    "sbl_balance": ("sbl", "shares", "tw_eve", "借券賣出餘額"),
    "sbl_sell": ("sbl", "shares", "tw_eve", "借券賣出 (that day's new SBL short sales)"),
    "per": ("per", "x", "tw_eve", "本益比 (TWSE; NaN when not meaningful, e.g. losses)"),
    "pbr": ("per", "x", "tw_eve", "股價淨值比 (TWSE)"),
    "dividend_yield": ("per", "%", "tw_eve", "殖利率 (TWSE)"),
    "revenue": ("revenue", "NTD", "revenue", "月營收 of the latest published month (金控 can be negative)"),
    "revenue_yoy": ("revenue", "ratio", "revenue", "monthly revenue / same month a year earlier - 1 (NaN if base <= 0)"),
    "revenue_mom": ("revenue", "ratio", "revenue", "monthly revenue / previous month - 1 (NaN if base <= 0)"),
}

PRICE_FIELDS = ("open", "high", "low", "close", "volume")

# Per-day flows (not levels): shown only on the first bar where the record becomes available, NaN on later bars
# without a new record (carrying yesterday's net buy forward would invent trades). Levels are carried forward.
FLOWS = {"taiex_turnover", "mkt_foreign_net", "mkt_trust_net", "mkt_dealer_net",
         "foreign_net", "trust_net", "dealer_net", "inst_net", "sbl_sell"}

# ------------------------------------------------------------------------------------------------ alignment core


def _dt64(x) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(x).values, dtype="datetime64[ns]")


def align(raw: pd.DataFrame, index, tol: int = DAILY_TOL, repeat: bool = True) -> pd.DataFrame:
    """Lookahead-safe alignment. ``raw`` is indexed by availability date; returns ``raw`` reindexed to ``index`` (sorted)
    where the row for t is the last raw row with avail <= t, NaN if none or if it is more than ``tol`` calendar days
    old. repeat=False (flows): a raw row is shown only at the first t that picks it, NaN afterwards.
    Rows sharing an availability date: the last one in ``raw``'s order wins."""
    idx = pd.DatetimeIndex(index)
    raw = raw.astype(float)
    raw = raw.iloc[np.argsort(_dt64(raw.index), kind="stable")]
    raw = raw[~raw.index.duplicated(keep="last")]
    if raw.empty:
        return pd.DataFrame(np.nan, index=idx, columns=raw.columns)
    a, t = _dt64(raw.index), _dt64(idx)
    pos = np.searchsorted(a, t, side="right") - 1
    ok = pos >= 0
    ok[ok] = (t[ok] - a[pos[ok]]) <= np.timedelta64(tol, "D")
    if not repeat:
        ok[1:] &= pos[1:] != pos[:-1]
    vals = raw.to_numpy()[np.where(ok, pos, 0)]
    vals[~ok] = np.nan
    return pd.DataFrame(vals, index=idx, columns=raw.columns)


def calendar(end=None) -> pd.DatetimeIndex:
    """Taiwan trading dates from START (optionally <= end): union of the cached Yahoo stock bars (data/raw) and
    FinMind's TAIEX dates (complete; Yahoo misses ~18 TW sessions since 2010). No network; memoised per process."""
    if "__calendar__" not in _cache:
        _cache["__calendar__"] = _build_calendar()
    cal = _cache["__calendar__"]
    return cal if end is None else cal[cal <= pd.Timestamp(end)]


def _build_calendar() -> pd.DatetimeIndex:
    parts = []
    for f in RAW_DIR.glob("*.csv"):
        parts.append(pd.read_csv(f, usecols=["date"], parse_dates=["date"])["date"])
    taiex = _fm_path("market", "taiex")
    if taiex.exists():
        parts.append(pd.to_datetime(pd.read_csv(taiex, usecols=["date"])["date"]))
    if not parts:
        raise FileNotFoundError("no price cache: run `python -m stocklab.data` or `python -m stocklab.external` first")
    cal = pd.DatetimeIndex(pd.concat(parts).unique()).sort_values()
    cal = cal[cal >= pd.Timestamp(START)].as_unit("ns")
    cal.name = "date"
    return cal


# ------------------------------------------------------------------------------------------------ downloads


def _throttle(kind):
    wait = PAUSE[kind] - (time.monotonic() - _last_call.get(kind, -1e9))
    if wait > 0:
        time.sleep(wait)
    _last_call[kind] = time.monotonic()


def _cutoff() -> pd.Timestamp:
    """Drop Yahoo bars dated on/after this day: they may still be in progress (Taipei now - 8 h)."""
    return pd.Timestamp((datetime.now(TPE) - timedelta(hours=8)).date())


def _fm_path(kind, key, code=None):
    return EXT_DIR / "finmind" / kind / (f"{key}.csv.gz" if code is None else f"{key}/{code}.csv.gz")


def finmind(dataset, data_id=None, start=START, end=None, max_wait_min=75) -> pd.DataFrame:
    """One FinMind v4 request (whole history for one stock). Waits and retries when the hourly limit is hit."""
    params = {"dataset": dataset, "start_date": start, "end_date": end or datetime.now(TPE).strftime("%Y-%m-%d")}
    if data_id:
        params["data_id"] = data_id
    token = os.environ.get("FINMIND_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    import requests

    waited = 0
    while True:
        _throttle("finmind")
        try:
            r = requests.get(FINMIND_URL, params=params, headers=headers, timeout=120)
            j = r.json()
        except Exception as e:  # network hiccup / non-JSON reply
            j, r = {"msg": repr(e)}, None
        status = r.status_code if r is not None else -1
        if status == 200 and "data" in j:
            return pd.DataFrame(j["data"])
        msg = str(j.get("msg", ""))
        if (status in (402, 429) or "limit" in msg.lower() or r is None) and waited < max_wait_min:
            print(f"  FinMind {dataset} {data_id}: {status} {msg[:80]} -> waiting 5 min", flush=True)
            time.sleep(300)
            waited += 5
            continue
        raise RuntimeError(f"FinMind {dataset} {data_id}: HTTP {status} {msg}")


def _save_fm(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.drop(columns=[c for c in ("stock_name", "InternationalCode", "note", "Note", "country") if c in df])
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, compression="gzip", encoding="utf-8")
    tmp.replace(path)


def _manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _set_manifest(path, start):
    m = _manifest()
    m[path.relative_to(EXT_DIR).as_posix()] = str(start)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=0, sort_keys=True), encoding="utf-8")


def _cached_start(path):
    """Start date that was requested for a cached file (files cached before the manifest existed: 2010-01-01)."""
    if not path.exists():
        return None
    return _manifest().get(path.relative_to(EXT_DIR).as_posix(), "2010-01-01")


def download_yahoo(refresh=False):
    """All YAHOO symbols in one yfinance call from START; re-fetched only if missing or cached from a later start."""
    import yfinance as yf

    out = EXT_DIR / "yahoo"
    out.mkdir(parents=True, exist_ok=True)
    todo = {}
    for n, (sym, _, _) in YAHOO.items():
        have = _cached_start(out / f"{n}.csv.gz")
        if refresh or have is None or have > START:
            todo[n] = sym
    if not todo:
        return
    print(f"Yahoo: {len(todo)} symbols from {START}", flush=True)
    raw = yf.download(list(todo.values()), start=START, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    cutoff = _cutoff()
    for name, sym in todo.items():
        if sym not in raw.columns.get_level_values(0):
            print(f"  Yahoo {sym}: no data", flush=True)
            continue
        df = raw[sym].rename(columns=str.lower)[list(PRICE_FIELDS)].dropna(subset=["close"])
        df.index = pd.DatetimeIndex(df.index).tz_localize(None) if df.index.tz is not None else pd.DatetimeIndex(df.index)
        df = df[df.index < cutoff]
        if df.empty:
            print(f"  Yahoo {sym}: empty after cleaning", flush=True)
            continue
        df.index.name = "date"
        path = out / f"{name}.csv.gz"
        df.to_csv(path, float_format="%.8g", compression="gzip")
        _set_manifest(path, START)


def _fetch_fm(dataset, data_id, path, key, refresh):
    """Download one FinMind series into ``path``. If it is cached from a later start than wanted, only the missing
    early segment is requested and prepended (once: the manifest remembers the requested start)."""
    first = FM_FIRST.get(key, START)
    want = START if first is None else first
    have = _cached_start(path)
    if refresh or have is None:
        df = finmind(dataset, data_id, start=want)
        _save_fm(df, path)
        _set_manifest(path, want)
        return f"{len(df)} rows"
    if first is None or have <= want:
        return None
    seg = finmind(dataset, data_id, start=want, end=str((pd.Timestamp(have) - pd.Timedelta(days=1)).date()))
    old = _read_csv(path)
    if len(seg):
        if len(old):
            seg = seg[[c for c in old.columns if c in seg.columns]]
        _save_fm(pd.concat([seg, old], ignore_index=True), path)
    _set_manifest(path, want)
    return f"+{len(seg)} earlier rows"


def download_finmind_market(refresh=False):
    for key, (ds, data_id) in FM_MARKET.items():
        msg = _fetch_fm(ds, data_id, _fm_path("market", key), key, refresh)
        if msg:
            print(f"FinMind market {key}: {msg}", flush=True)


def download_finmind_stocks(codes=None, refresh=False):
    codes = list(codes or CODES)
    for key, ds in FM_STOCK.items():
        n = 0
        for c in codes:
            n += _fetch_fm(ds, c, _fm_path("stock", key, c), key, refresh) is not None
        if n:
            print(f"FinMind {key} ({ds}): {n} stocks fetched", flush=True)


def _taifex_month(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    import requests

    _throttle("taifex")
    form = {"down_type": "", "queryStartDate": start.strftime("%Y/%m/%d"), "queryEndDate": end.strftime("%Y/%m/%d")}
    r = requests.post(TAIFEX_PC_URL, data=form, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    text = r.content.decode("cp950", errors="replace")
    if "alert(" in text:
        raise RuntimeError(f"TAIFEX put/call {form}: server refused ({text[-200:]!r})")
    df = pd.read_csv(io.StringIO(text), index_col=False)
    df = df.iloc[:, :7]
    df.columns = ["date", "put_vol", "call_vol", "pc_ratio_vol", "put_oi", "call_oi", "pc_ratio_oi"]
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d").dt.strftime("%Y-%m-%d")
    return df


def download_taifex_pc(refresh=False):
    """TAIFEX 台指選擇權 put/call ratios, one calendar month per POST (longer ranges are refused), 2 s apart.
    Only missing months are fetched: those before the cached start (backfill) and, with refresh=True, those from the
    last cached month on."""
    path = EXT_DIR / "taifex" / "pc_ratio.csv"
    old = pd.read_csv(path) if path.exists() else None
    today = pd.Timestamp(datetime.now(TPE).date())
    start = pd.Timestamp(START).replace(day=1)
    if old is None:
        months = list(pd.date_range(start, today, freq="MS"))
    else:
        have = pd.Timestamp(_cached_start(path)).replace(day=1)
        months = list(pd.date_range(start, have - pd.Timedelta(days=1), freq="MS"))
        if refresh:
            months += list(pd.date_range(pd.Timestamp(old["date"].max()).replace(day=1), today, freq="MS"))
    if not months:
        return
    print(f"TAIFEX put/call: {len(months)} monthly requests", flush=True)
    parts = [] if old is None else [old]
    for m in months:
        parts.append(_taifex_month(m, min(m + pd.offsets.MonthEnd(0), today)))
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates("date", keep="last").sort_values("date")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    _set_manifest(path, START)


def update_fm(dataset, data_id, path, lookback_days=10):
    """Daily update of one cached FinMind series: re-fetch from `lookback_days` before its last date and replace that
    window. Returns the number of rows dated after the previous last date (None if not cached: download_all first)."""
    if not path.exists():
        return None
    old = _read_csv(path)
    if old.empty:
        return None
    last = pd.Timestamp(old["date"].max())
    start = last - pd.Timedelta(days=lookback_days)
    new = finmind(dataset, data_id, start=start.strftime("%Y-%m-%d"))
    if new.empty:
        return 0
    new = new[[c for c in old.columns if c in new.columns]]
    keep = old[pd.to_datetime(old["date"]) < start]
    _save_fm(pd.concat([keep, new], ignore_index=True), path)
    return int((pd.to_datetime(new["date"]) > last).sum())


def update_recent(codes=None, lookback_days=10, stocks=True):
    """Daily update with few requests: Yahoo re-fetched whole (one call), TAIFEX from the last cached month, every
    cached FinMind series from `lookback_days` before its last date. Monthly revenue only from the 1st to the 15th
    (when new figures appear) or when its cache is over 20 days old."""
    download_yahoo(refresh=True)
    for key, (ds, data_id) in FM_MARKET.items():
        if FM_FIRST.get(key, START) is None:  # static tables
            continue
        print(f"FinMind market {key}: +{update_fm(ds, data_id, _fm_path('market', key), lookback_days)} rows", flush=True)
    download_taifex_pc(refresh=True)
    if stocks:
        today = datetime.now(TPE)
        for key, ds in FM_STOCK.items():
            n = 0
            for c in codes or CODES:
                p = _fm_path("stock", key, c)
                if key == "revenue" and today.day > 15 and p.exists() and time.time() - p.stat().st_mtime < 20 * 86400:
                    continue
                n += update_fm(ds, c, p, lookback_days) or 0
            print(f"FinMind {key}: +{n} rows", flush=True)
    _cache.clear()


def last_session() -> pd.Timestamp:
    """Latest TWSE trading date in the cached FinMind TAIEX series (complete calendar, updated by update_recent)."""
    return pd.Timestamp(_read_csv(_fm_path("market", "taiex"))["date"].max())


def download_all(refresh=False, codes=None, stocks=True):
    """Download every source into data/external/ (only what is missing unless refresh=True). Unregistered FinMind
    allows ~300 requests / hour; a full per-stock download is ~300 requests, so expect it to pause for the limit."""
    download_yahoo(refresh)
    download_finmind_market(refresh)
    download_taifex_pc(refresh)
    if stocks:
        download_finmind_stocks(codes, refresh)
    _cache.clear()


# ------------------------------------------------------------------------------------------------ raw readers
# Each returns a float DataFrame indexed by availability date (see module docstring).


def _read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not cached: run `python -m stocklab.external` first")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _memo(key, fn):
    """Processed raw frames are memoised per process (small: a few float columns per series)."""
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def _by_date(df, cols=None):
    df = df.copy()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("date"))).as_unit("ns")
    df.index.name = "date"
    return df if cols is None else df[cols]


def _net(df, names):
    """Net (buy - sell) summed over the given investor-type names, one row per date."""
    sub = df[df["name"].isin(names)]
    return (sub["buy"] - sub["sell"]).groupby(pd.to_datetime(sub["date"])).sum()


def yahoo_raw(name) -> pd.DataFrame:
    """Unaligned Yahoo OHLCV, indexed by the bar's own (exchange-local) date == availability date."""
    return _memo(("yahoo", name), lambda: _by_date(_read_csv(EXT_DIR / "yahoo" / f"{name}.csv.gz"), list(PRICE_FIELDS)).astype(float))


# FinMind investor names: 外資及陸資 / 外資自營商 (split out from 2017-12-18); 自營商 is one "Dealer" line until
# 2014-11-28, then 自行買賣 "Dealer_self" + 避險 "Dealer_Hedging".
FOREIGN = ["Foreign_Investor", "Foreign_Dealer_Self"]
DEALER = ["Dealer", "Dealer_self", "Dealer_Hedging"]


def _group_taiex():
    d = _by_date(_read_csv(_fm_path("market", "taiex")))
    return pd.DataFrame({"taiex": d["close"], "taiex_turnover": d["Trading_money"]})


def _group_taiex_tr():
    d = _by_date(_read_csv(_fm_path("market", "taiex_tr")))
    return pd.DataFrame({"taiex_tr": d["price"]})


def _group_usdtwd():
    d = _by_date(_read_csv(_fm_path("market", "usdtwd")))
    mid = (d["spot_buy"] + d["spot_sell"]) / 2
    return pd.DataFrame({"usdtwd": mid.where((d["spot_buy"] > 0) & (d["spot_sell"] > 0))})


def _group_inst_total():
    d = _read_csv(_fm_path("market", "inst_total"))
    out = pd.DataFrame(
        {"mkt_foreign_net": _net(d, FOREIGN), "mkt_trust_net": _net(d, ["Investment_Trust"]), "mkt_dealer_net": _net(d, DEALER)}
    )
    out.index = pd.DatetimeIndex(out.index).as_unit("ns")
    return out


def _group_margin_total():
    d = _read_csv(_fm_path("market", "margin_total"))
    p = d.pivot_table(index="date", columns="name", values="TodayBalance", aggfunc="last")
    p.index = pd.DatetimeIndex(pd.to_datetime(p.index)).as_unit("ns")
    out = pd.DataFrame(index=p.index)
    out["mkt_margin_lots"] = p.get("MarginPurchase")
    out["mkt_short_lots"] = p.get("ShortSale")
    out["mkt_margin_value"] = p.get("MarginPurchaseMoney")  # NTD throughout (checked: 15k-55k NTD per lot)
    return out


def _oi_net(sub):
    return (sub["long_open_interest_balance_volume"] - sub["short_open_interest_balance_volume"]).groupby(
        pd.to_datetime(sub["date"])
    ).sum()


def _group_fut_inst_tx():
    d = _read_csv(_fm_path("market", "fut_inst_tx"))
    who = d["institutional_investors"].astype(str)
    out = pd.DataFrame(
        {
            "fut_foreign_net_oi": _oi_net(d[who.str.startswith("外資")]),
            "fut_trust_net_oi": _oi_net(d[who.str.startswith("投信")]),
            "fut_dealer_net_oi": _oi_net(d[who.str.startswith("自營")]),
        }
    )
    out.index = pd.DatetimeIndex(out.index).as_unit("ns")
    return out


def _group_opt_inst_txo():
    d = _read_csv(_fm_path("market", "opt_inst_txo"))
    f = d[d["institutional_investors"].astype(str).str.startswith("外資")]
    cp = f["call_put"].astype(str)
    out = pd.DataFrame({"opt_foreign_call_net_oi": _oi_net(f[cp == "買權"]), "opt_foreign_put_net_oi": _oi_net(f[cp == "賣權"])})
    out.index = pd.DatetimeIndex(out.index).as_unit("ns")
    return out


def _group_pc_ratio():
    d = pd.read_csv(EXT_DIR / "taifex" / "pc_ratio.csv") if (EXT_DIR / "taifex" / "pc_ratio.csv").exists() else None
    if d is None:
        raise FileNotFoundError("TAIFEX put/call not cached: run `python -m stocklab.external` first")
    return _by_date(d, ["pc_ratio_vol", "pc_ratio_oi"])


_MARKET_GROUPS = {
    "taiex": _group_taiex,
    "taiex_tr": _group_taiex_tr,
    "usdtwd": _group_usdtwd,
    "inst_total": _group_inst_total,
    "margin_total": _group_margin_total,
    "fut_inst_tx": _group_fut_inst_tx,
    "opt_inst_txo": _group_opt_inst_txo,
    "pc_ratio": _group_pc_ratio,
}


def market_raw(column) -> pd.Series:
    """Unaligned series for one load_market column, indexed by availability date (== source date for these)."""
    group = MARKET[column][0]
    if group == "yahoo":
        return yahoo_raw(column)["close"].rename(column)
    return market_group(group)[column].dropna()


def market_group(group) -> pd.DataFrame:
    return _memo(("market", group), lambda: _MARKET_GROUPS[group]().astype(float))


def revenue_avail(year, month, cal=None) -> pd.Timestamp:
    """Availability date of month (year, month)'s revenue: first TW trading day on/after the 10th of the next month,
    then REVENUE_EXTRA_LAG more trading days. NaT if the calendar does not reach that far yet."""
    cal = calendar() if cal is None else cal
    deadline = pd.Timestamp(year, month, REVENUE_DEADLINE_DAY) + pd.DateOffset(months=1)
    i = cal.searchsorted(deadline, side="left") + REVENUE_EXTRA_LAG
    return cal[i] if i < len(cal) else pd.NaT


def _stock_inst(code):
    d = _read_csv(_fm_path("stock", "inst", code))
    if d.empty:
        return pd.DataFrame(columns=["foreign_net", "trust_net", "dealer_net", "inst_net"])
    out = pd.DataFrame({"foreign_net": _net(d, FOREIGN), "trust_net": _net(d, ["Investment_Trust"]), "dealer_net": _net(d, DEALER)})
    out = out.fillna(0.0)  # a name missing on a date means no trades by that investor type
    out["inst_net"] = out.sum(axis=1)
    out.index = pd.DatetimeIndex(out.index).as_unit("ns")
    return out


def _stock_holding(code):
    d = _read_csv(_fm_path("stock", "holding", code))
    if d.empty:
        return pd.DataFrame(columns=["foreign_ratio", "shares_issued"])
    d = _by_date(d)
    return pd.DataFrame({"foreign_ratio": d["ForeignInvestmentSharesRatio"], "shares_issued": d["NumberOfSharesIssued"]})


def _stock_margin(code):
    d = _read_csv(_fm_path("stock", "margin", code))
    if d.empty:
        return pd.DataFrame(columns=["margin_balance", "short_balance"])
    d = _by_date(d)
    return pd.DataFrame({"margin_balance": d["MarginPurchaseTodayBalance"] * 1000.0, "short_balance": d["ShortSaleTodayBalance"] * 1000.0})


def _stock_sbl(code):
    d = _read_csv(_fm_path("stock", "sbl", code))
    if d.empty:
        return pd.DataFrame(columns=["sbl_balance", "sbl_sell"])
    d = _by_date(d)
    return pd.DataFrame({"sbl_balance": d["SBLShortSalesCurrentDayBalance"], "sbl_sell": d["SBLShortSalesShortSales"]})


def _stock_per(code):
    d = _read_csv(_fm_path("stock", "per", code))
    if d.empty:
        return pd.DataFrame(columns=["per", "pbr", "dividend_yield"])
    d = _by_date(d)
    return pd.DataFrame(
        {"per": d["PER"].where(d["PER"] > 0), "pbr": d["PBR"].where(d["PBR"] > 0), "dividend_yield": d["dividend_yield"]}
    )


def _stock_revenue(code, cal=None):
    d = _read_csv(_fm_path("stock", "revenue", code))
    cols = ["revenue", "revenue_yoy", "revenue_mom", "year", "month"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    cal = calendar() if cal is None else cal
    d = d.drop_duplicates(["revenue_year", "revenue_month"], keep="last")
    per = pd.PeriodIndex.from_fields(year=d["revenue_year"], month=d["revenue_month"], freq="M")
    rev = pd.Series(d["revenue"].to_numpy(dtype=float), index=per).sort_index()
    # financial holdings can report negative monthly net revenue: keep it, but growth needs a positive base
    base = rev.where(rev > 0)
    yoy = rev / base.reindex(rev.index - 12).to_numpy() - 1
    mom = rev / base.reindex(rev.index - 1).to_numpy() - 1
    avail = [revenue_avail(p.year, p.month, cal) for p in rev.index]
    out = pd.DataFrame(
        {"revenue": rev.to_numpy(), "revenue_yoy": yoy.to_numpy(), "revenue_mom": mom.to_numpy(),
         "year": rev.index.year, "month": rev.index.month},
        index=pd.DatetimeIndex(avail).as_unit("ns"),
    )
    return out[out.index.notna()]


_STOCK_GROUPS = {
    "inst": _stock_inst,
    "holding": _stock_holding,
    "margin": _stock_margin,
    "sbl": _stock_sbl,
    "per": _stock_per,
    "revenue": _stock_revenue,
}


def stock_raw(code, group) -> pd.DataFrame:
    """Unaligned per-stock frame for one group, indexed by availability date (for revenue: the publication-rule date,
    with the revenue month in columns year/month)."""
    return _memo(("stock", code, group), lambda: _STOCK_GROUPS[group](code).astype(float))


# ------------------------------------------------------------------------------------------------ public loaders
#
# at_close=False (default): value at t = latest record public before the open of the TW session after t (avail <= t).
#   US closes of calendar date t itself are included (they print at 04:00-06:00 Taipei on t+1).
# at_close=True: only what is public by the TW close (13:30) of t, i.e. what a signal computed on the bar close (e.g.
#   in MultiCharts) can see: TW close prices of t, everything else from dates < t (avail shifted by one calendar day).
# end: drop every date after `end` (inclusive bound). Strategies must pass end=data_end(data) so that the runner's
#   truncation lookahead check, which cuts `data`, also cuts the external inputs.


def _tol(timing):
    return MONTHLY_TOL if timing == "revenue" else DAILY_TOL


def _shift(raw, timing, at_close):
    if at_close and timing != "tw":
        raw = raw.copy()
        raw.index = raw.index + pd.Timedelta(days=1)
    return raw


def _index(index, end=None):
    idx = calendar() if index is None else pd.DatetimeIndex(index)
    return idx if end is None else idx[idx <= pd.Timestamp(end)]


def data_end(data) -> pd.Timestamp:
    """Last date present in a strategy's ``data`` (dict code -> DataFrame, or one DataFrame / Series). Pass it as
    ``end=`` to every loader: e.g. ``ext.load_market(end=ext.data_end(data))``."""
    frames = data.values() if isinstance(data, dict) else [data]
    return max(f.index[-1] for f in frames if len(f))


def load_market(columns=None, index=None, at_close=False, end=None) -> pd.DataFrame:
    """Market-wide / global series (MARKET) aligned to TW dates (default index: calendar()).

    Yahoo columns are adjusted closes; for at_close=False the US ones at t are the US close of calendar date t or the
    last one before it. Units, sources and publication timing per column: MARKET[col] = (group, unit, timing, desc)."""
    columns = list(MARKET) if columns is None else list(columns)
    index = _index(index, end)
    out = {}
    for col in columns:
        g, _, timing, _ = MARKET[col]
        raw = market_raw(col).to_frame(col)
        out[col] = align(_shift(raw, timing, at_close), index, _tol(timing), col not in FLOWS)[col]
    return pd.DataFrame(out, index=index)


def load_ohlcv(name, index=None, at_close=False, end=None) -> pd.DataFrame:
    """Aligned OHLCV for a Yahoo series (keys of YAHOO) or "taiex" (FinMind TAIEX OHLC, volume = traded value NTD)."""
    index = _index(index, end)
    if name == "taiex":
        d = _by_date(_read_csv(_fm_path("market", "taiex")))
        raw = pd.DataFrame({"open": d["open"], "high": d["max"], "low": d["min"], "close": d["close"], "volume": d["Trading_money"]})
        timing = "tw"
    else:
        raw, timing = yahoo_raw(name), YAHOO[name][2]
    return align(_shift(raw, timing, at_close), index, _tol(timing))


def load_stock_fields(code, fields=None, index=None, at_close=False, end=None) -> pd.DataFrame:
    """STOCK_FIELDS (all or the given ones) for one stock, aligned to TW dates: daily chip / valuation data of d from
    bar d (published that evening); monthly revenue from revenue_avail(). Units: STOCK_FIELDS[field][1]."""
    fields = list(STOCK_FIELDS) if fields is None else list(fields)
    index = _index(index, end)
    parts = []
    for f in fields:
        g, _, timing, _ = STOCK_FIELDS[f]
        parts.append(align(_shift(stock_raw(code, g)[[f]], timing, at_close), index, _tol(timing), f not in FLOWS))
    return pd.concat(parts, axis=1)


def load_stock_panel(field, codes=None, index=None, at_close=False, end=None) -> pd.DataFrame:
    """One STOCK_FIELDS field for many stocks (default: the 50 in universe.CODES) as a (TW dates x codes) frame,
    aligned exactly like load_stock_fields. NaN where a stock has no data yet (before listing / before the source)."""
    codes = list(codes or CODES)
    index = _index(index, end)
    g, _, timing, _ = STOCK_FIELDS[field]
    return pd.DataFrame(
        {c: align(_shift(stock_raw(c, g)[[field]], timing, at_close), index, _tol(timing), field not in FLOWS)[field]
         for c in codes},
        index=index,
    )


def load_price_panel(field="close", codes=None, index=None, end=None) -> pd.DataFrame:
    """Cleaned adjusted prices (stocklab.data.load) of many stocks as one (TW dates x codes) frame; field in
    open/high/low/close/volume. Bar d is known at 13:30 on d -> avail d. NaN before listing; on dates a stock has no
    bar, prices carry the last bar forward (at most DAILY_TOL days) and volume is NaN."""
    from .data import load

    codes = list(codes or CODES)
    index = _index(index, end)
    out = {}
    for c in codes:
        df = load(c)[[field]]
        df.index = pd.DatetimeIndex(df.index).as_unit("ns")
        out[c] = align(df, index, repeat=field != "volume")[field]
    return pd.DataFrame(out, index=index)


def industry(codes=None) -> pd.Series:
    """Industry label (TWSE 產業別, the most specific one, e.g. 半導體業 rather than 電子工業) per stock, as of the
    download date -- a static categorical, not point-in-time."""
    codes = list(codes or CODES)
    d = _read_csv(_fm_path("market", "stock_info"))
    d = d[d["stock_id"].astype(str).isin(codes) & d["type"].isin(["twse", "tpex"])]
    d = d.sort_values("date", ascending=False)
    specific = d[~d["industry_category"].isin(["電子工業", "化學生技醫療"])]
    lab = specific.groupby(specific["stock_id"].astype(str))["industry_category"].first()
    lab = lab.combine_first(d.groupby(d["stock_id"].astype(str))["industry_category"].first())
    return lab.reindex(codes).rename("industry")


# ------------------------------------------------------------------------------------------------ coverage


def _cov_row(name, source, raw_index, aligned, cal, monthly=False):
    raw_index = pd.DatetimeIndex(raw_index).dropna()
    if raw_index.empty:
        return {"series": name, "source": source, "first": None, "last": None, "n": 0, "tw_gap%": np.nan, "nan%": np.nan}
    first, last = raw_index.min(), raw_index.max()
    span = cal[(cal >= first) & (cal <= last)]
    gap = np.nan if monthly else 100 * (1 - span.isin(raw_index).mean())
    nan = 100 * aligned.reindex(span).isna().mean()
    return {"series": name, "source": source, "first": first.date(), "last": last.date(), "n": len(raw_index),
            "tw_gap%": round(gap, 2), "nan%": round(nan, 2)}


def coverage(codes=None) -> pd.DataFrame:
    """Per series: first / last availability date, number of records, % of TW trading days in [first, last] with no
    record on that day (daily series; US gaps include US holidays), and % NaN after alignment within [first, last].
    Per-stock fields: first = earliest over stocks (first_latest = latest start, i.e. the newest listing), last =
    latest, n / gap / nan = medians over stocks."""
    cal = calendar()
    rows = []
    mk = load_market(index=cal)
    for col, (g, *_rest) in MARKET.items():
        src = f"Yahoo {YAHOO[col][0]}" if g == "yahoo" else ("TAIFEX pcRatio" if g == "pc_ratio" else f"FinMind {FM_MARKET[g][0]}")
        rows.append(_cov_row(col, src, market_raw(col).index, mk[col], cal))
    codes = list(codes or CODES)
    for field, (g, *_rest) in STOCK_FIELDS.items():
        per = []
        tol = MONTHLY_TOL if g == "revenue" else DAILY_TOL
        for c in codes:
            raw = stock_raw(c, g)[field].dropna()
            al = align(raw.to_frame(), cal, tol, field not in FLOWS)[field]
            per.append(_cov_row(field, f"FinMind {FM_STOCK[g]}", raw.index, al, cal, g == "revenue"))
        per = pd.DataFrame(per)
        rows.append({"series": f"{field} [{per['n'].gt(0).sum()}/{len(codes)}]", "source": per["source"].iloc[0],
                     "first": per["first"].dropna().min(), "last": per["last"].dropna().max(), "n": int(per["n"].median()),
                     "tw_gap%": per["tw_gap%"].median(), "nan%": per["nan%"].median(),
                     "first_latest": per["first"].dropna().max()})
    return pd.DataFrame(rows)


def cache_size_mb() -> float:
    return sum(f.stat().st_size for f in EXT_DIR.rglob("*") if f.is_file()) / 2**20


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download external data into data/external/ and print coverage.")
    ap.add_argument("--refresh", action="store_true", help="re-download even if cached")
    ap.add_argument("--no-stocks", action="store_true", help="skip the per-stock FinMind datasets")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    download_all(refresh=args.refresh, stocks=not args.no_stocks)
    pd.set_option("display.width", 200)
    print(coverage().to_string(index=False))
    print(f"cache size: {cache_size_mb():.1f} MB in {EXT_DIR}")
