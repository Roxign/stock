"""Taiwan short-selling (融券) rules for the backtest. Sources and approximations: research/short_rules.md.

Per security and trading day:
  can_open_short   a new or larger short may be filled at this bar's open
  must_cover       an open short must be bought back at this bar's open
  flat_restricted  平盤以下不得融券賣出 applies: a short sale fills only if the open is >= the previous close
plus costs: 融券保證金成數 (normally 90%), 擔保維持率 130%, 融券手續費 0.08%, and the seller's transaction tax.

Short-sale suspensions (停止融券賣出, around ex-dividend book closures and shareholder meetings) come from the
exchange's daily margin report: FinMind TaiwanStockMarginPurchaseShortSale "Note" contains "X" on report dates whose
NEXT trading day is suspended. The legal last cover day (融券最後回補日) falls inside the suspension; we force the
cover at the open of the first suspended day, one or two days early (slightly conservative). Suspensions are
announced about two weeks ahead (停資停券預告), so treating them as known is not lookahead. A security is eligible
from its first margin-report record (new listings wait about six months).
"""

from __future__ import annotations

import pandas as pd

from . import external as ext
from .universe import CODES

DIR = ext.EXT_DIR / "shortrules"
MAINTENANCE = 1.30
SHORT_FEE = 0.0008
COMMISSION = 0.001425
STOCK_TAX = 0.003

# 平盤以下不得融券賣出 for every security (inclusive): 2008 financial crisis, 2015 market crash.
FLAT_BAN_ALL = [("2008-09-22", "2009-01-02"), ("2015-08-24", "2015-09-18")]
# From 2013-09-23 ordinary stocks may be shorted below the previous close, except on the day after a limit-down close
# (limit 7%, 10% from 2015-06-01). Before that the restriction applied to all stocks outside 台灣50/中型100/資訊科技;
# lacking historical index membership we apply it to all stocks before 2013-09-23 (conservative); ETFs were exempt.
FLAT_RULE_LIFTED = "2013-09-23"
LIMIT_CHANGE = "2015-06-01"
# Temporary rule: after a close-to-close drop of 3.5% or more, the next day may not be shorted below the previous close.
DROP_35 = [("2020-03-20", "2020-06-09"), ("2022-10-21", "2023-02-23")]
# 融券保證金成數 changes (inclusive); 90% otherwise.
MARGIN_RATES = [("2022-10-01", "2022-10-11", 1.00), ("2022-10-12", "2023-02-23", 1.20), ("2025-04-07", "2025-05-25", 1.30)]
DEFAULT_MARGIN = 0.90


def _path(code):
    return DIR / f"{code}.csv.gz"


def download_notes(codes=None, refresh=False):
    """Daily margin-report fields needed for the short calendar (date, Note, short balance/limit)."""
    DIR.mkdir(parents=True, exist_ok=True)
    for code in codes or CODES:
        p = _path(code)
        if p.exists() and not refresh:
            continue
        d = ext.finmind("TaiwanStockMarginPurchaseShortSale", code, start=ext.START)
        keep = ["date", "Note", "ShortSaleTodayBalance", "ShortSaleLimit"]
        d[[c for c in keep if c in d.columns]].to_csv(p, index=False, compression="gzip")
        print(code, len(d), flush=True)


def update_notes(codes=None, lookback_days=10):
    """Daily update: re-fetch the notes from `lookback_days` before the cached last date (full download if missing)."""
    for code in codes or CODES:
        p = _path(code)
        if not p.exists():
            download_notes([code])
            continue
        old = pd.read_csv(p, dtype={"Note": str})
        start = pd.Timestamp(old["date"].max()) - pd.Timedelta(days=lookback_days)
        d = ext.finmind("TaiwanStockMarginPurchaseShortSale", code, start=start.strftime("%Y-%m-%d"))
        if d.empty:
            continue
        d = d[[c for c in old.columns if c in d.columns]]
        keep = old[pd.to_datetime(old["date"]) < start]
        pd.concat([keep, d], ignore_index=True).to_csv(p, index=False, compression="gzip")


def notes(code) -> pd.DataFrame:
    d = pd.read_csv(_path(code), parse_dates=["date"], dtype={"Note": str}).set_index("date").sort_index()
    d["Note"] = d["Note"].fillna("").str.strip()
    return d


def _is_etf(code):
    return not code.isdigit() or code.startswith("00")


def _in(index, windows):
    out = pd.Series(False, index=index)
    for a, b in windows:
        out |= (index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b))
    return out


def short_calendar(code, df) -> pd.DataFrame:
    """Rule calendar on df's trading days (df: the security's adjusted OHLC frame)."""
    idx = df.index
    n = notes(code)
    eligible = pd.Series(idx >= n.index[0], index=idx)
    # A note on report date d describes the next trading day: map it forward onto the security's own calendar.
    nxt = idx.searchsorted(n.index, side="right")
    ok = nxt < len(idx)
    flags = pd.DataFrame({"X": n["Note"].str.contains("X").to_numpy()[ok], "halt": n["Note"].str.contains("!").to_numpy()[ok]},
                         index=idx[nxt[ok]])
    flags = flags.groupby(level=0).max().reindex(idx, fill_value=False)
    suspended = flags["X"] | flags["halt"] | ~eligible
    must_cover = suspended & ~suspended.shift(fill_value=True)

    ret = df["close"].pct_change()
    limit = pd.Series(0.065, index=idx).where(idx < pd.Timestamp(LIMIT_CHANGE), 0.095)
    after_limit_down = (ret <= -limit).shift(fill_value=False)
    after_drop_35 = (ret <= -0.035).shift(fill_value=False) & _in(idx, DROP_35)
    flat = _in(idx, FLAT_BAN_ALL) | after_drop_35
    if not _is_etf(code):
        flat |= pd.Series(idx < pd.Timestamp(FLAT_RULE_LIFTED), index=idx) | after_limit_down
    return pd.DataFrame({"can_open_short": ~suspended, "must_cover": must_cover, "flat_restricted": flat}, index=idx)


def margin_rate(dates) -> pd.Series:
    idx = pd.DatetimeIndex(dates)
    out = pd.Series(DEFAULT_MARGIN, index=idx)
    for a, b, r in MARGIN_RATES:
        out[(idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))] = r
    return out


def fees(code) -> dict:
    """One-way cost rates: buy (commission), sell (commission + transaction tax), short (extra 融券手續費 on short sales)."""
    if _is_etf(code):
        from . import etf

        tax = 0.0 if etf.ETFS.get(code, {}).get("tax_exempt") else etf.ETF_TAX
    else:
        tax = STOCK_TAX
    return {"buy": COMMISSION, "sell": COMMISSION + tax, "short": SHORT_FEE}


def eligible(code, date) -> bool:
    return pd.Timestamp(date) >= notes(code).index[0]


if __name__ == "__main__":
    import sys

    download_notes(sys.argv[1:] or None)
