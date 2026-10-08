from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .universe import CODES, EXTRA_CODES

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
START = "2008-01-01"  # two years before the 2010 evaluation start, so 240-day indicators are warm
COLS = ["open", "high", "low", "close", "volume"]
ALL_CODES = CODES + EXTRA_CODES  # the 50 plus the representative stocks used for development


def session_cutoff() -> pd.Timestamp:
    """Bars dated on/after this are dropped: today's bar is final only after the 14:30 Taipei after-hours session."""
    now = datetime.now(timezone(timedelta(hours=8)))
    today = pd.Timestamp(now.date())
    return today if (now.hour, now.minute) < (14, 30) else today + pd.Timedelta(days=1)


def download(codes=CODES, start=START):
    """Download dividend/split-adjusted daily OHLCV from Yahoo and cache as CSV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tickers = [f"{c}.TW" for c in codes]
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    for c, t in zip(codes, tickers):
        df = raw[t].rename(columns=str.lower)[COLS].dropna(subset=["open", "high", "low", "close"])
        df = df[(df["volume"] > 0) & (df["high"] >= df["low"])]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[df.index < session_cutoff()]
        df.index.name = "date"
        df.to_csv(RAW_DIR / f"{c}.csv", float_format="%.4f")


LIMIT_CHANGE = pd.Timestamp("2015-06-01")


def abnormal(df):
    """Bars whose close-to-close move exceeds the daily price limit then in force (7%, 10% from 2015-06-01) plus tolerance."""
    limit = np.where(df.index < LIMIT_CHANGE, 0.075, 0.105)
    return np.flatnonzero(df["close"].pct_change().abs().to_numpy() > limit)


def clean(df, cluster_gap=120, early_bars=500):
    """Adjusted prices can't legitimately move past the price limit, so such moves are data artifacts.

    For stocks listed after the sample start, a cluster of them in the first bars is pre-listing (興櫃) data with no
    price limit: drop through the cluster. Every remaining one is a corporate action Yahoo didn't adjust (減資,
    spin-off) or a bad bar: rescale history before it so that day's return becomes zero.
    """
    if df.index[0] > pd.Timestamp(START) + pd.Timedelta(days=14):
        early = abnormal(df)
        early = early[early < early_bars]
        cut = max((b for a, b in zip(early[:-1], early[1:]) if b - a <= cluster_gap), default=-1)
        df = df.iloc[cut + 1 :]
    df = df.copy()
    cols = [df.columns.get_loc(c) for c in ("open", "high", "low", "close")]
    for i in abnormal(df):
        df.iloc[:i, cols] *= df["close"].iloc[i] / df["close"].iloc[i - 1]
    return df


def top_up(codes, until=None, lookback_days=14):
    """Append the latest bars Yahoo doesn't have yet from the exchange's daily quotes (FinMind TaiwanStockPrice).

    Only when the exchange close on Yahoo's last date matches Yahoo's close, i.e. no dividend adjustment is pending
    (adjusted and raw prices coincide for the most recent bars). The next full download replaces these bars.
    Codes whose cache already reaches `until` (e.g. external.last_session()) are skipped without a request.
    """
    from . import external as ext

    added = {}
    for c in codes:
        p = RAW_DIR / f"{c}.csv"
        df = pd.read_csv(p, index_col="date", parse_dates=True)
        last = df.index[-1]
        if until is not None and last >= pd.Timestamp(until):
            continue
        f = ext.finmind("TaiwanStockPrice", c, start=(last - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d"))
        if f is None or f.empty:
            continue
        f = f.assign(date=pd.to_datetime(f["date"])).set_index("date")
        if last not in f.index or abs(f.loc[last, "close"] / df.loc[last, "close"] - 1) > 0.001:
            continue
        new = f[(f.index > last) & (f.index < session_cutoff()) & (f["Trading_Volume"] > 0)]
        if new.empty:
            continue
        bars = pd.DataFrame({"open": new["open"], "high": new["max"], "low": new["min"], "close": new["close"],
                             "volume": new["Trading_Volume"]}, index=new.index.rename("date"))
        pd.concat([df, bars]).to_csv(p, float_format="%.4f")
        added[c] = [d.strftime("%Y-%m-%d") for d in bars.index]
    return added


def load(code):
    return clean(pd.read_csv(RAW_DIR / f"{code}.csv", index_col="date", parse_dates=True))


def load_all(codes=CODES):
    return {c: load(c) for c in codes}


if __name__ == "__main__":
    download(ALL_CODES)
    for c in ALL_CODES:
        df = load(c)
        print(c, df.index.min().date(), df.index.max().date(), len(df))
