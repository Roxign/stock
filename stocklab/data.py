from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .universe import CODES

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
START = "2008-01-01"  # two years before the 2010 evaluation start, so 240-day indicators are warm
COLS = ["open", "high", "low", "close", "volume"]


def download(codes=CODES, start=START):
    """Download dividend/split-adjusted daily OHLCV from Yahoo and cache as CSV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tickers = [f"{c}.TW" for c in codes]
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    for c, t in zip(codes, tickers):
        df = raw[t].rename(columns=str.lower)[COLS].dropna(subset=["open", "high", "low", "close"])
        df = df[(df["volume"] > 0) & (df["high"] >= df["low"])]
        df.index = pd.to_datetime(df.index).tz_localize(None)
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


def load(code):
    return clean(pd.read_csv(RAW_DIR / f"{code}.csv", index_col="date", parse_dates=True))


def load_all(codes=CODES):
    return {c: load(c) for c in codes}


if __name__ == "__main__":
    download()
    for c in CODES:
        df = load(c)
        print(c, df.index.min().date(), df.index.max().date(), len(df))
