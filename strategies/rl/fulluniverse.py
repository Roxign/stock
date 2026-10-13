"""Every TWSE common stock since 2008, delisted ones included, for a survivorship-free test of momentum.

Stock list: FinMind TaiwanStockInfo (type twse, 4-digit codes, no ETF / DR / ETN) plus FinMind TaiwanStockDelisting.
Prices:
  * still listed -> Yahoo "<code>.TW" in batches (auto_adjust off: the raw close gives the traded value used to rank
    liquidity point-in-time; OHLC are then scaled by Adj Close / Close for dividends and splits);
  * delisted (Yahoo has dropped them) -> FinMind TaiwanStockPrice (raw, with Trading_money) adjusted with
    TaiwanStockDividendResult (factor = reference price / previous close on each ex-date).
Both then go through stocklab.data.clean (pre-listing bars, unadjusted capital reductions / bad prints).

    python -m strategies.rl.fulluniverse          # resumable; files in data/rl/full/<code>.csv

About 200 delisted stocks x 2 FinMind requests; without a token FinMind allows ~300 requests an hour, and
stocklab.external.finmind waits automatically when the limit is hit.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from stocklab import external as ext
from stocklab.data import ROOT, clean, session_cutoff

DIR = ROOT / "data" / "rl" / "full"
DELIST = ROOT / "data" / "rl" / "delisting.csv"
NOT_STOCKS = {"ETF", "存託憑證", "ETN", "受益證券", "Index", "大盤", "創新板股票", "創新版股票"}


def stock_list():
    info = pd.read_csv(ext.EXT_DIR / "finmind" / "market" / "stock_info.csv.gz", dtype={"stock_id": str})
    if not DELIST.exists():
        ext.finmind("TaiwanStockDelisting", None, start="2008-01-01").to_csv(DELIST, index=False)
    dl = pd.read_csv(DELIST, dtype={"stock_id": str})
    dl = dl[dl["stock_id"].str.fullmatch(r"\d{4}")]
    tw = info[(info["type"] == "twse") & info["stock_id"].str.fullmatch(r"\d{4}") & ~info["industry_category"].isin(NOT_STOCKS)]
    known_other = set(info.loc[info["type"] != "twse", "stock_id"]) - set(tw["stock_id"])
    delisted = sorted((set(dl["stock_id"]) - known_other))          # TWSE or unknown market (older delistings)
    current = sorted(set(tw["stock_id"]) - set(dl["stock_id"]))
    return current, delisted, dict(zip(dl["stock_id"], dl["date"]))


def save(code, df):
    DIR.mkdir(parents=True, exist_ok=True)
    df = df[["open", "high", "low", "close", "volume", "value"]].dropna(subset=["open", "high", "low", "close"])
    df = df[(df["volume"] > 0) & (df["high"] >= df["low"]) & (df["close"] > 0)]
    if len(df) < 30:
        return False
    df.index.name = "date"
    df.to_csv(DIR / f"{code}.csv", float_format="%.4f")
    return True


def from_yahoo(codes):
    import yfinance as yf

    got, failed = [], []
    for i in range(0, len(codes), 50):
        batch = [c for c in codes[i : i + 50] if not (DIR / f"{c}.csv").exists()]
        if not batch:
            continue
        raw = yf.download([f"{c}.TW" for c in batch], start="2008-01-01", auto_adjust=False, progress=False, group_by="ticker", threads=True)
        for c in batch:
            try:
                d = raw[f"{c}.TW"].dropna(subset=["Close"])
            except KeyError:
                failed.append(c)
                continue
            if d.empty:
                failed.append(c)
                continue
            k = d["Adj Close"] / d["Close"]
            df = pd.DataFrame({"open": d["Open"] * k, "high": d["High"] * k, "low": d["Low"] * k, "close": d["Adj Close"],
                               "volume": d["Volume"], "value": d["Close"] * d["Volume"]})
            df.index = pd.to_datetime(df.index).tz_localize(None)
            (got if save(c, df) else failed).append(c)
        print(f"  Yahoo {min(i + 50, len(codes))}/{len(codes)}：成功 {len(got)}、失敗 {len(failed)}", flush=True)
        time.sleep(1)
    return got, failed


def from_finmind(code):
    p = ext.finmind("TaiwanStockPrice", code, start="2008-01-01")
    if p.empty:
        return False
    p["date"] = pd.to_datetime(p["date"])
    p = p.set_index("date").sort_index()
    df = pd.DataFrame({"open": p["open"], "high": p["max"], "low": p["min"], "close": p["close"],
                       "volume": p["Trading_Volume"], "value": p["Trading_money"]}).astype(float)
    df = df[(df["open"] > 0) & (df["close"] > 0)]
    dv = ext.finmind("TaiwanStockDividendResult", code, start="2008-01-01")
    f = np.ones(len(df))
    for _, r in dv.iterrows():
        if r["before_price"] > 0 and r["after_price"] > 0:
            f[df.index < pd.Timestamp(r["date"])] *= r["after_price"] / r["before_price"]
    for col in ("open", "high", "low", "close"):
        df[col] = df[col] * f
    return save(code, df)


def load_full(code):
    df = pd.read_csv(DIR / f"{code}.csv", index_col="date", parse_dates=True)
    df = df[df.index < session_cutoff()]  # today's bar is final only after the 14:30 after-hours session
    try:
        return clean(df)
    except Exception:
        return df


def main():
    current, delisted, dates = stock_list()
    print(f"上市中 {len(current)} 檔（Yahoo），已下市 {len(delisted)} 檔（FinMind）", flush=True)
    got, failed = from_yahoo(current)
    todo = [c for c in delisted + failed if not (DIR / f"{c}.csv").exists()]
    print(f"Yahoo 沒有的 {len(failed)} 檔改用 FinMind；FinMind 待抓 {len(todo)} 檔（每檔 2 次請求）", flush=True)
    for k, c in enumerate(todo):
        try:
            ok = from_finmind(c)
        except RuntimeError as e:
            ok = False
            print(f"  {c}: {e}", flush=True)
        if k % 10 == 9 or k == len(todo) - 1:
            print(f"  FinMind {k + 1}/{len(todo)}（最後一檔 {c} {'成功' if ok else '沒有資料'}）", flush=True)
    have = sorted(p.stem for p in DIR.glob("*.csv"))
    print(f"完成：共 {len(have)} 檔", flush=True)


if __name__ == "__main__":
    main()
