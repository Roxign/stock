"""Export the published dl_cross_stock strategy's daily target exposure of each stock as ASCII price files for
MultiCharts QuoteManager, to be loaded as data2 under each stock's chart (see multicharts/dl_cross_stock_attn.txt).

  C:\\Users\\user\\.venvs\\stock\\Scripts\\python.exe -m strategies.dl_cross_stock.export_signals            # all 50
  C:\\Users\\user\\.venvs\\stock\\Scripts\\python.exe -m strategies.dl_cross_stock.export_signals 2330 2317  # some

Writes exports/dl_cross_stock_attn/<code>_XSA.csv (not committed), one row per trading day of the stock:
  Date,Time,Open,High,Low,Close,Volume
  2026/10/02,13:30:00,175,175,175,175,1
with Open = High = Low = Close = 100 + 100 x target exposure (100 = flat, 125 / 150 / 175 = 25 / 50 / 75 %,
200 = fully invested; the offset keeps every price positive). The row dated t is the exposure decided after the close
of t (it uses US closes of calendar day t and TWSE evening statistics of t, so run this after ~06:00 Taipei on t+1 and
before the 09:00 open); MultiCharts fills it at the open of t+1.

Also prints the latest per-stock targets and the current 10-stock portfolio of the weights() version (10 % each,
rebalanced on the first trading day of each month), for placing orders by hand.
"""

import sys

import pandas as pd

from stocklab.data import ROOT, load_all

from .strategy import STRATEGIES

OFFSET = 100.0
SCALE = 100.0
SUFFIX = "XSA"


def main(codes=None):
    s = STRATEGIES[0]
    data = load_all()
    pos = s["positions"](data)
    out_dir = ROOT / "exports" / s["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for code in sorted(codes or pos):
        p = pos[code].reindex(data[code].index).fillna(0.0).clip(0.0, 1.0)
        px = (OFFSET + SCALE * p).round(2)
        df = pd.DataFrame({"Date": p.index.strftime("%Y/%m/%d"), "Time": "13:30:00", "Open": px.values,
                           "High": px.values, "Low": px.values, "Close": px.values, "Volume": 1})
        df.to_csv(out_dir / f"{code}_{SUFFIX}.csv", index=False)
        prev = float(p.iloc[-2]) if len(p) > 1 else float("nan")
        rows.append((code, p.index[-1].date(), prev, float(p.iloc[-1])))
    print(f"wrote {len(rows)} files to {out_dir}")
    print(f"{'代號':<6}{'最後日期':<12}{'前一日目標':>10}{'最新目標':>10}  動作（下一個交易日開盤）")
    for code, d, prev, cur in rows:
        act = "不變" if abs(cur - prev) < 1e-9 else ("加碼" if cur > prev else "減碼")
        print(f"{code:<6}{str(d):<12}{prev:>10.2f}{cur:>10.2f}  {act}")
    w = s["weights"](data)
    cur = {c: float(v.iloc[-1]) for c, v in w.items() if len(v) and v.iloc[-1] > 0}
    print("\n組合版（weights）目前持股，每檔占總權益：")
    for c in sorted(cur, key=lambda c: -cur[c]):
        print(f"  {c}: {cur[c]:.0%}")
    print(f"  現金: {max(0.0, 1 - sum(cur.values())):.0%}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
