"""Export the published dl_revenue_flow_gbdt strategy as ASCII price files for MultiCharts QuoteManager, to be loaded
as data2 under each stock's chart (see multicharts/dl_revenue_flow_gbdt.txt).

  python -m strategies.dl_revenue_flow.export_signals            # all 50 stocks
  python -m strategies.dl_revenue_flow.export_signals 2330 2317  # some stocks (the model still uses all 50)

Run it after the close of a trading day, once FinMind has the evening data (三大法人, 融資融券, 借券, 本益比 of that
day: ~21:30 Taipei) and the US close (~05:00 Taipei next morning), i.e. between ~06:00 and the 09:00 open; refresh the
data first (stocklab.data / stocklab.external downloads). Writes to exports/dl_revenue_flow_gbdt/ (not committed):

  <code>_RF.csv   per-stock target exposure (the `positions` of the strategy: 0.5 or 1.0)
  <code>_RFW.csv  portfolio version (the `weights`: top-20 by forecast) as a fraction of one 1/20 capital slot
                  (1.0 = in the top 20, 0 = not held; 0.4 = 1/50 equal weight before 2012)

one row per trading day of the stock:
  Date,Time,Open,High,Low,Close,Volume
  2026/10/02,13:30:00,150,150,150,150,1
with Open = High = Low = Close = 100 + 100 x value (the offset keeps every price positive). The row dated t is decided
after the close of t; MultiCharts fills it at the open of t+1. The targets change only on revenue release days (the
second TW trading day on/after the 10th of each month), so in practice this needs to run once a month.
Also prints the latest targets and the current top-20 list, for placing the orders by hand.
"""

import sys

import pandas as pd

from stocklab.data import ROOT, load_all

from .strategy import SID, TOP_K, positions, weights

OFFSET = 100.0
SCALE = 100.0


def _write(series, path):
    px = (OFFSET + SCALE * series.clip(0.0, 1.0)).round(2)
    df = pd.DataFrame({"Date": series.index.strftime("%Y/%m/%d"), "Time": "13:30:00", "Open": px.values,
                       "High": px.values, "Low": px.values, "Close": px.values, "Volume": 1})
    df.to_csv(path, index=False)


def main(codes=None):
    data = load_all()
    pos = positions(data)
    w = weights(data)
    out_dir = ROOT / "exports" / SID
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for code in sorted(codes or pos):
        idx = data[code].index
        p = pos[code].reindex(idx).fillna(0.0)
        slot = (w[code].reindex(idx).fillna(0.0) * TOP_K).clip(0.0, 1.0)
        _write(p, out_dir / f"{code}_RF.csv")
        _write(slot, out_dir / f"{code}_RFW.csv")
        prev = float(p.iloc[-2]) if len(p) > 1 else float("nan")
        rows.append((code, idx[-1].date(), prev, float(p.iloc[-1]), float(slot.iloc[-1])))
    print(f"wrote {2 * len(rows)} files to {out_dir}")
    print(f"{'代號':<6}{'最後日期':<12}{'前一日曝險':>10}{'最新曝險':>10}{'投組持有':>8}  動作（下一個交易日開盤）")
    for code, d, prev, cur, slot in rows:
        act = "不變" if abs(cur - prev) < 1e-9 else ("加碼" if cur > prev else "減碼")
        print(f"{code:<6}{str(d):<12}{prev:>10.2f}{cur:>10.2f}{'是' if slot > 0.99 else '':>8}  {act}")
    held = sorted(c for c in w if w[c].iloc[-1] > 0)
    print(f"投組版目前持有 {len(held)} 檔（各占總資金 1/{TOP_K}）: {', '.join(held)}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
