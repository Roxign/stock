"""Export the daily target exposure of dl_trend_labels_tcn for MultiCharts (data2 signal series).

    C:\\Users\\user\\.venvs\\stock\\Scripts\\python.exe -m strategies.dl_trend_labels.export_signals

Recomputes the walk-forward positions on the latest data in data/raw (the TCN cannot run inside MultiCharts) and
writes one ASCII price file per stock to exports/dl_trend_labels_tcn/TL_<code>.txt (not committed):

    Date,Time,Open,High,Low,Close,Volume
    2026/10/02,13:30,1.0000,1.0000,1.0000,1.0000,1

Open = High = Low = Close = the target exposure decided at that bar's close (0.5 or 1.0; 1.0 before the first model);
it is traded at the NEXT bar's open, exactly like the Python backtest. Import each file in QuoteManager as a new
symbol (e.g. TL_2330) and add it to the stock's chart as data2 -- see multicharts/dl_trend_labels_tcn.txt.
"""

from pathlib import Path

from stocklab.data import ROOT, load_all

from .strategy import SID, positions

OUT = ROOT / "exports" / SID


def main():
    data = load_all()
    pos = positions(data)
    OUT.mkdir(parents=True, exist_ok=True)
    for code, s in sorted(pos.items()):
        s = s.reindex(data[code].index).fillna(1.0).clip(0.0, 1.0)
        lines = ["Date,Time,Open,High,Low,Close,Volume"]
        lines += [f"{d:%Y/%m/%d},13:30,{v:.4f},{v:.4f},{v:.4f},{v:.4f},1" for d, v in s.items()]
        (OUT / f"TL_{code}.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    last = max(s.index[-1] for s in pos.values())
    print(f"wrote {len(pos)} files to {OUT} (last bar {last:%Y-%m-%d})")
    for code in sorted(pos):
        print(f"  {code}: target {pos[code].iloc[-1]:.2f}")


if __name__ == "__main__":
    main()
