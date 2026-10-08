"""Export the daily target exposure of dl_market_state_mlp for MultiCharts (data2 signal series).

    C:\\Users\\user\\.venvs\\stock\\Scripts\\python.exe -m strategies.dl_market_state.export_signals

The market-state MLP needs external data (三大法人 / 融資 / put-call / TAIFEX positions / US & Asian markets) that
MultiCharts does not have, so Python recomputes the walk-forward exposure on the latest data in data/raw and writes one
ASCII price file per stock to exports/dl_market_state_mlp/MS_<code>.txt (not committed), plus MS_MARKET.txt (the
common exposure on the union calendar; every stock's file holds the same value on the same date):

    Date,Time,Open,High,Low,Close,Volume
    2026/10/02,13:30,1.0000,1.0000,1.0000,1.0000,1

Open = High = Low = Close = the target exposure decided at that bar's close (0.5 / 0.75 / 1.0; 1.0 before the first
model in 2012); it is traded at the NEXT bar's open, exactly like the Python backtest. Import each file in QuoteManager
as a new symbol (e.g. MS_2330) and add it to the stock's chart as data2 -- see multicharts/dl_market_state_mlp.txt.
"""

from stocklab.data import ROOT, load_all

from .core import to_positions
from .strategy import SID, common_exposure

OUT = ROOT / "exports" / SID


def _write(path, s):
    lines = ["Date,Time,Open,High,Low,Close,Volume"]
    lines += [f"{d:%Y/%m/%d},13:30,{v:.4f},{v:.4f},{v:.4f},{v:.4f},1" for d, v in s.items()]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main():
    data = load_all()
    g = common_exposure(data)
    pos = to_positions(g, data)          # identical to strategy.positions(data), without retraining twice
    OUT.mkdir(parents=True, exist_ok=True)
    _write(OUT / "MS_MARKET.txt", g.clip(0.0, 1.0))
    for code, s in sorted(pos.items()):
        _write(OUT / f"MS_{code}.txt", s.reindex(data[code].index).fillna(1.0).clip(0.0, 1.0))
    print(f"wrote {len(pos) + 1} files to {OUT} (last bar {g.index[-1]:%Y-%m-%d}, common target {g.iloc[-1]:.2f})")


if __name__ == "__main__":
    main()
