"""Template: copy this folder to strategies/<family>/ (no leading underscore) to register strategies.

Contract for each entry in STRATEGIES:
  id          unique snake_case id
  label       short Traditional Chinese name shown in the viewer
  family      group name shown in the viewer
  description Traditional Chinese markdown: rules, parameters, how they were chosen, caveats
  multicharts path (relative to this folder) of the PowerLanguage source, or None
  positions   fn(data: dict[code, DataFrame]) -> dict[code, Series]
              DataFrame columns: open, high, low, close, volume (adjusted, daily, DatetimeIndex)
              Series value at bar t = target exposure in [0, 1] decided using data up to bar t's close only.
              The engine trades the change at bar t+1's open. Prefer discrete values (0 / 0.5 / 1).
"""

import pandas as pd

from stocklab.indicators import sma


def ma_cross(data, fast=20, slow=60):
    out = {}
    for code, df in data.items():
        out[code] = (sma(df["close"], fast) > sma(df["close"], slow)).astype(float)
    return out


STRATEGIES = [
    {
        "id": "example_ma_cross",
        "label": "均線交叉範例",
        "family": "範例",
        "description": "收盤價 20 日均線在 60 日均線之上時持有，否則空手。",
        "multicharts": None,
        "positions": ma_cross,
    }
]
