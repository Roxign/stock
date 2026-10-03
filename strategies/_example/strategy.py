"""Template: copy this folder to strategies/<family>/ (no leading underscore) to register strategies.

Contract for each entry in STRATEGIES:
  id          unique snake_case id
  label       short Traditional Chinese name shown in the viewer
  family      group name shown in the viewer
  description Traditional Chinese markdown: rules, parameters, how they were chosen, caveats
  multicharts path (relative to this folder) of the PowerLanguage source, or None
  positions   fn(data: dict[code, DataFrame]) -> dict[code, Series]
              DataFrame columns: open, high, low, close, volume (adjusted, daily, DatetimeIndex)
              Series value at bar t = target exposure in [0, 1], decided from prices up to bar t's close plus any
              external data published before bar t+1's open (e.g. 三大法人 published the evening of t, US close of
              date t) — stocklab.external aligns every series this way by default.
              The engine trades the change at bar t+1's open. Prefer discrete values (0 / 0.5 / 1).
  weights     optional fn(data) -> dict[code, Series]: target weight of TOTAL portfolio equity per stock (sum <= 1,
              rest is cash) for the single-account portfolio scoreboard. Cross-sectional strategies (rank/select among
              the 50) should provide it; without it, positions get equal capital slots (weight = exposure / 50).

Optional cross-validation hooks (stocklab/cv.py, `evaluate.py --cv`):
  param_grid    list of parameter dicts (keep it small, <= ~40) covering the neighbourhood you actually explored
  build         fn(params) -> positions fn; build(default params) must reproduce `positions` exactly
                -> "retune" CV: each fold is tested with the parameters that did best on the other 7 folds
  cv_positions  fn(data, folds: dict[fold, (start, end)]) -> dict[code, Series] for models that are trained:
                positions inside each fold come from a model fitted WITHOUT that fold (purge every training sample
                whose label window overlaps the fold, plus an embargo after it) -> "purged" CV

Any data used besides `data` (e.g. stocklab.external) must be cut at the last date present in `data`, so the
truncation lookahead check (stocklab.runner.check_lookahead) also truncates it. Log every configuration you try with
stocklab.trials.log_trial.
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
