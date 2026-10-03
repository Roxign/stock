"""KDJ round, experiment K5: strategy-level ablation, in-sample only (data truncated at 2020-12-31).

  python -m strategies.kdj_macd_dl.research.kdj_ablation_is [out.csv]

Production mapping (floor / gate, thresholds = 20th / 40th percentile of in-training predictions, 10-bar EMA) on the
J base rule, with the network's feature set varied, plus controls:
  logistic   logistic regression on the same features (is the non-linearity needed?)
  wrongstock each stock gets ANOTHER stock's model signal (same persistence / exposure statistics, no link to its
             own indicators) -> separates information from "being invested more often"
  rule       the base rule alone (0/1) and with a 0.5 floor (no model)
"""

import sys

import numpy as np

from .. import core
from ..strategy import floor_from, gate_from
from .kdj_features import auc_report, lr_trainer, with_features
from .kdj_harness import baseline_rows, data, evaluate, show


def wrong_stock(signals, shift=7):
    codes = sorted(signals)
    out = {}
    for i, c in enumerate(codes):
        donor = signals[codes[(i + shift) % len(codes)]]
        s = signals[c].copy()
        for col in ("p", "ps", "q_lo", "q_hi"):
            s[col] = donor[col].reindex(s.index)
        mask = signals[c]["ps"].notna()     # keep this stock's own fallback period
        s.loc[~mask, ["ps", "q_lo", "q_hi"]] = np.nan
        out[c] = s
    return out


def with_base(signals, fn, stocks):
    ind = {s["code"]: s["ind"] for s in stocks}
    out = {}
    for c, s in signals.items():
        s = s.copy()
        s["base"] = fn(ind[c])
        out[c] = s
    return out


if __name__ == "__main__":
    d = data()
    stocks0, _ = core.prepare(d, ("kd",))
    rows = baseline_rows()
    sig_main = None
    for fset, model in (("kdj_ma_atr", "mlp"), ("kdj_ma_atr", "lr"), ("kd_ma_atr", "mlp"), ("kdj", "mlp"),
                        ("kd", "mlp"), ("ma_atr_only", "mlp"), ("macd_ma_atr", "mlp")):
        stocks, names = with_features(stocks0, fset)
        sig, _ = core.walk_forward(None, trainer=lr_trainer if model == "lr" else None, stocks=(stocks, names))
        auc, _ = auc_report(stocks, sig)
        tag = f"{fset}/{model} (AUC {auc:.4f})"
        rows.append(evaluate(floor_from(sig), f"floor {tag}"))
        rows.append(evaluate(gate_from(sig), f"gate  {tag}"))
        if fset == "kdj_ma_atr" and model == "mlp":
            sig_main = sig
            for sh in (7, 19):
                w = wrong_stock(sig, sh)
                rows.append(evaluate(floor_from(w), f"floor wrongstock+{sh}"))
                rows.append(evaluate(gate_from(w), f"gate  wrongstock+{sh}"))
            kd_base = with_base(sig, core.base_rule_kd, stocks)
            rows.append(evaluate(floor_from(kd_base), "floor kdj_ma_atr/mlp on OLD KD rule"))
            rows.append(evaluate(gate_from(kd_base), "gate  kdj_ma_atr/mlp on OLD KD rule"))
        print(tag, flush=True)
    rule = {c: s["base"] for c, s in sig_main.items()}
    rows.append(evaluate(rule, "rule 0/1 (J rule)"))
    rows.append(evaluate({c: 0.5 + 0.5 * r for c, r in rule.items()}, "rule + floor (no model)"))
    show(rows, sys.argv[1] if len(sys.argv) > 1 else None)
