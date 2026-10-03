"""Shared registry of every configuration any agent evaluated, so the final results can be deflated for multiple testing.

Each call appends one JSON line to research/trials.jsonl. Log every configuration you backtest, not just the winners.
"""

import json
import math
from datetime import datetime

import numpy as np

from .data import ROOT

TRIALS = ROOT / "research" / "trials.jsonl"


def log_trial(agent, name, config, metrics, period="is"):
    """agent: your family folder; name: short id of the configuration; config: dict of settings;
    metrics: dict with at least median 'sharpe' (and ideally 'cagr', 'mdd', 'exposure') across the 50 stocks."""
    TRIALS.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": datetime.now().isoformat(timespec="seconds"), "agent": agent, "name": name, "period": period,
           "config": config, "metrics": metrics}
    with open(TRIALS, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")


def load_trials():
    if not TRIALS.exists():
        return []
    return [json.loads(line) for line in TRIALS.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm_ppf(p):
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * math.erfc(-mid / math.sqrt(2)) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def deflated_sharpe(sharpe, n_obs, trial_sharpes, skew=0.0, kurt=3.0):
    """Probability that the true Sharpe exceeds the best Sharpe expected from luck across the trials
    (Bailey & López de Prado 2014). Sharpe values are per-period (not annualised); n_obs = number of returns."""
    n = len(trial_sharpes)
    if n < 2:
        return math.nan
    var = float(np.var(trial_sharpes, ddof=1))
    g = 0.5772156649
    sr0 = math.sqrt(var) * ((1 - g) * _norm_ppf(1 - 1 / n) + g * _norm_ppf(1 - 1 / (n * math.e)))
    z = (sharpe - sr0) * math.sqrt(n_obs - 1) / math.sqrt(1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2)
    return 0.5 * math.erfc(-z / math.sqrt(2))
