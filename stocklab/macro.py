"""Exchange rates (USD, JPY vs TWD) and the Fed / Bank of Japan policy rates, aligned to the Taiwan trading calendar.

Sources (no account needed): stocklab.external for USD/TWD (Bank of Taiwan board rate) and USD/JPY (Yahoo), FinMind
InterestRate for central-bank policy decisions (one row per meeting). Timing:
  policy rate decided at a meeting on date d       used from TW date d + 1 (Fed decisions land ~02:00 Taipei on d+1;
                                                   the BoJ announces around noon Tokyo on d)
  FX of date d                                      as in stocklab.external (available for TW date d)
Always pass end=ext.data_end(data) from a strategy so the lookahead check also cuts these series.
"""

from __future__ import annotations

import pandas as pd

from . import external as ext

MACRO_DIR = ext.EXT_DIR / "macro"
POLICY = {"fed_rate": "FED", "boj_rate": "BOJ"}


def download_policy(refresh=False):
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    for name, cb in POLICY.items():
        p = MACRO_DIR / f"{cb}.csv"
        if p.exists() and not refresh:
            continue
        d = ext.finmind("InterestRate", cb, start="2000-01-01")
        d[["date", "interest_rate"]].to_csv(p, index=False)
        print(name, len(d), d["date"].iloc[-1], d["interest_rate"].iloc[-1], flush=True)


def _policy(name) -> pd.Series:
    d = pd.read_csv(MACRO_DIR / f"{POLICY[name]}.csv", parse_dates=["date"]).sort_values("date")
    return pd.Series(d["interest_rate"].astype(float).to_numpy(), index=d["date"] + pd.Timedelta(days=1))


def _on(index, s: pd.Series) -> pd.Series:
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def load_macro(index=None, end=None) -> pd.DataFrame:
    """FX levels and 20-day returns, policy rates and their differential, on the TW calendar."""
    idx = pd.DatetimeIndex(index if index is not None else ext.calendar(end=end))
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    fx = ext.load_market(["usdtwd", "usdjpy"], index=idx, end=end)
    out = pd.DataFrame(index=idx)
    out["usdtwd"] = fx["usdtwd"]
    out["usdjpy"] = fx["usdjpy"]
    out["jpytwd"] = out["usdtwd"] / out["usdjpy"]
    for name in POLICY:
        out[name] = _on(idx, _policy(name))
    out["rate_diff"] = out["fed_rate"] - out["boj_rate"]
    for c in ("usdtwd", "usdjpy", "jpytwd"):
        out[f"{c}_ret20"] = out[c] / out[c].shift(20) - 1
    out["rate_diff_chg120"] = out["rate_diff"] - out["rate_diff"].shift(120)
    return out


def carry_stress(m: pd.DataFrame, yen_jump=-0.03, twd_drop=0.015) -> pd.Series:
    """True when the yen surges (USD/JPY down >= 3% in 20 days: the carry-trade unwind pattern) or the TWD weakens
    quickly (USD/TWD up >= 1.5% in 20 days: typical of foreign outflows from Taiwan stocks)."""
    return (m["usdjpy_ret20"] <= yen_jump) | (m["usdtwd_ret20"] >= twd_drop)


if __name__ == "__main__":
    download_policy(refresh=True)
    m = load_macro()
    print(m.dropna().tail(3).T)
