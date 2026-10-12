"""KDJ / MACD low-point ranking (the site's front page): how deep each security's KDJ and MACD are after the latest
close, and what followed similar low points in the past.

低點分數 (0-100, higher = lower point) = average of
  KDJ 低檔   100 - (K + D) / 2 with the Taiwan KDJ(9,3,3); 100 = K and D both at 0
  MACD 低檔  100 x (1 - percentile of today's DIF / close among the last 500 sessions); 100 = DIF (EMA12 - EMA26)
             at its lowest relative to the price in two years, so stocks of any price level compare
Flags (today): KDJ 低檔 (D < 20), J < 0, MACD 零軸下 (DIF and MACD line both < 0), K 勾頭向上, 柱狀體勾頭向上, and
同步翻揚 = the user's buy rule (K and the histogram turning up while both MACD lines are below zero).
History, 2010 on: episodes where KDJ and MACD were both low (D < 20 and both MACD lines < 0; first day of each
episode), with the move from the next open to the open 20 sessions later versus any day of the same security; and the
same forward move by score band pooled over all securities, in- and out-of-sample (does a lower point help?).

  python -m stocklab.screen      # writes docs/data/low.json (daily_update.py runs it after the signals)
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import rules
from .data import ROOT

OUT = ROOT / "docs" / "data" / "low.json"
RANK_WINDOW = 500
H = 20
BANDS = [0, 30, 50, 70, 80, 90, 100.01]
PERIODS = {"is": ("2010-01-01", "2020-12-31"), "oos": ("2021-01-01", None)}
REBOUND_RULE = "kdj_macd_rebound_turn"
TPE = timezone(timedelta(hours=8))


def frame(df) -> pd.DataFrame:
    """Daily indicator values, low-point score, flags and the forward move after each close."""
    ind = rules.Indicators(df)
    idx = df.index
    close = df["close"]
    f = pd.DataFrame({"k": ind.k, "d": ind.d, "j": ind.j, "dif": ind.dif, "macd": ind.sig, "osc": ind.osc}, index=idx)
    f["dif_pct"] = f["dif"] / close * 100
    f["macd_low"] = (1 - f["dif_pct"].rolling(RANK_WINDOW, min_periods=250).rank(pct=True)) * 100
    f["kdj_low"] = 100 - (f["k"] + f["d"]) / 2
    f["score"] = (f["kdj_low"] + f["macd_low"]) / 2
    f["both_low"] = (f["d"] < 20) & (f["dif"] < 0) & (f["macd"] < 0)
    o = df["open"].to_numpy(dtype=float)
    fwd = np.full(len(o), np.nan)
    fwd[:-(H + 1)] = o[H + 1:] / o[1:-H] - 1  # bought at the next open, held H sessions
    f["fwd"] = fwd
    return f


def _flags(f) -> list:
    k, osc = f["k"].to_numpy(), f["osc"].to_numpy()
    last = f.iloc[-1]
    k_up = k[-1] > k[-2] and k[-2] <= k[-3]
    osc_up = osc[-1] > osc[-2] and osc[-2] <= osc[-3]
    zone_neg = last["dif"] < 0 and last["macd"] < 0
    out = []
    if last["d"] < 20:
        out.append("kdj_low")
    if last["j"] < 0:
        out.append("j_neg")
    if zone_neg:
        out.append("macd_neg")
    if k_up:
        out.append("k_up")
    if osc_up:
        out.append("osc_up")
    if k_up and osc_up and zone_neg:
        out.append("rebound")
    return out


def _period(f, p):
    a, b = PERIODS[p]
    return f.loc[a:b] if b else f.loc[a:]


def _history(f) -> dict | None:
    g = f.loc["2010-01-01":].dropna(subset=["fwd"])
    if len(g) < 250:
        return None
    start = g["both_low"] & ~g["both_low"].shift(1, fill_value=False)
    ep = g.loc[start, "fwd"]
    out = {"n": int(len(ep)), "base": float(g["fwd"].mean()), "base_up": float((g["fwd"] > 0).mean())}
    if len(ep):
        out |= {"mean": float(ep.mean()), "up": float((ep > 0).mean())}
    return out


def _low_days(f) -> int:
    b = f["both_low"].to_numpy()
    n = 0
    for v in b[::-1]:
        if not v:
            break
        n += 1
    return n


def _rule_today(signals, code):
    e = (signals or {}).get("stocks", {}).get(code, {}).get("sig", {}).get(REBOUND_RULE)
    if not e:
        return None
    buy = [t for t in e.get("trig", []) if t["side"] > 0]
    return {"a": e["a"], "side": e["side"], "buy": min(buy, key=lambda t: abs(t["d"])) if buy else None}


def _round(x, n=2):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), n)


def build(out=OUT, verbose=True) -> dict:
    from .etf import ETFS
    from .runner import load_everything
    from .universe import EXTRA_NAMES, NAMES, SECTORS

    t0 = time.time()
    data = load_everything()
    asof = max(df.index[-1] for df in data.values())
    sig_path = ROOT / "docs" / "data" / "signals.json"
    signals = json.loads(sig_path.read_text(encoding="utf-8")) if sig_path.exists() else None
    from .etf import CODES as ETF_CODES
    from .universe import EXTRA_CODES

    rows, pooled = [], {p: [] for p in PERIODS}
    for code, df in data.items():
        if len(df) < 300:
            continue
        f = frame(df)
        for p in PERIODS:
            g = _period(f, p).dropna(subset=["fwd", "score"])
            if len(g) > 60:
                pooled[p].append(pd.DataFrame({"score": g["score"], "ex": g["fwd"] - g["fwd"].mean(), "up": g["fwd"] > 0,
                                               "start": g["both_low"] & ~g["both_low"].shift(1, fill_value=False)}))
        last = f.iloc[-1]
        if df.index[-1] < asof or not math.isfinite(last["score"]):
            continue
        cl = df["close"].to_numpy()
        rows.append({
            "code": code, "name": NAMES.get(code) or EXTRA_NAMES.get(code) or ETFS[code]["name"],
            "kind": "etf" if code in ETF_CODES else "extra" if code in EXTRA_CODES else "stock", "sector": SECTORS.get(code),
            "close": _round(cl[-1], 4), "chg": _round(cl[-1] / cl[-2] - 1, 4),
            "score": _round(last["score"], 1), "kdj_low": _round(last["kdj_low"], 1), "macd_low": _round(last["macd_low"], 1),
            "k": _round(last["k"], 1), "d": _round(last["d"], 1), "j": _round(last["j"], 1),
            "dif_pct": _round(last["dif_pct"], 2), "macd_pct": _round(last["macd"] / cl[-1] * 100, 2),
            "flags": _flags(f), "low_days": _low_days(f), "hist": _history(f), "rule": _rule_today(signals, code),
        })
    rows.sort(key=lambda r: -r["score"])
    bands, episodes = {}, {}
    for p, parts in pooled.items():
        allp = pd.concat(parts, ignore_index=True)
        base_up = float(allp["up"].mean())
        bands[p] = []
        for lo, hi in zip(BANDS[:-1], BANDS[1:]):
            g = allp[(allp["score"] >= lo) & (allp["score"] < hi)]
            bands[p].append({"lo": lo, "hi": min(hi, 100), "n": int(len(g)), "ex": _round(g["ex"].mean(), 5),
                             "up": _round(g["up"].mean(), 4), "base_up": _round(base_up, 4)})
        ep = allp[allp["start"]]
        episodes[p] = {"n": int(len(ep)), "ex": _round(ep["ex"].mean(), 5), "up": _round(ep["up"].mean(), 4),
                       "base_up": _round(base_up, 4)}
    doc = {"asof": asof.strftime("%Y-%m-%d"), "generated": datetime.now(TPE).isoformat(timespec="minutes"),
           "horizon": H, "rank_window": RANK_WINDOW, "rows": rows, "bands": bands, "episodes": episodes}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    if verbose:
        print(f"low-point ranking as of {doc['asof']}: {len(rows)} securities -> {out} ({out.stat().st_size / 1e3:.0f} kB, "
              f"{time.time() - t0:.0f}s)", flush=True)
    return doc


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    d = build()
    for p in PERIODS:
        print(p, d["episodes"][p], [(b["lo"], b["n"], b["ex"], b["up"]) for b in d["bands"][p]])
    for r in d["rows"][:12]:
        print(r["code"], r["name"], r["score"], r["k"], r["d"], r["j"], r["dif_pct"], r["flags"], r["low_days"], r["hist"], r["rule"])
