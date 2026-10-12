"""KDJ / MACD low-point and high-point rankings (the site's front page): how deep (or how high) each security's KDJ and
MACD are after the latest close, and what followed similar points in the past. The high side is the mirror image
(高點分數 = (K + D) / 2 and the DIF percentile itself) for sell / short candidates.

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
SPARK = 60  # sessions in each row's thumbnail (about three months)
BANDS = [0, 30, 50, 70, 80, 90, 100.01]
PERIODS = {"is": ("2010-01-01", "2020-12-31"), "oos": ("2021-01-01", None)}
REBOUND_RULE = "kdj_macd_rebound_turn"
RECENT = 5  # a divergence shows on the row if it happened within the last RECENT sessions
DIVS = ("kd_top_div", "kd_top_div_hidden", "dif_top_div", "kd_bot_div", "kd_bot_div_hidden", "dif_bot_div")
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
    f["kdj_high"] = (f["k"] + f["d"]) / 2
    f["macd_high"] = 100 - f["macd_low"]
    f["score_hi"] = (f["kdj_high"] + f["macd_high"]) / 2
    f["both_high"] = (f["d"] > 80) & (f["dif"] > 0) & (f["macd"] > 0)
    o = df["open"].to_numpy(dtype=float)
    fwd = np.full(len(o), np.nan)
    fwd[:-(H + 1)] = o[H + 1:] / o[1:-H] - 1  # bought at the next open, held H sessions
    f["fwd"] = fwd
    return f


def _flags(f, ind) -> list:
    k, osc = f["k"].to_numpy(), f["osc"].to_numpy()
    last = f.iloc[-1]
    k_up = k[-1] > k[-2] and k[-2] <= k[-3]
    osc_up = osc[-1] > osc[-2] and osc[-2] <= osc[-3]
    k_down = k[-1] < k[-2] and k[-2] >= k[-3]
    osc_down = osc[-1] < osc[-2] and osc[-2] >= osc[-3]
    zone_neg = last["dif"] < 0 and last["macd"] < 0
    zone_pos = last["dif"] > 0 and last["macd"] > 0
    out = []
    if last["d"] > 80:
        out.append("kdj_high")
    if last["j"] > 100:
        out.append("j_hi")
    if zone_pos:
        out.append("macd_pos")
    if k_down:
        out.append("k_down")
    if osc_down:
        out.append("osc_down")
    if k_down and osc_down and zone_pos:
        out.append("weak")
    for cid in DIVS:  # within the last RECENT sessions
        if rules.condition(ind, {"id": cid})[-RECENT:].any():
            out.append(cid)
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


def _history(f, col="both_low") -> dict | None:
    g = f.loc["2010-01-01":].dropna(subset=["fwd"])
    if len(g) < 250:
        return None
    start = g[col] & ~g[col].shift(1, fill_value=False)
    ep = g.loc[start, "fwd"]
    out = {"n": int(len(ep)), "base": float(g["fwd"].mean()), "base_up": float((g["fwd"] > 0).mean())}
    if len(ep):
        out |= {"mean": float(ep.mean()), "up": float((ep > 0).mean())}
    return out


def _streak_now(b) -> int:
    n = 0
    for v in b[::-1]:
        if not v:
            break
        n += 1
    return n


def _low_days(f, col="both_low") -> int:
    b = f[col].to_numpy()
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
    sell = [t for t in e.get("trig", []) if t["side"] < 0]
    near = lambda ts: min(ts, key=lambda t: abs(t["d"])) if ts else None
    return {"a": e["a"], "side": e["side"], "buy": near(buy), "sell": near(sell)}


def _short_status(code, asof) -> str:
    """'ok' shortable at the next session, 'x' 停止融券 (or 停止買賣) per the latest margin report, '' unknown."""
    from . import shortrules

    try:
        n = shortrules.notes(code)
    except (FileNotFoundError, KeyError):
        return ""
    if not len(n) or n.index[-1] < asof:
        return ""
    note = n["Note"].iloc[-1]
    return "x" if ("X" in note or "!" in note) else "ok"


def _spark(df, f) -> dict:
    """Last SPARK sessions for the row thumbnail: closes, K, D, and MACD (DIF, histogram) as % of the close."""
    t, g = df.iloc[-SPARK:], f.iloc[-SPARK:]
    r = lambda v, n: [round(float(x), n) if n else int(round(float(x))) for x in v]
    return {"d0": t.index[0].strftime("%Y-%m-%d"), "d1": t.index[-1].strftime("%Y-%m-%d"), "dates": [d.strftime("%m/%d") for d in t.index],
            "c": r(t["close"], 2), "k": r(g["k"], 0), "d": r(g["d"], 0),
            "dif": r(g["dif"] / t["close"] * 100, 2), "osc": r(g["osc"] / t["close"] * 100, 2)}


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
    pooled_hi = {p: [] for p in PERIODS}
    for code, df in data.items():
        if len(df) < 300:
            continue
        f = frame(df)
        ind = rules.Indicators(df)
        for p in PERIODS:
            g = _period(f, p).dropna(subset=["fwd", "score"])
            if len(g) > 60:
                pooled[p].append(pd.DataFrame({"score": g["score"], "ex": g["fwd"] - g["fwd"].mean(), "up": g["fwd"] > 0,
                                               "start": g["both_low"] & ~g["both_low"].shift(1, fill_value=False)}))
                pooled_hi[p].append(pd.DataFrame({"score": g["score_hi"], "ex": g["fwd"] - g["fwd"].mean(), "up": g["fwd"] > 0,
                                                  "start": g["both_high"] & ~g["both_high"].shift(1, fill_value=False)}))
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
            "score_hi": _round(last["score_hi"], 1), "kdj_high": _round(last["kdj_high"], 1), "macd_high": _round(last["macd_high"], 1),
            "flags": _flags(f, ind), "low_days": _low_days(f), "high_days": _low_days(f, "both_high"),
            "stuck_hi": _streak_now(f["k"].to_numpy() > 80), "stuck_lo": _streak_now(f["k"].to_numpy() < 20),
            "hist": _history(f), "hist_hi": _history(f, "both_high"), "rule": _rule_today(signals, code),
            "short": _short_status(code, asof),
            "spark": _spark(df, f),
        })
    rows.sort(key=lambda r: -r["score"])
    # thumbnails share the trading calendar: keep the dates once, per row only where they differ (suspensions)
    common = max({tuple(r["spark"]["dates"]) for r in rows}, key=lambda t: sum(tuple(r["spark"]["dates"]) == t for r in rows))
    for r in rows:
        if tuple(r["spark"]["dates"]) == common:
            del r["spark"]["dates"]
    def summarize(pool):
        bands, episodes = {}, {}
        for p, parts in pool.items():
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
        return bands, episodes

    bands, episodes = summarize(pooled)
    bands_hi, episodes_hi = summarize(pooled_hi)
    doc = {"asof": asof.strftime("%Y-%m-%d"), "generated": datetime.now(TPE).isoformat(timespec="minutes"),
           "horizon": H, "rank_window": RANK_WINDOW, "spark": SPARK, "spark_dates": list(common), "rows": rows, "bands": bands, "episodes": episodes,
           "bands_hi": bands_hi, "episodes_hi": episodes_hi}
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
