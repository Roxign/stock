"""Buy / sell notifications: each strategy's latest decision per security, price triggers for the next session, and
how good each strategy's past buy and sell points were.

Timing is the backtest's: a strategy decides its target exposure at a bar's close and the order fills at the next
session's open. After the close of the last bar (asof), target[asof] versus target[asof - 1] is the order for the next
open (買進 / 加碼 / 賣出 / 減碼 / 放空 / 加空 / 回補 / 減空).

Price triggers: when a strategy decides each security from that security's own bars only (checked by rerunning it on
one security at a time, not assumed), a hypothetical next bar is appended (open = last close, close = x, high / low =
the larger / smaller of the two, volume = 20-day average) for x on a grid over the ±10% daily price limit, and the
switching points are refined by bisection and rounded to the tick. The result lists the closing prices that would
change the next decision, e.g. "收盤 ≤ 512 → 賣出" (filled at the open after that close). The bar has no wicks, so
rules that use the high / low (KDJ) can come out slightly different; the signal after the actual close is what counts.

Signal quality: the move from each past buy / sell fill (open) to the open 5 and 20 sessions later, minus that
security's average move over the same period, pooled by universe, plus statistics of the completed trades.

  python -m stocklab.signals        # from cached positions (python daily_update.py refreshes them first)
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import backtest as bt
from .data import ROOT
from .runner import discover, fees_of, load_everything, pos_path, universe_data, universes_of

OUT = ROOT / "docs" / "data" / "signals.json"
GRID = np.round(np.linspace(-0.10, 0.10, 41), 4)  # next close vs last close, 0.5% steps over the price limit
BISECT = 8                 # 0.5% / 2^8 = 0.002%: finer than any tick
WINDOW = 800               # trailing bars used for trigger scans (indicators and rule states settle well within this)
CHECK = 60                 # ...accepted only if its decisions match the full history on the last CHECK bars
PER_STOCK_FAMILIES = {"kdj_macd_rule", "kdj_macd_rebound", "trend", "mean_reversion", "long_short"}
PER_STOCK_IDS = {"kdj_macd_dl_base"}  # the no-DL rule inside a DL family (its siblings would retrain models)
WEIGHT_FAMILIES = {"multi_asset"}
HORIZONS = (5, 20)
MIN_STEP = 0.25            # exposure change that counts as a buy / sell point in the quality statistics
QUALITY_PERIODS = ("is", "oos")
EPS = 1e-6
TPE = timezone(timedelta(hours=8))


def action(held, tgt):
    """Order for the next open when the held exposure becomes tgt: (label, side), side +1 = buy order, -1 = sell."""
    if abs(tgt - held) < EPS:
        return ("續抱" if held > EPS else "續空" if held < -EPS else "空手"), 0
    flat_h, flat_t = abs(held) < EPS, abs(tgt) < EPS
    if held >= -EPS and tgt >= -EPS:
        return (("買進" if flat_h else "加碼"), 1) if tgt > held else (("賣出" if flat_t else "減碼"), -1)
    if held <= EPS and tgt <= EPS:
        return (("放空" if flat_h else "加空"), -1) if tgt < held else (("回補" if flat_t else "減空"), 1)
    return ("賣出並放空", -1) if tgt < 0 else ("回補並買進", 1)


def tick(px, etf=False):
    """TWSE tick size (stocks; ETFs 0.01 below 50, else 0.05)."""
    if etf:
        return 0.01 if px < 50 else 0.05
    for limit, t in ((10, 0.01), (50, 0.05), (100, 0.1), (500, 0.5), (1000, 1.0)):
        if px < limit:
            return t
    return 5.0


def on_tick(px, etf, up):
    t = tick(px, etf)
    n = math.ceil(px / t - 1e-9) if up else math.floor(px / t + 1e-9)
    return round(n * t, 2)


def _decide(s, code, df):
    return bt.clean_target(s["positions"]({code: df})[code], df.index)


def _with_bar(df, close, volume):
    bar = df.iloc[[-1]].copy()
    o = float(df["close"].iloc[-1])
    bar.index = pd.DatetimeIndex([df.index[-1] + pd.offsets.BDay(1)], name=df.index.name)
    bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"] = o, max(o, close), min(o, close), close, volume
    return pd.concat([df, bar])


def per_stock(s, udata, raw, n=2):
    """True if one-security runs reproduce the universe run (the strategy looks at nothing but the security)."""
    for c in list(udata)[:n]:
        one = _decide(s, c, udata[c]).to_numpy()
        if not np.allclose(one, bt.clean_target(raw[c], udata[c].index).to_numpy(), atol=EPS):
            return False
    return True


def _window(s, code, df, ref):
    if len(df) > WINDOW + CHECK:
        w = df.iloc[-WINDOW:]
        if np.allclose(_decide(s, code, w).iloc[-CHECK:].to_numpy(), ref[-CHECK:], atol=EPS):
            return w
    return df


def scan(s, code, df, cur, etf):
    """Closing-price ranges for the next session that change the decision from `cur`."""
    c0 = float(df["close"].iloc[-1])
    vol = float(df["volume"].iloc[-20:].mean())
    memo = {}

    def f(r):
        if r not in memo:
            memo[r] = float(_decide(s, code, _with_bar(df, c0 * (1 + r), vol)).iloc[-1])
        return memo[r]

    vals = [f(r) for r in GRID]
    segs, lo = [], None  # [first return with the value, last return with it, value]; None = beyond the grid
    for k in range(len(GRID)):
        if k == len(GRID) - 1:
            segs.append((lo, None, vals[k]))
        elif abs(vals[k + 1] - vals[k]) > EPS:
            a, b = float(GRID[k]), float(GRID[k + 1])
            for _ in range(BISECT):
                m = (a + b) / 2
                a, b = (m, b) if abs(f(m) - vals[k]) < EPS else (a, m)
            segs.append((lo, a, vals[k]))
            lo = b
    out = []
    for r_lo, r_hi, v in segs:
        if abs(v - cur) < EPS:
            continue
        p_lo = None if r_lo is None else on_tick(c0 * (1 + r_lo), etf, up=True)
        p_hi = None if r_hi is None else on_tick(c0 * (1 + r_hi), etf, up=False)
        if p_lo is not None and p_hi is not None and p_lo > p_hi:
            continue  # narrower than a tick
        if p_hi is not None and c0 > p_hi:
            dist = p_hi / c0 - 1
        elif p_lo is not None and c0 < p_lo:
            dist = p_lo / c0 - 1
        else:
            dist = 0.0
        label, side = action(cur, v)
        out.append({"lo": p_lo, "hi": p_hi, "to": round(v, 4), "a": label, "side": side, "d": round(dist, 4)})
    return out


def _since(tgt, o, c, dates):
    """The position held during the last bar and the fill that set its level (the last sign change or exposure step of
    at least MIN_STEP): date, open price, sessions held, gross return to the last close, exposure."""
    held = np.r_[0.0, tgt[:-1]]  # exposure during bar i (decided at the close of i - 1, filled at i's open)
    T = len(tgt) - 1
    side = np.sign(round(float(held[T]), 6))
    if side == 0:
        return None
    sgn = np.sign(np.round(held, 6))
    i = T
    while i > 1 and sgn[i - 1] == side and abs(held[i] - held[i - 1]) < MIN_STEP:
        i -= 1
    return {"date": dates[i].strftime("%Y-%m-%d"), "px": round(float(o[i]), 4), "days": int(T - i + 1),
            "ret": round(float(side * (c[T] / o[i] - 1)), 4), "level": round(float(held[T]), 4)}


def _quality_one(df, tgt, fees, period):
    b = bt.period_bounds(df.index, period)
    if b is None:
        return None
    i0, i1 = b
    o = df["open"].to_numpy()
    held = np.r_[0.0, tgt[:-1]]          # exposure during bar i (decided at the close of i - 1)
    d = np.r_[0.0, np.diff(held)]        # change filled at bar i's open
    idx = np.arange(i0 + 1, i1 + 1)
    out = {"years": (i1 - i0 + 1) / 252, "buys": int((d[idx] >= MIN_STEP).sum()), "sells": int((d[idx] <= -MIN_STEP).sum())}
    for h in HORIZONS:
        fwd = np.full(len(o), np.nan)
        fwd[:-h] = o[h:] / o[:-h] - 1
        ok = idx[idx <= i1 - h]
        base = float(np.nanmean(fwd[i0:i1 - h + 1])) if i1 - h >= i0 else math.nan
        up = fwd[i0:i1 - h + 1]
        out[f"base_up{h}"] = (float((up > 0).sum()), int(np.isfinite(up).sum()))
        out[f"buy{h}"] = list(fwd[ok[d[ok] >= MIN_STEP]] - base)
        out[f"sell{h}"] = list(fwd[ok[d[ok] <= -MIN_STEP]] - base)
        out[f"buy_up{h}"] = list((fwd[ok[d[ok] >= MIN_STEP]] > 0).astype(float))
        out[f"sell_dn{h}"] = list((fwd[ok[d[ok] <= -MIN_STEP]] < 0).astype(float))
    trades, entry, cur = [], None, 0.0
    for i in range(i0, i1 + 1):  # engine convention: flat at i0, fills at opens
        t = held[i]
        if abs(t - cur) < 1e-9:
            continue
        if cur != 0 and (t == 0 or (t > 0) != (cur > 0)) and entry:
            trades.append((bt._trade_ret(entry[1], entry[2], o[i], fees), i - entry[0]))
            entry = None
        if t != 0 and (cur == 0 or (t > 0) != (cur > 0)):
            entry = (i, 1 if t > 0 else -1, o[i])
        cur = t
    out["trades"] = trades
    return out


def _pool(parts):
    parts = [p for p in parts if p]
    if not parts:
        return None
    cat = lambda k: [x for p in parts for x in p[k]]
    mean = lambda v: round(float(np.mean(v)), 5) if len(v) else None
    years = sum(p["years"] for p in parts)
    tr = [t for p in parts for t in p["trades"]]
    out = {"codes": len(parts), "buys": sum(p["buys"] for p in parts), "sells": sum(p["sells"] for p in parts)}
    out["buy_rate"] = round(out["buys"] / years, 3) if years else None
    for h in HORIZONS:
        out[f"buy{h}"], out[f"sell{h}"] = mean(cat(f"buy{h}")), mean(cat(f"sell{h}"))
        out[f"buy_up{h}"], out[f"sell_dn{h}"] = mean(cat(f"buy_up{h}")), mean(cat(f"sell_dn{h}"))
        ups, n = sum(p[f"base_up{h}"][0] for p in parts), sum(p[f"base_up{h}"][1] for p in parts)
        out[f"base_up{h}"] = round(ups / n, 5) if n else None
    out["trades"] = len(tr)
    out["win"] = mean([float(r > 0) for r, _ in tr])
    out["avg"] = mean([r for r, _ in tr])
    out["hold"] = round(float(np.mean([n for _, n in tr])), 1) if tr else None
    return out


def robust(quality, side, min_signals=30):
    """True if every universe / period with data (50 stocks, extra stocks, ETFs; in- and out-of-sample) shows the right
    sign for the 20-session excess move after this side's signals: above zero after buys, below zero after sells."""
    key, count, sign = ("buy20", "buys", 1) if side > 0 else ("sell20", "sells", -1)
    vals, n = [], 0
    for kind in ("stocks", "extra", "etfs"):
        for per in QUALITY_PERIODS:
            q = quality.get(kind, {}).get(per)
            if q and q.get(key) is not None:
                vals.append(q[key] * sign)
                n += q[count]
    return len(vals) >= 2 and n >= min_signals and all(v > 0 for v in vals)


def _kind(code):
    from .etf import CODES as ETF_CODES
    from .universe import EXTRA_CODES

    return "etfs" if code in ETF_CODES else "extra" if code in EXTRA_CODES else "stocks"


NOTE_TEXT = {"X": "停止融券", "O": "停止融資", "!": "停止買賣"}


def _short_note(code, asof):
    """Note of the margin report dated asof (it describes the next session); "" if that report isn't cached."""
    from . import shortrules

    try:
        n = shortrules.notes(code)
    except (FileNotFoundError, KeyError):
        return ""
    return n["Note"].iloc[-1] if len(n) and n.index[-1] == asof else ""


def macro_snapshot(end):
    from . import macro

    m = macro.load_macro(end=end).dropna(subset=["usdtwd", "usdjpy", "fed_rate", "boj_rate"])
    if m.empty:
        return None
    last = m.iloc[-1]
    f = lambda x, n=4: None if pd.isna(x) else round(float(x), n)
    hist = m.iloc[-260:]
    return {
        "date": m.index[-1].strftime("%Y-%m-%d"),
        "usdtwd": f(last["usdtwd"]), "usdjpy": f(last["usdjpy"], 2), "jpytwd": f(last["jpytwd"], 5),
        "fed": f(last["fed_rate"], 3), "boj": f(last["boj_rate"], 3), "diff": f(last["rate_diff"], 3),
        "usdtwd20": f(last["usdtwd_ret20"]), "usdjpy20": f(last["usdjpy_ret20"]), "jpytwd20": f(last["jpytwd_ret20"]),
        "diff120": f(last["rate_diff_chg120"], 3), "stress": bool(macro.carry_stress(m).iloc[-1]),
        "hist": {"d": [d.strftime("%Y-%m-%d") for d in hist.index], "usdtwd": [f(x) for x in hist["usdtwd"]],
                 "usdjpy": [f(x, 2) for x in hist["usdjpy"]], "diff": [f(x, 3) for x in hist["rate_diff"]]},
    }


def build(out=OUT, ids=None, verbose=True):
    t0 = time.time()
    data = load_everything()
    asof = max(df.index[-1] for df in data.values())
    strategies = [s for s in discover() if ids is None or s["id"] in ids]
    from .etf import CODES as ETF_CODES
    from .etf import ETFS
    from .universe import EXTRA_NAMES, NAMES

    stocks = {}
    for c, df in data.items():
        cl = df["close"].to_numpy()
        stocks[c] = {"name": NAMES.get(c) or EXTRA_NAMES.get(c) or ETFS[c]["name"], "kind": _kind(c),
                     "date": df.index[-1].strftime("%Y-%m-%d"), "close": round(float(cl[-1]), 4),
                     "chg": round(float(cl[-1] / cl[-2] - 1), 4), "note": _short_note(c, asof), "sig": {}}
        stocks[c]["note_text"] = "、".join(t for k, t in NOTE_TEXT.items() if k in stocks[c]["note"])
    meta, quality = {}, {}
    for s in strategies:
        sid = s["id"]
        t = time.time()
        meta[sid] = {"triggers": False, "asof": None}
        parts = {}
        for u in universes_of(s, data):
            path = pos_path(sid, u)
            if not path.exists():
                continue
            pos = pd.read_pickle(path)
            udata = universe_data(s, data, u)
            raw = s["positions"](udata) if s["family_dir"] in PER_STOCK_FAMILIES or sid in PER_STOCK_IDS else None
            # multi-asset strategies: their per-ETF positions only mark holdings, the weights are the actual orders
            wts = s["weights"](udata) if s.get("weights") and s["family_dir"] in WEIGHT_FAMILIES else None
            scanned = raw is not None and per_stock(s, udata, raw)
            meta[sid]["triggers"] |= scanned
            for c, df in udata.items():
                p = pos.get(c)
                if p is None or len(p) < 3:
                    continue
                last = min(p.index[-1], df.index[-1])
                d = df.loc[:last]
                tgt = bt.clean_target(wts[c] if wts is not None and c in wts else p, d.index).to_numpy()
                o, cl = d["open"].to_numpy(), d["close"].to_numpy()
                held, cur = float(tgt[-2]), float(tgt[-1])
                label, side = action(held, cur)
                e = {"a": label, "side": side, "held": round(held, 4), "to": round(cur, 4)}
                if last < asof:
                    e["stale"] = last.strftime("%Y-%m-%d")
                since = _since(tgt, o, cl, d.index)
                if since:
                    e["since"] = since
                note = stocks[c]["note"]
                if side and "!" in note:
                    e["warn"] = "下一交易日停止買賣"
                elif cur < -EPS and side < 0 and "X" in note:
                    e["warn"] = "下一交易日停止融券賣出，無法放空"
                elif held < -EPS and "X" in note:
                    e["warn"] = "下一交易日停止融券：空單會在開盤強制回補"
                elif cur < held - EPS and cur < -EPS and c not in ETF_CODES and stocks[c]["chg"] <= -0.095:
                    e["warn"] = "今日跌停：明日開盤若低於今日收盤（平盤以下）不得融券賣出"
                if scanned and last == asof:
                    ref = bt.clean_target(raw[c], df.index).to_numpy()
                    e["trig"] = scan(s, c, _window(s, c, df, ref), float(ref[-1]), c in ETF_CODES)
                stocks[c]["sig"][sid] = e
                parts.setdefault(_kind(c), []).append((c, d, tgt))
            meta[sid]["asof"] = max(meta[sid]["asof"] or "", max(p.index[-1] for p in pos.values()).strftime("%Y-%m-%d"))
        quality[sid] = {k: {per: _pool([_quality_one(d, tgt, fees_of(c), per) for c, d, tgt in lst]) for per in QUALITY_PERIODS}
                        for k, lst in parts.items()}
        if verbose:
            print(f"  {sid}: {time.time() - t:.1f}s{' (triggers)' if meta[sid]['triggers'] else ''}", flush=True)
    doc = {
        "asof": asof.strftime("%Y-%m-%d"),
        "generated": datetime.now(TPE).isoformat(timespec="minutes"),
        "horizons": list(HORIZONS),
        "min_step": MIN_STEP,
        "macro": macro_snapshot(asof),
        "strategies": {sid: {**meta[sid], "quality": quality[sid], "robust_buy": robust(quality[sid], 1),
                             "robust_sell": robust(quality[sid], -1)} for sid in meta},
        "stocks": stocks,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_clean(doc), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    if verbose:
        print(f"signals as of {doc['asof']} -> {out} ({out.stat().st_size / 1e3:.0f} kB, {time.time() - t0:.0f}s)")
    return doc


def _clean(x):
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.generic):
        return _clean(x.item())
    return x


def load(path=OUT):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(doc=None, sids=None, codes=None):
    """Plain-text notification: orders for the next open, then triggers within 3% (for a future push channel)."""
    doc = doc or load()
    lines = [f"{doc['asof']} 收盤後訊號（下一個交易日開盤執行）"]
    near = []
    for c, st in doc["stocks"].items():
        if codes and c not in codes:
            continue
        for sid, e in st["sig"].items():
            if sids and sid not in sids:
                continue
            if e["side"]:
                lines.append(f"【{e['a']}】{c} {st['name']}｜{sid}｜收盤 {st['close']:g}" + (f"｜{e['warn']}" if e.get("warn") else ""))
            for t in e.get("trig", []):
                if abs(t["d"]) <= 0.03:
                    if t["lo"] is None and t["hi"] is None:
                        rng, dist = "任何價格", "必定觸發"
                    else:
                        rng = f"≤ {t['hi']:g}" if t["lo"] is None else f"≥ {t['lo']:g}" if t["hi"] is None else f"{t['lo']:g}～{t['hi']:g}"
                        dist = "平盤即觸發" if t["d"] == 0 else f"{t['d']:+.1%}"
                    near.append(f"  {c} {st['name']}｜{sid}｜明日收盤 {rng} → {t['a']}（{dist}）")
    if near:
        lines += ["", "接近觸發（明日收盤變動 3% 以內）："] + near
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    build()
