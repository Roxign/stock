"""Rule lab engine: long-only KDJ / MACD / moving-average rules described as data, mirrored by docs/rules.js.

A rule (JSON) is
  {"name": ..., "buy":  {"mode": "all" | "any", "within": W, "conds": [{"id": "k_up"}, {"id": "j_below", "v": 0}, ...]},
                "sell": {...same...},
                "stop": 10, "take": 0, "trail": 0, "hold": 0,      # percent / percent / percent / bars, 0 = off
                "kdj_n": 9, "macd": [12, 26, 9]}
Condition ids, labels and PowerLanguage templates are in docs/rule_catalog.json (shared with the browser).

Semantics (identical in both engines):
  mode "all"  every condition was true at least once in the last W bars (W = 1: on the same bar)
  mode "any"  at least one condition is true on the bar (W is ignored)
  no conditions on a side: that side never fires
  0/1 position decided at each bar's close, filled at the next open by the backtest engine; while flat a buy signal
  goes long; while long, exits are checked from the bar after the signal (the fill bar) on: sell signal, close below
  entry x (1 - stop%), close above entry x (1 + take%), close below the highest close since entry x (1 - trail%),
  or `hold` bars since the signal. Entry = open of the bar after the buy signal. No re-entry on the exit bar.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from .data import ROOT
from .indicators import macd, sma, tw_kdj

CATALOG_PATH = ROOT / "docs" / "rule_catalog.json"


@lru_cache(maxsize=1)
def catalog() -> dict:
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {c["id"]: c for c in data["conditions"]}


def _prev(x, n=1):
    out = np.full(len(x), np.nan)
    if n < len(x):
        out[n:] = x[:-n]
    return out


def _turn(x, up):
    x1, x2 = _prev(x), _prev(x, 2)
    return (x > x1) & (x1 <= x2) if up else (x < x1) & (x1 >= x2)


def _cross(a, b, up):
    a1, b1 = _prev(a), _prev(b)
    return (a > b) & (a1 <= b1) if up else (a < b) & (a1 >= b1)


class Indicators:
    """KDJ, MACD and moving averages of one security, computed once and shared by all conditions."""

    def __init__(self, df, kdj_n=9, macd_p=(12, 26, 9)):
        k = tw_kdj(df, kdj_n)
        m = macd(df["close"], *macd_p)
        self.close_s = df["close"]
        self.close = df["close"].to_numpy(dtype=float)
        self.k, self.d, self.j = (k[c].to_numpy() for c in ("k", "d", "j"))
        self.dif, self.sig, self.osc = (m[c].to_numpy() for c in ("dif", "macd", "osc"))
        self._ma = {}

    def ma(self, n):
        n = int(n)
        if n not in self._ma:
            self._ma[n] = sma(self.close_s, n).to_numpy()
        return self._ma[n]


def _param(c, name):
    spec = catalog()[c["id"]].get("params", {}).get(name, {})
    return float(c.get(name, spec.get("default", math.nan)))


def condition(ind: Indicators, c) -> np.ndarray:
    k, d, j, dif, sg, osc, cl = ind.k, ind.d, ind.j, ind.dif, ind.sig, ind.osc, ind.close
    cid = c["id"]
    with np.errstate(invalid="ignore"):
        if cid == "kd_golden":
            out = _cross(k, d, True)
        elif cid == "kd_dead":
            out = _cross(k, d, False)
        elif cid in ("k_up", "k_down"):
            out = _turn(k, cid == "k_up")
        elif cid in ("j_up", "j_down"):
            out = _turn(j, cid == "j_up")
        elif cid == "kd_bull":
            out = k > d
        elif cid == "kd_bear":
            out = k < d
        elif cid in ("k_below", "k_above", "d_below", "d_above", "j_below", "j_above"):
            x = {"k": k, "d": d, "j": j}[cid[0]]
            v = _param(c, "v")
            out = x < v if cid.endswith("below") else x > v
        elif cid == "macd_golden":
            out = _cross(dif, sg, True)
        elif cid == "macd_dead":
            out = _cross(dif, sg, False)
        elif cid in ("osc_up", "osc_down"):
            out = _turn(osc, cid == "osc_up")
        elif cid == "osc_rising":
            out = osc > _prev(osc)
        elif cid == "osc_falling":
            out = osc < _prev(osc)
        elif cid in ("dif_up", "dif_down"):
            out = _turn(dif, cid == "dif_up")
        elif cid == "osc_pos":
            out = osc > 0
        elif cid == "osc_neg":
            out = osc < 0
        elif cid == "dif_pos":
            out = dif > 0
        elif cid == "dif_neg":
            out = dif < 0
        elif cid == "zone_pos":
            out = (dif > 0) & (sg > 0)
        elif cid == "zone_neg":
            out = (dif < 0) & (sg < 0)
        elif cid in ("close_above_ma", "close_below_ma"):
            m = ind.ma(_param(c, "n"))
            out = cl > m if cid == "close_above_ma" else cl < m
        elif cid in ("ma_rising", "ma_falling"):
            m = ind.ma(_param(c, "n"))
            out = m > _prev(m) if cid == "ma_rising" else m < _prev(m)
        else:
            raise KeyError(f"unknown condition {cid!r}")
    return np.asarray(out, dtype=bool)


def _recent(a, w):
    """True where `a` was true at least once in the last w bars (this one included)."""
    if w <= 1:
        return a
    idx = np.where(a, np.arange(len(a)), -1)
    last = np.maximum.accumulate(idx)
    return (last >= 0) & (np.arange(len(a)) - last < w)


def side_signal(ind: Indicators, side, n) -> np.ndarray:
    conds = (side or {}).get("conds") or []
    if not conds:
        return np.zeros(n, dtype=bool)
    arrs = [condition(ind, c) for c in conds]
    if side.get("mode", "all") == "any":
        return np.logical_or.reduce(arrs)
    w = max(1, int(side.get("within", 1) or 1))
    return np.logical_and.reduce([_recent(a, w) for a in arrs])


def positions_one(df, rule, ind: Indicators | None = None) -> pd.Series:
    ind = ind or Indicators(df, int(rule.get("kdj_n", 9)), tuple(rule.get("macd", (12, 26, 9))))
    n = len(df)
    buy, sell = side_signal(ind, rule.get("buy"), n), side_signal(ind, rule.get("sell"), n)
    out = state_machine(df["open"].to_numpy(dtype=float), ind.close, buy, sell, rule)
    return pd.Series(out, index=df.index)


def state_machine(o, c, buy, sell, rule) -> np.ndarray:
    """0/1 targets from buy / sell signal arrays plus the rule's exit protection (see the module docstring)."""
    n = len(c)
    stop, take, trail = (float(rule.get(x) or 0) / 100 for x in ("stop", "take", "trail"))
    hold = int(rule.get("hold") or 0)
    out = np.zeros(n)
    on, sig, entry, peak = False, -1, math.nan, -math.inf
    for t in range(n):
        if not on:
            if buy[t]:
                on, sig, entry, peak = True, t, math.nan, -math.inf
        else:
            if t == sig + 1:
                entry = o[t]
            peak = max(peak, c[t])
            ex = bool(sell[t])
            if stop and c[t] < entry * (1 - stop):
                ex = True
            if take and c[t] > entry * (1 + take):
                ex = True
            if trail and c[t] < peak * (1 - trail):
                ex = True
            if hold and t - sig >= hold:
                ex = True
            if ex:
                on = False
        out[t] = 1.0 if on else 0.0
    return out


def positions(data, rule):
    return {code: positions_one(df, rule) for code, df in data.items()}


def _fmt(c):
    spec = catalog()[c["id"]]
    text = spec["label"]
    for p, ps in spec.get("params", {}).items():
        v = c.get(p, ps["default"])
        text = text.replace("{" + p + "}", f"{v:g}")
    return text


def describe(rule) -> str:
    """One-paragraph Traditional Chinese description of a rule."""
    def side(s, what):
        conds = (s or {}).get("conds") or []
        if not conds:
            return f"{what}：無（只靠出場保護）" if what == "賣出" else f"{what}：無"
        if s.get("mode", "all") == "any":
            return f"{what}：任一符合 — " + "、".join(_fmt(c) for c in conds)
        w = int(s.get("within", 1) or 1)
        when = "同一根K棒全部符合" if w == 1 else f"最近 {w} 根K棒內都出現過"
        return f"{what}：{when} — " + "、".join(_fmt(c) for c in conds)

    risk = [f"{label} {rule[k]:g}%" for k, label in (("stop", "收盤停損"), ("take", "收盤停利"), ("trail", "移動停損"))
            if rule.get(k)]
    if rule.get("hold"):
        risk.append(f"最多持有 {int(rule['hold'])} 天")
    params = f"KDJ({rule.get('kdj_n', 9)},3,3)、MACD({','.join(str(x) for x in rule.get('macd', (12, 26, 9)))})"
    return "；".join([side(rule.get("buy"), "買進"), side(rule.get("sell"), "賣出"), "出場保護：" + ("、".join(risk) or "無"),
                     params]) + "。收盤判斷，下一個交易日開盤成交，只做多。"
