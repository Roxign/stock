"""Divergence (背離) and KD saturation (鈍化) study with the rule-lab engine.

  python -m strategies.kdj_macd_lab.divergence_study

Part A, event study on the 50 stocks and the 55 extra stocks, 2010-2020 and 2021-: what happened after each event
(first day of a divergence, K crossing 80 / 20, saturation confirmed after 3 days, saturation ending). Move from the
next open to the open 5 / 20 / 60 sessions later, minus that stock's average move over the same period.
Part B, the same buy or sell rule with different exits / entries (the user's rebound rule as the base), run with the
real engine: Sharpe, CAGR, drawdown against buy-and-hold, 8 blocked folds. Every variant is logged to
research/trials.jsonl. Results: research/divergence_saturation.md.
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import pandas as pd

from stocklab import backtest as bt
from stocklab import rules

HORIZONS = (5, 20, 60)
PERIODS = {"is": ("2010-01-01", "2020-12-31"), "oos": ("2021-01-01", None)}

EVENTS = {
    "KD 頂背離（股價創新高、K 沒過前高）": {"id": "kd_top_div"},
    "K 回到前高、股價沒過前高（買盤不足）": {"id": "kd_top_div_hidden"},
    "MACD 頂背離（股價創新高、DIF 沒過前高）": {"id": "dif_top_div"},
    "DIF 回到前高、股價沒過前高": {"id": "dif_top_div_hidden"},
    "KD 底背離（股價創新低、K 沒破前低）": {"id": "kd_bot_div"},
    "K 回到前低、股價沒破前低（賣壓減弱）": {"id": "kd_bot_div_hidden"},
    "MACD 底背離（股價創新低、DIF 沒破前低）": {"id": "dif_bot_div"},
    "DIF 回到前低、股價沒破前低": {"id": "dif_bot_div_hidden"},
    "K 剛站上 80（進入高檔）": {"id": "k_above", "v": 80},
    "高檔鈍化確認（K 連續 3 天 > 80）": {"id": "k_high_stuck", "v": 80, "n": 3},
    "高檔鈍化結束（連續 3 天 > 80 後跌破）": {"id": "high_stuck_end", "v": 80, "n": 3},
    "K 剛跌破 20（進入低檔）": {"id": "k_below", "v": 20},
    "低檔鈍化確認（K 連續 3 天 < 20）": {"id": "k_low_stuck", "v": 20, "n": 3},
    "低檔鈍化結束（連續 3 天 < 20 後站上）": {"id": "low_stuck_end", "v": 20, "n": 3},
}

BUY = {"mode": "all", "within": 1, "conds": [{"id": "k_up"}, {"id": "osc_up"}, {"id": "zone_neg"}]}
SELL = {"mode": "all", "within": 1, "conds": [{"id": "k_down"}, {"id": "osc_down"}, {"id": "zone_pos"}]}
ONE = lambda *c: {"mode": "all", "within": 1, "conds": list(c)}
OR = lambda base, *alts: {**base, "or": list(alts)}
EXITS = {
    "原規則：K、柱狀體同步轉弱且零軸上": SELL,
    "原規則＋「K 回到前高、股價沒過前高」": OR(SELL, ONE({"id": "kd_top_div_hidden"})),
    "原規則＋KD 頂背離": OR(SELL, ONE({"id": "kd_top_div"})),
    "原規則＋MACD 頂背離": OR(SELL, ONE({"id": "dif_top_div"})),
    "只用「K 回到前高、股價沒過前高」": ONE({"id": "kd_top_div_hidden"}),
    "只用 KD 頂背離": ONE({"id": "kd_top_div"}),
    "只用 MACD 頂背離": ONE({"id": "dif_top_div"}),
    "K > 80 就賣": ONE({"id": "k_above", "v": 80}),
    "K > 80 且 KD 死亡交叉": ONE({"id": "kd_dead"}, {"id": "k_above", "v": 80}),
    "高檔鈍化結束才賣（連續 3 天 > 80 後跌破）": ONE({"id": "high_stuck_end", "v": 80, "n": 3}),
    "高檔鈍化結束＋DIF 轉弱": ONE({"id": "high_stuck_end", "v": 80, "n": 3}, {"id": "dif_down"}),
    "不設賣出訊號（只靠停損）": {"mode": "all", "conds": []},
}
ENTRIES = {
    "原規則：K、柱狀體同步翻揚且零軸下": BUY,
    "原規則＋KD 底背離": OR(BUY, ONE({"id": "kd_bot_div"})),
    "原規則＋MACD 底背離": OR(BUY, ONE({"id": "dif_bot_div"})),
    "只用 KD 底背離": ONE({"id": "kd_bot_div"}),
    "只用「K 回到前低、股價沒破前低」": ONE({"id": "kd_bot_div_hidden"}),
    "只用 MACD 底背離": ONE({"id": "dif_bot_div"}),
    "K < 20 就買": ONE({"id": "k_below", "v": 20}),
    "K < 20 且 KD 黃金交叉": ONE({"id": "kd_golden"}, {"id": "k_below", "v": 20}),
    "低檔鈍化結束才買（連續 3 天 < 20 後站上）": ONE({"id": "low_stuck_end", "v": 20, "n": 3}),
    "低檔鈍化結束＋DIF 翻揚": {"mode": "all", "within": 3, "conds": [{"id": "low_stuck_end", "v": 20, "n": 3}, {"id": "dif_up"}]},
}


def _period(index, p):
    a, b = PERIODS[p]
    m = index >= pd.Timestamp(a)
    return m & (index <= pd.Timestamp(b)) if b else m


def event_study(data, groups):
    """rows[(event, group, period, h)] = list of excess moves; base up-rates per (group, period, h)."""
    out, base_up = {}, {}
    for code, df in data.items():
        g = groups[code]
        ind = rules.Indicators(df)
        o = df["open"].to_numpy(dtype=float)
        fwd = {}
        for h in HORIZONS:
            f = np.full(len(o), np.nan)
            f[:-(h + 1)] = o[h + 1:] / o[1:-h] - 1
            fwd[h] = f
        for p in PERIODS:
            m = _period(df.index, p) & np.isfinite(fwd[max(HORIZONS)])
            if m.sum() < 120:
                continue
            for h in HORIZONS:
                v = fwd[h][m]
                base_up.setdefault((g, p, h), []).append(((v > 0).sum(), len(v)))
            for name, c in EVENTS.items():
                a = rules.condition(ind, c)
                start = a & ~np.r_[False, a[:-1]]
                for h in HORIZONS:
                    base = np.nanmean(fwd[h][m])
                    sel = start & m
                    out.setdefault((name, g, p, h), []).extend(fwd[h][sel] - base)
                    out.setdefault((name, g, p, h, "up"), []).extend(fwd[h][sel] > 0)
    return out, {k: sum(x for x, _ in v) / sum(n for _, n in v) for k, v in base_up.items()}


def _variants():
    for name, sell in EXITS.items():
        yield "exit", name, {"buy": BUY, "sell": sell, "stop": 10}
    for name, buy in ENTRIES.items():
        yield "entry", name, {"buy": buy, "sell": SELL, "stop": 10}


def strategy_tests(data, groups):
    from stocklab.runner import fees_of
    from strategies.kdj_macd_lab.search import fast_metrics

    inds = {c: rules.Indicators(df) for c, df in data.items()}
    periods = ["is", "oos"] + list(bt.FOLDS)
    bh = {}
    for c, df in data.items():
        for p in periods:
            b = bt.period_bounds(df.index, p)
            if b:
                yrs = (df.index[b[1]] - df.index[b[0]]).days / 365.25
                bh[(c, p)] = (b, yrs, fast_metrics(df["open"].to_numpy(float), inds[c].close, np.ones(len(df)), b[0], b[1], yrs, fees_of(c)))
    rows = []
    for kind, name, rule in _variants():
        res = {}
        quality = {"buy": [], "sell": []}
        for c, df in data.items():
            o = df["open"].to_numpy(float)
            tgt = rules.positions_one(df, rule, inds[c]).to_numpy()
            d = np.r_[0.0, np.diff(np.r_[0.0, tgt[:-1]])]
            f20 = np.full(len(o), np.nan)
            f20[:-20] = o[20:] / o[:-20] - 1
            for p in periods:
                if (c, p) not in bh:
                    continue
                b, yrs, mb = bh[(c, p)]
                m = fast_metrics(o, inds[c].close, tgt, b[0], b[1], yrs, fees_of(c))
                res.setdefault((groups[c], p), []).append((m["sharpe"], mb["sharpe"], m["cagr"], mb["cagr"], m["mdd"], mb["mdd"], m["exposure"], m["trades"]))
                if p == "oos" and groups[c] == "50":
                    i0, i1 = b
                    base = np.nanmean(f20[i0:i1 - 19])
                    idx = np.arange(i0 + 1, i1 - 19)
                    quality["buy"] += list(f20[idx[d[idx] > 0.5]] - base)
                    quality["sell"] += list(f20[idx[d[idx] < -0.5]] - base)
        row = {"kind": kind, "name": name, "rule": rule}
        for (g, p), v in res.items():
            a = np.array(v)
            row[f"{g}_{p}"] = {"sharpe": float(np.median(a[:, 0])), "bh_sharpe": float(np.median(a[:, 1])),
                               "cagr": float(np.median(a[:, 2])), "bh_cagr": float(np.median(a[:, 3])),
                               "mdd": float(np.median(a[:, 4])), "bh_mdd": float(np.median(a[:, 5])),
                               "exposure": float(np.median(a[:, 6])), "trades": float(np.median(a[:, 7])),
                               "beat": float(np.mean(a[:, 0] > a[:, 1]))}
        row["folds_won"] = int(sum(row[f"50_{f}"]["sharpe"] > row[f"50_{f}"]["bh_sharpe"] for f in bt.FOLDS if f"50_{f}" in row))
        row["oos_buy20"] = float(np.mean(quality["buy"])) if quality["buy"] else None
        row["oos_sell20"] = float(np.mean(quality["sell"])) if quality["sell"] else None
        rows.append(row)
    return rows


def main():
    from stocklab.data import ALL_CODES, ROOT, load_all
    from stocklab.trials import log_trial
    from stocklab.universe import CODES

    t0 = time.time()
    data = load_all(ALL_CODES)
    groups = {c: "50" if c in CODES else "55" for c in data}
    ev, base_up = event_study(data, groups)
    events = []
    for name in EVENTS:
        for g in ("50", "55"):
            for p in PERIODS:
                r = {"event": name, "group": g, "period": p}
                for h in HORIZONS:
                    v = np.array(ev.get((name, g, p, h), []), dtype=float)
                    u = np.array(ev.get((name, g, p, h, "up"), []), dtype=float)
                    r[f"n"] = int(len(v))
                    r[f"ex{h}"] = float(np.nanmean(v)) if len(v) else None
                    r[f"up{h}"] = float(np.mean(u)) if len(u) else None
                    r[f"base_up{h}"] = base_up.get((g, p, h))
                events.append(r)
    print(f"events done in {time.time() - t0:.0f}s", flush=True)
    tests = strategy_tests(data, groups)
    for r in tests:
        log_trial("kdj_macd_lab", f"div_sat:{r['kind']}:{r['name']}", r["rule"],
                  {k: r["50_is"][k] for k in ("sharpe", "cagr", "mdd", "exposure")}, "is")
    out = {"events": events, "tests": tests}
    path = ROOT / "data" / "divergence_study.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    pct = lambda x: "—" if x is None or not math.isfinite(x) else f"{x * 100:+.1f}%"
    for r in events:
        if r["group"] == "50":
            print(f"{r['event']:<36} {r['period']:>3} n={r['n']:>5} ex5 {pct(r['ex5'])} ex20 {pct(r['ex20'])} ex60 {pct(r['ex60'])} up20 {r['up20'] or 0:.0%} (base {r['base_up20'] or 0:.0%})")
    for r in tests:
        a, b, x = r["50_is"], r["50_oos"], r.get("55_oos", {})
        print(f"{r['kind']:<5} {r['name']:<34} IS S {a['sharpe']:.2f}/{a['bh_sharpe']:.2f} CAGR {a['cagr']:.1%} MDD {a['mdd']:.0%} | "
              f"OOS S {b['sharpe']:.2f}/{b['bh_sharpe']:.2f} CAGR {b['cagr']:.1%} MDD {b['mdd']:.0%} exp {b['exposure']:.0%} | "
              f"55 OOS S {x.get('sharpe', float('nan')):.2f}/{x.get('bh_sharpe', float('nan')):.2f} | folds {r['folds_won']}/8 | "
              f"buy20 {pct(r['oos_buy20'])} sell20 {pct(r['oos_sell20'])}")
    print(f"wrote {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
