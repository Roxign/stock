"""Systematic search over KDJ + MACD rule combinations with the rule-lab engine, selected in-sample only.

  python -m strategies.kdj_macd_lab.search

Grid (6,930 rules):
  buy   one KDJ trigger x one MACD context (or none) x window 1 / 3 bars x optional "60-day MA rising" filter
  sell  one KDJ exit x one MACD context (or none), or no sell signal at all
  exit protection   none / 10% closing stop / 15% trailing stop
Ranking: median over the 50 stocks of (rule Sharpe - buy-and-hold Sharpe) in 2010-2020, among rules with at least
five completed trades per stock (median). The top rules are then run with the real engine out of sample (2021-),
on the 55 extra stocks and in the 8 blocked folds. Every rule is logged to research/trials.jsonl (agent
kdj_macd_lab) for the multiple-testing deflation. Results: RESEARCH.md in this folder.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from stocklab import backtest as bt
from stocklab import rules

BUY_KDJ = [{"id": "kd_golden"}, {"id": "k_up"}, {"id": "j_up"}, {"id": "j_below", "v": 0}, {"id": "k_below", "v": 20}]
BUY_MACD = [None, {"id": "zone_neg"}, {"id": "zone_pos"}, {"id": "osc_up"}, {"id": "macd_golden"}, {"id": "osc_rising"}]
WINDOWS = [1, 3]
FILTERS = [None, {"id": "ma_rising", "n": 60}]
SELL_KDJ = [{"id": "kd_dead"}, {"id": "k_down"}, {"id": "j_down"}, {"id": "j_above", "v": 100}, {"id": "k_above", "v": 80}]
SELL_MACD = [None, {"id": "zone_pos"}, {"id": "osc_down"}, {"id": "macd_dead"}]
RISK = [{}, {"stop": 10}, {"trail": 15}]
TOP = 20
MIN_TRADES = 5


def buy_sides():
    out = []
    for t, m, f in itertools.product(BUY_KDJ, BUY_MACD, FILTERS):
        for w in (WINDOWS if m else [1]):
            conds = [t] + ([m] if m else []) + ([f] if f else [])
            out.append({"mode": "all", "within": w, "conds": conds})
    return out


def sell_sides():
    out = [{"mode": "all", "within": 1, "conds": []}]
    for x, m in itertools.product(SELL_KDJ, SELL_MACD):
        out.append({"mode": "all", "within": 1, "conds": [x] + ([m] if m else [])})
    return out


def grid():
    return [{"buy": b, "sell": s, **r} for b, s, r in itertools.product(buy_sides(), sell_sides(), RISK)]


def fast_metrics(o, c, tgt, i0, i1, years, fees):
    """Exact engine metrics for a 0/1 target (fractional shares, fills at the open): Sharpe, CAGR, MDD, exposure,
    completed trades."""
    h = tgt[i0 - 1:i1]                     # held during bars i0..i1
    hp = np.r_[0.0, h[:-1]]                # held before each open (the engine starts flat at i0)
    oi, ci, cp = o[i0:i1 + 1], c[i0:i1 + 1], c[i0 - 1:i1]
    lr = hp * np.log(oi / cp) + h * np.log(ci / oi)
    lr = lr - np.where(h > hp, math.log1p(fees["buy"]), 0.0) + np.where(h < hp, math.log1p(-fees["sell"]), 0.0)
    r = np.expm1(lr)
    sd = r.std(ddof=1)
    eq = np.exp(np.cumsum(lr))
    mdd = float(np.min(eq / np.maximum.accumulate(np.maximum(eq, 1.0)) - 1))
    return {"sharpe": float(r.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0,
            "cagr": float(eq[-1] ** (1 / years) - 1), "mdd": mdd, "exposure": float(h.mean()),
            "trades": int(((h == 0) & (hp == 1)).sum())}


_W = {}


def _init(codes, period):
    from stocklab.data import load_all
    from stocklab.runner import fees_of

    data = load_all(codes)
    for code, df in data.items():
        b = bt.period_bounds(df.index, period)
        if b is None:  # listed too late for this period (e.g. 7769 in 2010-2020)
            continue
        ind = rules.Indicators(df)
        years = (df.index[b[1]] - df.index[b[0]]).days / 365.25
        o = df["open"].to_numpy(dtype=float)
        _W[code] = {"ind": ind, "o": o, "b": b, "years": years, "fees": fees_of(code), "cache": {},
                    "bh": fast_metrics(o, ind.close, np.ones(len(df)), b[0], b[1], years, fees_of(code))}


def _signal(w, side, n):
    conds = side["conds"]
    if not conds:
        return np.zeros(n, dtype=bool)
    arrs = []
    for c in conds:
        key = json.dumps(c, sort_keys=True)
        if key not in w["cache"]:
            w["cache"][key] = rules.condition(w["ind"], c)
        arrs.append(rules._recent(w["cache"][key], max(1, side.get("within", 1))))
    return np.logical_and.reduce(arrs)


def _evaluate(batch):
    out = []
    for k, rule in batch:
        rows = []
        for w in _W.values():
            n = len(w["o"])
            tgt = rules.state_machine(w["o"], w["ind"].close, _signal(w, rule["buy"], n), _signal(w, rule["sell"], n), rule)
            m = fast_metrics(w["o"], w["ind"].close, tgt, *w["b"], w["years"], w["fees"])
            rows.append((m["sharpe"], m["sharpe"] - w["bh"]["sharpe"], m["cagr"], m["cagr"] - w["bh"]["cagr"], m["mdd"],
                         m["exposure"], m["trades"]))
        a = np.array(rows)
        med = np.median(a, axis=0)
        out.append({"k": k, "sharpe": med[0], "d_sharpe": med[1], "cagr": med[2], "d_cagr": med[3], "mdd": med[4],
                    "exposure": med[5], "trades": med[6], "beat_sharpe": float((a[:, 1] > 0).mean())})
    return out


def scan(rule_list, codes, period="is", workers=8):
    batches = [list(enumerate(rule_list))[i::workers * 8] for i in range(workers * 8)]
    rows = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(codes, period)) as ex:
        for part in ex.map(_evaluate, batches):
            rows += part
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def validate(rule, data, codes, periods):
    """Real-engine medians per period for one rule on the given codes."""
    from stocklab.runner import fees_of

    pos = rules.positions({c: data[c] for c in codes}, rule)
    out = {}
    for p in periods:
        ms, bhs = [], []
        for c in codes:
            r = bt.run_strategy(data[c], pos[c], p, fees_of(c))
            b = bt.run_buy_hold(data[c], p, fees_of(c))
            if r and b:
                ms.append(r["metrics"])
                bhs.append(b["metrics"])
        if ms:
            out[p] = {k: float(np.median([m[k] for m in ms])) for k in ("cagr", "mdd", "sharpe")}
            out[p]["bh_sharpe"] = float(np.median([m["sharpe"] for m in bhs]))
            out[p]["bh_cagr"] = float(np.median([m["cagr"] for m in bhs]))
            out[p]["beat_sharpe"] = float(np.mean([m["sharpe"] > b["sharpe"] for m, b in zip(ms, bhs)]))
    return out


def main():
    from stocklab.data import load_all
    from stocklab.trials import deflated_sharpe, log_trial
    from stocklab.universe import CODES, EXTRA_CODES

    t0 = time.time()
    rule_list = grid()
    print(f"{len(rule_list)} rules x {len(CODES)} stocks, in-sample 2010-2020", flush=True)
    res = scan(rule_list, CODES)
    print(f"scanned in {time.time() - t0:.0f}s", flush=True)
    res["rule"] = [rule_list[k] for k in res["k"]]
    for _, r in res.iterrows():
        log_trial("kdj_macd_lab", f"grid:{r['k']}", r["rule"],
                  {"sharpe": r["sharpe"], "cagr": r["cagr"], "mdd": r["mdd"], "exposure": r["exposure"]}, "is")
    ok = res[res["trades"] >= MIN_TRADES].sort_values("d_sharpe", ascending=False)
    print(f"{len(ok)} rules with >= {MIN_TRADES} trades per stock; median d_sharpe quantiles:",
          ok["d_sharpe"].quantile([0.5, 0.9, 0.99, 1.0]).round(3).to_dict(), flush=True)

    best = ok.iloc[0]
    daily = res["sharpe"].to_numpy() / math.sqrt(252)
    dsr = deflated_sharpe(best["sharpe"] / math.sqrt(252), 2700, daily)
    data = load_all(CODES + EXTRA_CODES)
    top = []
    for _, r in ok.head(TOP).iterrows():
        v50 = validate(r["rule"], data, CODES, ["is", "oos"] + list(bt.FOLDS))
        vx = validate(r["rule"], data, EXTRA_CODES, ["is", "oos"])
        folds_won = sum(v50[f]["sharpe"] > v50[f]["bh_sharpe"] for f in bt.FOLDS if f in v50)
        top.append({"k": int(r["k"]), "rule": r["rule"], "text": rules.describe(r["rule"]), "is50": v50["is"], "oos50": v50["oos"],
                    "is55": vx.get("is"), "oos55": vx.get("oos"), "folds_won": int(folds_won), "trades": float(r["trades"]),
                    "exposure": float(r["exposure"])})
        print(f"#{r['k']}: IS dS {r['d_sharpe']:+.3f} | OOS50 S {v50['oos']['sharpe']:.2f} vs {v50['oos']['bh_sharpe']:.2f} | "
              f"55 IS {vx['is']['sharpe']:.2f} vs {vx['is']['bh_sharpe']:.2f}, OOS {vx['oos']['sharpe']:.2f} vs {vx['oos']['bh_sharpe']:.2f} | "
              f"folds {folds_won}/8 | {rules.describe(r['rule'])[:120]}", flush=True)
    out = {"n_rules": len(rule_list), "n_eligible": int(len(ok)), "dsr_best": dsr,
           "quantiles_d_sharpe": ok["d_sharpe"].quantile([0.5, 0.9, 0.99, 1.0]).round(4).tolist(), "top": top}
    from stocklab.data import ROOT
    path = ROOT / "data" / "lab_search.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    res.drop(columns=["rule"]).to_csv(ROOT / "data" / "lab_search_all.csv", index=False)
    print(f"deflated Sharpe probability of the best rule: {dsr:.3f}; wrote {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
