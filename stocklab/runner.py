import importlib
import math
import time

import numpy as np
import pandas as pd

from . import backtest as bt
from .data import ROOT

POS_CACHE = ROOT / "data" / "positions"
FAMILY_ORDER = ["kdj_macd_rule", "kdj_macd_dl", "trend", "mean_reversion"]
BASELINES = [
    {"id": "buy_hold", "label": "買進持有", "family": "基準", "description": "期初一次全部買進，持有到期末，不做任何操作。", "multicharts": None},
    {"id": "dca", "label": "定期定額", "family": "基準", "description": "把本金平均分成每月一份，於每月第一個交易日開盤買進，持有到期末。另計算資金加權報酬率 (XIRR)。", "multicharts": None},
]


def discover(families=None):
    """Every strategies/<family>/strategy.py exposes a STRATEGIES list; folders starting with _ load only when named."""
    out = []
    if families is None:
        found = [f.parent.name for f in (ROOT / "strategies").glob("*/strategy.py") if not f.parent.name.startswith("_")]
        families = sorted(found, key=lambda f: (FAMILY_ORDER.index(f) if f in FAMILY_ORDER else len(FAMILY_ORDER), f))
    for family in families:
        f = ROOT / "strategies" / family / "strategy.py"
        mod = importlib.import_module(f"strategies.{family}.strategy")
        for s in mod.STRATEGIES:
            out.append({**s, "folder": f.parent, "family_dir": family})
    ids = [s["id"] for s in out]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate strategy ids: {dup}")
    return out


def compute_positions(strategy, data, use_cache=False):
    path = POS_CACHE / f"{strategy['id']}.pkl"
    if use_cache and path.exists():
        return pd.read_pickle(path)
    t = time.time()
    pos = strategy["positions"](data)
    validate(strategy["id"], pos, data)
    POS_CACHE.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(pos, path)
    print(f"  {strategy['id']}: positions in {time.time() - t:.1f}s")
    return pos


def validate(sid, pos, data):
    missing = set(data) - set(pos)
    if missing:
        raise ValueError(f"{sid}: no positions for {sorted(missing)}")
    for code, s in pos.items():
        if not isinstance(s, pd.Series):
            raise TypeError(f"{sid}/{code}: positions must be a pandas Series")
        s = s.reindex(data[code].index)
        if ((s < -1e-9) | (s > 1 + 1e-9)).any():
            raise ValueError(f"{sid}/{code}: positions must be within [0, 1]")


def check_lookahead(strategy, data, cuts=("2014-06-30", "2018-03-30", "2022-09-30"), tol=0.01):
    """Recompute positions (and portfolio weights, if provided) on data truncated at each cut date;
    any change before the cut means future data leaked in."""
    ok = True
    for key in ("positions", "weights"):
        fn = strategy.get(key)
        if fn is None:
            continue
        full = fn(data)
        for cut in cuts:
            cut = pd.Timestamp(cut)
            part = fn({c: df.loc[:cut] for c, df in data.items() if df.index[0] < cut})
            diffs, total = 0, 0
            for code, p in part.items():
                a = full[code].reindex(data[code].index).loc[:cut].fillna(0)
                b = p.reindex(a.index).fillna(0)
                diffs += int((a - b).abs().gt(1e-6).sum())
                total += len(a)
            frac = diffs / max(total, 1)
            status = "OK" if frac <= tol else "LEAK?"
            ok &= frac <= tol
            print(f"  lookahead {strategy['id']}/{key} cut={cut.date()}: {diffs}/{total} bars differ ({frac:.3%}) {status}")
    return ok


def evaluate(strategies, data, periods=("is", "oos", "full"), use_cache=False):
    """Returns results[sid][period][code] = run result dict (or None)."""
    results = {}
    for b in BASELINES:
        runner = bt.run_buy_hold if b["id"] == "buy_hold" else bt.run_dca
        results[b["id"]] = {p: {c: runner(df, p) for c, df in data.items()} for p in periods}
    for s in strategies:
        pos = compute_positions(s, data, use_cache)
        results[s["id"]] = {p: {c: bt.run_strategy(df, pos[c], p) for c, df in data.items()} for p in periods}
    return results


def leaderboard(results, labels, periods=("is", "oos", "full")):
    rows = []
    for p in periods:
        bh = results["buy_hold"][p]
        dca = results["dca"][p]
        for sid, by_p in results.items():
            res = by_p[p]
            codes = [c for c, r in res.items() if r]
            m = {c: res[c]["metrics"] for c in codes}
            beat_bh = np.mean([m[c]["cagr"] > bh[c]["metrics"]["cagr"] for c in codes])
            beat_bh_sharpe = np.mean([m[c]["sharpe"] > bh[c]["metrics"]["sharpe"] for c in codes])
            beat_dca = np.mean([m[c]["cagr"] > dca[c]["metrics"]["cagr"] for c in codes])
            def med(k):
                v = [x for x in (m[c].get(k, math.nan) for c in codes) if not math.isnan(x)]
                return float(np.median(v)) if v else math.nan

            rows.append({
                "period": p, "id": sid, "label": labels.get(sid, sid), "n": len(codes),
                "cagr_med": med("cagr"), "cagr_mean": float(np.mean([m[c]["cagr"] for c in codes])),
                "mdd_med": med("mdd"), "sharpe_med": med("sharpe"), "exposure_med": med("exposure"),
                "orders_med": med("orders"), "trades_med": med("trades"), "win_med": med("win"),
                "beat_bh": float(beat_bh), "beat_bh_sharpe": float(beat_bh_sharpe), "beat_dca": float(beat_dca),
            })
    return pd.DataFrame(rows)


def print_leaderboard(df):
    pct = lambda x: "" if pd.isna(x) else f"{x:6.1%}"
    num = lambda x: "" if pd.isna(x) else f"{x:5.2f}"
    for p, g in df.groupby("period", sort=False):
        print(f"\n=== {bt.PERIOD_LABELS[p]} ===")
        print(f"{'策略':<28}{'n':>3} {'CAGR中位':>8} {'CAGR平均':>8} {'MDD中位':>8} {'Sharpe':>6} {'持股比':>7} {'調整數':>6} {'交易數':>6} {'勝率':>7} {'勝B&H%':>7} {'Sharpe勝B&H%':>8} {'勝定額%':>7}")
        for _, r in g.sort_values("cagr_med", ascending=False).iterrows():
            print(f"{r['id']:<28}{r['n']:>3} {pct(r['cagr_med']):>8} {pct(r['cagr_mean']):>8} {pct(r['mdd_med']):>8} {num(r['sharpe_med']):>6} "
                  f"{pct(r['exposure_med']):>7} {num(r['orders_med']):>6} {num(r['trades_med']):>6} {pct(r['win_med']):>7} {pct(r['beat_bh']):>7} {pct(r['beat_bh_sharpe']):>8} {pct(r['beat_dca']):>7}")


def portfolio_results(strategies, data, periods, use_cache=True):
    """results[period][id] for the 0050 / equal-weight benchmarks and every strategy run as one 50-stock account.
    A strategy may provide 'weights' (cross-sectional target weights of total equity); otherwise its per-stock
    positions get equal capital slots."""
    from . import portfolio as pf

    out = {}
    for p in periods:
        out[p] = pf.benchmarks(data, p)
        for s in strategies:
            if s.get("weights"):
                w = s["weights"](data)
            else:
                w = pf.per_stock_weights(compute_positions(s, data, use_cache), data)
            out[p][s["id"]] = pf.run(data, w, p)
    return out


def print_portfolio(results):
    from .portfolio import BENCH_LABELS

    pct = lambda x: "" if x is None or pd.isna(x) else f"{x:6.1%}"
    for p, rows in results.items():
        print(f"\n=== 投資組合（50 檔同一帳戶，可持現金）{bt.PERIOD_LABELS[p]} ===")
        print(f"{'策略':<28} {'CAGR':>7} {'MDD':>7} {'Sharpe':>6} {'平均持股':>7} {'年換手':>7}")
        live = [(k, r) for k, r in rows.items() if r]
        for k, r in sorted(live, key=lambda kv: -kv[1]["metrics"]["cagr"]):
            m = r["metrics"]
            name = BENCH_LABELS.get(k, k)
            print(f"{name:<28} {pct(m['cagr']):>7} {pct(m['mdd']):>7} {m['sharpe']:6.2f} {pct(m.get('exposure')):>7} {pct(m.get('turnover')):>7}")
