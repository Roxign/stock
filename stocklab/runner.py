import importlib
import math
import time

import numpy as np
import pandas as pd

from . import backtest as bt
from .data import ROOT

POS_CACHE = ROOT / "data" / "positions"
FAMILY_ORDER = ["kdj_macd_rule", "kdj_macd_rebound", "kdj_macd_lab", "kdj_macd_dl", "dl_position", "dl_revenue_flow", "dl_market_state",
                "dl_cross_stock", "dl_trend_labels", "trend", "mean_reversion"]
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


# Families not run on the extra stocks: they need per-stock FinMind chip / revenue data, downloaded for the 50 only.
EXTRA_SKIP_FAMILIES = {"dl_revenue_flow", "dl_cross_stock"}


def universe_data(strategy, data, universe=None):
    """The securities a strategy trades: 'stocks' (default, the 50), 'etfs', 'all', or with universe='extra' the
    representative stocks outside the 50 (an out-of-sample check of the stock strategies)."""
    from .etf import CODES as ETF_CODES
    from .universe import CODES as STOCK_CODES, EXTRA_CODES

    u = universe or strategy.get("universe", "stocks")
    want = {"stocks": set(STOCK_CODES), "etfs": set(ETF_CODES), "all": set(STOCK_CODES) | set(ETF_CODES),
            "extra": set(EXTRA_CODES)}[u]
    return {c: df for c, df in data.items() if c in want}


def runs_on_extra(strategy, data=None):
    """Stock strategies also run on the extra stocks (when they are loaded), except EXTRA_SKIP_FAMILIES."""
    from .universe import EXTRA_CODES

    loaded = data is None or any(c in data for c in EXTRA_CODES)
    return loaded and strategy.get("universe", "stocks") == "stocks" and strategy.get("family_dir") not in EXTRA_SKIP_FAMILIES


def universes_of(strategy, data):
    return [None] + (["extra"] if runs_on_extra(strategy, data) else [])


def pos_path(sid, universe=None):
    return POS_CACHE / (f"{sid}.pkl" if universe is None else f"{sid}@{universe}.pkl")


def fees_of(code):
    from .shortrules import fees

    return fees(code)


def compute_positions(strategy, data, use_cache=False, universe=None):
    """Strategy targets after Taiwan short-selling rules (bt.enforce_short_rules), cached per strategy (and per
    universe: data/positions/<id>@extra.pkl for the extra stocks)."""
    path = pos_path(strategy["id"], universe)
    if use_cache and path.exists():
        return pd.read_pickle(path)
    t = time.time()
    udata = universe_data(strategy, data, universe)
    pos = strategy["positions"](udata)
    validate(strategy["id"], pos, udata)
    pos = bt.effective_positions(pos, udata)
    POS_CACHE.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(pos, path)
    print(f"  {strategy['id']}{'@' + universe if universe else ''}: positions in {time.time() - t:.1f}s", flush=True)
    return pos


def refresh_positions(strategies, data, families=None):
    """Recompute and cache every strategy's positions (main universe and extra stocks). A failing strategy keeps its
    previous cache and is reported, so one broken model doesn't stop the daily update."""
    failed = []
    for s in strategies:
        if families and s["family_dir"] not in families:
            continue
        for u in universes_of(s, data):
            try:
                compute_positions(s, data, use_cache=False, universe=u)
            except Exception as e:  # noqa: BLE001
                failed.append(f"{s['id']}{'@' + u if u else ''}")
                print(f"  {s['id']}: FAILED {type(e).__name__}: {e}", flush=True)
    if failed:
        print(f"positions not refreshed: {failed}", flush=True)
    return failed


def validate(sid, pos, data):
    missing = set(data) - set(pos)
    if missing:
        raise ValueError(f"{sid}: no positions for {sorted(missing)}")
    for code, s in pos.items():
        if not isinstance(s, pd.Series):
            raise TypeError(f"{sid}/{code}: positions must be a pandas Series")
        s = s.reindex(data[code].index)
        if ((s < -1 - 1e-9) | (s > 1 + 1e-9)).any():
            raise ValueError(f"{sid}/{code}: positions must be within [-1, 1]")


def check_lookahead(strategy, data, cuts=("2014-06-30", "2018-03-30", "2022-09-30"), tol=0.01):
    """Recompute positions (and portfolio weights, if provided) on data truncated at each cut date;
    any change before the cut means future data leaked in."""
    ok = True
    data = universe_data(strategy, data)
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
        results[b["id"]] = {p: {c: runner(df, p, fees_of(c)) for c, df in data.items()} for p in periods}
    for s in strategies:
        results[s["id"]] = {p: {} for p in periods}
        for u in universes_of(s, data):
            if u and use_cache and not pos_path(s["id"], u).exists():
                continue  # cached mode never trains a model for the extra stocks; run refresh_positions for that
            pos = compute_positions(s, data, use_cache, u)
            for c, df in universe_data(s, data, u).items():
                f = fees_of(c)
                for p in periods:
                    results[s["id"]][p][c] = bt.run_strategy(df, pos[c], p, f)
    return results


def leaderboard(results, labels, periods=("is", "oos", "full"), codes=None):
    """Medians across securities; codes restricts the set (e.g. the 50 stocks or the ETFs). Rows with no securities
    in the set are skipped."""
    rows = []
    keep = None if codes is None else set(codes)
    for p in periods:
        bh = results["buy_hold"][p]
        dca = results["dca"][p]
        for sid, by_p in results.items():
            res = by_p[p]
            codes = [c for c, r in res.items() if r and (keep is None or c in keep)]
            if not codes:
                continue
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
    weights = {}
    for s in strategies:
        if s.get("weights"):
            weights[s["id"]] = s["weights"](universe_data(s, data))
        else:
            weights[s["id"]] = pf.per_stock_weights(compute_positions(s, data, use_cache), data)
    for p in periods:
        out[p] = pf.benchmarks(data, p)
        for sid, w in weights.items():
            out[p][sid] = pf.run(data, w, p)
    return out


def load_everything(extra=True):
    """The 50 stocks plus the ETFs (0050, gold, oil, US Treasuries) and, with extra=True, the representative stocks
    outside the 50, keyed by code."""
    from .data import ALL_CODES, load_all
    from .universe import CODES
    from .etf import load_all_etfs

    return load_all(ALL_CODES if extra else CODES) | load_all_etfs()


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
