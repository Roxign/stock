"""Run every strategy and baseline, then export data for the GitHub Pages viewer in docs/.

  python build_site.py             # recompute all positions
  python build_site.py --cached    # reuse data/positions/*.pkl
  python build_site.py --download  # refresh prices from Yahoo first
"""

import argparse
import json
import math
import shutil
from datetime import datetime

import pandas as pd

from stocklab import backtest as bt
from stocklab import cv
from stocklab.data import RAW_DIR, ROOT, download, load_all
from stocklab.runner import BASELINES, POS_CACHE, discover, evaluate, leaderboard, print_leaderboard
from stocklab.universe import DATA_DATE, NAMES, WEIGHTS

DOCS = ROOT / "docs"
PERIODS = ("full", "is", "oos")
FOLDS = cv.FOLDS


def cross_validated(strategy, data, use_cache):
    """cv.cross_validate slimmed to per-fold metrics, cached because retune/purged CV re-runs the strategy many times."""
    path = POS_CACHE / f"{strategy['id']}~cv.pkl"
    if use_cache and path.exists():
        return pd.read_pickle(path)
    out = cv.cross_validate(strategy, data)
    if out is None:
        return None
    slim = {"mode": out["mode"], "chosen": out["chosen"],
            "results": {f: {c: ({"metrics": r["metrics"]} if r else None) for c, r in by_code.items()}
                        for f, by_code in out["results"].items()}}
    POS_CACHE.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(slim, path)
    return slim


def jsonable(x):
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else round(x, 6)
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if hasattr(x, "item"):
        return jsonable(x.item())
    return x


def encode_positions(pos, index):
    """Only the bars where the target exposure changes, as [bar index, exposure]."""
    s = bt.clean_target(pos, index).round(4).to_numpy()
    out, prev = [], 0.0
    for i, v in enumerate(s):
        if v != prev:
            out.append([i, float(v)])
            prev = v
    return out


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--families", help="comma-separated strategies/<family> folders (default: all)")
    args = ap.parse_args()

    if args.download or not any(RAW_DIR.glob("*.csv")):
        download()
    data = load_all()
    strategies = discover(args.families.split(",") if args.families else None)
    results = evaluate(strategies, data, PERIODS + FOLDS, use_cache=args.cached)
    labels = {b["id"]: b["label"] for b in BASELINES} | {s["id"]: s["label"] for s in strategies}
    main_results = dict(results)
    cv_meta, table_ids = {}, ["dca"]
    for s in strategies:
        table_ids.append(s["id"])
        out = cross_validated(s, data, args.cached)
        if out:
            cid = f"{s['id']}~cv"
            results[cid] = out["results"]
            labels[cid] = f"{s['label']}（交叉驗證）"
            cv_meta[s["id"]] = {"mode": out["mode"], "chosen": out["chosen"]}
            table_ids.append(cid)
    board = pd.concat([leaderboard(main_results, labels, PERIODS), leaderboard(results, labels, FOLDS)], ignore_index=True)
    print_leaderboard(board[board["period"].isin(PERIODS)])
    cv_rows, cv_bh = cv.fold_table(results, table_ids, labels)
    cv.print_fold_table(cv_rows, cv_bh)

    for sub in ("data/stocks", "multicharts", "research"):
        shutil.rmtree(DOCS / sub, ignore_errors=True)

    meta = [dict(b, research=None) for b in BASELINES]
    for s in strategies:
        m = {k: s[k] for k in ("id", "label", "family", "description")}
        m["multicharts"] = m["research"] = None
        if s.get("multicharts"):
            dst = DOCS / "multicharts" / f"{s['id']}.txt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(s["folder"] / s["multicharts"], dst)
            m["multicharts"] = f"multicharts/{s['id']}.txt"
        notes = s["folder"] / "RESEARCH.md"
        if notes.exists():
            dst = DOCS / "research" / f"{s['family_dir']}.md"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(notes, dst)
            m["research"] = f"research/{s['family_dir']}.md"
        meta.append(m)

    positions = {s["id"]: pd.read_pickle(POS_CACHE / f"{s['id']}.pkl") for s in strategies}
    for code, df in data.items():
        write_json(DOCS / "data" / "stocks" / f"{code}.json", {
            "code": code,
            "name": NAMES[code],
            "d": [d.strftime("%Y-%m-%d") for d in df.index],
            "o": df["open"].round(4).tolist(),
            "h": df["high"].round(4).tolist(),
            "l": df["low"].round(4).tolist(),
            "c": df["close"].round(4).tolist(),
            "v": (df["volume"] / 1000).round().astype(int).tolist(),
            "pos": {sid: encode_positions(p[code], df.index) for sid, p in positions.items()},
        })

    write_json(DOCS / "data" / "summary.json", {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data_date": DATA_DATE,
        "engine": {"capital": bt.CAPITAL, "buy_fee": bt.BUY_FEE, "sell_fee": bt.SELL_FEE, "warmup": bt.WARMUP},
        "periods": {p: {"start": s, "end": e, "label": bt.PERIOD_LABELS[p]} for p, (s, e) in bt.ALL_PERIODS.items()},
        "folds": [{"id": f, "label": bt.FOLD_LABELS[f]} for f in FOLDS],
        "cv": cv_meta,
        "cv_table": {"rows": cv_rows, "bh": cv_bh},
        "strategies": meta,
        "stocks": [{"code": c, "name": NAMES[c], "weight": WEIGHTS[c], "start": df.index[0].strftime("%Y-%m-%d"),
                    "end": df.index[-1].strftime("%Y-%m-%d")} for c, df in data.items()],
        "board": board.to_dict("records"),
        "metrics": {sid: {p: {c: r["metrics"] for c, r in by_code.items() if r} for p, by_code in by_p.items()}
                    for sid, by_p in results.items()},
    })
    print(f"\nwrote {DOCS / 'data'}")


if __name__ == "__main__":
    main()
