"""Evaluate strategies against 買進持有 and 定期定額 without touching the website files.

  python evaluate.py --family kdj_macd_rule              # in-sample only (default)
  python evaluate.py --family kdj_macd_rule --periods is,oos
  python evaluate.py --ids trend_sma_half --lookahead    # also run the truncation lookahead check
  python evaluate.py --family X --controls               # add B1 固定曝險 / B2 別檔訊號 / B3 波動目標 controls
  python evaluate.py --family X --portfolio              # 50-stock single-account scoreboard vs 0050
  python evaluate.py --family X --cv                     # 8-fold blocked cross-validation (stocklab/cv.py)
"""

import argparse

from stocklab import cv
from stocklab.controls import evaluate_controls
from stocklab.data import RAW_DIR, download
from stocklab.runner import (check_lookahead, compute_positions, discover, evaluate, leaderboard, load_everything, portfolio_results,
                             print_leaderboard, print_portfolio)

from stocklab.etf import CODES as ETF_CODES  # noqa: E402
from stocklab.universe import CODES as STOCK_CODES  # noqa: E402

UNIVERSES = {"stocks": ("股票（50 檔）", STOCK_CODES), "etfs": ("ETF（0050、黃金、石油、美債）", ETF_CODES)}

ap = argparse.ArgumentParser()
ap.add_argument("--family", help="strategies/<family> folder name")
ap.add_argument("--ids", help="comma-separated strategy ids")
ap.add_argument("--periods", default="is", help="comma-separated: is,oos,full")
ap.add_argument("--lookahead", action="store_true")
ap.add_argument("--cached", action="store_true", help="reuse cached positions")
ap.add_argument("--controls", action="store_true", help="also evaluate the B1/B2/B3 controls for each strategy")
ap.add_argument("--portfolio", action="store_true", help="also print the single-account portfolio scoreboard")
ap.add_argument("--cv", action="store_true", help="8-fold cross-validation: fixed params, plus retune/purged when supported")
args = ap.parse_args()

if not any(RAW_DIR.glob("*.csv")):
    download()
data = load_everything()
strategies = discover([args.family] if args.family else None)
if args.ids:
    want = set(args.ids.split(","))
    strategies = [s for s in strategies if s["id"] in want]
if not strategies:
    raise SystemExit("no matching strategies")

if args.lookahead:
    for s in strategies:
        check_lookahead(s, data)

periods = tuple(args.periods.split(","))
results = evaluate(strategies, data, periods, use_cache=args.cached)
labels = {s["id"]: s["label"] for s in strategies}
if args.controls:
    for s in strategies:
        results |= evaluate_controls(s["id"], compute_positions(s, data, use_cache=True), data, periods)
used = {s.get("universe", "stocks") for s in strategies}
shown = [u for u in UNIVERSES if u in used or "all" in used]
for u in shown:
    print(f"\n##### {UNIVERSES[u][0]}")
    print_leaderboard(leaderboard(results, labels, periods, codes=UNIVERSES[u][1]))
if args.portfolio:
    print_portfolio(portfolio_results(strategies, data, periods))
if args.cv:
    fold_res = evaluate(strategies, data, cv.FOLDS, use_cache=True)
    ids = [s["id"] for s in strategies]
    for s in strategies:
        out = cv.cross_validate(s, data)
        if out:
            fold_res[f"{s['id']}~cv"] = out["results"]
            ids.append(f"{s['id']}~cv")
            if out["chosen"]:
                print(f"\n{s['id']} 每折選出的參數（grid {out['grid_size']} 組）:")
                for f, p in out["chosen"].items():
                    print(f"  {f}: {p}")
    for u in shown:
        print(f"\n##### {UNIVERSES[u][0]}")
        rows, bh = cv.fold_table(fold_res, ["dca"] + ids, labels, codes=UNIVERSES[u][1])
        cv.print_fold_table(rows, bh)
