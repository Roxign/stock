"""Evaluate strategies against 買進持有 and 定期定額 without touching the website files.

  python evaluate.py --family kdj_macd_rule              # in-sample only (default)
  python evaluate.py --family kdj_macd_rule --periods is,oos
  python evaluate.py --ids trend_sma_half --lookahead    # also run the truncation lookahead check
  python evaluate.py --family X --controls               # add B1 固定曝險 / B2 別檔訊號 / B3 波動目標 controls
  python evaluate.py --family X --portfolio              # 50-stock single-account scoreboard vs 0050
"""

import argparse

from stocklab.controls import evaluate_controls
from stocklab.data import RAW_DIR, download, load_all
from stocklab.runner import (check_lookahead, compute_positions, discover, evaluate, leaderboard, portfolio_results,
                             print_leaderboard, print_portfolio)

ap = argparse.ArgumentParser()
ap.add_argument("--family", help="strategies/<family> folder name")
ap.add_argument("--ids", help="comma-separated strategy ids")
ap.add_argument("--periods", default="is", help="comma-separated: is,oos,full")
ap.add_argument("--lookahead", action="store_true")
ap.add_argument("--cached", action="store_true", help="reuse cached positions")
ap.add_argument("--controls", action="store_true", help="also evaluate the B1/B2/B3 controls for each strategy")
ap.add_argument("--portfolio", action="store_true", help="also print the single-account portfolio scoreboard")
args = ap.parse_args()

if not any(RAW_DIR.glob("*.csv")):
    download()
data = load_all()
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
print_leaderboard(leaderboard(results, labels, periods))
if args.portfolio:
    print_portfolio(portfolio_results(strategies, data, periods))
