"""Evaluate strategies against 買進持有 and 定期定額 without touching the website files.

  python evaluate.py --family kd_macd_rule              # in-sample only (default)
  python evaluate.py --family kd_macd_rule --periods is,oos
  python evaluate.py --ids kd_macd_cross --lookahead    # also run the truncation lookahead check
"""

import argparse

from stocklab.data import RAW_DIR, download, load_all
from stocklab.runner import check_lookahead, discover, evaluate, leaderboard, print_leaderboard

ap = argparse.ArgumentParser()
ap.add_argument("--family", help="strategies/<family> folder name")
ap.add_argument("--ids", help="comma-separated strategy ids")
ap.add_argument("--periods", default="is", help="comma-separated: is,oos,full")
ap.add_argument("--lookahead", action="store_true")
ap.add_argument("--cached", action="store_true", help="reuse cached positions")
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
print_leaderboard(leaderboard(results, labels, periods))
