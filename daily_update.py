"""Daily update after the Taiwan close: data -> positions -> signals -> website (-> git push).

  python daily_update.py                  # data, positions, signals, site
  python daily_update.py --push           # also commit and push the website data (GitHub Pages)
  python daily_update.py --only signals,site
  python daily_update.py --skip data
  python daily_update.py --fast           # keep the deep-learning positions (retraining them takes about an hour)

Run after 14:30 Taipei (bars dated today are dropped before that). Yahoo can lag a day; stocklab.data.top_up fills
the gap from the exchange quotes on FinMind. FinMind allows ~300 requests an hour without an account (600 with a
free token in the FINMIND_TOKEN environment variable); a daily run makes about 400, so it may pause for the limit.
"""

import argparse
import subprocess
import sys
import time

from stocklab.data import ROOT

STAGES = ["data", "positions", "signals", "site"]
DL_FAMILIES = {"kdj_macd_dl", "dl_position", "dl_revenue_flow", "dl_market_state", "dl_cross_stock", "dl_trend_labels"}


def stage_data():
    from stocklab import data, etf, macro, shortrules
    from stocklab import external as ext
    from stocklab.universe import CODES

    data.download(data.ALL_CODES)
    ext.update_recent(codes=CODES)
    latest = ext.last_session()
    added = data.top_up(data.ALL_CODES, until=latest)
    print(f"latest session {latest.date()}; topped up from FinMind: {len(added)} codes", flush=True)
    stale = [c for c in data.ALL_CODES if data.load(c).index[-1] < latest]
    if stale:
        print(f"  still behind {latest.date()}: {stale}", flush=True)
    etf.download_etfs(refresh=True)
    shortrules.update_notes(data.ALL_CODES + etf.CODES)
    macro.download_policy(refresh=True)


def stage_positions(fast=False):
    from stocklab.runner import discover, load_everything, refresh_positions

    strategies = discover()
    if fast:  # their signals then show the date their positions end ("部位停在 …")
        strategies = [s for s in strategies if s["family_dir"] not in DL_FAMILIES]
    refresh_positions(strategies, load_everything())


def stage_signals():
    from stocklab import signals

    signals.build()


def stage_site():
    subprocess.run([sys.executable, str(ROOT / "build_site.py"), "--cached"], check=True, cwd=ROOT)


def push():
    from stocklab import signals

    asof = signals.load()["asof"]
    subprocess.run(["git", "add", "docs"], check=True, cwd=ROOT)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        print("nothing to commit")
        return
    subprocess.run(["git", "commit", "-m", f"Daily signals as of {asof}"], check=True, cwd=ROOT)
    subprocess.run(["git", "push"], check=True, cwd=ROOT)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help=f"comma-separated stages from {STAGES}")
    ap.add_argument("--skip", help="comma-separated stages to skip")
    ap.add_argument("--push", action="store_true", help="commit and push docs/ afterwards")
    ap.add_argument("--fast", action="store_true", help="skip retraining the deep-learning strategies")
    args = ap.parse_args()
    run = args.only.split(",") if args.only else STAGES
    run = [s for s in STAGES if s in run and s not in (args.skip or "").split(",")]
    for name in run:
        t = time.time()
        print(f"== {name}", flush=True)
        if name == "positions":
            stage_positions(args.fast)
        else:
            globals()[f"stage_{name}"]()
        print(f"== {name} done in {time.time() - t:.0f}s", flush=True)
    if args.push:
        push()
