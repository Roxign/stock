"""Run every strategy and baseline, then export data for the GitHub Pages viewer in docs/.

  python build_site.py             # recompute all positions
  python build_site.py --cached    # reuse data/positions/*.pkl
  python build_site.py --download  # refresh prices from Yahoo first
"""

import argparse
import base64
import json
import math
import shutil
from datetime import datetime

import pandas as pd

from stocklab import backtest as bt
from stocklab import cv
from stocklab.data import RAW_DIR, ROOT, download
from stocklab.etf import CODES as ETF_CODES, ETFS
from stocklab.portfolio import BENCH_LABELS
from stocklab.runner import (BASELINES, POS_CACHE, discover, evaluate, fees_of, leaderboard, load_everything, pos_path,
                             portfolio_results, print_leaderboard, print_portfolio, runs_on_extra, universes_of)
from stocklab.universe import CODES as STOCK_CODES, DATA_DATE, EXTRA_CODES, EXTRA_NAMES, NAMES, SECTORS, WEIGHTS

DOCS = ROOT / "docs"
PERIODS = ("full", "is", "oos")
FOLDS = cv.FOLDS
UNIVERSES = {"stocks": ("股票（50 檔）", STOCK_CODES), "extra": ("代表股（另 55 檔）", EXTRA_CODES),
             "etfs": ("ETF（0050、黃金、石油、美債）", ETF_CODES)}
DOCS_COPIED = ("diagnosis.md", "dl_literature.md", "data_sources.md", "short_rules.md", "etf_data.md", "macro_fx_rates.md",
               "signals.md")


def latest_block(sid, board, cv_rows, cv_meta, extra_board=None, data_end=None):
    """Markdown table of the current numbers, prepended to each description so hand-written figures can't go stale."""
    def cell(row):
        if row is None:
            return "—"
        return f"{row['cagr_med']:.1%} / {row['mdd_med']:.1%} / {row['sharpe_med']:.2f}"

    def row(i, p, b=board):
        hit = b[(b["id"] == i) & (b["period"] == p)]
        return None if hit.empty else hit.iloc[0]

    what = "ETF" if (board["universe"] == "etfs").all() else "50 檔"
    lines = [
        f"**最新回測（每次更新資料時自動產生，資料至 {data_end}）**：{what}中位數，年化報酬 / 最大回撤 / Sharpe",
        "",
        "| | 樣本內 2010–2020 | 樣本外 2021–今 |",
        "|---|---|---|",
        f"| 本策略 | {cell(row(sid, 'is'))} | {cell(row(sid, 'oos'))} |",
        f"| 買進持有 | {cell(row('buy_hold', 'is'))} | {cell(row('buy_hold', 'oos'))} |",
    ]
    if extra_board is not None and row(sid, "is", extra_board) is not None:
        lines += [
            f"| 本策略：另 55 檔代表股（開發時沒用過） | {cell(row(sid, 'is', extra_board))} | {cell(row(sid, 'oos', extra_board))} |",
            f"| 買進持有：另 55 檔代表股 | {cell(row('buy_hold', 'is', extra_board))} | {cell(row('buy_hold', 'oos', extra_board))} |",
        ]
    by_id = {r["id"]: r for r in cv_rows}
    if sid in by_id:
        r = by_id[sid]
        text = f"交叉驗證 8 折（固定參數）：Sharpe 勝過買進持有 {r['folds_sharpe']}/8 折、年化報酬勝 {r['folds_cagr']}/8 折、最大回撤較小 {r['folds_mdd']}/8 折"
        cvr = by_id.get(f"{sid}~cv")
        if cvr:
            how = "每折重新訓練" if cv_meta[sid]["mode"] == "purged" else "每折改用其他 7 折最佳參數"
            text += f"；{how}：Sharpe 勝 {cvr['folds_sharpe']}/8 折、回撤較小 {cvr['folds_mdd']}/8 折"
        lines += ["", text + "。"]
    lines += ["", "下方說明中的數字是研究當時的結果，資料更新後可能略有差異，請以本表與總覽頁為準。", "", "---", ""]
    return "\n".join(lines)


def name_of(code):
    return NAMES.get(code) or EXTRA_NAMES.get(code) or ETFS[code]["name"]


def kind_of(code):
    return "etf" if code in ETF_CODES else "extra" if code in EXTRA_CODES else "stock"


def portfolio_export(pf_results):
    """Portfolio metrics per period, plus weekly equity curves (multiple of starting capital) for each period."""
    out = {"labels": BENCH_LABELS, "periods": {}, "equity": {}}
    for p, rows in pf_results.items():
        out["periods"][p] = {k: r["metrics"] for k, r in rows.items() if r}
        out["equity"][p] = {}
        for k, r in rows.items():
            if not r:
                continue
            w = (r["equity"] / bt.CAPITAL).resample("W-FRI").last().dropna()
            out["equity"][p][k] = [[d.strftime("%Y-%m-%d"), round(float(v), 4)] for d, v in w.items()]
    return out


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
    ap.add_argument("--cached", action="store_true", help="reuse cached positions and cross-validation (daily update)")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--families", help="comma-separated strategies/<family> folders (default: all)")
    args = ap.parse_args()

    if args.download or not any(RAW_DIR.glob("*.csv")):
        download()
    data = load_everything()
    data_end = max(df.index[-1] for df in data.values()).strftime("%Y-%m-%d")
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
    boards, cv_tables = [], {}
    for u, (_, codes) in UNIVERSES.items():
        b = pd.concat([leaderboard(main_results, labels, PERIODS, codes=codes), leaderboard(results, labels, FOLDS, codes=codes)],
                      ignore_index=True)
        b["universe"] = u
        boards.append(b)
        rows, bh = cv.fold_table(results, table_ids, labels, codes=codes)
        cv_tables[u] = {"rows": rows, "bh": bh}
        print(f"\n##### {UNIVERSES[u][0]}")
        print_leaderboard(b[b["period"].isin(PERIODS)])
        cv.print_fold_table(rows, bh)
    board = pd.concat(boards, ignore_index=True)
    pf_results = portfolio_results(strategies, data, PERIODS, use_cache=True)
    print_portfolio(pf_results)

    for sub in ("data/stocks", "multicharts", "research"):
        shutil.rmtree(DOCS / sub, ignore_errors=True)

    for name in DOCS_COPIED:
        src = ROOT / "research" / name
        if src.exists():
            dst = DOCS / "research" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

    meta = [dict(b, research=None, universe="all") for b in BASELINES]
    for s in strategies:
        u = s.get("universe", "stocks")
        m = {k: s[k] for k in ("id", "label", "family", "description")}
        m["universe"] = u
        m["portfolio_weights"] = bool(s.get("weights"))
        m["extra"] = runs_on_extra(s, data) and pos_path(s["id"], "extra").exists()
        if s.get("lab_rule"):  # opens in the rule lab (docs/rules.js decodes the same base64url JSON)
            raw = json.dumps(s["lab_rule"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            m["lab_link"] = "#/lab?r=" + base64.urlsafe_b64encode(raw).decode().rstrip("=")
        key = "etfs" if u == "etfs" else "stocks"
        m["description"] = latest_block(s["id"], board[board["universe"] == key], cv_tables[key]["rows"], cv_meta,
                                        board[board["universe"] == "extra"] if m["extra"] else None, data_end) + s["description"]
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

    positions = {}
    for s in strategies:
        positions[s["id"]] = {}
        for u in universes_of(s, data):
            if pos_path(s["id"], u).exists():
                positions[s["id"]].update(pd.read_pickle(pos_path(s["id"], u)))
    for code, df in data.items():
        write_json(DOCS / "data" / "stocks" / f"{code}.json", {
            "code": code,
            "name": name_of(code),
            "kind": kind_of(code),
            "sector": SECTORS.get(code),
            "fees": fees_of(code),
            "d": [d.strftime("%Y-%m-%d") for d in df.index],
            "o": df["open"].round(4).tolist(),
            "h": df["high"].round(4).tolist(),
            "l": df["low"].round(4).tolist(),
            "c": df["close"].round(4).tolist(),
            "v": (df["volume"] / 1000).round().astype(int).tolist(),
            "pos": {sid: encode_positions(p[code], df.index) for sid, p in positions.items() if code in p},
        })

    write_json(DOCS / "data" / "summary.json", {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data_date": DATA_DATE,
        "data_end": data_end,
        "engine": {"capital": bt.CAPITAL, "buy_fee": bt.BUY_FEE, "sell_fee": bt.SELL_FEE, "warmup": bt.WARMUP},
        "periods": {p: {"start": s, "end": e, "label": bt.PERIOD_LABELS[p]} for p, (s, e) in bt.ALL_PERIODS.items()},
        "folds": [{"id": f, "label": bt.FOLD_LABELS[f]} for f in FOLDS],
        "cv": cv_meta,
        "cv_tables": cv_tables,
        "universes": {u: {"label": lab, "codes": list(codes)} for u, (lab, codes) in UNIVERSES.items()},
        "strategies": meta,
        "stocks": [{"code": c, "name": name_of(c), "kind": kind_of(c), "sector": SECTORS.get(c), "weight": WEIGHTS.get(c),
                    "start": df.index[0].strftime("%Y-%m-%d"), "end": df.index[-1].strftime("%Y-%m-%d")} for c, df in data.items()],
        "board": board.to_dict("records"),
        "portfolio": portfolio_export(pf_results),
        "metrics": {sid: {p: {c: r["metrics"] for c, r in by_code.items() if r} for p, by_code in by_p.items()}
                    for sid, by_p in results.items()},
    })
    print(f"\nwrote {DOCS / 'data'}")


if __name__ == "__main__":
    main()
