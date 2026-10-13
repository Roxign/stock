"""Robustness of the two tradable patterns found in study.py:

  A. first disposition (第一次處置): buy at the close of the first restricted day, sell at the close of the last one
  B. second disposition (第二次處置): short at the first restricted day's open, cover at the next open -- only when the
     open is at or above the previous close, because 融券 may not be sold below the previous close (平盤以下不得融券)

Checks: market-adjusted returns, each year separately, a t-statistic over calendar months (events cluster in hot
markets), liquid stocks only, and a control group of equally hot stocks that were not restricted.

    python -m strategies.events.followup
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from stocklab.data import ROOT
from strategies.rl.common import save

from .study import EV, market, mseg, prices

OUT = __import__("pathlib").Path(__file__).parent / "results" / "followup.json"
COST = 0.001425 * 2 + 0.003           # buy + sell commission and tax
COST_SHORT = COST + 0.0008            # + 融券 fee


def stats(x, dates=None):
    x = pd.Series(np.asarray(x, float), index=pd.DatetimeIndex(dates) if dates is not None else None).dropna()
    out = {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()), "win": float((x > 0).mean()),
           "t": float(x.mean() / x.std() * math.sqrt(len(x))) if len(x) > 2 else np.nan}
    if dates is not None:
        m = x.groupby(x.index.to_period("M")).mean()
        out |= {"months": int(len(m)), "t_month": float(m.mean() / m.std() * math.sqrt(len(m))) if len(m) > 2 else np.nan}
    return out


def line(name, s):
    tm = f"，以月份計 t = {s['t_month']:.1f}（{s['months']} 個月）" if "t_month" in s else ""
    return f"  {name:<30} n={s['n']:>5}  平均 {s['mean']:+.2%}  中位 {s['median']:+.2%}  賺錢比例 {s['win']:.0%}  t = {s['t']:.1f}{tm}"


def main():
    data = prices()
    mk = market(data)
    ev = pd.read_csv(EV / "disposition_events.csv", dtype={"code": str}, parse_dates=["announce"])
    res = {}

    # ---------------- A: buy the first disposition's first-day close, sell at its end
    a = ev[ev["kind"] == "第一次處置"].copy()
    a["net"] = a["buy_Sclose_to_E"] - COST
    a["net_ab"] = a["buy_Sclose_to_E_ab"] - COST
    liq = {}
    for i, r in a.iterrows():
        df = data[r["code"]]
        j = df.index.searchsorted(r["announce"], side="right") - 1
        liq[i] = float(df["value"].iloc[max(0, j - 20) : j + 1].mean())
    a["liq"] = pd.Series(liq)
    print("A. 第一次處置：首日收盤買、處置最後一天收盤賣（已扣成本）")
    res["A"] = {"all": stats(a["net"], a["announce"]), "abnormal": stats(a["net_ab"], a["announce"])}
    print(line("全部", res["A"]["all"]))
    print(line("扣掉同期大盤（等權）", res["A"]["abnormal"]))
    for lab, m in (("成交金額前一半（流動性較好）", a["liq"] >= a["liq"].median()), ("首日跌超過 3%", a["dayS"] < -0.03),
                   ("首日沒跌", a["dayS"] >= 0)):
        res["A"][lab] = stats(a.loc[m, "net"], a.loc[m, "announce"])
        print(line(lab, res["A"][lab]))
    res["A"]["by_year"] = {}
    print("  逐年：", end="")
    for y, g in a.groupby("year"):
        s = stats(g["net"])
        res["A"]["by_year"][int(y)] = s
        print(f"{y} {s['mean']:+.1%}（{s['n']}）", end="  ")
    print()

    # control: equally hot stocks not restricted, bought at the next close and held the same number of days
    lo, hi = a["runup"].quantile([0.25, 0.75])
    days = int(a["days"].median())
    disp = set(zip(ev["code"], ev["announce"]))
    ctl = []
    for c, df in data.items():
        cl = df["close"].to_numpy()
        run = cl[10:] / cl[:-10] - 1
        for k in np.flatnonzero((run >= lo) & (run <= hi)):
            i = k + 10
            if i + days >= len(cl) or (c, df.index[i]) in disp or df.index[i].year < 2010:
                continue
            ctl.append((df.index[i], cl[i + days] / cl[i + 1] - 1 - COST))
    ctl = pd.DataFrame(ctl, columns=["date", "net"])
    res["A"]["control"] = stats(ctl["net"], ctl["date"])
    print(line(f"對照：同樣漲多但沒被處置，隔天收盤買、持有 {days - 1} 天", res["A"]["control"]))

    # ---------------- B: short the second disposition at the open when the open is not below the previous close
    b = ev[ev["kind"] == "第二次處置"].copy()
    b["net"] = b["short_S_to_S1"] - COST_SHORT
    ok = b["gap"] >= 0
    print("\nB. 第二次處置：首日開盤融券、隔天開盤回補（已扣成本）")
    res["B"] = {"all": stats(b["net"], b["announce"]), "shortable": stats(b.loc[ok, "net"], b.loc[ok, "announce"]),
                "share_shortable": float(ok.mean())}
    print(line("全部（忽略平盤以下不得融券）", res["B"]["all"]))
    print(line(f"只算開盤不低於平盤的（{ok.mean():.0%}）", res["B"]["shortable"]))
    save(OUT, res)



def engine_timing(data, ev):
    """Pattern A with the backtester's convention: decided at a close, filled at the next open."""
    rows = []
    for _, r in ev[ev["kind"] == "第一次處置"].iterrows():
        df = data[r["code"]]
        idx = df.index
        iA = idx.searchsorted(r["announce"], side="right") - 1
        iS, iE = iA + 1, iA + int(r["days"])
        if iE + 1 >= len(idx) or iE <= iS:
            continue
        o, c = df["open"].to_numpy(), df["close"].to_numpy()
        rows.append({"code": r["code"], "announce": r["announce"], "S": idx[iS], "E": idx[iE],
                     "close_close": c[iE] / c[iS] - 1 - COST, "open_open": o[iE + 1] / o[iS + 1] - 1 - COST})
    return pd.DataFrame(rows)


def portfolio(data, trades, slots=10):
    """One account: each event gets 1/slots of equity from the open after the first restricted day to the open after
    the last one; events beyond `slots` concurrent positions are skipped. Uses stocklab.portfolio (own fees)."""
    from stocklab import portfolio as pf

    codes = sorted(set(trades["code"]))
    W = {c: pd.Series(0.0, index=data[c].index) for c in codes}
    busy = []
    taken = 0
    for _, t in trades.sort_values("S").iterrows():
        busy = [e for e in busy if e > t["S"]]
        if len(busy) >= slots:
            continue
        w = W[t["code"]]
        span = (w.index >= t["S"]) & (w.index < t["E"])
        w[span] = w[span] + 1.0 / slots
        busy.append(t["E"])
        taken += 1
    sub = {c: data[c] for c in codes}
    out = {}
    for p in ("is", "oos", "full"):
        r = pf.simulate(sub, W, p)
        m = r["metrics"]
        e = pf.simulate({"0050": pf.load_etf()}, {"0050": pd.Series(1.0, index=pf.load_etf().index)}, p)["metrics"]
        out[p] = {"cagr": float(m["cagr"]), "sharpe": float(m["sharpe"]), "mdd": float(m["mdd"]), "exposure": float(m["exposure"]),
                  "etf_cagr": float(e["cagr"]), "etf_sharpe": float(e["sharpe"]), "etf_mdd": float(e["mdd"])}
    return out, taken


def more():
    data = prices()
    ev = pd.read_csv(EV / "disposition_events.csv", dtype={"code": str}, parse_dates=["announce"])
    t = engine_timing(data, ev)
    res = {}
    print("A 改成隔天開盤進出（決定在收盤、隔天開盤成交，和回測引擎相同）")
    for lab, m in (("全部", slice(None)), ("2010–2020", t["announce"] < "2021-01-01"),
                   ("2021–2026/08", (t["announce"] >= "2021-01-01") & (t["announce"] < "2026-08-10")),
                   ("2026/08 新制（5 天）", t["announce"] >= "2026-08-10")):
        g = t.loc[m]
        res[lab] = {"close_close": stats(g["close_close"], g["announce"]), "open_open": stats(g["open_open"], g["announce"])}
        print(line(f"{lab}・收盤進出", res[lab]["close_close"]))
        print(line(f"{lab}・隔天開盤進出", res[lab]["open_open"]))
    for slots in (5, 10, 20):
        pr, taken = portfolio(data, t, slots)
        res[f"portfolio_{slots}"] = pr | {"taken": taken}
        print(f"帳戶回測（最多同時 {slots} 檔，每檔 1/{slots}，實際做了 {taken} 次）："
              + "；".join(f"{p} 年化 {v['cagr']:+.1%} Sharpe {v['sharpe']:.2f} MDD {v['mdd']:.0%} 平均持股 {v['exposure']:.0%}（0050 {v['etf_cagr']:+.1%}/{v['etf_sharpe']:.2f}/{v['etf_mdd']:.0%}）" for p, v in pr.items()), flush=True)
    old = {}
    import json

    if OUT.exists():
        old = json.loads(OUT.read_text(encoding="utf-8"))
    save(OUT, old | {"engine_timing": res})



def overlay(data, trades, slots=10, base="0050"):
    """Idle money sits in 0050; each event takes 1/slots of equity out of 0050 for its holding period."""
    from stocklab import portfolio as pf

    etf = pf.load_etf()
    codes = sorted(set(trades["code"]))
    W = {c: pd.Series(0.0, index=data[c].index) for c in codes}
    busy = []
    for _, t in trades.sort_values("S").iterrows():
        busy = [e for e in busy if e > t["S"]]
        if len(busy) >= slots:
            continue
        w = W[t["code"]]
        span = (w.index >= t["S"]) & (w.index < t["E"])
        w[span] = w[span] + 1.0 / slots
        busy.append(t["E"])
    used = pd.DataFrame(W).reindex(etf.index).fillna(0).sum(axis=1)
    W[base] = (1 - used).clip(lower=0)
    sub = {c: data[c] for c in codes} | {base: etf}
    out = {}
    for p in ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "is", "oos", "full"):
        m = pf.simulate(sub, W, p)["metrics"]
        e = pf.simulate({base: etf}, {base: pd.Series(1.0, index=etf.index)}, p)["metrics"]
        out[p] = {"cagr": float(m["cagr"]), "sharpe": float(m["sharpe"]), "mdd": float(m["mdd"]), "turnover": float(m["turnover"]),
                  "etf_cagr": float(e["cagr"]), "etf_sharpe": float(e["sharpe"]), "etf_mdd": float(e["mdd"])}
    return out


def more2():
    data = prices()
    ev = pd.read_csv(EV / "disposition_events.csv", dtype={"code": str}, parse_dates=["announce"])
    t = engine_timing(data, ev)
    res = {}
    for slip in (0.0, 0.003, 0.006):
        x = t["open_open"] - slip
        res[f"slippage_{slip}"] = stats(x, t["announce"])
        print(line(f"每筆再多扣滑價 {slip:.1%}", res[f"slippage_{slip}"]))
    for slots in (5, 10):
        o = overlay(data, t, slots)
        res[f"overlay_{slots}"] = o
        wins = sum(o[f]["sharpe"] > o[f]["etf_sharpe"] for f in ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"))
        winc = sum(o[f]["cagr"] > o[f]["etf_cagr"] for f in ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"))
        print(f"0050＋處置反彈（最多 {slots} 檔）：" + "；".join(f"{p} {v['cagr']:+.1%}/{v['sharpe']:.2f}/MDD {v['mdd']:.0%}（0050 {v['etf_cagr']:+.1%}/{v['etf_sharpe']:.2f}/{v['etf_mdd']:.0%}）"
                                                    for p, v in o.items() if p in ("is", "oos", "full")) + f"；8 段中年化勝 {winc}、Sharpe 勝 {wins}", flush=True)
    import json

    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    save(OUT, old | {"overlay": res})


if __name__ == "__main__":
    import sys

    {"more": more, "more2": more2}.get((sys.argv[1:] or ["main"])[0], main)()
