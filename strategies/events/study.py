"""Do stocks fall on the day trading restrictions start? Event study of TWSE 處置 (disposition) and 注意 (attention).

    python -m strategies.events.study

Prices: every TWSE common stock since 2008, delisted ones included (data/rl/full, strategies/rl/fulluniverse.py).
For a disposition announced after the close of day A, restricted from day S (normally A + 1) to day E:

  run-up   close(A) / close(A-10)            what got the stock restricted
  day A    close(A) / close(A-1)             the announcement day (rules are mechanical, so it can be anticipated)
  gap      open(S) / close(A)                overnight into the first restricted day (cannot be avoided once announced)
  intraday close(S) / open(S)                the first restricted day after the open (a holder can still sell at the open)
  day S    close(S) / close(A)               "限制交易當天"
  period   close(E) / close(A)               the whole restricted period
  after    close(E+5) / close(E), close(E+20) / close(E)

Every segment is also shown minus the same segment of the equal-weight average of all stocks that day (abnormal).
Control: stock-days with a 10-day run-up in the same range but no disposition announced, to tell the restriction
itself apart from the usual pull-back after a spike.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from stocklab.data import ROOT
from strategies.rl.common import save
from strategies.rl.fulluniverse import DIR as FULL, load_full

EV = ROOT / "data" / "events"
OUT = __import__("pathlib").Path(__file__).parent / "results" / "study.json"
SEGS = ["runup", "dayA", "gap", "intraday", "dayS", "period", "after5", "after20"]
LABEL = {"runup": "公告前 10 天", "dayA": "公告當天", "gap": "限制首日開盤跳空", "intraday": "限制首日開盤→收盤",
         "dayS": "限制首日（收盤對收盤）", "period": "整段處置期間", "after5": "結束後 5 天", "after20": "結束後 20 天",
         "next_gap": "隔天開盤跳空", "next_intra": "隔天開盤→收盤", "next_day": "隔天（收盤對收盤）", "next5": "之後 5 天", "next20": "之後 20 天"}


def prices():
    data = {}
    for p in FULL.glob("*.csv"):
        df = load_full(p.stem)
        if len(df) > 30:
            data[p.stem] = df
    return data


def market(data):
    """Equal-weight average, per calendar day, of each stock's close-close, gap and intraday returns."""
    cc, gp, it = {}, {}, {}
    for c, df in data.items():
        pc = df["close"].shift()
        cc[c] = (df["close"] / pc - 1).clip(-0.2, 0.2)
        gp[c] = (df["open"] / pc - 1).clip(-0.2, 0.2)
        it[c] = (df["close"] / df["open"] - 1).clip(-0.2, 0.2)
    f = lambda d: pd.DataFrame(d).mean(axis=1)
    return {"cc": f(cc), "gap": f(gp), "intra": f(it)}


def seg(df, i0, i1, kind="cc"):
    """Return from bar i0 to bar i1 of the stock: close->close by default; 'gap' open(i1)/close(i0); 'intra' close(i1)/open(i1)."""
    n = len(df)
    if i0 < 0 or i1 >= n or i1 < i0:
        return np.nan
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    if kind == "gap":
        return o[i1] / c[i0] - 1
    if kind == "intra":
        return c[i1] / o[i1] - 1
    return c[i1] / c[i0] - 1


def mseg(mk, idx, i0, i1, kind="cc"):
    """The equal-weight market's return over the same dates (compounded close-close days, or that day's gap/intraday)."""
    if i0 < 0 or i1 >= len(idx) or i1 < i0:
        return np.nan
    if kind in ("gap", "intra"):
        return float(mk["gap" if kind == "gap" else "intra"].get(idx[i1], np.nan))
    r = mk["cc"].reindex(idx[i0 + 1 : i1 + 1]).fillna(0)
    return float(np.prod(1 + r) - 1)


def summary(df, cols, by=None):
    out = {}
    groups = [("全部", df)] if by is None else list(df.groupby(by))
    for g, d in groups:
        row = {}
        for c in cols:
            x = d[c].dropna()
            ab = d[f"{c}_ab"].dropna() if f"{c}_ab" in d else None
            if len(x) < 5:
                continue
            row[c] = {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()), "neg": float((x < 0).mean()),
                      "t": float(x.mean() / x.std() * math.sqrt(len(x))) if x.std() > 0 else 0.0}
            if ab is not None and len(ab) >= 5:
                row[c] |= {"ab_mean": float(ab.mean()), "ab_neg": float((ab < 0).mean()),
                           "ab_t": float(ab.mean() / ab.std() * math.sqrt(len(ab))) if ab.std() > 0 else 0.0}
        out[str(g)] = row
    return out


def show(title, s):
    print(f"\n{title}")
    for g, row in s.items():
        if len(s) > 1:
            print(f"  [{g}]")
        for c, v in row.items():
            ab = f"｜扣大盤 {v['ab_mean']:+.2%}（下跌比例 {v['ab_neg']:.0%}，t = {v['ab_t']:.1f}）" if "ab_mean" in v else ""
            print(f"    {LABEL.get(c, c):<14} n={v['n']:>5}  平均 {v['mean']:+.2%}  中位 {v['median']:+.2%}  下跌比例 {v['neg']:.0%}  t = {v['t']:.1f}{ab}")


def disposition(data, mk):
    ev = pd.read_csv(EV / "punish.csv", dtype={"code": str}, parse_dates=["announce", "start", "end"])
    ev = ev[ev["code"].isin(data.keys())].copy()
    rows = []
    for _, e in ev.iterrows():
        df = data[e["code"]]
        idx = df.index
        iA = idx.searchsorted(e["announce"], side="right") - 1
        iS = idx.searchsorted(e["start"])
        iE = idx.searchsorted(e["end"], side="right") - 1 if pd.notna(e["end"]) else iS
        if iA < 11 or iS >= len(idx) or iS != iA + 1:
            continue
        r = {"code": e["code"], "announce": e["announce"], "year": e["announce"].year, "minutes": e["minutes"], "measure": e["measure"],
             "days": iE - iS + 1,
             "runup": seg(df, iA - 10, iA), "dayA": seg(df, iA - 1, iA), "gap": seg(df, iA, iS, "gap"), "intraday": seg(df, iA, iS, "intra"),
             "dayS": seg(df, iA, iS), "period": seg(df, iA, iE), "after5": seg(df, iE, iE + 5), "after20": seg(df, iE, iE + 20)}
        # tradable paths after the announcement (fills at opens / closes the investor can actually reach)
        o, cl = df["open"].to_numpy(), df["close"].to_numpy()
        r["short_S_to_S1"] = 1 - o[iS + 1] / o[iS] if iS + 1 < len(o) else np.nan          # short at S open, cover next open
        r["buy_Sclose_to_E"] = cl[iE] / cl[iS] - 1 if iE > iS else np.nan                  # buy after the first-day drop
        r["buy_Sclose_to_E_ab"] = r["buy_Sclose_to_E"] - mseg(mk, idx, iS, iE) if iE > iS else np.nan
        r["sell_open_vs_hold"] = o[iS] / cl[iE] - 1 if iE >= iS else np.nan                # holder sells at S open instead of holding to E
        for k, (a, b, kind) in {"runup": (iA - 10, iA, "cc"), "dayA": (iA - 1, iA, "cc"), "gap": (iA, iS, "gap"), "intraday": (iA, iS, "intra"),
                                "dayS": (iA, iS, "cc"), "period": (iA, iE, "cc"), "after5": (iE, iE + 5, "cc"), "after20": (iE, iE + 20, "cc")}.items():
            r[f"{k}_ab"] = r[k] - mseg(mk, idx, a, b, kind)
        rows.append(r)
    return pd.DataFrame(rows)


def attention(data, mk, disp):
    ev = pd.read_csv(EV / "notice.csv", dtype={"code": str}, parse_dates=["date"])
    ev = ev[ev["code"].isin(data.keys())].copy()
    disp_days = set(zip(disp["code"], disp["announce"])) if len(disp) else set()
    rows = []
    for _, e in ev.iterrows():
        df = data[e["code"]]
        idx = df.index
        i = idx.searchsorted(e["date"], side="right") - 1
        if i < 11 or i + 1 >= len(idx) or idx[i] != e["date"]:
            continue
        r = {"code": e["code"], "date": e["date"], "year": e["date"].year, "clause": str(e["clause"]), "disposed_same_day": (e["code"], e["date"]) in disp_days,
             "runup": seg(df, i - 10, i), "next_gap": seg(df, i, i + 1, "gap"), "next_intra": seg(df, i, i + 1, "intra"),
             "next_day": seg(df, i, i + 1), "next5": seg(df, i, i + 5), "next20": seg(df, i, i + 20)}
        for k, (a, b, kind) in {"next_gap": (i, i + 1, "gap"), "next_intra": (i, i + 1, "intra"), "next_day": (i, i + 1, "cc"),
                                "next5": (i, i + 5, "cc"), "next20": (i, i + 20, "cc")}.items():
            r[f"{k}_ab"] = r[k] - mseg(mk, idx, a, b, kind)
        rows.append(r)
    return pd.DataFrame(rows)


def control(data, mk, disp, years=range(2010, 2027)):
    """Stock-days whose 10-day run-up falls in the disposition events' inter-quartile range, without a disposition
    announced that day: what the next day looks like for a similarly hot stock that was not restricted."""
    lo, hi = disp["runup"].quantile([0.25, 0.75])
    disp_days = set(zip(disp["code"], disp["announce"]))
    rows = []
    for c, df in data.items():
        cl, op = df["close"].to_numpy(), df["open"].to_numpy()
        run = cl[10:] / cl[:-10] - 1
        for k in np.flatnonzero((run >= lo) & (run <= hi)):
            i = k + 10
            if i + 1 >= len(cl) or (c, df.index[i]) in disp_days or df.index[i].year not in years:
                continue
            day = cl[i + 1] / cl[i] - 1
            m = mk["cc"].get(df.index[i + 1], np.nan)
            rows.append({"code": c, "date": df.index[i], "runup": run[k], "dayS": day, "dayS_ab": day - m,
                         "gap": op[i + 1] / cl[i] - 1, "gap_ab": op[i + 1] / cl[i] - 1 - mk["gap"].get(df.index[i + 1], np.nan),
                         "intraday": cl[i + 1] / op[i + 1] - 1, "intraday_ab": cl[i + 1] / op[i + 1] - 1 - mk["intra"].get(df.index[i + 1], np.nan)})
    return pd.DataFrame(rows), (float(lo), float(hi))


def main():
    data = prices()
    mk = market(data)
    res = {}
    d = disposition(data, mk)
    print(f"處置事件：{len(d)} 筆（{d['code'].nunique()} 檔股票，{d['year'].min()}–{d['year'].max()}）")
    res["disposition"] = summary(d, SEGS)
    show("處置股（全部）", res["disposition"])
    # by the announcement's own label: 2026-08 onward both levels match every ~2 minutes for 5 days, so the matching
    # interval no longer tells them apart
    d["kind"] = np.select([d["measure"].str.contains("第一次"), d["measure"].str.contains("第二次")], ["第一次處置", "第二次處置"], "其他（督導會報、人工管制等）")
    res["disposition_by_kind"] = summary(d, ["gap", "intraday", "dayS", "period", "after20"], by="kind")
    show("依處置程度", res["disposition_by_kind"])
    d["era"] = np.select([d["announce"] < pd.Timestamp("2018-01-01"), d["announce"] < pd.Timestamp("2026-08-10")],
                         ["2010–2017", "2018–2026/08（約 10 天）"], "2026/08 新制（5 天、約 2 分鐘撮合）")
    res["disposition_by_era"] = summary(d, ["gap", "intraday", "dayS", "period"], by="era")
    show("依年代", res["disposition_by_era"])
    cost_rt, cost_short = 0.001425 * 2 + 0.003, 0.001425 * 2 + 0.003 + 0.0008
    tr = {}
    for kind, g in d.groupby("kind"):
        x = {"放空：首日開盤融券、隔天開盤回補": g["short_S_to_S1"] - cost_short,
             "買進：首日收盤買、處置結束收盤賣": g["buy_Sclose_to_E"] - cost_rt,
             "持有者：首日開盤賣掉，相對於抱到結束": g["sell_open_vs_hold"]}
        tr[kind] = {}
        for name, v in x.items():
            v = v.dropna()
            tr[kind][name] = {"n": int(len(v)), "mean": float(v.mean()), "median": float(v.median()), "pos": float((v > 0).mean()),
                              "t": float(v.mean() / v.std() * math.sqrt(len(v)))}
    res["tradable"] = tr
    print("\n實際可操作的做法（已扣手續費、證交稅、融券費；處置股不能當沖）")
    for kind, rows in tr.items():
        print(f"  [{kind}]")
        for name, v in rows.items():
            print(f"    {name:<26} n={v['n']:>5}  平均 {v['mean']:+.2%}  中位 {v['median']:+.2%}  賺錢比例 {v['pos']:.0%}  t = {v['t']:.1f}")
    c, rng = control(data, mk, d)
    res["control_range"] = rng
    res["control"] = summary(c, ["gap", "intraday", "dayS"])
    show(f"對照組：同樣漲了 {rng[0]:.0%}～{rng[1]:.0%}（10 天）但沒被處置的股票，隔天", res["control"])
    a = attention(data, mk, d)
    print(f"\n注意事件：{len(a)} 筆")
    res["attention"] = summary(a, ["next_gap", "next_intra", "next_day", "next5", "next20"])
    show("注意股（公告隔天起）", res["attention"])
    a["first_clause"] = a["clause"].str.split("、").str[0].replace({"nan": "其他"})
    res["attention_by_clause"] = summary(a, ["next_day", "next5"], by="first_clause")
    show("注意股依原因（第幾款）", res["attention_by_clause"])
    d.to_csv(EV / "disposition_events.csv", index=False)
    a.to_csv(EV / "attention_events.csv", index=False)
    save(OUT, res)


if __name__ == "__main__":
    main()
