"""Lookahead checks for stocklab.external: every aligned value must come from the correct (earlier) raw record.

Run with `python tests/test_external.py` (or pytest). Needs the cache from `python -m stocklab.external`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stocklab import external as ext  # noqa: E402

T = pd.Timestamp


def _raw_net(code, date, names):
    """Net buy straight from the cached FinMind file, independent of the loader's code path."""
    d = pd.read_csv(ext._fm_path("stock", "inst", code))
    d = d[(d["date"] == date) & d["name"].isin(names)]
    return float((d["buy"] - d["sell"]).sum())


def test_align_synthetic():
    raw = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-05", "2024-01-09"]))
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-30"])
    got = ext.align(raw, idx, tol=14)["v"].to_numpy()
    want = np.array([np.nan, 1, 1, 2, 2, 3, np.nan])  # 01-30 is 21 days after the last record > tol
    assert np.array_equal(got, want, equal_nan=True), got
    # flows: each record only on the first bar that picks it up
    got = ext.align(raw, idx, tol=14, repeat=False)["v"].to_numpy()
    want = np.array([np.nan, 1, np.nan, 2, np.nan, 3, np.nan])
    assert np.array_equal(got, want, equal_nan=True), got


def test_us_close_alignment():
    """US close of date d prints 04:00-06:00 Taipei on d+1 < next TW open, so TW bar t sees US dates <= t only."""
    sox = ext.yahoo_raw("sox")["close"]
    m = ext.load_market(["sox"])["sox"]
    cases = {
        "2024-01-12": "2024-01-12",  # Fri: same-date US close (known Sat 05:00, before Mon open)
        "2024-01-15": "2024-01-12",  # Mon: US closed (MLK day) -> previous Friday, never Tue 01-16
        "2024-01-16": "2024-01-16",
        "2024-02-05": "2024-02-05",  # last TW bar before Lunar New Year: NOT the US closes of 02-06..02-14
        "2024-02-15": "2024-02-15",  # first TW bar after LNY
    }
    for tw, us in cases.items():
        assert m[T(tw)] == sox[T(us)], (tw, m[T(tw)], us, sox[T(us)])
    assert m[T("2024-02-05")] != sox[T("2024-02-06")]
    # at_close: only what is known at the TW close of t -> US date strictly before t
    mc = ext.load_market(["sox"], at_close=True)["sox"]
    assert mc[T("2024-01-16")] == sox[T("2024-01-12")]
    assert mc[T("2024-02-15")] == sox[T("2024-02-14")]


def test_tw_daily_chip_alignment():
    """三大法人 / 融資 for TW date d are published the evening of d -> used from bar d, never earlier."""
    f = ext.load_stock_fields("2330", ["foreign_net", "margin_balance"])
    for d in ["2015-06-01", "2020-03-19", "2024-02-15", "2026-10-02"]:
        assert f.loc[T(d), "foreign_net"] == _raw_net("2330", d, ext.FOREIGN), d
    raw_m = pd.read_csv(ext._fm_path("stock", "margin", "2330")).set_index("date")["MarginPurchaseTodayBalance"]
    assert f.loc[T("2024-02-05"), "margin_balance"] == raw_m["2024-02-05"] * 1000
    assert f.loc[T("2024-02-15"), "margin_balance"] == raw_m["2024-02-15"] * 1000
    # before the per-stock 三大法人 history starts (2012-05-02) there is nothing
    assert np.isnan(f.loc[T("2012-04-30"), "foreign_net"])
    fc = ext.load_stock_fields("2330", ["foreign_net"], at_close=True)["foreign_net"]
    assert fc[T("2024-02-15")] == _raw_net("2330", "2024-02-05", ext.FOREIGN)


def test_revenue_alignment():
    """Revenue of month m: first TW trading day >= the 10th of m+1, plus REVENUE_EXTRA_LAG (1) trading day."""
    assert ext.REVENUE_EXTRA_LAG == 1
    raw = pd.read_csv(ext._fm_path("stock", "revenue", "2330"))
    rev = raw.set_index(["revenue_year", "revenue_month"])["revenue"].astype(float)
    r = ext.load_stock_fields("2330", ["revenue"])["revenue"]
    # Jan 2024: Feb 10 falls in the Lunar New Year closure; first trading day >= 10th is 02-15, +1 -> 02-16
    assert ext.revenue_avail(2024, 1) == T("2024-02-16")
    assert r[T("2024-02-15")] == rev[(2023, 12)]
    assert r[T("2024-02-16")] == rev[(2024, 1)]
    # Feb 2024: Mar 10 is a Sunday -> 03-11 (Mon) -> +1 -> 03-12
    assert ext.revenue_avail(2024, 2) == T("2024-03-12")
    assert r[T("2024-03-11")] == rev[(2024, 1)]
    assert r[T("2024-03-12")] == rev[(2024, 2)]
    yoy = ext.load_stock_fields("2330", ["revenue_yoy"])["revenue_yoy"]
    assert abs(yoy[T("2024-03-12")] - (rev[(2024, 2)] / rev[(2023, 2)] - 1)) < 1e-12


def test_market_columns_generic():
    """Every market column on random dates: aligned value == raw value at the last raw date <= t (independent slicing)."""
    cal = ext.calendar()
    rng = np.random.default_rng(0)
    dates = cal[np.sort(rng.choice(len(cal), 40, replace=False))]
    m = ext.load_market(index=dates)
    for col in ext.MARKET:
        raw = ext.market_raw(col)
        prev = None
        for t in dates:
            past = raw.loc[:t]
            stale = past.empty or (t - past.index[-1]).days > ext.DAILY_TOL
            reused = col in ext.FLOWS and prev is not None and not past.empty and past.index[-1] <= prev
            if stale or reused:
                assert np.isnan(m.loc[t, col]), (col, t)
            else:
                assert m.loc[t, col] == past.iloc[-1], (col, t, m.loc[t, col], past.iloc[-1])
            prev = t


def test_stock_fields_generic():
    """Same independent check for every per-stock field of a few stocks (incl. late listings)."""
    cal = ext.calendar()
    rng = np.random.default_rng(1)
    dates = cal[np.sort(rng.choice(len(cal), 40, replace=False))]
    for code in ["2330", "2881", "6669"]:
        f = ext.load_stock_fields(code, index=dates)
        for field, (group, _, timing, _) in ext.STOCK_FIELDS.items():
            raw = ext.stock_raw(code, group)[field]  # NaN records (e.g. PER of a loss-maker) count as the latest value
            tol = ext.MONTHLY_TOL if timing == "revenue" else ext.DAILY_TOL
            prev = None
            for t in dates:
                past = raw.loc[:t]
                stale = past.empty or (t - past.index[-1]).days > tol
                reused = field in ext.FLOWS and prev is not None and not past.empty and past.index[-1] <= prev
                if stale or reused:
                    assert np.isnan(f.loc[t, field]), (code, field, t)
                else:
                    assert np.array_equal(f.loc[t, field], past.iloc[-1], equal_nan=True), (code, field, t)
                prev = t


def test_truncation_invariance():
    """Values at t do not depend on data / calendar after t (the project's truncation check)."""
    cal = ext.calendar()
    cut = T("2019-06-28")
    full = ext.load_market()
    short = ext.load_market(index=cal[cal <= cut])
    pd.testing.assert_frame_equal(full.loc[:cut], short)
    fs = ext.load_stock_fields("2454")
    ss = ext.load_stock_fields("2454", index=cal[cal <= cut])
    pd.testing.assert_frame_equal(fs.loc[:cut], ss)


def test_end_and_data_end():
    """Strategies cut external data at data_end(data); the result must equal the uncut frame up to that date."""
    from stocklab.data import load

    data = {c: load(c).loc[:"2019-06-28"] for c in ["2330", "2454"]}
    end = ext.data_end(data)
    assert end == T("2019-06-28")
    m = ext.load_market(["sox", "mkt_foreign_net", "pc_ratio_oi"], end=end)
    assert m.index[-1] == end and len(m) == len(ext.calendar(end))
    pd.testing.assert_frame_equal(m, ext.load_market(["sox", "mkt_foreign_net", "pc_ratio_oi"]).loc[:end])
    f = ext.load_stock_fields("2330", index=data["2330"].index, end=end)
    assert f.index.equals(data["2330"].index)
    p = ext.load_stock_panel("revenue_yoy", codes=["2330", "2454"], end=end)
    assert p.index[-1] == end
    pd.testing.assert_frame_equal(p, ext.load_stock_panel("revenue_yoy", codes=["2330", "2454"]).loc[:end])


def test_panels_shape():
    cal = ext.calendar()
    p = ext.load_stock_panel("foreign_ratio")
    assert p.index.equals(cal) and list(p.columns) == ext.CODES
    q = ext.load_price_panel("close", codes=["2330", "2454"])
    assert q.index.equals(cal)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
