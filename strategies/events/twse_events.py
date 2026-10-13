"""TWSE 注意股票 (attention) and 處置股票 (disposition) announcements since 2010, from the exchange's own website.

    python -m strategies.events.twse_events        # data/events/punish.csv and notice.csv (resumable, ~11 min)

punish  : announcement date, code, name, count, condition, period start/end, measure (第一次處置 / 第二次處置 ...),
          matching interval in minutes (5 or 20, parsed from the measure text), full text
notice  : date, code, name, cumulative count, reason text, clause (款) number, close, PE

Requests are spaced 3 seconds apart. Dates on the site are in the ROC calendar (113/01/12 = 2024-01-12).
"""

from __future__ import annotations

import json
import re
import time
from datetime import date

import pandas as pd
import requests

from stocklab.data import ROOT

DIR = ROOT / "data" / "events"
BASE = "https://www.twse.com.tw/rwd/zh/announcement"
HEAD = {"User-Agent": "Mozilla/5.0 (research; stock lab)"}
NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def roc(s):
    """'113/01/12' or '113.01.12' -> Timestamp."""
    y, m, d = re.split(r"[/.]", s.strip())
    return pd.Timestamp(int(y) + 1911, int(m), int(d))


def get(kind, start, end):
    url = f"{BASE}/{kind}?startDate={start:%Y%m%d}&endDate={end:%Y%m%d}&response=json"
    for attempt in range(5):
        try:
            r = requests.get(url, headers=HEAD, timeout=60)
            j = r.json()
            if j.get("stat") == "OK":
                return j.get("data", []), j.get("fields", [])
            if "沒有符合" in str(j.get("stat")):
                return [], []
        except (requests.RequestException, json.JSONDecodeError):
            pass
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"TWSE {kind} {start:%Y%m%d}-{end:%Y%m%d}: no data after retries")


def cn(t):
    """Chinese numeral up to 99 ('五', '二十', '二十五') or digits -> int."""
    if t.isdigit():
        return int(t)
    if "十" in t:
        hi, _, lo = t.partition("十")
        return (NUM[hi] if hi else 1) * 10 + (NUM[lo] if lo else 0)
    return NUM.get(t)


def minutes(text):
    m = re.search(r"每([一二三四五六七八九十]+|\d+)分鐘", text.replace(" ", ""))
    return cn(m.group(1)) if m else None


def punish(years):
    rows = []
    for y in years:
        data, fields = get("punish", date(y, 1, 1), date(y, 12, 31))
        for r in data:
            d = dict(zip(fields, r))
            period = re.split(r"[～~]", d["處置起迄時間"])
            text = re.sub(r"<[^>]+>", "", d.get("處置內容", ""))
            rows.append({"announce": roc(d["公布日期"]), "code": str(d["證券代號"]).strip(), "name": d["證券名稱"].strip(),
                         "count": d.get("累計"), "condition": d.get("處置條件", ""), "start": roc(period[0]),
                         "end": roc(period[1]) if len(period) > 1 else None, "measure": d.get("處置措施", ""),
                         "minutes": minutes(text), "text": text.replace("\n", " ")})
        print(f"  處置 {y}：{len(data)} 筆", flush=True)
        time.sleep(3)
    return pd.DataFrame(rows)


def notice(months):
    rows = []
    for start in months:
        end = (start + pd.offsets.MonthEnd(0)).date()
        data, fields = get("notice", start.date(), end)
        for r in data:
            d = dict(zip(fields, r))
            text = re.sub(r"<[^>]+>", "", d.get("注意交易資訊", ""))
            clause = re.findall(r"第([一二三四五六七八九十]+)款", text)
            rows.append({"date": roc(d["日期"]), "code": str(d["證券代號"]).strip(), "name": d["證券名稱"].strip(),
                         "count": d.get("累計次數"), "clause": "、".join(clause), "close": d.get("收盤價"), "text": text})
        print(f"  注意 {start:%Y-%m}：{len(data)} 筆", flush=True)
        time.sleep(3)
    return pd.DataFrame(rows)


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    this_year = pd.Timestamp.today().year
    p = DIR / "punish.csv"
    if not p.exists():
        punish(range(2010, this_year + 1)).to_csv(p, index=False)
    n = DIR / "notice.csv"
    done = pd.read_csv(n, parse_dates=["date"]) if n.exists() else pd.DataFrame(columns=["date"])
    months = pd.date_range("2010-01-01", pd.Timestamp.today(), freq="MS")
    have = set(pd.to_datetime(done["date"]).dt.to_period("M")) if len(done) else set()
    todo = [m for m in months if m.to_period("M") not in have or m.to_period("M") == pd.Timestamp.today().to_period("M")]
    for k in range(0, len(todo), 12):  # save every year so an interruption keeps what was fetched
        part = notice(todo[k : k + 12])
        done = pd.concat([done[~pd.to_datetime(done["date"]).dt.to_period("M").isin([m.to_period("M") for m in todo[k : k + 12]])], part])
        done.to_csv(n, index=False)
    print(f"完成：處置 {len(pd.read_csv(p))} 筆，注意 {len(done)} 筆", flush=True)


if __name__ == "__main__":
    main()
