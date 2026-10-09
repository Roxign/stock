"""Rules promoted from the rule lab (網站「規則實驗室」) to official strategies: every entry of rules.json becomes a
strategy with daily signals and closing-price triggers. Engine and semantics: stocklab/rules.py, mirrored by
docs/rules.js, so the numbers match the lab's.

rules.json: [{"id": "short_id", "name": "顯示名稱", "note": "optional markdown", "rule": {...rule JSON from the lab...}}]
"""

import json
from pathlib import Path

from stocklab import rules

ENTRIES = json.loads(Path(__file__).with_name("rules.json").read_text(encoding="utf-8"))


def _positions(rule):
    return lambda data: rules.positions(data, rule)


STRATEGIES = [
    {
        "id": f"lab_{e['id']}",
        "label": e["name"],
        "family": "規則實驗室",
        "description": "**規則**：" + rules.describe(e["rule"]) + "\n\n" + e.get("note", "")
                       + "\n\n可在網站「規則實驗室」開啟這個規則、修改條件，並匯出 MultiCharts 程式碼。\n",
        "multicharts": None,
        "positions": _positions(e["rule"]),
        "lab_rule": e["rule"],
    }
    for e in ENTRIES
]
