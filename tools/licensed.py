"""
tools/licensed.py — 查詢公司或機構是否為合法業者（白名單）。

假投資詐騙常自稱「某某投顧」「某某資產管理」。合法的投信投顧業者
可以在投信投顧公會網站（sitca.org.tw）查到。

目前讀 data/raw/licensed.csv（請從公會網站整理後放入），
沒有的話改用 data/seed/licensed_sample.csv。
CSV 只需要一個欄位：名稱。
"""

import csv
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def _load_licensed() -> list[str]:
    path = config.RAW_DIR / "licensed.csv"
    if not path.exists():
        path = config.SEED_DIR / "licensed_sample.csv"
    with path.open(encoding="utf-8-sig") as f:
        return [row["名稱"].strip() for row in csv.DictReader(f) if row.get("名稱")]


# 公司名稱裡的通用字，比對前先拿掉，只留下品牌名
_GENERIC_WORDS = ["股份有限公司", "有限公司", "證券", "投資顧問", "投顧", "投資信託", "投信", "資產管理", "期貨"]


def _brand(name: str) -> str:
    """「範例證券投資顧問股份有限公司」→「範例」；「範例投顧」→「範例」"""
    for word in _GENERIC_WORDS:
        name = name.replace(word, "")
    return name.strip()


def lookup_licensed_company(name: str) -> dict:
    """
    比對品牌名：「範例投顧」可以比對到「範例證券投資顧問股份有限公司」。
    注意：查不到不代表一定是詐騙，可能只是名單不完整，所以回傳時會提醒。
    """
    name = name.strip()
    brand = _brand(name)
    matches = [n for n in _load_licensed() if brand and brand == _brand(n)]
    return {
        "query": name,
        "found": bool(matches),
        "matches": matches[:5],
        "note": "查無資料不等於詐騙，可能名稱寫法不同或名單不完整",
    }
