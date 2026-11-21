"""
tools/blacklist.py — 查詢網域是否在涉詐黑名單上。

資料來源（依優先順序）：
  1. data/raw/ 裡由 crawlers/update_data.py 下載的 165 開放資料 CSV
  2. 如果還沒下載，改用 data/seed/blacklist_sample.csv（假網域，只供開發測試）

165 的 CSV 欄位是「網站名稱、網址、件數、統計起始日期、統計結束日期」。
"""

import csv
from functools import lru_cache

import config
from tools.extract import get_domain


@lru_cache(maxsize=1)  # 只在第一次呼叫時讀檔，之後直接用記憶體裡的結果
def _load_blacklist() -> dict[str, str]:
    """回傳 {網域: 資料來源}。"""
    files = sorted(config.RAW_DIR.glob("165_*.csv"))
    if not files:
        files = [config.SEED_DIR / "blacklist_sample.csv"]

    domains: dict[str, str] = {}
    for path in files:
        with path.open(encoding="utf-8-sig") as f:  # utf-8-sig 可處理政府 CSV 常見的 BOM
            for row in csv.DictReader(f):
                raw = row.get("網址") or row.get("網域") or ""
                if not raw:
                    continue
                url = raw if raw.startswith("http") else "http://" + raw
                domain = get_domain(url)
                if domain:
                    domains[domain] = path.name
    return domains


def check_blacklist(domain: str) -> dict:
    """查一個網域。也會比對上層網域，例如 a.scam.com 會命中 scam.com。"""
    blacklist = _load_blacklist()
    domain = domain.lower().strip()
    parts = domain.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in blacklist:
            return {"domain": domain, "listed": True, "matched": candidate, "source": blacklist[candidate]}
    return {"domain": domain, "listed": False, "checked_entries": len(blacklist)}
