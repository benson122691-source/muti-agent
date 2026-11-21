"""
tools/extract.py — 從訊息裡抽出網址、網域、LINE ID 等資訊。

這些都是純文字處理，不連網路、不呼叫模型，所以又快又便宜。
"""

import re
from urllib.parse import urlparse

_URL = re.compile(r"https?://[^\s<>\"'，。！？）)]+|www\.[^\s<>\"'，。！？）)]+", re.IGNORECASE)
_LINE_ID = re.compile(r"(?:line\s*id|賴|加line|line)\s*[:：@]?\s*@?([a-zA-Z0-9._-]{4,20})", re.IGNORECASE)
_PHONE = re.compile(r"09\d{2}[-\s]?\d{3}[-\s]?\d{3}")

# 常見短網址服務：真正的目的地被藏起來，需要展開才知道
SHORT_URL_DOMAINS = {"bit.ly", "reurl.cc", "tinyurl.com", "lihi.cc", "lihi1.cc", "ppt.cc", "t.co", "goo.su"}


def extract_urls(text: str) -> list[str]:
    urls = _URL.findall(text)
    return [u if u.lower().startswith("http") else "http://" + u for u in urls]


def get_domain(url: str) -> str:
    """https://www.Example.com/abc → example.com"""
    host = urlparse(url).hostname or ""
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def extract_contacts(text: str) -> dict:
    """抽出聯絡方式。詐騙常要求「加 LINE」把對話移到私人管道。"""
    return {
        "line_ids": sorted(set(_LINE_ID.findall(text))),
        "phone_count": len(_PHONE.findall(text)),
        "mentions_line": bool(re.search(r"line|賴", text, re.IGNORECASE)),
    }


def analyze_urls(text: str) -> dict:
    """給連結查核 agent 用的工具：列出訊息中所有網址和它們的網域。"""
    urls = extract_urls(text)
    return {
        "urls": [
            {"url": u, "domain": get_domain(u), "is_short_url": get_domain(u) in SHORT_URL_DOMAINS}
            for u in urls
        ]
    }
