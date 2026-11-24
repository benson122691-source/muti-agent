"""
tools/extract.py — 從訊息裡抽出網址、網域、LINE ID 等資訊。

這些都是純文字處理，不連網路、不呼叫模型，所以又快又便宜。
"""

import re
from urllib.parse import urlparse

_URL = re.compile(r"https?://[^\s<>\"'，。！？）)]+|www\.[^\s<>\"'，。！？）)]+", re.IGNORECASE)
_LINE_ID = re.compile(r"(?:line\s*id|賴|加line|line)\s*[:：@]?\s*@?([a-zA-Z0-9._-]{4,20})", re.IGNORECASE)
_PHONE = re.compile(r"09\d{2}[-\s]?\d{3}[-\s]?\d{3}")

# 截圖裡的長網址常被換行切斷，例如：
#   https://muti-agent-
#   production.up.railway.app
# 規則：網址結尾是 - . / 這種「明顯沒寫完」的字元，下一行又是英數字開頭，就把兩行接起來
_BROKEN_URL = re.compile(r"((?:https?://|www\.)[^\s]*[-./_])[ \t]*\r?\n[ \t]*([A-Za-z0-9][^\s]*)", re.IGNORECASE)

# 網域的最後一段（.com、.tw、.app…）至少要 2 個英文字母，才算完整的網址
_COMPLETE_HOST = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}$")
# 直接用 IP 當網址（例如 http://1.2.3.4/login）也算完整；詐騙網站偶爾會這樣做
_IP_HOST = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

# 常見短網址服務：真正的目的地被藏起來，需要展開才知道
SHORT_URL_DOMAINS = {"bit.ly", "reurl.cc", "tinyurl.com", "lihi.cc", "lihi1.cc", "ppt.cc", "t.co", "goo.su"}


def rejoin_broken_urls(text: str) -> str:
    """把被換行切斷的網址接回一行。可能斷不只一次，所以重複到沒有可以接的為止。"""
    while True:
        joined = _BROKEN_URL.sub(r"\1\2", text)
        if joined == text:
            return joined
        text = joined


def is_complete_url(url: str) -> bool:
    """
    網址看起來完整嗎？例如 https://muti-agent- 就不完整（沒有 .com 之類的結尾）。
    不完整的網址查黑名單、抓網頁都沒有意義，只會得到「查不到」。
    """
    host = get_domain(url)
    if _IP_HOST.match(host):
        return True
    return bool(_COMPLETE_HOST.match(host)) and not host.endswith("-")


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
    """
    給連結查核用的工具：列出訊息中所有網址和它們的網域。
    is_complete=False 的網址可能被切斷了，不要拿去查黑名單或抓網頁。
    """
    urls = extract_urls(rejoin_broken_urls(text))
    return {
        "urls": [
            {
                "url": u,
                "domain": get_domain(u),
                "is_short_url": get_domain(u) in SHORT_URL_DOMAINS,
                "is_complete": is_complete_url(u),
            }
            for u in urls
        ]
    }
