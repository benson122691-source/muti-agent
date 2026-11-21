"""
sandbox/service.py — 隔離抓取服務：唯一可以打開可疑網頁的程式。

部署時跑在獨立的容器裡（見 docker-compose.yml）：
  - 這個容器沒有 ANTHROPIC_API_KEY，也連不到資料庫
  - 只回傳整理過的欄位，絕不回傳整頁 HTML

啟動（本機開發）：
    uvicorn sandbox.service:app --port 8001

兩個端點：
  POST /fetch  {"url": "..."}     → 展開短網址、抓網頁摘要
  POST /whois  {"domain": "..."}  → 查網域註冊日期（用公開的 RDAP 服務）
"""

import ipaddress
import re
import socket
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="隔離抓取服務")

MAX_BYTES = 500_000      # 最多讀 500KB，避免被超大網頁拖垮
MAX_REDIRECTS = 5
TIMEOUT_SEC = 10
USER_AGENT = "Mozilla/5.0 (scam-checker sandbox)"


class FetchRequest(BaseModel):
    url: str


class WhoisRequest(BaseModel):
    domain: str


def is_public_host(host: str) -> bool:
    """
    防止 SSRF：不准抓內網位址（例如 localhost、192.168.x.x）。
    否則有人可以貼一個指向內網的網址，讓我們的伺服器去打內部服務。
    """
    try:
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        return True
    except socket.gaierror:
        return False


def summarize_html(html: str, final_url: str) -> dict:
    """只取出判斷需要的特徵。title 限制 80 字，降低 prompt injection 的空間。"""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip()[:80] if title_match else ""
    lowered = html.lower()
    return {
        "final_url": final_url,
        "final_domain": urlparse(final_url).hostname,
        "title": title,
        "has_password_field": 'type="password"' in lowered or "type='password'" in lowered,
        "form_count": lowered.count("<form"),
        "links_to_line": "line.me" in lowered,
        "mentions_crypto": bool(re.search(r"usdt|bitcoin|比特幣|虛擬貨幣", lowered)),
        "html_size": len(html),
    }


@app.post("/fetch")
def fetch(req: FetchRequest) -> dict:
    url = req.url if req.url.startswith("http") else "http://" + req.url
    host = urlparse(url).hostname or ""
    if not is_public_host(host):
        return {"error": "拒絕抓取：網址無法解析或指向內部網路"}

    redirect_chain = []
    try:
        with httpx.Client(follow_redirects=False, timeout=TIMEOUT_SEC, headers={"User-Agent": USER_AGENT}) as client:
            # 自己一步步跟轉址，每一步都檢查目的地是否為公開網址
            for _ in range(MAX_REDIRECTS + 1):
                response = client.get(url)
                if response.is_redirect:
                    url = str(response.next_request.url)
                    redirect_chain.append(url)
                    if not is_public_host(urlparse(url).hostname or ""):
                        return {"error": "拒絕抓取：轉址到內部網路", "redirect_chain": redirect_chain}
                    continue
                html = response.text[:MAX_BYTES]
                result = summarize_html(html, url)
                result["status_code"] = response.status_code
                result["redirect_chain"] = redirect_chain
                return result
        return {"error": "轉址次數太多", "redirect_chain": redirect_chain}
    except httpx.HTTPError as e:
        return {"error": f"無法連線：{e.__class__.__name__}", "redirect_chain": redirect_chain}


@app.post("/whois")
def whois(req: WhoisRequest) -> dict:
    domain = req.domain.lower().strip()
    try:
        response = httpx.get(f"https://rdap.org/domain/{domain}", timeout=TIMEOUT_SEC, follow_redirects=True)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError) as e:
        return {"domain": domain, "error": f"查不到註冊資料：{e.__class__.__name__}"}

    registered = next(
        (e["eventDate"] for e in data.get("events", []) if e.get("eventAction") == "registration"), None
    )
    if not registered:
        return {"domain": domain, "error": "註冊資料中沒有註冊日期"}
    reg_date = datetime.fromisoformat(registered.replace("Z", "+00:00"))
    age_days = (datetime.now(timezone.utc) - reg_date).days
    return {"domain": domain, "registered": registered[:10], "age_days": age_days, "is_new_domain": age_days < 90}
