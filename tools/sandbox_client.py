"""
tools/sandbox_client.py — 透過「隔離抓取服務」查看可疑網址。

重要的安全設計：
  主程式（和所有 agent）自己永遠不直接打開可疑網頁，
  而是請 sandbox/service.py 這個獨立服務去開。那個服務：
    - 跑在另一個容器裡，沒有 API 金鑰、不能連資料庫
    - 只回傳整理過的欄位（標題、最終網址、有沒有密碼欄位…），不回傳整頁 HTML

  這樣就算網頁裡藏了「忽略之前的指示」之類的 prompt injection，
  模型也只會看到一個短短的標題字串，而且被標記成資料。

服務沒開的時候，工具會回傳錯誤訊息，agent 會把該項判斷為 unknown，系統不會當掉。
"""

import httpx

import config


def _post(path: str, payload: dict) -> dict:
    try:
        response = httpx.post(
            f"{config.SANDBOX_URL}{path}", json=payload, timeout=config.SANDBOX_TIMEOUT_SEC
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        return {"error": f"隔離抓取服務無法使用：{e.__class__.__name__}"}


def fetch_page_safely(url: str) -> dict:
    """展開短網址並抓取網頁摘要。"""
    return _post("/fetch", {"url": url})


def domain_age(domain: str) -> dict:
    """查網域註冊多久了。詐騙網站常是最近幾週才註冊的。"""
    return _post("/whois", {"domain": domain})
