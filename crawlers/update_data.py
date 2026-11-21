"""
crawlers/update_data.py — 下載 165 開放資料，更新黑名單。

執行方式：
    python -m crawlers.update_data

GitHub Actions 每天會自動跑一次（見 .github/workflows/update-data.yml）。

設定方法：
  1. 打開資料集頁面，例如 https://data.gov.tw/dataset/160055
  2. 在「CSV」下載按鈕上按右鍵 → 複製連結
  3. 貼到 .env 的 URL_165_FAKE_INVEST_CSV
  另一個資料集「165反詐騙諮詢專線_遭停止解析涉詐網站」範圍更廣，
  用一樣的方式貼到 URL_165_SUSPENDED_DOMAINS_CSV。

安全機制：
  新下載的筆數如果比舊檔少了一半以上，可能是來源出問題（網站改版、下載不完整），
  這時保留舊檔、不覆蓋，並以錯誤碼結束，讓排程發出告警。
"""

import csv
import io
import sys

import httpx

import config

SOURCES = {
    "165_fake_invest.csv": config.URL_165_FAKE_INVEST_CSV,
    "165_suspended_domains.csv": config.URL_165_SUSPENDED_DOMAINS_CSV,
}


def count_rows(text: str) -> int:
    return sum(1 for _ in csv.DictReader(io.StringIO(text)))


def update_one(filename: str, url: str) -> bool:
    """下載一個資料集。成功回傳 True。"""
    if not url:
        print(f"[略過] {filename}：.env 沒有設定下載網址")
        return True

    try:
        response = httpx.get(url, timeout=60, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as e:
        print(f"[失敗] {filename}：{e}")
        return False

    text = response.content.decode("utf-8-sig")
    new_rows = count_rows(text)
    path = config.RAW_DIR / filename

    if path.exists():
        old_rows = count_rows(path.read_text(encoding="utf-8-sig"))
        if new_rows < old_rows * 0.5:
            print(f"[異常] {filename}：新檔 {new_rows} 筆，舊檔 {old_rows} 筆，保留舊檔")
            return False

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[完成] {filename}：{new_rows} 筆")
    return True


def main() -> None:
    results = [update_one(name, url) for name, url in SOURCES.items()]
    if not all(results):
        sys.exit(1)  # 讓 GitHub Actions 標示為失敗，你會收到通知


if __name__ == "__main__":
    main()
