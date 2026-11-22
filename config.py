"""
config.py — 全專案的設定都集中在這裡。

想調整模型、步數上限、檔案路徑，只要改這個檔案（或在 .env 設定同名的環境變數）。
其他檔案一律從這裡 import，不要在程式各處寫死數字。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # 讀取專案根目錄的 .env 檔（如果有的話）

# ---------- 路徑 ----------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # 爬蟲下載下來的原始資料
SEED_DIR = DATA_DIR / "seed"        # 專案內附的範例資料（開發用）
MODEL_PATH = ROOT / "ml" / "model.joblib"   # 訓練好的話術分類器
TRACE_DIR = ROOT / "logs" / "traces"        # 每次查核的完整記錄
FEEDBACK_PATH = DATA_DIR / "feedback.jsonl" # 使用者回報「判斷錯了」

# ---------- 模型 ----------
# 用哪一家的模型：claude 或 gemini（對應的轉接層在 llm.py 和 llm_gemini.py）
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude").lower()

# 主要判斷（彙整、審查、單一 agent）用較強的模型；
# 分流和查核 agent 的工作比較單純，用便宜快速的模型。
# 沒在 .env 指定的話，依照 LLM_PROVIDER 用下面的預設值。
_DEFAULT_MODELS = {
    "claude": ("claude-sonnet-5-5", "claude-haiku-5-5"),
    "gemini": ("gemini-3.8-flash", "gemini-3.8-flash"),
}
_default_strong, _default_fast = _DEFAULT_MODELS.get(LLM_PROVIDER, _DEFAULT_MODELS["claude"])
MODEL_STRONG = os.getenv("MODEL_STRONG", _default_strong)
MODEL_FAST = os.getenv("MODEL_FAST", _default_fast)

# 每百萬 token 的價格（美元）。請到官方價格頁（Anthropic 或 Google）查最新數字後填入 .env。
# 用 Gemini 免費方案的話維持 0 即可。
# 沒填的話，評估報告只會顯示 token 數，不會算出金額。
PRICE_INPUT_PER_MTOK = {
    MODEL_STRONG: float(os.getenv("PRICE_STRONG_INPUT", "0")),
    MODEL_FAST: float(os.getenv("PRICE_FAST_INPUT", "0")),
}
PRICE_OUTPUT_PER_MTOK = {
    MODEL_STRONG: float(os.getenv("PRICE_STRONG_OUTPUT", "0")),
    MODEL_FAST: float(os.getenv("PRICE_FAST_OUTPUT", "0")),
}
USD_TO_TWD = float(os.getenv("USD_TO_TWD", "32"))

# ---------- Agent 限制（防止繞圈子、亂花錢） ----------
MAX_STEPS_PER_AGENT = 6        # 每個 agent 最多跟模型來回幾次
MAX_REVIEW_RETRIES = 1         # 審查不通過時，最多退回重查幾次
MAX_LLM_CHECKS_PER_DAY = int(os.getenv("MAX_LLM_CHECKS_PER_DAY", "200"))
# 超過每日上限時，API 自動降級成 V0（只用 ML 分類器），不會直接停機。

# ---------- 隔離抓取服務 ----------
# 可疑網頁只能由這個獨立服務去打開（見 sandbox/service.py）。
SANDBOX_URL = os.getenv("SANDBOX_URL", "http://localhost:8001")
SANDBOX_TIMEOUT_SEC = 15

# ---------- 開放資料下載網址 ----------
# 到 data.gov.tw 的資料集頁面，對「CSV 下載」按右鍵複製連結，貼到 .env。
URL_165_FAKE_INVEST_CSV = os.getenv("URL_165_FAKE_INVEST_CSV", "")
URL_165_SUSPENDED_DOMAINS_CSV = os.getenv("URL_165_SUSPENDED_DOMAINS_CSV", "")

# ---------- 風險判斷門檻 ----------
# V0（純 ML）用分類器的機率直接決定風險等級。
V0_HIGH_THRESHOLD = 0.75
V0_MEDIUM_THRESHOLD = 0.45
