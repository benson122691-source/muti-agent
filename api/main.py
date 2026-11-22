"""
api/main.py — 網頁後端（FastAPI）。

啟動：
    uvicorn api.main:app --reload --port 8000
然後打開 http://localhost:8000

端點：
  GET  /               網頁介面（web/index.html）
  POST /api/check      查核訊息
  POST /api/feedback   使用者回報判斷錯誤
  GET  /api/health     健康檢查（部署平台用來確認服務活著）
"""

import json
import time
from datetime import date
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

import config
from agents.pipeline import check
from llm import make_llm
from privacy import mask_pii
from schemas import CheckRequest, FeedbackRequest

app = FastAPI(title="詐騙訊息多 Agent 查核系統")
WEB_DIR = Path(__file__).parent.parent / "web"

# 依 LLM_PROVIDER 建立 Claude 或 Gemini；沒有 API 金鑰時是 None，仍可運作，只是只能用 V0
llm = make_llm()

# 每日 LLM 查核次數（存在記憶體，服務重啟會歸零；上線後可改存資料庫）
_daily_count = {"date": date.today(), "count": 0}


def _pick_version(requested: str) -> tuple[str, str]:
    """決定實際要用的版本。回傳 (版本, 降級原因)。"""
    if requested == "v0":
        return "v0", ""
    if llm is None:
        key = "GEMINI_API_KEY" if config.LLM_PROVIDER == "gemini" else "ANTHROPIC_API_KEY"
        return "v0", f"未設定 {key}，已改用 V0（只用分類器）"

    if _daily_count["date"] != date.today():
        _daily_count.update(date=date.today(), count=0)
    if _daily_count["count"] >= config.MAX_LLM_CHECKS_PER_DAY:
        return "v0", "已達今日 LLM 查核上限，已改用 V0（只用分類器）"

    _daily_count["count"] += 1
    return requested, ""


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "llm_enabled": llm is not None, "provider": config.LLM_PROVIDER}


@app.post("/api/check")
async def api_check(req: CheckRequest):
    version, downgrade_reason = _pick_version(req.version)
    started = time.time()
    report, trace = await check(req.text, version, llm)
    return {
        "trace_id": trace.trace_id,
        "version_used": version,
        "downgrade_reason": downgrade_reason,
        "elapsed_sec": round(time.time() - started, 2),
        "tokens": trace.total_tokens(),
        "cost_twd": round(trace.cost_twd(), 3),
        "report": report.model_dump(),
    }


@app.post("/api/feedback")
def api_feedback(req: FeedbackRequest):
    """回報先存起來，等人工複核後再加入案例庫和訓練資料（見 docs/程式導讀.md 的回饋流程）。"""
    config.FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": time.time(), **req.model_dump()}
    record["note"] = mask_pii(record["note"])
    with config.FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True}
