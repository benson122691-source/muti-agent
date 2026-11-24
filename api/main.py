"""
api/main.py — 網頁後端（FastAPI）。

啟動：
    uvicorn api.main:app --reload --port 8000
然後打開 http://localhost:8000

端點：
  GET  /                  網頁介面（web/index.html）
  POST /api/check         查核文字訊息
  POST /api/check-image   查核截圖：先讀出圖片裡的文字，再走跟 /api/check 一樣的流程
  POST /api/feedback      使用者回報判斷錯誤
  GET  /api/health        健康檢查（部署平台用來確認服務活著）
"""

import asyncio
import json
import logging
import time
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import config
from agents.pipeline import check
from llm import make_llm
from privacy import mask_pii
from schemas import CheckRequest, FeedbackRequest
from tracing import Trace
from vision import MAX_IMAGE_BYTES, ImageError, read_screenshot

# 讓 tracing.py 的每一步都印到伺服器記錄
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

app = FastAPI(title="詐騙訊息多 Agent 查核系統")
WEB_DIR = Path(__file__).parent.parent / "web"
VERSIONS = ("v0", "v1", "v2", "v3", "v4")

# 依 LLM_PROVIDER 建立 Claude 或 Gemini；沒有 API 金鑰時是 None，仍可運作，只是只能用 V0
llm = make_llm()

# 每日 LLM 查核次數（存在記憶體，服務重啟會歸零；上線後可改存資料庫）
_daily_count = {"date": date.today(), "count": 0}


def _daily_limit_reached() -> bool:
    if _daily_count["date"] != date.today():
        _daily_count.update(date=date.today(), count=0)
    return _daily_count["count"] >= config.MAX_LLM_CHECKS_PER_DAY


def _pick_version(requested: str) -> tuple[str, str]:
    """決定實際要用的版本。回傳 (版本, 降級原因)。"""
    if requested == "v0":
        return "v0", ""
    if llm is None:
        key = "GEMINI_API_KEY" if config.LLM_PROVIDER == "gemini" else "ANTHROPIC_API_KEY"
        return "v0", f"未設定 {key}，已改用 V0（只用分類器）"
    if _daily_limit_reached():
        return "v0", "已達今日 LLM 查核上限，已改用 V0（只用分類器）"

    _daily_count["count"] += 1
    return requested, ""


async def _run_check(text: str, requested_version: str, trace: Trace | None = None) -> dict:
    """文字查核和截圖查核共用的部分：選版本、執行、出錯就降級成 V0。"""
    version, downgrade_reason = _pick_version(requested_version)
    started = time.time()
    try:
        # wait_for：超過時間上限就中斷，不讓使用者一直等
        report, trace = await asyncio.wait_for(check(text, version, llm, trace), timeout=config.CHECK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        downgrade_reason = f"查核超過 {config.CHECK_TIMEOUT_SEC} 秒，模型可能太忙，已改用 V0（只用分類器），請稍後再試"
        version = "v0"
        report, trace = await check(text, "v0", llm, trace)
    except Exception as e:  # noqa: BLE001
        # 例如分流或單一 agent 遇到模型暫時無法使用：改用 V0，讓使用者至少拿到結果
        downgrade_reason = f"模型暫時無法使用（{e.__class__.__name__}），已改用 V0（只用分類器），請稍後再試"
        version = "v0"
        report, trace = await check(text, "v0", llm, trace)
    return {
        "trace_id": trace.trace_id,
        "version_used": version,
        # complete=False 代表「使用者要的版本沒跑完，被迫改用 V0」，網頁不會把結果當成正式結論顯示
        "complete": not downgrade_reason,
        "downgrade_reason": downgrade_reason,
        "elapsed_sec": round(time.time() - started, 2),
        "tokens": trace.total_tokens(),
        "cost_twd": round(trace.cost_twd(), 3),
        "report": report.model_dump(),
    }


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": message})


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "llm_enabled": llm is not None, "provider": config.LLM_PROVIDER}


@app.post("/api/check")
async def api_check(req: CheckRequest):
    return await _run_check(req.text, req.version)


@app.post("/api/check-image")
async def api_check_image(image: UploadFile = File(...), version: str = Form("v4")):
    """
    截圖查核。和文字查核的差別只有第一步：先用模型把圖片轉成文字。
    讀圖一定要用模型，所以沒有 API 金鑰或超過每日上限時，直接請使用者改貼文字。
    """
    if version not in VERSIONS:
        return _error(422, f"版本只能是 {', '.join(VERSIONS)}")
    if llm is None:
        return _error(400, "截圖查核需要模型讀圖，目前未設定 API 金鑰，請改為貼上文字")
    if _daily_limit_reached():
        return _error(429, "已達今日查核上限，請明天再試，或改為貼上文字（會用 V0 查核）")

    # 最多讀「上限 + 1 byte」：超過上限就知道太大，不用把整個超大檔案讀進記憶體
    data = await image.read(MAX_IMAGE_BYTES + 1)
    trace = Trace()
    try:
        text = await asyncio.wait_for(read_screenshot(data, llm, trace), timeout=config.LLM_REQUEST_TIMEOUT_SEC * 2)
    except ImageError as e:
        return _error(400, str(e))
    except Exception as e:  # noqa: BLE001 — 包含逾時、模型太忙
        trace.log("image_reader", "error", f"讀圖失敗：{e}")
        return _error(503, "讀取圖片失敗，模型可能暫時太忙，請稍後再試，或改為貼上文字")
    finally:
        del data  # 圖片讀完就丟，不存檔

    result = await _run_check(text, version, trace)
    result["extracted_text"] = text  # 讓使用者確認模型讀到的內容對不對
    return result


@app.post("/api/feedback")
def api_feedback(req: FeedbackRequest):
    """回報先存起來，等人工複核後再加入案例庫和訓練資料（見 docs/程式導讀.md 的回饋流程）。"""
    config.FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": time.time(), **req.model_dump()}
    record["note"] = mask_pii(record["note"])
    with config.FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True}
