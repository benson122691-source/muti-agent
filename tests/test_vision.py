"""
tests/test_vision.py — 測試截圖查核：圖片檢查、讀圖、上傳 API。

不需要真的圖片辨識：用 FakeLLM 假裝「讀到了某段文字」，測的是我們自己的程式邏輯。
"""

import asyncio
import base64

import pytest
from fastapi.testclient import TestClient
from google.genai import types

import config
from api import main
from llm import FakeLLM, claude_image_message
from llm_gemini import GeminiLLM
from tests.test_gemini import FakeGeminiClient
from tracing import Trace
from vision import ImageError, detect_image_type, read_screenshot, validate_image

# 只要開頭的「特徵碼」對，就會被當成 PNG / JPG（測試不需要真正的圖片內容）
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
JPG = b"\xff\xd8\xff" + b"\x00" * 100
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 100

READ_RESULT = "【圖片類型】LINE 對話截圖\n【文字內容】名師帶你抓飆股，保證獲利30%，加LINE\n【網址】無"


# ---------------------------------------------------------------------------
# 圖片檢查
# ---------------------------------------------------------------------------


def test_detect_image_type_by_content():
    assert detect_image_type(PNG) == "image/png"
    assert detect_image_type(JPG) == "image/jpeg"
    assert detect_image_type(WEBP) == "image/webp"
    assert detect_image_type(b"%PDF-1.7 not an image") is None


def test_validate_rejects_bad_images():
    with pytest.raises(ImageError, match="沒有收到"):
        validate_image(b"")
    with pytest.raises(ImageError, match="只支援"):
        validate_image(b"GIF89a" + b"\x00" * 100)
    with pytest.raises(ImageError, match="太大"):
        validate_image(PNG + b"\x00" * (config.MAX_IMAGE_MB * 1024 * 1024))


def test_read_screenshot_returns_model_text():
    fake = FakeLLM({"image_reader": [READ_RESULT]})
    text = asyncio.run(read_screenshot(PNG, fake, Trace()))
    assert "保證獲利" in text


def test_read_screenshot_rejoins_broken_urls():
    fake = FakeLLM({"image_reader": ["【網址】https://fake-invest-\ndemo.example/join"]})
    text = asyncio.run(read_screenshot(PNG, fake, Trace()))
    assert "https://fake-invest-demo.example/join" in text


def test_read_screenshot_rejects_empty_result():
    fake = FakeLLM({"image_reader": ["   "]})
    with pytest.raises(ImageError, match="讀不到"):
        asyncio.run(read_screenshot(PNG, fake, Trace()))


# ---------------------------------------------------------------------------
# 各家模型的圖片格式
# ---------------------------------------------------------------------------


def test_gemini_sends_image_bytes_and_skips_thoughts():
    client = FakeGeminiClient([types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[
            types.Part(text="（思考過程）", thought=True),
            types.Part(text=READ_RESULT),
        ]))],
    )])
    text = asyncio.run(GeminiLLM(client=client).read_image(
        model="gemini-test", system="讀圖", prompt="請讀取", image=PNG, mime_type="image/png", trace=Trace(),
    ))
    assert text == READ_RESULT  # 思考過程不能混進讀圖結果

    image_part = client.requests[0]["contents"][0].parts[0]
    assert image_part.inline_data.mime_type == "image/png"
    assert image_part.inline_data.data == PNG


def test_claude_image_message_format():
    message = claude_image_message("請讀取", PNG, "image/png")
    image_block = message["content"][0]
    assert image_block["type"] == "image"
    assert image_block["source"]["media_type"] == "image/png"
    assert base64.b64decode(image_block["source"]["data"]) == PNG


# ---------------------------------------------------------------------------
# 上傳 API
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path / "traces")
    return TestClient(main.app)


def upload(client, data=PNG, version="v0"):
    return client.post("/api/check-image", files={"image": ("shot.png", data, "image/png")}, data={"version": version})


def test_check_image_reads_then_checks(client, monkeypatch):
    fake = FakeLLM({"image_reader": [READ_RESULT]})
    monkeypatch.setattr(main, "llm", fake)
    response = upload(client)  # 用 V0 查核，才不用替每個 agent 寫劇本

    assert response.status_code == 200
    data = response.json()
    assert "保證獲利" in data["extracted_text"]
    assert data["version_used"] == "v0"
    assert data["report"]["risk_level"] in ("low", "medium", "high")
    assert fake.calls == ["image_reader"]


def test_check_image_needs_model(client, monkeypatch):
    monkeypatch.setattr(main, "llm", None)
    response = upload(client)
    assert response.status_code == 400
    assert "API 金鑰" in response.json()["detail"]


def test_check_image_rejects_non_image(client, monkeypatch):
    monkeypatch.setattr(main, "llm", FakeLLM({"image_reader": [READ_RESULT]}))
    response = upload(client, data=b"%PDF-1.7 not an image")
    assert response.status_code == 400
    assert "只支援" in response.json()["detail"]


def test_check_image_model_failure_returns_503(client, monkeypatch):
    monkeypatch.setattr(main, "llm", FakeLLM({}))  # 沒有讀圖劇本 → 模擬模型失敗
    response = upload(client)
    assert response.status_code == 503
    assert "改為貼上文字" in response.json()["detail"]


def test_check_image_respects_daily_limit(client, monkeypatch):
    monkeypatch.setattr(main, "llm", FakeLLM({"image_reader": [READ_RESULT]}))
    monkeypatch.setattr(config, "MAX_LLM_CHECKS_PER_DAY", 0)
    response = upload(client)
    assert response.status_code == 429


def test_check_image_rejects_unknown_version(client, monkeypatch):
    monkeypatch.setattr(main, "llm", FakeLLM({"image_reader": [READ_RESULT]}))
    assert upload(client, version="v9").status_code == 422
