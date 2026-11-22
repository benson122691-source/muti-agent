"""
tests/test_api.py — 測試 API 端點。沒有 API 金鑰時，所有請求都應該降級成 V0。
"""

import pytest
from fastapi.testclient import TestClient

import config
from api import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path / "traces")
    monkeypatch.setattr(config, "FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(main, "llm", None)  # 模擬沒有 API 金鑰
    return TestClient(main.app)


def test_homepage(client):
    assert "詐騙訊息查核" in client.get("/").text


def test_check_falls_back_to_v0_without_key(client):
    data = client.post("/api/check", json={"text": "保證獲利加LINE", "version": "v3"}).json()
    assert data["version_used"] == "v0"
    assert "ANTHROPIC_API_KEY" in data["downgrade_reason"]
    assert data["report"]["risk_level"] in ("low", "medium", "high")


def test_empty_text_rejected(client):
    assert client.post("/api/check", json={"text": ""}).status_code == 422


def test_feedback_masks_pii(client):
    client.post("/api/feedback", json={"trace_id": "abc", "is_scam": True, "note": "他給我 0912345678"})
    saved = config.FEEDBACK_PATH.read_text(encoding="utf-8")
    assert "0912345678" not in saved


def test_check_falls_back_to_v0_when_model_fails(client, monkeypatch):
    """模型在查核途中失敗（例如分流就遇到 503），API 應該改用 V0，而不是回 500。"""
    from llm import FakeLLM
    monkeypatch.setattr(main, "llm", FakeLLM({}))  # 沒有任何劇本 → 第一個 agent 就失敗
    response = client.post("/api/check", json={"text": "保證獲利加LINE", "version": "v1"})
    assert response.status_code == 200
    data = response.json()
    assert data["version_used"] == "v0"
    assert "模型暫時無法使用" in data["downgrade_reason"]
