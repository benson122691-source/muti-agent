"""
tests/test_v4.py — 測試 V4 精簡版：程式跑工具的規則（evidence.py）和只呼叫 1 次模型的判讀。

隔離抓取服務（sandbox）在測試時沒有啟動，所以把抓網頁、查網域年齡換成假的回傳值。
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

import config
from agents import evidence
from agents.evidence import check_cases, check_identity, check_links, check_script, gather_evidence
from agents.pipeline import check
from api import main
from llm import FakeLLM, fake_tool_call

SCAM_TEXT = "名師帶你抓AI飆股，保證獲利30%，加LINE ID: stock888 https://ai-stock-demo.example/join"


@pytest.fixture(autouse=True)
def fake_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)
    monkeypatch.setattr(evidence, "fetch_page_safely", lambda url: {"final_domain": "example.com", "title": "首頁"})
    monkeypatch.setattr(evidence, "domain_age", lambda domain: {"age_days": 3000, "is_new_domain": False})


def run(coro):
    return asyncio.run(coro)


def draft(risk="high", scam_type="投資"):
    return fake_tool_call("submit_report_draft", {
        "risk_level": risk, "scam_type": scam_type, "summary": "測試摘要", "advice": "請撥打 165",
    })


# ---------------------------------------------------------------------------
# 規則：每一類查核
# ---------------------------------------------------------------------------


def test_script_rule_flags_multiple_tactics():
    assert check_script("限時加入，保證獲利，入金USDT").signal == "scam"
    assert check_script("明天早上九點開會").signal == "clean"


def test_case_rule():
    assert check_cases("AI妖股免費贈書加官方LINE").signal == "scam"
    assert check_cases("週末要不要一起去吃火鍋").signal == "unknown"  # 不像已知案例 ≠ 安全


def test_link_rule_blacklist_hit():
    finding = check_links(SCAM_TEXT)
    assert finding.signal == "scam" and finding.confidence == 0.95


def test_link_rule_new_domain_with_password_page(monkeypatch):
    monkeypatch.setattr(evidence, "fetch_page_safely", lambda url: {"final_domain": "new.example", "has_password_field": True})
    monkeypatch.setattr(evidence, "domain_age", lambda domain: {"age_days": 10, "is_new_domain": True})
    finding = check_links("請登入 https://new.example/login")
    assert finding.signal == "scam" and finding.confidence == 0.7


def test_link_rule_sandbox_down_is_unknown(monkeypatch):
    monkeypatch.setattr(evidence, "fetch_page_safely", lambda url: {"error": "隔離抓取服務無法使用"})
    assert check_links("看這裡 https://unknown.example").signal == "unknown"


def test_link_rule_incomplete_url_is_unknown(monkeypatch):
    """被切斷的網址不能拿去查：查了只會得到「查不到」，誤導判斷。"""
    calls = []
    monkeypatch.setattr(evidence, "fetch_page_safely", lambda url: calls.append(url) or {})
    finding = check_links("請點 https://muti-agent-")
    assert finding.signal == "unknown"
    assert "不完整" in finding.evidence[0]
    assert calls == []  # 完全沒有去抓網頁


def test_link_rule_no_url_skipped():
    assert check_links("沒有網址的訊息") is None


def test_identity_rule():
    assert check_identity("我們是王氏投顧，帶你穩賺").signal == "scam"     # 查不到合法登記
    assert check_identity("我們是範例投顧，帶你穩賺").signal == "unknown"  # 合法，但不代表沒被冒名
    assert check_identity("明天早上九點開會") is None


def test_gather_evidence_runs_relevant_checks():
    findings = run(gather_evidence(SCAM_TEXT))
    assert set(findings) == {"script_analyzer", "case_matcher", "link_checker", "identity_checker"}
    assert set(run(gather_evidence("明天早上九點開會"))) == {"script_analyzer", "case_matcher"}


# ---------------------------------------------------------------------------
# V4 流程
# ---------------------------------------------------------------------------


def test_v4_calls_model_once():
    fake = FakeLLM({"judge": [[draft("high")]]})
    report, trace = run(check(SCAM_TEXT, "v4", llm=fake))

    assert fake.calls == ["judge"]  # 整個查核只呼叫模型 1 次
    assert report.risk_level == "high"
    assert {f.agent for f in report.findings} == {"script_analyzer", "case_matcher", "link_checker", "identity_checker"}


def test_v4_judge_sees_evidence():
    """判讀 agent 收到的任務裡要有工具查到的證據。"""
    seen = {}

    class SpyLLM(FakeLLM):
        async def create(self, **kwargs):
            seen["task"] = kwargs["messages"][0]["content"]
            return await super().create(**kwargs)

    run(check(SCAM_TEXT, "v4", llm=SpyLLM({"judge": [[draft()]]})))
    assert "<evidence>" in seen["task"] and "黑名單" in seen["task"]


def test_v4_judge_failure_falls_back_to_rules():
    report, _ = run(check(SCAM_TEXT, "v4", llm=FakeLLM({})))  # 沒有劇本 → 判讀失敗
    assert report.risk_level == "high"  # 黑名單命中（0.95）→ 規則判斷為 high
    assert "規則產生" in report.summary


# ---------------------------------------------------------------------------
# API：未完成的查核要標示出來
# ---------------------------------------------------------------------------


def test_api_marks_complete_and_incomplete(monkeypatch):
    client = TestClient(main.app)

    monkeypatch.setattr(main, "llm", FakeLLM({"judge": [[draft()]]}))
    data = client.post("/api/check", json={"text": SCAM_TEXT}).json()  # 不指定版本 → 預設 V4
    assert data["version_used"] == "v4" and data["complete"] is True

    monkeypatch.setattr(main, "llm", None)  # 沒有金鑰 → 被迫用 V0
    data = client.post("/api/check", json={"text": SCAM_TEXT, "version": "v4"}).json()
    assert data["version_used"] == "v0" and data["complete"] is False

    data = client.post("/api/check", json={"text": SCAM_TEXT, "version": "v0"}).json()
    assert data["complete"] is True  # 使用者自己選 V0，不算未完成
