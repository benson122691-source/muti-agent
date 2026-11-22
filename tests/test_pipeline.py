"""
tests/test_pipeline.py — 用 FakeLLM 測試 agent 流程。

不需要 API 金鑰、不花錢。FakeLLM 會照我們寫好的「劇本」回答，
所以測的是「我們的程式邏輯對不對」，不是「模型聰不聰明」（那是 evals/ 的工作）。

執行：pytest
"""

import asyncio

import pytest

import config
from agents.pipeline import check
from llm import FakeLLM, fake_text, fake_tool_call

SCAM_TEXT = "名師帶你抓飆股保證獲利，加LINE https://ai-stock-demo.example/join"


@pytest.fixture(autouse=True)
def tmp_traces(tmp_path, monkeypatch):
    """測試時把記錄檔寫到暫存資料夾，不要弄髒 logs/。"""
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)


def finding(agent, signal="scam", confidence=0.9):
    return {"agent": agent, "signal": signal, "confidence": confidence, "evidence": [f"{agent} 的證據"], "sources": []}


def routing(has_url=True, identity=False):
    return fake_tool_call(
        "submit_routing_decision",
        {"scam_type": "投資", "has_url": has_url, "run_identity_check": identity, "reason": "測試"},
    )


def draft(risk="high"):
    return fake_tool_call(
        "submit_report_draft",
        {"risk_level": risk, "scam_type": "投資", "summary": "測試摘要", "advice": "請撥打 165"},
    )


def review(approved, recheck=()):
    return fake_tool_call(
        "submit_review_result",
        {"approved": approved, "problems": [] if approved else ["證據不足"], "recheck_agents": list(recheck)},
    )


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------


def test_v0_runs_without_llm():
    report, trace = run(check(SCAM_TEXT, "v0", llm=None))
    assert report.findings[0].agent == "ml_classifier"
    assert "保證獲利" in " ".join(report.findings[0].evidence)
    assert trace.total_tokens() == 0  # V0 不應該用到任何 token


def test_v2_runs_checkers_and_aggregates():
    fake = FakeLLM({
        "router": [[routing(has_url=True)]],
        # 連結查核：先呼叫一個工具，再提交結果 → 測試工具迴圈有正常運作
        "link_checker": [
            [fake_tool_call("check_blacklist", {"domain": "ai-stock-demo.example"})],
            [fake_tool_call("submit_finding", finding("link_checker"))],
        ],
        "script_analyzer": [[fake_tool_call("submit_finding", finding("script_analyzer"))]],
        "case_matcher": [[fake_tool_call("submit_finding", finding("case_matcher", "unknown", 0.3))]],
        "aggregator": [[draft("high")]],
    })
    report, trace = run(check(SCAM_TEXT, "v2", llm=fake))

    assert report.risk_level == "high"
    assert {f.agent for f in report.findings} == {"link_checker", "script_analyzer", "case_matcher"}
    assert "identity_checker" not in fake.calls  # 分流說不用查身分，就不該啟動
    # 確認黑名單工具真的被執行，而且結果有記錄下來
    tool_results = [e for e in trace.events if e["kind"] == "tool_result"]
    assert tool_results[0]["data"]["output"]["listed"] is True


def test_v2_skips_link_checker_when_no_url():
    fake = FakeLLM({
        "router": [[routing(has_url=False)]],
        "script_analyzer": [[fake_tool_call("submit_finding", finding("script_analyzer", "clean", 0.8))]],
        "case_matcher": [[fake_tool_call("submit_finding", finding("case_matcher", "unknown", 0.2))]],
        "aggregator": [[draft("low")]],
    })
    report, _ = run(check("明天早上九點開會", "v2", llm=fake))
    assert "link_checker" not in fake.calls


def test_failed_checker_becomes_unknown_instead_of_crashing():
    fake = FakeLLM({
        "router": [[routing(has_url=False)]],
        "script_analyzer": [[fake_tool_call("submit_finding", finding("script_analyzer"))]],
        "case_matcher": [],  # 沒有劇本 → FakeLLM 會丟錯，模擬 agent 當掉
        "aggregator": [[draft("medium")]],
    })
    report, _ = run(check(SCAM_TEXT.split("http")[0], "v2", llm=fake))
    failed = [f for f in report.findings if f.agent == "case_matcher"][0]
    assert failed.signal == "unknown"


def test_v3_approved_on_first_review():
    fake = FakeLLM({
        "router": [[routing(has_url=False)]],
        "script_analyzer": [[fake_tool_call("submit_finding", finding("script_analyzer"))]],
        "case_matcher": [[fake_tool_call("submit_finding", finding("case_matcher"))]],
        "aggregator": [[draft("high")]],
        "reviewer": [[review(True)]],
    })
    report, _ = run(check("保證獲利加LINE", "v3", llm=fake))
    assert report.reviewed is True


def test_v3_rejected_then_rechecks_only_named_agent():
    fake = FakeLLM({
        "router": [[routing(has_url=False)]],
        "script_analyzer": [
            [fake_tool_call("submit_finding", finding("script_analyzer", "unknown", 0.3))],
            [fake_tool_call("submit_finding", finding("script_analyzer", "scam", 0.9))],  # 重查後的結果
        ],
        "case_matcher": [[fake_tool_call("submit_finding", finding("case_matcher", "unknown", 0.2))]],
        "aggregator": [[draft("low")], [draft("high")]],
        "reviewer": [[review(False, recheck=["script_analyzer"])], [review(True)]],
    })
    report, _ = run(check("保證獲利加LINE", "v3", llm=fake))

    assert report.reviewed is True
    assert report.risk_level == "high"
    assert fake.calls.count("script_analyzer") == 2  # 被點名的重查了
    assert fake.calls.count("case_matcher") == 1     # 沒被點名的不重查


def test_v3_still_rejected_is_marked_and_not_low():
    fake = FakeLLM({
        "router": [[routing(has_url=False)]],
        "script_analyzer": [
            [fake_tool_call("submit_finding", finding("script_analyzer", "clean"))],
            [fake_tool_call("submit_finding", finding("script_analyzer", "clean"))],
        ],
        "case_matcher": [
            [fake_tool_call("submit_finding", finding("case_matcher", "unknown"))],
            [fake_tool_call("submit_finding", finding("case_matcher", "unknown"))],
        ],
        "aggregator": [[draft("low")], [draft("low")]],
        "reviewer": [[review(False)], [review(False)]],
    })
    report, _ = run(check("某則訊息", "v3", llm=fake))
    assert report.reviewed is False
    assert report.risk_level == "medium"  # 不確定時寧可保守


def test_v1_single_agent():
    report_input = {
        "risk_level": "high",
        "scam_type": "投資",
        "summary": "測試",
        "findings": [finding("single_agent")],
        "advice": "請撥打 165",
        "reviewed": False,
    }
    fake = FakeLLM({
        "single_agent": [
            [fake_tool_call("classify_script", {"text": SCAM_TEXT})],
            [fake_tool_call("submit_report", report_input)],
        ]
    })
    report, _ = run(check(SCAM_TEXT, "v1", llm=fake))
    assert report.risk_level == "high"


def test_agent_retries_after_invalid_submission():
    """模型第一次提交格式錯誤（confidence 超過 1），應該收到錯誤訊息並重交。"""
    bad = finding("script_analyzer")
    bad["confidence"] = 5
    fake = FakeLLM({
        "router": [[routing(has_url=False)]],
        "script_analyzer": [
            [fake_tool_call("submit_finding", bad)],
            [fake_tool_call("submit_finding", finding("script_analyzer"))],
        ],
        "case_matcher": [[fake_tool_call("submit_finding", finding("case_matcher"))]],
        "aggregator": [[draft("high")]],
    })
    report, _ = run(check("保證獲利", "v2", llm=fake))
    assert fake.calls.count("script_analyzer") == 2


def test_agent_reminded_when_it_only_talks():
    """模型只回文字、沒呼叫工具時，應該被提醒，而不是直接結束。"""
    fake = FakeLLM({
        "router": [[fake_text("我覺得是投資詐騙")], [routing(has_url=False)]],
        "script_analyzer": [[fake_tool_call("submit_finding", finding("script_analyzer"))]],
        "case_matcher": [[fake_tool_call("submit_finding", finding("case_matcher"))]],
        "aggregator": [[draft("high")]],
    })
    run(check("保證獲利", "v2", llm=fake))
    assert fake.calls.count("router") == 2


# ---------------------------------------------------------------------------
# 模型暫時無法使用時的備援（例如 Gemini 回 503）
# ---------------------------------------------------------------------------


def test_aggregator_failure_falls_back_to_rules():
    fake = FakeLLM({
        "router": [[routing(has_url=False)]],
        "script_analyzer": [[fake_tool_call("submit_finding", finding("script_analyzer", "scam", 0.9))]],
        "case_matcher": [[fake_tool_call("submit_finding", finding("case_matcher", "unknown", 0.2))]],
        "aggregator": [],  # 沒有劇本 → 丟錯，模擬模型暫時無法使用
    })
    report, _ = run(check("保證獲利加LINE", "v2", llm=fake))
    assert report.risk_level == "high"  # 有一個信心 0.9 的 scam 訊號
    assert report.scam_type == "投資"   # 沿用分流的判斷
    assert "規則產生" in report.summary


def test_reviewer_failure_keeps_report_unreviewed():
    fake = FakeLLM({
        "router": [[routing(has_url=False)]],
        "script_analyzer": [[fake_tool_call("submit_finding", finding("script_analyzer"))]],
        "case_matcher": [[fake_tool_call("submit_finding", finding("case_matcher"))]],
        "aggregator": [[draft("high")]],
        "reviewer": [],  # 審查 agent 失敗
    })
    report, _ = run(check("保證獲利加LINE", "v3", llm=fake))
    assert report.reviewed is False
    assert report.risk_level == "high"
