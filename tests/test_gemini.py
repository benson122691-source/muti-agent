"""
tests/test_gemini.py — 測試 Gemini 轉接層（llm_gemini.py）。

不連網路：用一個假的 client 代替 Google 的伺服器，
檢查「我們送出去的格式」和「收到後轉換的結果」對不對。
"""

import asyncio
from types import SimpleNamespace

import pytest
from google.genai import errors, types

import config
import llm_gemini
from agents.definitions import make_checkers, make_single_agent
from llm import make_llm
from llm_gemini import GeminiLLM, inline_refs
from tracing import Trace


def gemini_reply(*parts, prompt_tokens=100, output_tokens=20):
    """做出一個長得像 Gemini 回傳的回應。"""
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens, candidates_token_count=output_tokens, thoughts_token_count=5
        ),
    )


def call_part(name, args, signature=None):
    return types.Part(function_call=types.FunctionCall(name=name, args=args), thought_signature=signature)


class FakeGeminiClient:
    """假的 Gemini client：依序回傳準備好的回應，並記下每次收到的請求。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    async def _generate(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    monkeypatch.setattr(llm_gemini, "RETRY_WAIT_SEC", 0)


FINDING = {"agent": "script_analyzer", "signal": "scam", "confidence": 0.9, "evidence": ["保證獲利"], "sources": []}


def test_agent_loop_works_with_gemini():
    client = FakeGeminiClient([
        gemini_reply(call_part("classify_script", {"text": "保證獲利加LINE"}, signature=b"sig-1")),
        gemini_reply(call_part("submit_finding", FINDING, signature=b"sig-2")),
    ])
    trace = Trace()
    agent = make_checkers()["script_analyzer"]
    result = asyncio.run(agent.run("<message>保證獲利加LINE</message>", GeminiLLM(client=client), trace))

    assert result.signal == "scam"
    assert len(client.requests) == 2

    # 第二次請求的對話紀錄：使用者 → 模型（呼叫工具）→ 使用者（工具結果）
    second = client.requests[1]["contents"]
    assert [c.role for c in second] == ["user", "model", "user"]
    # 思考簽章要原封不動送回去，否則 Gemini 3 會回 400 錯誤
    assert second[1].parts[0].thought_signature == b"sig-1"
    # 工具結果要帶上工具名稱，而且內容是真的分類器結果
    response = second[2].parts[0].function_response
    assert response.name == "classify_script"
    assert "scam_probability" in response.response["result"]

    # token 有記錄（輸出 = 回覆 20 + 思考 5），兩次呼叫
    assert trace.total_tokens() == 2 * (100 + 25)


def test_tool_declarations_are_sent():
    client = FakeGeminiClient([gemini_reply(call_part("submit_finding", FINDING))])
    asyncio.run(make_checkers()["script_analyzer"].run("測試", GeminiLLM(client=client), Trace()))
    sent = client.requests[0]["config"]
    names = [f.name for f in sent.tools[0].function_declarations]
    assert names == ["classify_script", "submit_finding"]
    assert sent.automatic_function_calling.disable is True  # 工具要由 agent_loop 執行


def test_nested_schema_has_no_refs():
    """單一 agent 提交的 Report 有巢狀的 Finding，送給 Gemini 前要展開。"""
    schema = make_single_agent()._submit_tool()["input_schema"]
    assert "$ref" in str(schema)
    flat = inline_refs(schema)
    assert "$ref" not in str(flat) and "$defs" not in str(flat)
    assert "evidence" in str(flat["properties"]["findings"])


def test_text_only_reply_triggers_reminder():
    """Gemini 只回文字時，agent_loop 應該提醒它用提交工具。"""
    client = FakeGeminiClient([
        gemini_reply(types.Part(text="我覺得是詐騙")),
        gemini_reply(call_part("submit_finding", FINDING)),
    ])
    asyncio.run(make_checkers()["script_analyzer"].run("測試", GeminiLLM(client=client), Trace()))
    assert len(client.requests) == 2


def test_retries_on_rate_limit():
    client = FakeGeminiClient([
        errors.ClientError(429, {"error": {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}}),
        gemini_reply(call_part("submit_finding", FINDING)),
    ])
    result = asyncio.run(make_checkers()["script_analyzer"].run("測試", GeminiLLM(client=client), Trace()))
    assert result.signal == "scam"
    assert len(client.requests) == 2


def test_retries_when_model_is_busy():
    """Gemini 回 503（模型太忙）時也要重試。"""
    client = FakeGeminiClient([
        errors.ServerError(503, {"error": {"message": "high demand", "status": "UNAVAILABLE"}}),
        gemini_reply(call_part("submit_finding", FINDING)),
    ])
    result = asyncio.run(make_checkers()["script_analyzer"].run("測試", GeminiLLM(client=client), Trace()))
    assert result.signal == "scam"


def test_daily_quota_is_not_retried():
    """每日額度用完時，重試也沒用，應該馬上放棄。"""
    quota_error = errors.ClientError(429, {"error": {
        "code": 429, "message": "You exceeded your current quota", "status": "RESOURCE_EXHAUSTED",
        "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}],
    }})
    client = FakeGeminiClient([quota_error])
    with pytest.raises(errors.ClientError):
        asyncio.run(make_checkers()["script_analyzer"].run("測試", GeminiLLM(client=client), Trace()))
    assert len(client.requests) == 1


def test_other_errors_are_not_retried():
    client = FakeGeminiClient([errors.ClientError(400, {"error": {"message": "bad request"}})])
    with pytest.raises(errors.ClientError):
        asyncio.run(make_checkers()["script_analyzer"].run("測試", GeminiLLM(client=client), Trace()))
    assert len(client.requests) == 1


def test_make_llm_picks_provider(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert make_llm() is None  # 沒金鑰 → None，系統會降級成 V0

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert isinstance(make_llm(), GeminiLLM)
