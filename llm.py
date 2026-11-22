"""
llm.py — 呼叫模型的唯一入口。

為什麼要包一層，而不是在各處直接呼叫 anthropic？
  1. 每次呼叫都自動記錄 token 用量（算成本用）。
  2. 測試時可以換成 FakeLLM，不用花錢、不用網路，也能測整個流程。
  3. 可以換成別家模型：Gemini 的轉接層在 llm_gemini.py。

agent_loop.py 只認得 `await llm.create(...)` 這個介面，
不管背後是 Claude、Gemini 還是 FakeLLM。

要用哪一家，由 .env 的 LLM_PROVIDER 決定（claude 或 gemini），其他程式呼叫 make_llm() 就好。
"""

import os
from types import SimpleNamespace

import anthropic

import config
from tracing import Trace


def make_llm():
    """
    依照 config.LLM_PROVIDER 建立模型。
    對應的 API 金鑰沒設定時回傳 None，呼叫端會自動降級成 V0（只用分類器）。
    """
    if config.LLM_PROVIDER == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            return None
        from llm_gemini import GeminiLLM  # 用到才載入，沒裝 Google 套件也不影響 Claude 版本
        return GeminiLLM()

    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    return ClaudeLLM()


class ClaudeLLM:
    """真正呼叫 Anthropic API。需要環境變數 ANTHROPIC_API_KEY。"""

    def __init__(self):
        self.client = anthropic.AsyncAnthropic()

    async def create(
        self, *, model: str, system: str, messages: list, tools: list, trace: Trace, agent_name: str = ""
    ):
        # agent_name 只有 FakeLLM 會用到，這裡收下但不使用，讓兩邊的介面一致
        response = await self.client.messages.create(
            model=model,
            max_tokens=1500,
            system=system,
            messages=messages,
            tools=tools,
        )
        trace.add_usage(model, response.usage.input_tokens, response.usage.output_tokens)
        return response


# ---------------------------------------------------------------------------
# 以下是測試用的假模型
# ---------------------------------------------------------------------------


def fake_tool_call(name: str, tool_input: dict, call_id: str = "call_1"):
    """做出一個長得像 Claude 回傳的「我要呼叫工具」區塊。"""
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=tool_input)


def fake_text(text: str):
    return SimpleNamespace(type="text", text=text)


class FakeLLM:
    """
    照劇本回答的假模型。

    用法：
        fake = FakeLLM({
            "router": [ [fake_tool_call("submit_routing", {...})] ],
            "link_checker": [ [fake_tool_call("check_blacklist", {...})],
                              [fake_tool_call("submit_finding", {...})] ],
        })
    key 是 agent 名稱，value 是「依序回傳的回應清單」，每個回應是一串內容區塊。
    """

    def __init__(self, scripts: dict[str, list[list]]):
        # 複製一份，避免修改到呼叫端的資料
        self.scripts = {name: list(replies) for name, replies in scripts.items()}
        self.calls: list[str] = []  # 記錄被哪些 agent 呼叫過，方便測試檢查

    async def create(self, *, model, system, messages, tools, trace, agent_name: str = ""):
        self.calls.append(agent_name)
        replies = self.scripts.get(agent_name)
        if not replies:
            raise RuntimeError(f"FakeLLM 沒有為 agent「{agent_name}」準備回應")
        content = replies.pop(0)
        trace.add_usage(model, 100, 50)  # 假裝用了一些 token
        stop_reason = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
        return SimpleNamespace(content=content, stop_reason=stop_reason)
