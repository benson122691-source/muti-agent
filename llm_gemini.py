"""
llm_gemini.py — 讓 agent 也能用 Google Gemini。

整個專案的 agent 迴圈（agent_loop.py）是照 Claude 的格式寫的：
  - 工具：{"name", "description", "input_schema"}
  - 模型回覆：一串區塊，每個區塊是 type="text" 或 type="tool_use"
  - 工具結果：{"type": "tool_result", "tool_use_id", "content"}

Gemini 的格式不一樣，所以這個檔案是一個「轉接頭」：
  送出前：Claude 格式 → Gemini 格式
  收到後：Gemini 格式 → Claude 格式

agent_loop.py 完全不用改，它根本不知道背後換成了 Gemini。

一個 Gemini 3 特有的重點：思考簽章（thought signature）
  Gemini 呼叫工具時，回覆裡會附一段加密的「思考簽章」。下一輪必須把它原封不動送回去，
  否則 API 會回 400 錯誤。所以我們把 Gemini 原本的 Part 物件存在區塊的 raw_part 裡，
  送回去時直接用原本的 Part，不要自己重組。
"""

import asyncio
import json
import os
import uuid
from types import SimpleNamespace

from google import genai
from google.genai import errors, types

from tracing import Trace

# 這些錯誤通常等一下就會好，值得重試：
#   429 = 超過使用次數限制（免費方案很常遇到）
#   500 = Google 伺服器內部錯誤
#   503 = 模型暫時太忙（high demand）
RETRYABLE_CODES = {429, 500, 503}
MAX_RETRIES = 3          # 最多重試幾次
RETRY_WAIT_SEC = 5       # 第一次重試前等幾秒，之後每次加倍（5、10、20 秒）


class GeminiLLM:
    """呼叫 Gemini API。需要環境變數 GEMINI_API_KEY（在 Google AI Studio 申請）。"""

    def __init__(self, client: genai.Client | None = None):
        self.client = client or genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    async def create(
        self, *, model: str, system: str, messages: list, tools: list, trace: Trace, agent_name: str = ""
    ):
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=[to_gemini_tool(t) for t in tools])],
            # 關掉 SDK 的「自動執行工具」：工具要由我們的 agent_loop 來執行和記錄
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            max_output_tokens=4000,  # Gemini 的思考過程也算輸出 token，所以給多一點
        )
        response = await self._call_with_retry(model, to_gemini_contents(messages), config, trace, agent_name)

        usage = response.usage_metadata
        if usage:
            output = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
            trace.add_usage(model, usage.prompt_token_count or 0, output)
        return to_claude_response(response)

    async def _call_with_retry(self, model, contents, config, trace: Trace, agent_name: str):
        wait = RETRY_WAIT_SEC
        for attempt in range(MAX_RETRIES + 1):
            try:
                return await self.client.aio.models.generate_content(model=model, contents=contents, config=config)
            except errors.APIError as e:
                if e.code not in RETRYABLE_CODES or attempt == MAX_RETRIES:
                    raise
                trace.log(agent_name, "error", f"Gemini 回傳 {e.code}，{wait} 秒後重試")
                await asyncio.sleep(wait)
                wait *= 2


# ---------------------------------------------------------------------------
# 格式轉換：Claude → Gemini
# ---------------------------------------------------------------------------


def to_gemini_tool(tool: dict) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=tool["name"],
        description=tool["description"],
        parameters_json_schema=inline_refs(tool["input_schema"]),
    )


def inline_refs(schema: dict) -> dict:
    """
    把 JSON Schema 裡的 $ref 展開成實際內容。

    Pydantic 產生的巢狀 schema 會長這樣：
        {"properties": {"findings": {"items": {"$ref": "#/$defs/Finding"}}}, "$defs": {"Finding": {...}}}
    Gemini 不一定支援 $ref，所以直接把 Finding 的內容貼到 $ref 的位置。
    """
    defs = schema.get("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]  # "#/$defs/Finding" → "Finding"
                return resolve(defs[name])
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


def to_gemini_contents(messages: list) -> list[types.Content]:
    """
    轉換對話紀錄。三種訊息：
      1. 使用者文字（字串）            → role="user" 的文字
      2. 模型回覆（區塊清單）          → role="model"，直接用原本的 Gemini Part（保留思考簽章）
      3. 工具結果（tool_result 清單）  → role="user" 的 function_response
    """
    contents = []
    call_info: dict[str, tuple[str, str | None]] = {}  # 我們的 id → (工具名稱, Gemini 原本的 id)

    for message in messages:
        content = message["content"]

        if isinstance(content, str):
            contents.append(types.Content(role="user", parts=[types.Part(text=content)]))

        elif message["role"] == "assistant":
            for block in content:
                if block.type == "tool_use":
                    call_info[block.id] = (block.name, block.gemini_id)
            contents.append(types.Content(role="model", parts=[block.raw_part for block in content]))

        else:  # 工具結果
            parts = []
            for result in content:
                name, gemini_id = call_info[result["tool_use_id"]]
                key = "error" if result.get("is_error") else "result"
                parts.append(types.Part(function_response=types.FunctionResponse(
                    id=gemini_id, name=name, response={key: _maybe_json(result["content"])},
                )))
            contents.append(types.Content(role="user", parts=parts))

    return contents


def _maybe_json(text: str):
    """工具結果是 JSON 字串時轉回物件，Gemini 比較好讀；不是 JSON 就保持文字。"""
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


# ---------------------------------------------------------------------------
# 格式轉換：Gemini → Claude
# ---------------------------------------------------------------------------


def to_claude_response(response: types.GenerateContentResponse):
    """把 Gemini 的回覆轉成 agent_loop.py 看得懂的樣子。"""
    parts = []
    if response.candidates and response.candidates[0].content:
        parts = response.candidates[0].content.parts or []

    blocks = []
    for part in parts:
        if part.function_call:
            call = part.function_call
            blocks.append(SimpleNamespace(
                type="tool_use",
                id=f"call_{uuid.uuid4().hex[:8]}",  # Gemini 不一定給 id，自己產生一個
                gemini_id=call.id,
                name=call.name,
                input=dict(call.args or {}),
                raw_part=part,
            ))
        elif part.text and not part.thought:  # 思考內容不當成回覆
            blocks.append(SimpleNamespace(type="text", text=part.text, raw_part=part))
        else:
            # 只有思考簽章、沒有內容的 Part 也要保留，送回去時才不會少簽章
            blocks.append(SimpleNamespace(type="thought", raw_part=part))

    if not any(b.type in ("tool_use", "text") for b in blocks):
        # 被安全機制擋下或回覆是空的：當成一段空文字，agent_loop 會提醒模型重交
        blocks.append(SimpleNamespace(type="text", text="", raw_part=types.Part(text="（無回覆）")))

    stop_reason = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return SimpleNamespace(content=blocks, stop_reason=stop_reason)
