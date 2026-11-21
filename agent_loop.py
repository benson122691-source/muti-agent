"""
agent_loop.py — 整個專案最核心的檔案：一個通用的 agent。

一個 agent ＝ 一段 system prompt ＋ 幾個工具 ＋ 一個「提交結果」的工具。

運作方式（就是之前講的 agent 迴圈）：
  1. 把任務和工具清單送給模型。
  2. 模型回傳「我要呼叫某個工具」→ 我們執行那個 Python 函式，把結果送回去。
  3. 重複第 2 步，直到模型呼叫「提交結果」工具（例如 submit_finding）。
  4. 提交的內容用 Pydantic 檢查格式，合格就回傳。

為什麼用「提交工具」來拿結果，而不是叫模型直接回一段 JSON 文字？
  因為工具有 input_schema，模型會照著欄位填，格式穩定很多；
  再加上 Pydantic 檢查，不合格就把錯誤訊息退回給模型重填。
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

import config
from tracing import Trace


@dataclass
class Tool:
    """一個工具 = 給模型看的說明 + 真正執行的 Python 函式。"""

    name: str
    description: str          # 模型靠這段文字決定要不要用這個工具，要寫清楚
    input_schema: dict        # JSON Schema，描述參數
    func: Callable[..., Any]  # 真正做事的函式，參數名稱要跟 input_schema 一致

    def to_api(self) -> dict:
        """轉成 Anthropic API 要的格式。"""
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


class Agent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[Tool],
        output_model: type[BaseModel],
        model: str,
        max_steps: int = config.MAX_STEPS_PER_AGENT,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = {t.name: t for t in tools}
        self.output_model = output_model
        self.model = model
        self.max_steps = max_steps
        # 提交工具的名稱，例如 output_model 是 Finding → submit_finding
        self.submit_name = "submit_" + _snake(output_model.__name__)

    def _submit_tool(self) -> dict:
        """根據 output_model 自動產生「提交結果」工具，欄位直接取自 Pydantic schema。"""
        return {
            "name": self.submit_name,
            "description": "完成任務時呼叫這個工具提交最終結果。提交後任務結束。",
            "input_schema": self.output_model.model_json_schema(),
        }

    async def run(self, task: str, llm, trace: Trace) -> BaseModel:
        messages = [{"role": "user", "content": task}]
        api_tools = [t.to_api() for t in self.tools.values()] + [self._submit_tool()]

        for step in range(self.max_steps):
            trace.log(self.name, "llm_call", {"step": step})
            response = await llm.create(
                model=self.model,
                system=self.system_prompt,
                messages=messages,
                tools=api_tools,
                trace=trace,
                agent_name=self.name,
            )
            # 模型的回覆要原封不動加回對話紀錄，下一輪它才知道自己剛剛做了什麼
            messages.append({"role": "assistant", "content": response.content})

            tool_calls = [block for block in response.content if block.type == "tool_use"]

            # 情況 A：模型只講話、沒呼叫工具 → 提醒它要用提交工具交結果
            if not tool_calls:
                messages.append(
                    {"role": "user", "content": f"請呼叫 {self.submit_name} 工具提交結果。"}
                )
                continue

            results = []
            for call in tool_calls:
                # 情況 B：模型提交結果 → 檢查格式，合格就結束
                if call.name == self.submit_name:
                    try:
                        output = self.output_model.model_validate(call.input)
                        trace.log(self.name, "output", output.model_dump())
                        return output
                    except ValidationError as e:
                        # 格式不合格：把錯誤告訴模型，讓它修正後重交
                        trace.log(self.name, "error", f"提交格式錯誤：{e}")
                        results.append(_tool_result(call.id, f"格式錯誤，請修正後重新提交：{e}", True))
                        continue

                # 情況 C：模型呼叫一般工具 → 執行並把結果送回去
                results.append(await self._execute(call, trace))

            messages.append({"role": "user", "content": results})

        # 跑到步數上限還沒交結果：不要讓整個系統卡住，丟出錯誤讓上層處理
        trace.log(self.name, "error", "超過步數上限")
        raise RuntimeError(f"agent {self.name} 超過 {self.max_steps} 步仍未提交結果")

    async def _execute(self, call, trace: Trace) -> dict:
        trace.log(self.name, "tool_call", {"tool": call.name, "input": call.input})
        tool = self.tools.get(call.name)
        if tool is None:
            return _tool_result(call.id, f"沒有名為 {call.name} 的工具", True)
        try:
            # 工具是一般（同步）函式，例如抓網頁可能要等好幾秒。
            # 放到另一個執行緒跑，等待期間其他 agent 才能繼續工作，平行執行才有意義。
            output = await asyncio.to_thread(tool.func, **call.input)
            text = json.dumps(output, ensure_ascii=False, default=str)
            trace.log(self.name, "tool_result", {"tool": call.name, "output": output})
            return _tool_result(call.id, text, False)
        except Exception as e:  # noqa: BLE001 — 工具壞掉時，把錯誤告訴模型，而不是讓程式當掉
            trace.log(self.name, "error", f"{call.name} 執行失敗：{e}")
            return _tool_result(call.id, f"工具執行失敗：{e}", True)


def _tool_result(call_id: str, content: str, is_error: bool) -> dict:
    return {"type": "tool_result", "tool_use_id": call_id, "content": content, "is_error": is_error}


def _snake(name: str) -> str:
    """RoutingDecision → routing_decision"""
    out = ""
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out += "_"
        out += ch.lower()
    return out
