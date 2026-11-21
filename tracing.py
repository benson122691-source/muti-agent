"""
tracing.py — 記錄一次查核過程中「每個 agent 做了什麼」。

除錯全靠這份記錄：哪個 agent 呼叫了哪個工具、拿到什麼結果、花了多少 token。
每次查核會存成 logs/traces/<trace_id>.jsonl，一行一個事件。

存檔前會先用 privacy.mask_pii() 遮罩手機號碼、身分證字號等個資。
"""

import json
import time
import uuid
from dataclasses import dataclass, field

import config
from privacy import mask_pii


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    events: list[dict] = field(default_factory=list)
    # 依模型分開記 token，因為不同模型價格不同
    input_tokens: dict[str, int] = field(default_factory=dict)
    output_tokens: dict[str, int] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def log(self, agent: str, kind: str, data) -> None:
        """記一筆事件。kind 例如：llm_call、tool_call、tool_result、output、error。"""
        self.events.append(
            {"t": round(time.time() - self.started_at, 3), "agent": agent, "kind": kind, "data": data}
        )

    def add_usage(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens[model] = self.input_tokens.get(model, 0) + input_tokens
        self.output_tokens[model] = self.output_tokens.get(model, 0) + output_tokens

    def total_tokens(self) -> int:
        return sum(self.input_tokens.values()) + sum(self.output_tokens.values())

    def cost_twd(self) -> float:
        """依 config 的價格算出新台幣成本。價格沒填就是 0。"""
        usd = 0.0
        for model, n in self.input_tokens.items():
            usd += n / 1_000_000 * config.PRICE_INPUT_PER_MTOK.get(model, 0)
        for model, n in self.output_tokens.items():
            usd += n / 1_000_000 * config.PRICE_OUTPUT_PER_MTOK.get(model, 0)
        return usd * config.USD_TO_TWD

    def save(self) -> None:
        config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = config.TRACE_DIR / f"{self.trace_id}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for e in self.events:
                line = json.dumps(e, ensure_ascii=False, default=str)
                f.write(mask_pii(line) + "\n")
