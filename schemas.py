"""
schemas.py — agent 之間傳遞資料的「格式合約」。

為什麼要固定格式？
  4 個查核 agent 各自工作，如果每個都用自己的格式回報，彙整 agent 就很難處理。
  所以規定大家都回傳同一種 Finding，最後的結果則統一是 Report。

Pydantic 會自動檢查欄位型別：模型如果回傳了不合格的資料，會直接報錯，
不會讓錯誤資料悄悄流到下一步。
"""

from typing import Literal

from pydantic import BaseModel, Field

# 詐騙類型（分流 agent 只能從這幾個裡面選）
ScamType = Literal["投資", "假冒官方或銀行", "網購", "交友", "求職", "其他", "非詐騙"]
RiskLevel = Literal["low", "medium", "high"]


class Finding(BaseModel):
    """一個查核 agent 的結論。"""

    agent: str = Field(description="哪個 agent 產生的，例如 link_checker")
    signal: Literal["scam", "clean", "unknown"] = Field(
        description="scam=發現詐騙訊號；clean=查過沒問題；unknown=資料不足無法判斷"
    )
    confidence: float = Field(ge=0, le=1, description="信心程度，0 到 1")
    evidence: list[str] = Field(description="具體證據，每條一句話")
    sources: list[str] = Field(default_factory=list, description="證據出處：資料集名稱或網址")


class RoutingDecision(BaseModel):
    """分流 agent 的決定：這則訊息是什麼類型、要啟動哪些查核。"""

    scam_type: ScamType
    has_url: bool = Field(description="訊息裡有沒有網址；沒有就不用跑連結查核")
    run_identity_check: bool = Field(description="有沒有自稱公司、老師、官方單位")
    reason: str


class Report(BaseModel):
    """最後交給使用者的查核報告。"""

    risk_level: RiskLevel
    scam_type: ScamType
    summary: str = Field(description="一兩句話的結論")
    findings: list[Finding] = Field(default_factory=list)
    advice: str = Field(description="建議使用者怎麼做")
    reviewed: bool = Field(default=False, description="是否通過審查 agent")


class ReportDraft(BaseModel):
    """
    彙整 agent 的輸出：報告的結論部分。

    為什麼不直接讓彙整 agent 輸出 Report？
      Report 裡有 findings，那是查核 agent 已經產生的資料，程式自己接上就好，
      不需要讓模型再抄一遍（浪費 token，還可能抄錯）。
    """

    risk_level: RiskLevel
    scam_type: ScamType
    summary: str
    advice: str


class ReviewResult(BaseModel):
    """審查 agent 的結果。"""

    approved: bool
    problems: list[str] = Field(default_factory=list, description="證據和結論對不上的地方")
    recheck_agents: list[str] = Field(
        default_factory=list, description="需要重查的 agent 名稱"
    )


class CheckRequest(BaseModel):
    """API 收到的請求。"""

    text: str = Field(min_length=1, max_length=5000)
    version: Literal["v0", "v1", "v2", "v3"] = "v3"


class FeedbackRequest(BaseModel):
    """使用者回報判斷錯誤。"""

    trace_id: str
    is_scam: bool = Field(description="使用者認為正確答案是：是不是詐騙")
    note: str = ""
