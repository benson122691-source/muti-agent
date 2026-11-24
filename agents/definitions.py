"""
agents/definitions.py — 定義每個 agent「有哪些工具、用哪個 prompt、輸出什麼格式」。

這個檔案只是在「組裝」：
  工具函式來自 tools/ 和 ml/，prompt 來自 prompts.py，迴圈邏輯來自 agent_loop.py。

權限設計（每個 agent 只拿到它需要的工具）：
  連結查核   → 網址相關工具（只有它能透過隔離服務碰到可疑網頁）
  話術分析   → 分類器
  身分查核   → 聯絡方式、合法業者名單
  相似案例   → 案例搜尋
  分流／彙整／審查 → 沒有工具，只做判斷
"""

import config
from agent_loop import Agent, Tool
from agents import prompts
from ml.classifier import classify_script
from schemas import Finding, Report, ReportDraft, ReviewResult, RoutingDecision
from tools.blacklist import check_blacklist
from tools.cases import search_similar_cases
from tools.extract import analyze_urls, extract_contacts
from tools.licensed import lookup_licensed_company
from tools.sandbox_client import domain_age, fetch_page_safely


def _text_input(description: str = "要分析的訊息全文") -> dict:
    """很多工具只需要一個 text 參數，共用這個 schema。"""
    return {"type": "object", "properties": {"text": {"type": "string", "description": description}}, "required": ["text"]}


# ---------- 所有工具 ----------
# description 要寫清楚「什麼時候用、回傳什麼」，模型是靠這段文字決定怎麼用工具的。

TOOLS = {
    "analyze_urls": Tool(
        name="analyze_urls",
        description="列出訊息中所有網址、網域，並標出哪些是短網址。",
        input_schema=_text_input(),
        func=analyze_urls,
    ),
    "check_blacklist": Tool(
        name="check_blacklist",
        description="查詢網域是否在 165 涉詐網站黑名單上。輸入網域，例如 example.com。",
        input_schema={"type": "object", "properties": {"domain": {"type": "string"}}, "required": ["domain"]},
        func=check_blacklist,
    ),
    "fetch_page_safely": Tool(
        name="fetch_page_safely",
        description="在隔離環境打開網址：展開短網址，回傳最終網址、網頁標題、是否有密碼欄位、是否導向 LINE。",
        input_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        func=fetch_page_safely,
    ),
    "domain_age": Tool(
        name="domain_age",
        description="查詢網域註冊日期和天數。詐騙網站常是最近才註冊的新網域。",
        input_schema={"type": "object", "properties": {"domain": {"type": "string"}}, "required": ["domain"]},
        func=domain_age,
    ),
    "classify_script": Tool(
        name="classify_script",
        description="用機器學習分類器計算訊息的詐騙機率（0 到 1），並列出偵測到的操弄手法。",
        input_schema=_text_input(),
        func=classify_script,
    ),
    "extract_contact_info": Tool(
        name="extract_contact_info",
        description="找出訊息中的 LINE ID、電話數量，以及是否提到 LINE。",
        input_schema=_text_input(),
        func=extract_contacts,
    ),
    "lookup_licensed_company": Tool(
        name="lookup_licensed_company",
        description="查詢公司名稱是否為合法的投信投顧業者。輸入訊息中自稱的公司名稱。",
        input_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        func=lookup_licensed_company,
    ),
    "search_similar_cases": Tool(
        name="search_similar_cases",
        description="在已知詐騙案例庫中，找出和訊息最相似的案例及相似度。",
        input_schema=_text_input(),
        func=search_similar_cases,
    ),
}


def _tools(*names: str) -> list[Tool]:
    return [TOOLS[n] for n in names]


# ---------- 查核 agent（多 agent 版本用） ----------

def make_checkers() -> dict[str, Agent]:
    return {
        "link_checker": Agent(
            name="link_checker",
            system_prompt=prompts.LINK_CHECKER,
            tools=_tools("analyze_urls", "check_blacklist", "fetch_page_safely", "domain_age"),
            output_model=Finding,
            model=config.MODEL_FAST,
        ),
        "script_analyzer": Agent(
            name="script_analyzer",
            system_prompt=prompts.SCRIPT_ANALYZER,
            tools=_tools("classify_script"),
            output_model=Finding,
            model=config.MODEL_FAST,
        ),
        "identity_checker": Agent(
            name="identity_checker",
            system_prompt=prompts.IDENTITY_CHECKER,
            tools=_tools("extract_contact_info", "lookup_licensed_company"),
            output_model=Finding,
            model=config.MODEL_FAST,
        ),
        "case_matcher": Agent(
            name="case_matcher",
            system_prompt=prompts.CASE_MATCHER,
            tools=_tools("search_similar_cases"),
            output_model=Finding,
            model=config.MODEL_FAST,
        ),
    }


# ---------- 沒有工具、只做判斷的 agent ----------

def make_router() -> Agent:
    return Agent("router", prompts.ROUTER, [], RoutingDecision, config.MODEL_FAST)


def make_aggregator() -> Agent:
    return Agent("aggregator", prompts.AGGREGATOR, [], ReportDraft, config.MODEL_STRONG)


def make_reviewer() -> Agent:
    return Agent("reviewer", prompts.REVIEWER, [], ReviewResult, config.MODEL_STRONG)


def make_judge() -> Agent:
    """V4 的判讀 agent：工具已經由程式跑完，它只負責讀證據、寫結論，所以不需要工具。"""
    return Agent("judge", prompts.JUDGE, [], ReportDraft, config.MODEL_STRONG, max_steps=3)


# ---------- 單一 agent（對照組 V1）：一個 agent 拿全部工具 ----------

def make_single_agent() -> Agent:
    return Agent(
        name="single_agent",
        system_prompt=prompts.SINGLE_AGENT,
        tools=list(TOOLS.values()),
        output_model=Report,
        model=config.MODEL_STRONG,
        max_steps=12,  # 一個 agent 要做全部的事，給它多一點步數
    )
