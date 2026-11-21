"""
agents/pipeline.py — 把 agent 串起來，做成 V0 到 V3 四個版本。

  V0：只用 ML 分類器，不呼叫 LLM（最便宜、最快，當作最低基準）
  V1：單一 agent 拿全部工具，自己決定怎麼查
  V2：分流 → 查核 agent 平行執行 → 彙整
  V3：V2 再加審查 agent，不通過就退回重查

四個版本用同一組工具、同一份測試集比較，才能知道「多 agent」和「審查」各自有沒有幫助。

對外只需要用 check()：
    report, trace = await check("訊息內容", version="v3", llm=ClaudeLLM())
"""

import asyncio
import json

import config
from agents.definitions import make_aggregator, make_checkers, make_reviewer, make_router, make_single_agent
from ml.classifier import find_tactics, scam_probability
from schemas import Finding, Report, RoutingDecision
from tools.extract import extract_urls
from tracing import Trace


def wrap_message(text: str) -> str:
    """把使用者內容包進 <message> 標籤，讓模型分清楚「資料」和「指令」。"""
    return f"請查核以下訊息：\n<message>\n{text}\n</message>"


# ============================================================
# V0：純 ML
# ============================================================

def run_v0(text: str, trace: Trace) -> Report:
    prob = scam_probability(text)
    tactics = find_tactics(text)
    if prob >= config.V0_HIGH_THRESHOLD:
        risk = "high"
    elif prob >= config.V0_MEDIUM_THRESHOLD:
        risk = "medium"
    else:
        risk = "low"

    finding = Finding(
        agent="ml_classifier",
        signal="scam" if risk != "low" else "clean",
        confidence=round(prob if risk != "low" else 1 - prob, 3),
        evidence=[f"分類器詐騙機率 {prob:.2f}"] + [f"偵測到手法：{t}" for t in tactics],
        sources=["ml/model.joblib"],
    )
    trace.log("v0", "output", finding.model_dump())
    return Report(
        risk_level=risk,
        scam_type="其他" if risk != "low" else "非詐騙",  # V0 不會判斷類型
        summary=f"分類器判斷詐騙機率為 {prob:.0%}" + (f"，偵測到：{'、'.join(tactics)}" if tactics else ""),
        findings=[finding],
        advice="有疑慮請撥打 165 反詐騙專線查證。" if risk != "low" else "目前未發現明顯的詐騙訊號，仍請保持警覺。",
    )


# ============================================================
# V1：單一 agent
# ============================================================

async def run_v1(text: str, llm, trace: Trace) -> Report:
    return await make_single_agent().run(wrap_message(text), llm, trace)


# ============================================================
# V2 / V3：多 agent
# ============================================================

async def _route(text: str, llm, trace: Trace) -> RoutingDecision:
    """分流。分流 agent 失敗時，保守地啟動全部查核，不要因此漏查。"""
    try:
        decision = await make_router().run(wrap_message(text), llm, trace)
    except Exception as e:  # noqa: BLE001
        trace.log("router", "error", f"分流失敗，改為全部查核：{e}")
        decision = RoutingDecision(scam_type="其他", has_url=True, run_identity_check=True, reason="分流失敗")

    # 是否有網址用程式再確認一次：這種事規則比模型可靠
    decision.has_url = decision.has_url or bool(extract_urls(text))
    return decision


def _pick_checkers(decision: RoutingDecision) -> list[str]:
    names = ["script_analyzer", "case_matcher"]  # 這兩個一定跑
    if decision.has_url:
        names.append("link_checker")             # 沒網址就跳過，省錢省時間
    if decision.run_identity_check:
        names.append("identity_checker")
    return names


async def _run_checkers(names: list[str], text: str, llm, trace: Trace, hint: str = "") -> dict[str, Finding]:
    """
    平行執行多個查核 agent。這裡就是「多 agent 比較快」的關鍵：
    asyncio.gather 讓它們同時等模型回覆，總時間約等於最慢的那一個，而不是全部加總。
    """
    checkers = make_checkers()
    task = wrap_message(text)
    if hint:
        task += f"\n\n審查 agent 指出上次的問題，請特別注意：\n{hint}"

    results = await asyncio.gather(
        *(checkers[name].run(task, llm, trace) for name in names),
        return_exceptions=True,  # 一個 agent 失敗不影響其他 agent
    )

    findings = {}
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            trace.log(name, "error", str(result))
            findings[name] = Finding(agent=name, signal="unknown", confidence=0, evidence=[f"查核失敗：{result}"])
        else:
            findings[name] = result
    return findings


async def _aggregate(text: str, findings: dict[str, Finding], llm, trace: Trace) -> Report:
    findings_json = json.dumps([f.model_dump() for f in findings.values()], ensure_ascii=False, indent=2)
    task = wrap_message(text) + f"\n\n查核結果：\n<findings>\n{findings_json}\n</findings>"
    draft = await make_aggregator().run(task, llm, trace)
    # findings 直接由程式接上，不靠模型抄寫
    return Report(**draft.model_dump(), findings=list(findings.values()))


async def run_multi(text: str, llm, trace: Trace, review: bool) -> Report:
    decision = await _route(text, llm, trace)
    names = _pick_checkers(decision)
    trace.log("pipeline", "routing", {"decision": decision.model_dump(), "checkers": names})

    findings = await _run_checkers(names, text, llm, trace)
    report = await _aggregate(text, findings, llm, trace)

    if not review:
        return report  # V2 到這裡結束

    # ---------- V3：審查 ----------
    reviewer = make_reviewer()
    for attempt in range(config.MAX_REVIEW_RETRIES + 1):
        task = wrap_message(text) + f"\n\n待審查的報告：\n<report>\n{report.model_dump_json(indent=2)}\n</report>"
        result = await reviewer.run(task, llm, trace)
        trace.log("pipeline", "review", {"attempt": attempt, "result": result.model_dump()})

        if result.approved:
            report.reviewed = True
            return report
        if attempt == config.MAX_REVIEW_RETRIES:
            break  # 已經退回過了，不再無限重查

        # 只重查審查 agent 點名、而且本來就有跑的 agent；沒點名就全部重查
        recheck = [n for n in result.recheck_agents if n in names] or names
        new_findings = await _run_checkers(recheck, text, llm, trace, hint="\n".join(result.problems))
        findings.update(new_findings)
        report = await _aggregate(text, findings, llm, trace)

    # 重查後仍不通過：保留報告，但標示未通過審查，並把風險至少拉到 medium（寧可保守）
    report.reviewed = False
    if report.risk_level == "low":
        report.risk_level = "medium"
        report.summary += "（審查未通過，結果不確定，請自行查證）"
    return report


# ============================================================
# 對外入口
# ============================================================

async def check(text: str, version: str, llm) -> tuple[Report, Trace]:
    trace = Trace()
    trace.log("pipeline", "start", {"version": version, "length": len(text)})

    if version == "v0":
        report = run_v0(text, trace)
    elif version == "v1":
        report = await run_v1(text, llm, trace)
    elif version == "v2":
        report = await run_multi(text, llm, trace, review=False)
    elif version == "v3":
        report = await run_multi(text, llm, trace, review=True)
    else:
        raise ValueError(f"未知的版本：{version}")

    trace.log("pipeline", "done", {"risk_level": report.risk_level, "tokens": trace.total_tokens()})
    trace.save()
    return report, trace
