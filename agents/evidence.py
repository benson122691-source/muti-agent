"""
agents/evidence.py — V4 的第一步：程式自己跑完所有工具，收集證據。

V3 的做法是讓每個查核 agent 自己決定「要不要查黑名單」「要不要跑分類器」，
所以要呼叫模型 10 次以上。但這些工具其實每次都該跑，根本不需要問模型。

V4 的做法：
  1. 這個檔案：用程式把所有工具跑完，再用「規則」把結果整理成 Finding（完全不用模型）
  2. pipeline.py 的 run_v4()：把所有證據一次交給判讀 agent，只呼叫模型 1 次

規則寫在這裡的好處是「透明」：每個 Finding 為什麼是 scam、clean 還是 unknown，
看程式就知道，不用猜模型在想什麼。
"""

import asyncio
import re

from ml.classifier import classify_script
from schemas import Finding
from tools.blacklist import check_blacklist
from tools.cases import search_similar_cases
from tools.extract import analyze_urls, extract_contacts
from tools.licensed import lookup_licensed_company
from tools.sandbox_client import domain_age, fetch_page_safely

MAX_URLS = 3       # 最多深入查幾個網址（抓網頁比較慢）
MAX_COMPANIES = 3  # 最多查幾個公司名稱

# 找出「某某投顧」「某某證券」這類自稱的公司名稱
_COMPANY = re.compile(r"[一-鿿A-Za-z0-9]{2,10}(?:投資顧問|投資信託|資產管理|投顧|投信|證券|期貨|金控)")


async def gather_evidence(text: str) -> dict[str, Finding]:
    """同時跑四類查核，回傳 {查核名稱: Finding}。沒有網址時不會有 link_checker。"""
    # 工具都是一般函式（可能要等網路），用 to_thread 同時執行，總時間 ≈ 最慢的那一個
    script, cases, link, identity = await asyncio.gather(
        asyncio.to_thread(check_script, text),
        asyncio.to_thread(check_cases, text),
        asyncio.to_thread(check_links, text),
        asyncio.to_thread(check_identity, text),
    )
    findings = {"script_analyzer": script, "case_matcher": cases}
    if link:
        findings["link_checker"] = link
    if identity:
        findings["identity_checker"] = identity
    return findings


# ---------------------------------------------------------------------------
# 四類查核：每個都是「跑工具 → 用規則判斷」
# ---------------------------------------------------------------------------


def check_script(text: str) -> Finding:
    """話術：分類器分數 + 關鍵字手法。"""
    result = classify_script(text)
    prob, tactics = result["scam_probability"], result["tactics"]
    evidence = [f"分類器詐騙機率 {prob:.2f}"] + [f"偵測到手法：{t}" for t in tactics]

    if prob >= 0.7 or len(tactics) >= 2:
        signal, confidence = "scam", max(prob, 0.7)
    elif tactics or prob >= 0.45:
        signal, confidence = "unknown", 0.4   # 只有一個弱訊號：正常促銷也可能出現
    else:
        signal, confidence = "clean", round(1 - prob, 2)
    return Finding(agent="script_analyzer", signal=signal, confidence=confidence,
                   evidence=evidence, sources=["ml/model.joblib", "關鍵字規則"])


def check_cases(text: str) -> Finding:
    """相似案例：相似度 0.3 以上才算像。相似度低不代表安全（案例庫可能沒收錄）。"""
    matches = search_similar_cases(text)["matches"]
    if matches and matches[0]["similarity"] >= 0.3:
        best = matches[0]
        return Finding(agent="case_matcher", signal="scam", confidence=min(0.9, 0.4 + best["similarity"]),
                       evidence=[f"和已知「{best['scam_type']}」詐騙相似度 {best['similarity']}：{best['text']}"],
                       sources=[best["source"]])
    top = matches[0]["similarity"] if matches else 0
    return Finding(agent="case_matcher", signal="unknown", confidence=0.3,
                   evidence=[f"案例庫中沒有明顯相似的案例（最高相似度 {top}）"])


def check_links(text: str) -> Finding | None:
    """網址：黑名單 → 隔離環境抓網頁 → 網域註冊時間。訊息沒有網址就回傳 None。"""
    urls = analyze_urls(text)["urls"][:MAX_URLS]
    if not urls:
        return None

    evidence, sources, unknown = [], [], False
    blacklisted, suspicious = 0, 0  # 黑名單命中（很確定）、新網域加可疑網頁（比較不確定）
    for u in urls:
        listed = check_blacklist(u["domain"])
        if listed["listed"]:
            blacklisted += 1
            evidence.append(f"網域 {u['domain']} 在涉詐網站黑名單上")
            sources.append(listed["source"])
            continue

        page = fetch_page_safely(u["url"])
        if "error" in page:
            unknown = True
            evidence.append(f"{u['domain']}：{page['error']}")
            continue
        final_domain = page.get("final_domain") or u["domain"]
        if final_domain != u["domain"]:
            evidence.append(f"{u['domain']} 實際導向 {final_domain}")
            if check_blacklist(final_domain)["listed"]:
                blacklisted += 1
                evidence.append(f"導向的網域 {final_domain} 在黑名單上")
                continue

        age = domain_age(final_domain)
        is_new = age.get("is_new_domain", False)
        if is_new:
            evidence.append(f"{final_domain} 註冊不到 90 天（{age['age_days']} 天）")
        if page.get("has_password_field"):
            evidence.append(f"{final_domain} 的網頁要求輸入密碼")
        if page.get("links_to_line"):
            evidence.append(f"{final_domain} 的網頁導向 LINE")
        if is_new and (page.get("has_password_field") or page.get("links_to_line")):
            suspicious += 1

    if blacklisted or suspicious:
        return Finding(agent="link_checker", signal="scam", confidence=0.95 if blacklisted else 0.7,
                       evidence=evidence, sources=sources)
    if unknown:
        return Finding(agent="link_checker", signal="unknown", confidence=0.3, evidence=evidence)
    return Finding(agent="link_checker", signal="clean", confidence=0.5,
                   evidence=evidence or ["網址不在黑名單上，網頁也沒有明顯可疑之處"])


def check_identity(text: str) -> Finding | None:
    """身分：自稱的投資公司是否合法、有沒有要求加 LINE。什麼都沒有就回傳 None。"""
    companies = list(dict.fromkeys(_COMPANY.findall(text)))[:MAX_COMPANIES]
    contacts = extract_contacts(text)
    if not companies and not contacts["line_ids"]:
        return None

    evidence, unlicensed = [], 0
    for name in companies:
        result = _lookup(name)
        if result["found"]:
            evidence.append(f"「{name}」查得到合法登記：{result['matches'][0]}")
        else:
            unlicensed += 1
            evidence.append(f"「{name}」在合法業者名單中查不到（名單可能不完整）")
    if contacts["line_ids"]:
        evidence.append(f"訊息提供 LINE ID：{'、'.join(contacts['line_ids'])}")

    if unlicensed:
        return Finding(agent="identity_checker", signal="scam", confidence=0.6, evidence=evidence,
                       sources=["合法業者名單"])
    return Finding(agent="identity_checker", signal="unknown", confidence=0.4, evidence=evidence)


def _lookup(name: str) -> dict:
    """
    正規表示式可能把前面的字也抓進來，例如「我們是範例投顧」。
    所以從頭開始一次去掉一個字再查：「我們是範例投顧」→「們是範例投顧」→「是範例投顧」→「範例投顧」。
    """
    for start in range(len(name) - 2):
        result = lookup_licensed_company(name[start:])
        if result["found"]:
            return result
    return {"found": False}
