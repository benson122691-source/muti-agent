"""
evals/run_eval.py — 用測試集評估各版本，產生對照表。

執行方式：
    python -m evals.run_eval                                # 預設：只跑 V0（不用 API 金鑰）
    python -m evals.run_eval --versions v0 v1 v2 v3 v4      # 五個版本都跑（需要 API 金鑰）
    python -m evals.run_eval --testset evals/testset.jsonl --repeats 3

測試集格式（JSONL，一行一筆）：
    {"text": "訊息內容", "is_scam": true, "scam_type": "投資"}

判斷規則：risk_level 為 medium 或 high 就算「判定為詐騙」。

輸出：
  - 終端機印出對照表
  - evals/results/<時間>_summary.md   對照表（可以直接貼進報告）
  - evals/results/<時間>_cases.csv    每一筆的預測結果（找錯誤原因用，搭配 trace_id 看 logs/traces/）

⚠️ evals/sample_testset.jsonl 是合成範例，只用來確認程式能跑。
   正式評估請換成真實資料，並保留一部分「最後才跑一次」的保留集。
"""

import argparse
import asyncio
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import config
from agents.pipeline import check
from llm import make_llm

RESULTS_DIR = Path(__file__).parent / "results"


def load_testset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def run_version(version: str, cases: list[dict], llm) -> list[dict]:
    """依序跑完所有案例。依序跑（而不是同時跑）才能量到準確的單次回應時間。"""
    rows = []
    for i, case in enumerate(cases, 1):
        started = time.time()
        try:
            report, trace = await check(case["text"], version, llm)
            rows.append({
                "version": version,
                "text": case["text"][:60],
                "is_scam": case["is_scam"],
                "true_type": case.get("scam_type", ""),
                "risk_level": report.risk_level,
                "pred_scam": report.risk_level in ("medium", "high"),
                "pred_type": report.scam_type,
                "latency": time.time() - started,
                "tokens": trace.total_tokens(),
                "cost_twd": trace.cost_twd(),
                "trace_id": trace.trace_id,
                "error": "",
            })
        except Exception as e:  # noqa: BLE001 — 單筆失敗要記下來，但不要中斷整個評估
            rows.append({"version": version, "text": case["text"][:60], "is_scam": case["is_scam"],
                         "pred_scam": None, "latency": time.time() - started, "error": str(e)})
        print(f"  [{version}] {i}/{len(cases)}", end="\r")
    print()
    return rows


def compute_metrics(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["pred_scam"] is not None]
    tp = sum(r["is_scam"] and r["pred_scam"] for r in ok)          # 真詐騙、抓到
    fn = sum(r["is_scam"] and not r["pred_scam"] for r in ok)      # 真詐騙、漏掉
    fp = sum(not r["is_scam"] and r["pred_scam"] for r in ok)      # 正常、誤判
    tn = sum(not r["is_scam"] and not r["pred_scam"] for r in ok)  # 正常、判對

    recall = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0

    scams = [r for r in ok if r["is_scam"] and r.get("true_type")]
    type_acc = sum(r["pred_type"] == r["true_type"] for r in scams) / len(scams) if scams else 0

    latencies = sorted(r["latency"] for r in ok) or [0]
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]

    return {
        "召回率": recall,
        "誤判率": false_positive_rate,
        "精確率": precision,
        "F1": f1,
        "類型正確率": type_acc,
        "P95秒數": p95,
        "平均tokens": statistics.mean(r["tokens"] for r in ok) if ok else 0,
        "平均成本NT$": statistics.mean(r["cost_twd"] for r in ok) if ok else 0,
        "失敗筆數": len(rows) - len(ok),
    }


def to_markdown(results: dict[str, dict]) -> str:
    columns = list(next(iter(results.values())).keys())
    lines = ["| 版本 | " + " | ".join(columns) + " |", "| --- " * (len(columns) + 1) + "|"]
    for version, m in results.items():
        cells = []
        for col in columns:
            value = m[col]
            if col in ("召回率", "誤判率", "精確率", "F1", "類型正確率"):
                cells.append(f"{value:.1%}")
            elif isinstance(value, float):
                cells.append(f"{value:.2f}")
            else:
                cells.append(str(value))
        lines.append(f"| {version.upper()} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=Path, default=Path(__file__).parent / "sample_testset.jsonl")
    parser.add_argument("--versions", nargs="+", default=["v0"], choices=["v0", "v1", "v2", "v3", "v4"])
    parser.add_argument("--repeats", type=int, default=1, help="LLM 有隨機性，建議跑 3 次取平均")
    args = parser.parse_args()

    needs_llm = any(v != "v0" for v in args.versions)
    llm = make_llm() if needs_llm else None
    if needs_llm and llm is None:
        key = "GEMINI_API_KEY" if config.LLM_PROVIDER == "gemini" else "ANTHROPIC_API_KEY"
        raise SystemExit(f"V1 到 V3 需要設定 {key}（寫在 .env）")
    if needs_llm:
        print(f"模型：{config.LLM_PROVIDER}（{config.MODEL_STRONG} / {config.MODEL_FAST}）")

    cases = load_testset(args.testset)
    print(f"測試集：{args.testset}（{len(cases)} 筆，詐騙 {sum(c['is_scam'] for c in cases)} 筆）")

    all_rows, results = [], {}
    for version in args.versions:
        runs = []
        for _ in range(args.repeats):
            rows = await run_version(version, cases, llm)
            all_rows += rows
            runs.append(compute_metrics(rows))
        # 多次執行取平均
        results[version] = {k: statistics.mean(r[k] for r in runs) for k in runs[0]}

    table = to_markdown(results)
    print("\n" + table)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RESULTS_DIR / f"{stamp}_summary.md").write_text(
        f"測試集：{args.testset.name}，{len(cases)} 筆，重複 {args.repeats} 次\n"
        f"模型：{config.LLM_PROVIDER}（{config.MODEL_STRONG} / {config.MODEL_FAST}）\n\n{table}\n", encoding="utf-8"
    )
    with (RESULTS_DIR / f"{stamp}_cases.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in all_rows for k in r}))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n結果已存到 {RESULTS_DIR}/{stamp}_*")


if __name__ == "__main__":
    asyncio.run(main())
