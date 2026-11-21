"""
tools/cases.py — 找出和這則訊息最像的已知詐騙案例。

做法：把所有案例和查詢訊息都轉成 TF-IDF 向量，用餘弦相似度排序。

為什麼先不用向量資料庫（Chroma）？
  案例只有幾百筆時，TF-IDF 已經夠快、而且不用額外安裝服務或呼叫 embedding API。
  案例變多、或需要「意思相近但用字不同」的比對時，再換成 embedding + Chroma。
  只要 search_similar_cases() 的輸入輸出不變，其他程式都不用改。

案例來源：data/seed/cases_seed.jsonl，格式：
    {"text": "案例內容", "scam_type": "投資", "source": "出處"}
"""

import json
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from ml.text import tokenize


@lru_cache(maxsize=1)
def _index():
    path = config.SEED_DIR / "cases_seed.jsonl"
    with path.open(encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    matrix = vectorizer.fit_transform([c["text"] for c in cases])
    return cases, vectorizer, matrix


def search_similar_cases(text: str, k: int = 3) -> dict:
    cases, vectorizer, matrix = _index()
    scores = cosine_similarity(vectorizer.transform([text]), matrix)[0]
    best = scores.argsort()[::-1][:k]  # 分數由高到低，取前 k 名
    return {
        "matches": [
            {
                "similarity": round(float(scores[i]), 3),
                "scam_type": cases[i]["scam_type"],
                "text": cases[i]["text"][:120],
                "source": cases[i].get("source", ""),
            }
            for i in best
            if scores[i] > 0
        ],
        "note": "相似度 0.3 以上才算明顯相似",
    }
