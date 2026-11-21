"""
ml/train.py — 訓練話術分類器。

執行方式：
    python -m ml.train                       # 用預設的訓練資料
    python -m ml.train --data 你的資料.jsonl  # 用自己標註的資料

訓練資料格式（JSONL，一行一筆）：
    {"text": "訊息內容", "is_scam": true}

流程：
  1. 讀資料，切成訓練集 80% / 驗證集 20%
  2. TF-IDF 把文字轉成數字向量（用 jieba 斷詞）
  3. 邏輯迴歸學習「哪些詞跟詐騙有關」
  4. 印出驗證集成績，存檔到 ml/model.joblib

注意：data/seed/train_seed.jsonl 是我手寫的合成範例，只是讓程式能先跑起來。
      正式訓練請換成你收集、標註的真實資料，數量越多越好（建議每類至少 500 筆）。
"""

import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import config
from ml.text import tokenize


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_model():
    return make_pipeline(
        # token_pattern=None：告訴 sklearn 我們自己處理斷詞
        # ngram_range=(1, 2)：除了單一詞，也看相鄰兩個詞的組合，例如「保證 獲利」
        TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), min_df=1),
        # class_weight="balanced"：詐騙和正常的資料量不一樣時，自動平衡權重
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=config.SEED_DIR / "train_seed.jsonl")
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    texts = [r["text"] for r in rows]
    labels = [int(r["is_scam"]) for r in rows]
    print(f"讀到 {len(rows)} 筆資料，其中詐騙 {sum(labels)} 筆")

    x_train, x_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = build_model()
    model.fit(x_train, y_train)

    print("\n驗證集成績：")
    print(classification_report(y_val, model.predict(x_val), target_names=["正常", "詐騙"], zero_division=0))

    # 驗證完之後，用全部資料重新訓練一次再存檔，讓模型看過最多資料
    model.fit(texts, labels)
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    print(f"模型已存到 {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
