"""
ml/text.py — 中文斷詞。

英文單字之間有空格，中文沒有，所以要先用 jieba 把句子切成詞，
TF-IDF 才能計算每個詞的重要性。

例如「保證獲利穩賺不賠」→ ["保證", "獲利", "穩賺不賠"]
"""

import logging

import jieba

jieba.setLogLevel(logging.WARNING)  # 關掉 jieba 啟動時的訊息


def tokenize(text: str) -> list[str]:
    return [w for w in jieba.lcut(text.lower()) if w.strip()]
