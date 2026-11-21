"""
ml/classifier.py — 話術分類器：給一則訊息「像不像詐騙」的分數，並標出操弄手法。

兩個部分：
  1. 機率分數：用 ml/train.py 訓練好的 TF-IDF + 邏輯迴歸模型。
  2. 手法標籤：用關鍵字規則找出常見的操弄手法。規則很好懂、好解釋，
     之後可以改成多標籤分類模型，再拿來和規則版比較。

這個分類器同時是：
  - V0 版本的全部（不用 LLM）
  - 話術分析 agent 的工具之一
"""

import re
from functools import lru_cache

import joblib

import config

# 操弄手法 → 對應的關鍵字（正規表示式）
# 參考刑事局公布的常見假投資廣告用語，以及 165 宣導資料整理
TACTIC_PATTERNS = {
    "保證獲利": r"保證(獲利|賺|收益)|穩賺|穩賺不賠|零風險|包賺|獲利\d+%|翻倍",
    "製造急迫感": r"限時|限額|名額有限|最後\d+|今天截止|立即|馬上|錯過",
    "要求加 LINE": r"加\s*line|加賴|line\s*群|官方\s*line|私訊",
    "冒用名人或官方": r"老師帶|名師|股神|官方唯一|唯一官方|財經專家|知名",
    "要求轉帳或虛擬貨幣": r"轉帳|匯款|usdt|虛擬貨幣|比特幣|儲值|保證金|手續費|解凍",
    "飆股話術": r"飆股|妖股|黑馬股|內線|主力|明牌",
    "要求提供個資": r"驗證碼|身分證|帳號密碼|信用卡號|網銀",
    "異常高報酬": r"日賺|月入\d+萬|躺著賺|被動收入|財富自由",
}


def find_tactics(text: str) -> list[str]:
    lowered = text.lower()
    return [name for name, pattern in TACTIC_PATTERNS.items() if re.search(pattern, lowered)]


@lru_cache(maxsize=1)
def _load_model():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到分類器模型 {config.MODEL_PATH}，請先執行：python -m ml.train"
        )
    return joblib.load(config.MODEL_PATH)


def scam_probability(text: str) -> float:
    model = _load_model()
    return float(model.predict_proba([text])[0][1])  # [0][1] = 第一筆資料、「詐騙」那一類的機率


def classify_script(text: str) -> dict:
    """給話術分析 agent 用的工具。"""
    return {
        "scam_probability": round(scam_probability(text), 3),
        "tactics": find_tactics(text),
        "note": "分數來自 TF-IDF 邏輯迴歸模型；手法標籤來自關鍵字規則",
    }
