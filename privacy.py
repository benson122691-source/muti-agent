"""
privacy.py — 遮罩個資，避免把使用者的電話、身分證字號存進記錄檔。

只用在「存檔」的時候。查核過程本身還是會看到原文，因為判斷詐騙需要這些資訊。
"""

import re

# 台灣手機：09 開頭共 10 碼，中間可能有 - 或空白
_PHONE = re.compile(r"09\d{2}[-\s]?\d{3}[-\s]?\d{3}")
# 身分證字號：1 個英文字母 + 1 或 2 + 8 個數字
_NATIONAL_ID = re.compile(r"(?<![A-Za-z0-9])[A-Z][12]\d{8}(?!\d)")
# 銀行帳號：連續 10 到 16 個數字
# 註：不用 \b，因為中文字也算「文字字元」，「帳號12345678901」中間沒有 \b 邊界
_ACCOUNT = re.compile(r"(?<!\d)\d{10,16}(?!\d)")


def mask_pii(text: str) -> str:
    text = _PHONE.sub("09**-***-***", text)
    text = _NATIONAL_ID.sub("[身分證已遮罩]", text)
    text = _ACCOUNT.sub("[帳號已遮罩]", text)
    return text
