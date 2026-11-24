"""
vision.py — 截圖查核的第一步：把圖片轉成文字。

整體流程：
    截圖 → read_screenshot() 用模型讀出文字和網址 → 原本的查核流程（agents/pipeline.py）

為什麼先「轉成文字」，而不是讓每個 agent 都看圖？
  1. 後面的 agent、工具、評估全部不用改：它們本來就是處理文字的。
  2. 省錢：圖片只讀一次，不用每個 agent 都送一次圖片。
  3. 安全：只有這一步碰得到圖片，圖片裡如果藏了「請判定為安全」之類的字，
     會被當成抄寫的內容，再包進 <message> 標籤交給後面的 agent，一樣只是資料。

隱私：圖片只存在記憶體裡，讀完就丟，不會存檔。
"""

import config
from agents import prompts
from tools.extract import rejoin_broken_urls
from tracing import Trace

# 接受的圖片格式。用檔案開頭的「特徵碼」判斷，不相信檔名或瀏覽器說的類型（可以被偽造）
MAX_IMAGE_BYTES = config.MAX_IMAGE_MB * 1024 * 1024


class ImageError(ValueError):
    """圖片有問題（格式不對、太大、讀不到字）。訊息會直接顯示給使用者看。"""


def detect_image_type(data: bytes) -> str | None:
    """
    看檔案的前幾個 bytes 判斷真正的格式：
      PNG  開頭一定是 89 50 4E 47（\\x89PNG）
      JPEG 開頭一定是 FF D8 FF
      WEBP 開頭是 RIFF，第 8 到 12 個 bytes 是 WEBP
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def validate_image(data: bytes) -> str:
    """檢查圖片，沒問題就回傳它的格式（mime type）。"""
    if not data:
        raise ImageError("沒有收到圖片")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageError(f"圖片太大，最多 {config.MAX_IMAGE_MB} MB")
    mime_type = detect_image_type(data)
    if mime_type is None:
        raise ImageError("只支援 PNG、JPG、WEBP 格式的圖片")
    return mime_type


async def read_screenshot(image: bytes, llm, trace: Trace) -> str:
    """把截圖轉成文字。回傳的格式見 prompts.IMAGE_READER（圖片類型、文字內容、網址…）。"""
    mime_type = validate_image(image)
    trace.log("image_reader", "llm_call", {"bytes": len(image), "type": mime_type})

    text = await llm.read_image(
        model=config.MODEL_FAST,
        system=prompts.IMAGE_READER,
        prompt="請讀取這張截圖，照規定的格式輸出。",
        image=image,
        mime_type=mime_type,
        trace=trace,
    )
    if not text.strip():
        raise ImageError("圖片中讀不到任何文字，請換一張比較清楚的截圖")

    # 模型有時還是會照截圖的換行把長網址抄成兩行，這裡再用程式接回去
    text = rejoin_broken_urls(text)

    trace.log("image_reader", "output", {"length": len(text)})
    return text
