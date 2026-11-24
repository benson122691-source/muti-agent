# 詐騙訊息多 Agent 查核系統

貼上一則可疑訊息，或上傳截圖，系統用多個 AI agent 同時查核，回傳風險等級、具體證據和建議。

本專案的重點不只是 demo，而是**用同一份測試集比較五個版本**，用數據回答「多 agent 到底值不值得」：

| 版本 | 架構 |
| --- | --- |
| V0 | 只用 ML 分類器（不呼叫 LLM） |
| V1 | 單一 agent，擁有全部工具 |
| V2 | 分流 → 4 個查核 agent 平行執行 → 彙整 |
| V3 | V2 ＋ 審查 agent，不通過就退回重查（呼叫模型 10 次以上） |
| V4 | 精簡版：程式自己跑完所有工具，再交給判讀 agent 寫結論（只呼叫模型 1 次，網站預設） |

```text
使用者訊息 → 分流 agent ─┬─ 連結查核（隔離環境）─┐
                          ├─ 話術分析            ├→ 彙整 agent → 審查 agent → 報告
                          ├─ 身分查核            │        ↑           │
                          └─ 相似案例 ───────────┘        └─ 退回重查 ─┘
```

想讀懂程式碼，請從 [docs/程式導讀.md](docs/程式導讀.md) 開始。

## 快速開始（本機）

```bash
# 1. 安裝套件（建議先建立虛擬環境：python -m venv .venv）
pip install -r requirements.txt

# 2. 產生範例資料、訓練分類器
python data/seed/make_seed.py
python -m ml.train

# 3. 設定模型和 API 金鑰（沒有金鑰也能跑，但只能用 V0）
cp .env.example .env      # 然後編輯 .env：LLM_PROVIDER 選 gemini 或 claude，填入對應的金鑰

# 4. 啟動（開兩個終端機）
uvicorn sandbox.service:app --port 8001     # 隔離抓取服務
uvicorn api.main:app --reload --port 8000   # 主程式
```

打開 http://localhost:8000 就能使用。

## 用 Docker 啟動

```bash
docker compose up --build
```

## 常用指令

| 指令 | 用途 |
| --- | --- |
| `pytest` | 跑單元測試（用假模型，不需要金鑰、不花錢） |
| `python -m evals.run_eval` | 評估 V0 |
| `python -m evals.run_eval --versions v0 v1 v2 v3 v4 --repeats 3` | 五個版本完整對照（需要金鑰） |
| `python -m ml.train --data 你的資料.jsonl` | 用自己的資料重新訓練分類器 |
| `python -m crawlers.update_data` | 下載最新的 165 開放資料 |

## 截圖查核

網頁可以上傳截圖（點選、拖曳，或直接 Ctrl+V 貼上），支援 PNG、JPG、WEBP，5 MB 以內。

```text
截圖 → 讀圖（模型讀出文字、網址、畫面細節）→ 原本的多 agent 查核流程
```

- 讀圖只做一次，後面的 agent 和工具都處理文字，不用改
- 圖片只存在記憶體裡，讀完就丟，不會存檔
- 讀圖需要模型，所以沒有 API 金鑰時只能貼文字

## 支援的模型

在 `.env` 的 `LLM_PROVIDER` 切換，其他程式都不用改：

| 設定值 | 需要的金鑰 | 申請地點 | 預設模型 |
| --- | --- | --- | --- |
| `gemini` | `GEMINI_API_KEY` | Google AI Studio（有免費額度） | gemini-3.5-flash / gemini-3.1-flash-lite |
| `claude` | `ANTHROPIC_API_KEY` | Claude Console | claude-sonnet-5-5 / claude-haiku-5-5 |

⚠️ Gemini 免費方案的內容可能被 Google 用來改進產品，並可能經人工審查。開發和測試合成資料沒問題；
公開上線讓別人貼真實訊息時，請改用付費方案。免費方案也有每分鐘次數限制，程式遇到 429 錯誤會自動等待重試。

## 安全設計

- **隔離抓取：** 可疑網頁只由 `sandbox/` 服務打開。該容器沒有 API 金鑰、檔案系統唯讀，只回傳摘要欄位，不回傳整頁 HTML。
- **防 prompt injection：** 使用者內容一律包在 `<message>` 標籤中，所有 prompt 都規定標籤內只是資料。
- **防 SSRF：** 隔離服務拒絕抓取內網位址，每一次轉址都會檢查。
- **成本上限：** 每個 agent 有步數上限，每日 LLM 查核有次數上限，超過就自動降級成 V0。
- **個資：** 記錄檔存檔前遮罩手機、身分證字號、帳號。

## ⚠️ 目前的資料是合成的

`data/seed/` 和 `evals/sample_testset.jsonl` 是手寫的範例，只是讓程式能先跑起來。
正式使用前請換成真實資料，步驟見 [docs/程式導讀.md](docs/程式導讀.md) 第 5 節。

查核結果僅供參考，如有疑慮請撥打 165 反詐騙諮詢專線。
