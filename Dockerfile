# 主程式和隔離抓取服務共用同一個映像檔，啟動指令不同（見 docker-compose.yml）
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# 建置時就產生範例資料並訓練分類器，容器啟動後馬上能用
RUN python data/seed/make_seed.py && python -m ml.train

# Railway 等平台會用 PORT 環境變數指定連接埠
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
