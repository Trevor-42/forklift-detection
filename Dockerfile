FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY webhook_server.py best.pt ./

ENV PORT=8080

CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 webhook_server:app
