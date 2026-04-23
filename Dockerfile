FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install rhombus CLI (used for cert-authenticated media downloads)
RUN curl -fsSL https://github.com/RhombusSystems/rhombus-cli/releases/download/v0.16.1/rhombus-cli_0.16.1_linux_amd64.tar.gz \
    | tar -xz -C /usr/local/bin rhombus \
    && chmod +x /usr/local/bin/rhombus

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY webhook_server.py best.pt ./

ENV PORT=8080

CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 webhook_server:app
