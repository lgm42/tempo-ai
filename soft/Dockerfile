FROM python:3.12-slim

# libgomp1 : OpenMP, requis par LightGBM
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tempo_ai ./tempo_ai

ENV TEMPO_DATA_DIR=/data \
    PORT=8080 \
    HOME=/tmp \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN mkdir -p /data && chown 568:568 /data
VOLUME /data
USER 568:568
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '8080'), timeout=4)"

CMD ["sh", "-c", "exec uvicorn tempo_ai.api:app --host 0.0.0.0 --port ${PORT} --no-access-log"]
