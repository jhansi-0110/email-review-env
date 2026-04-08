FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir \
    openenv-core \
    fastapi \
    uvicorn \
    pydantic \
    openai \
    websockets \
    python-dotenv \
    requests

RUN mkdir -p /app/outputs/evals /app/outputs/logs

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
