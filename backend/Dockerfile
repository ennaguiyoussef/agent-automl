# Agentic AutoML — backend image
# Reproducible build with uv; runs the FastAPI app under uvicorn.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Install dependencies first (cached layer, independent of app source).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application source.
COPY app ./app

# Runtime storage (overridden by env / mounted volumes in compose).
RUN mkdir -p /app/data /app/artifacts
ENV DATA_DIR=/app/data \
    ARTIFACTS_DIR=/app/artifacts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
