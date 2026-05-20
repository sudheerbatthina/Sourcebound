# ---------------------------------------------------------------------------
# Stage 1 — dependency layer (cached separately from source code)
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS deps

WORKDIR /install

# System libraries needed by unstructured / pdfplumber / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/deps -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 — runtime image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy system libs installed in stage 1
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages
COPY --from=deps /deps /usr/local

# Copy application source (everything except what's in .dockerignore)
COPY rag_assistant/ ./rag_assistant/
COPY api.py .
COPY frontend/ ./frontend/
COPY data/fetch_ai_overview.md /app/data/fetch_ai_overview.md

RUN mkdir -p /app/storage/data /app/storage/chroma_db /app/.cache

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
