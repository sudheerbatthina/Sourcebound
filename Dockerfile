FROM python:3.12-slim

WORKDIR /app

# Layer 1: system packages — cached until base image changes
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: Python dependencies — cached until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layer 3: application code — only layer that rebuilds on code changes
COPY . .

RUN mkdir -p /app/storage/data /app/storage/chroma_db /app/.cache

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
