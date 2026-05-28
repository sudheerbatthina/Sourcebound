# Sourcebound

Sourcebound is an open source, private Retrieval-Augmented Generation (RAG) assistant for document Q&A. It helps teams ask grounded questions over their own documents without sending source material into a public search workflow, with a healthcare policy use case focused on HIPAA-style access controls, citations, and audit-friendly evaluation.

Sourcebound extracts, chunks, embeds, and retrieves document content, then generates answers with source citations using OpenAI.

<!-- Suggested GitHub topics: rag, retrieval-augmented-generation, healthcare-ai, document-qa, fastapi, streamlit, chromadb, openai, mlflow, hipaa -->

---

## Why Sourcebound?

Organizations sit on policies, PDFs, playbooks, and operational documents that are hard to search and risky to expose. Healthcare teams face an even sharper version of that problem: staff need fast answers from trusted documents, but access control, PHI handling, and citation quality matter.

Sourcebound provides a private RAG reference implementation for document Q&A. It combines document ingestion, semantic retrieval, reranking, answer generation, API authentication, group-based access filtering, caching, and evaluation so teams can build useful assistants while keeping sensitive workflows under their own control.

---

## Features

| Feature | Detail |
|---|---|
| **PDF extraction** | `unstructured` (hi-res) for prose, `pdfplumber` for tables |
| **Section-aware chunking** | Groups content by Title/Header boundaries; large sections split at paragraphs |
| **Embeddings** | OpenAI `text-embedding-3-small` with file + Redis cache |
| **Vector store** | ChromaDB (persistent) with upsert |
| **Reranking** | Optional Cohere `rerank-english-v3.0` (falls back to vector scores) |
| **Answer generation** | OpenAI `gpt-4o-mini` with grounded system prompt |
| **API** | FastAPI with `X-API-Key` authentication and access-level filtering |
| **Chat UI** | Streamlit with citations and latency display |
| **Monitoring** | MLflow experiment tracking — params, metrics, per-trace artifacts |
| **Caching** | Redis (24 h TTL on answers) with file-based fallback |
| **Access control** | `user_group` on requests filters chunks by `access_level` metadata |
| **CI** | GitHub Actions: ruff lint + smoke tests (no API keys needed) |

---

## Project layout

```
rag_assistant/        # Core package
  config.py           # All settings, loaded from .env
  cache.py            # Redis (24 h TTL) + file fallback cache
  pdf_extractor.py    # PDF -> elements + tables
  chunker.py          # Section-aware chunking
  embedder.py         # Batched OpenAI embeddings
  vector_store.py     # ChromaDB ingestion
  retriever.py        # Vector search + optional Cohere rerank + access control
  generator.py        # Full RAG pipeline
api.py                # FastAPI service (X-API-Key authenticated)
app.py                # Streamlit chat UI
query.py              # CLI query tool
evals.py              # Evaluation harness + MLflow logging
scripts/
  ci_test.py          # Standalone smoke tests (no API keys needed)
.github/workflows/
  ci.yml              # GitHub Actions: lint + smoke tests on every push/PR
Dockerfile            # Multi-stage build for the FastAPI service
docker-compose.yml    # API + Redis services
.env.example          # Template for required environment variables
```

---

## Setup

### 1. Create a virtual environment

```bash
git clone <repo-url>
cd rag-assistant
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

### 3. Add documents and index

Place PDF files in `data/`, then run:

```bash
python -m rag_assistant.vector_store
```

---

## Running each component

### FastAPI service

```bash
uvicorn api:app --reload
# Available at http://localhost:8000
```

Swagger UI is available at [`/docs`](http://localhost:8000/docs) when the API service is running.

All endpoints except `/health` require an `X-API-Key` header:

```bash
# Health check (public)
curl http://localhost:8000/health

# Query (authenticated)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <YOUR_API_KEY>" \
  -d '{"question": "What is a business associate under HIPAA?", "user_group": "clinical"}'
```

### Streamlit chat UI

```bash
streamlit run app.py
```

### CLI query tool

```bash
python query.py "What must a valid HIPAA authorization include?"
```

### Evaluations

Full eval run (requires OpenAI key):
```bash
python evals.py
```

CI smoke tests only (no API keys needed):
```bash
python evals.py --ci
# or equivalently
python scripts/ci_test.py
```

View MLflow results after a full eval run:
```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

---

## Docker

### API + Redis via Docker Compose

```bash
cp .env.example .env   # fill in OPENAI_API_KEY
docker compose up --build
# API at http://localhost:8000, Redis at localhost:6379
```

The `REDIS_URL` is automatically set to `redis://redis:6379/0` inside the compose network.

### API standalone (file-cache fallback, no Redis)

```bash
docker build -t rag-api .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e API_KEYS=my-key \
  -v $(pwd)/chroma_db:/app/chroma_db:ro \
  -v $(pwd)/data:/app/data:ro \
  rag-api
```

---

## Access control

Every chunk has an `access_level` metadata field. Pass `user_group` on `/query` to filter:

| `user_group` | Sees |
|---|---|
| `public` | `public` chunks |
| `clinical` | `public` + `clinical` |
| `billing` | `public` + `billing` |
| `admin` | all levels |
| *(omitted)* | all levels |

Group mappings are defined in `rag_assistant/config.py` (`GROUP_ACCESS_MAP`). All existing chunks default to `public`; set `access_level` in metadata at index time to restrict them.

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key for embeddings + chat |
| `API_KEYS` | Yes | `dev-key-123` | Comma-separated valid `X-API-Key` values |
| `COHERE_API_KEY` | No | — | Enables Cohere reranking when set |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis connection URL |

See [.env.example](.env.example) for the full annotated template.
