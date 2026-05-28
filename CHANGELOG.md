# Changelog

All notable changes to Sourcebound will be documented in this file.

## v1.0.0 - 2026-05-28

- Initial open source release of Sourcebound.
- Private RAG assistant for document Q&A with a healthcare policy reference use case.
- PDF ingestion with `unstructured` for prose and `pdfplumber` for tables.
- Section-aware chunking for policy and operational documents.
- OpenAI embedding and grounded answer generation.
- Persistent ChromaDB vector storage with upsert support.
- Optional Cohere reranking for retrieved context.
- FastAPI service with `X-API-Key` authentication.
- Streamlit chat UI with citations and latency display.
- Group-based access filtering using document `access_level` metadata.
- Redis answer caching with file-based fallback.
- MLflow evaluation tracking with metrics and trace artifacts.
- Docker and Docker Compose support for API and Redis services.
- GitHub Actions CI with Ruff linting and smoke tests.
