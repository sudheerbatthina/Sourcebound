import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
_STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", "storage"))
STORAGE_DIR = _STORAGE_DIR          # public alias used by db.py
DATA_DIR = _STORAGE_DIR / "data"
CHROMA_DIR = str(_STORAGE_DIR / "chroma_db")
DATA_DIR.mkdir(parents=True, exist_ok=True)
Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(".cache")

# Collection
COLLECTION_NAME = "rag_chunks"
COLLECTION_PREFIX = os.environ.get("COLLECTION_PREFIX", "fetch_ai")


def get_collection_name(tenant_id: str) -> str:
    """Each tenant gets their own Chroma collection."""
    return f"{COLLECTION_PREFIX}_{tenant_id}"

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SECTION_MAX_CHARS = 1500

# Retrieval
TOP_K = 5

# Reranking (optional — requires COHERE_API_KEY in .env)
RERANK_TOP_N = 5
RERANK_CANDIDATES = 20
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# Embedding batching
EMBEDDING_BATCH_SIZE = 100

# MLflow
MLFLOW_EXPERIMENT_NAME = "rag-healthcare-assistant"
MLFLOW_TRACKING_URI = "mlruns"

# Redis (optional — falls back to file cache when unavailable)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ANSWER_CACHE_TTL = 86400  # 24 hours in seconds; embeddings have no TTL

# API authentication (comma-separated keys in .env, e.g. "key1,key2")
_raw_keys = os.getenv("API_KEYS", "dev-key-123")
API_KEYS: set[str] = {k.strip() for k in _raw_keys.split(",") if k.strip()}

# Access control — maps user_group → allowed access_levels
# Levels: public < clinical < billing < admin (each group sees its level + all below)
ACCESS_LEVEL_ORDER = ["public", "clinical", "billing", "admin"]

GROUP_ACCESS_MAP: dict[str, list[str]] = {
    "public":   ["public"],
    "clinical": ["public", "clinical"],
    "billing":  ["public", "billing"],
    "admin":    ["public", "clinical", "billing", "admin"],
}

# Default access level assigned to new chunks at index time
DEFAULT_ACCESS_LEVEL = "public"
