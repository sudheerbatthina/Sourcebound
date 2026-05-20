import json
import hashlib
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

try:
    import redis
    from rag_assistant.config import REDIS_URL, ANSWER_CACHE_TTL
    _redis = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
    _redis.ping()
    _BACKEND = "redis"
except Exception:
    _redis = None
    _BACKEND = "memory"

_mem_cache: dict = {}

SIMILARITY_THRESHOLD = 0.92
EMBED_MODEL = "text-embedding-3-small"

def _embed(text: str) -> list[float]:
    client = OpenAI()
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding

def _cosine(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _matches_metadata(answer: dict, answer_style: str | None, tenant_id: str | None, session_id: str | None) -> bool:
    """Keep cached answers scoped to the same response style and workspace context."""
    if answer_style and answer.get("answer_style") != answer_style:
        return False
    if tenant_id and answer.get("tenant_id") not in (None, tenant_id):
        return False
    if session_id and answer.get("session_id") not in (None, session_id):
        return False
    return True


def get_semantic_cache(
    question: str,
    answer_style: str | None = None,
    tenant_id: str | None = None,
    session_id: str | None = None,
):
    """Return cached answer if a semantically similar question was asked before."""
    q_emb = _embed(question)
    if _BACKEND == "redis":
        keys = _redis.keys("semcache:emb:*")
        for key in keys:
            raw = _redis.get(key)
            if not raw:
                continue
            cached_emb = json.loads(raw)
            if _cosine(q_emb, cached_emb) >= SIMILARITY_THRESHOLD:
                cache_id = key.replace("semcache:emb:", "")
                result = _redis.get(f"semcache:ans:{cache_id}")
                if result:
                    answer = json.loads(result)
                    if _matches_metadata(answer, answer_style, tenant_id, session_id):
                        return answer
    else:
        for cache_id, entry in _mem_cache.items():
            if _cosine(q_emb, entry["embedding"]) >= SIMILARITY_THRESHOLD:
                if _matches_metadata(entry["answer"], answer_style, tenant_id, session_id):
                    return entry["answer"]
    return None

def save_semantic_cache(question: str, answer: dict):
    """Save answer with question embedding for future semantic matching."""
    q_emb = _embed(question)
    cache_id = hashlib.md5(question.encode()).hexdigest()
    if _BACKEND == "redis":
        from rag_assistant.config import ANSWER_CACHE_TTL
        _redis.set(f"semcache:emb:{cache_id}", json.dumps(q_emb), ex=ANSWER_CACHE_TTL)
        _redis.set(f"semcache:ans:{cache_id}", json.dumps(answer), ex=ANSWER_CACHE_TTL)
    else:
        _mem_cache[cache_id] = {"embedding": q_emb, "answer": answer}
