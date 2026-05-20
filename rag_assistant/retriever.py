import os
import re
import logging

from dotenv import load_dotenv
from openai import OpenAI
import chromadb

from .config import (
    CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K,
    COHERE_API_KEY, RERANK_CANDIDATES, RERANK_TOP_N,
    GROUP_ACCESS_MAP, DEFAULT_ACCESS_LEVEL, get_collection_name,
)
from .hybrid_retriever import bm25_search, reciprocal_rank_fusion

load_dotenv()
logger = logging.getLogger(__name__)

try:
    import cohere as _cohere_module
    _COHERE_AVAILABLE = True
except ImportError:
    _COHERE_AVAILABLE = False


def _reranking_enabled() -> bool:
    key = COHERE_API_KEY or os.getenv("COHERE_API_KEY", "")
    return _COHERE_AVAILABLE and bool(key)


_APP_QUESTION_RE = re.compile(
    r"\b(fetch\s*ai|sourcebound|this\s+app|what\s+is\s+this|what\s+can\s+(i|you|we)\s+upload|"
    r"how\s+does\s+this\s+work|what\s+can\s+you\s+do|what\s+do\s+you\s+do|"
    r"tell\s+me\s+about\s+(this\s+)?app|about\s+you|who\s+are\s+you|"
    r"what\s+are\s+you|your\s+features?|what\s+is\s+fetch)\b",
    re.IGNORECASE,
)

APP_OVERVIEW_SOURCE = "fetch_ai_overview.md"


def is_app_question(query: str) -> bool:
    return bool(_APP_QUESTION_RE.search(query))


def _allowed_levels(user_group: str | None) -> list[str]:
    """Return the access levels this group may see.

    Unknown groups default to public-only. None means no filtering (admin
    shortcut used internally during indexing).
    """
    if user_group is None:
        return list(GROUP_ACCESS_MAP["admin"])  # unrestricted
    return GROUP_ACCESS_MAP.get(user_group, [DEFAULT_ACCESS_LEVEL])


def retrieve(
    query: str,
    top_k: int = TOP_K,
    user_group: str | None = None,
    session_id: str = "global",
    use_hybrid: bool = True,
    tenant_id: str = "default",
) -> list[dict]:
    """Find the most relevant chunks for a query.

    Args:
        query:      Natural-language question.
        top_k:      Final number of results to return.
        user_group: Access-control group (public/clinical/billing/admin/None).
        session_id: Scope retrieval to this session plus global chunks.
                    "global" → only global chunks.
                    Any other value → global + that session's chunks.

    If COHERE_API_KEY is set, retrieves RERANK_CANDIDATES from Chroma then
    reranks with Cohere to return top_k. Falls back to pure vector search
    when the key is absent or cohere is not installed.
    """
    use_reranking = _reranking_enabled()
    n_candidates = RERANK_CANDIDATES if use_reranking else top_k

    client = OpenAI()
    query_embedding = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    ).data[0].embedding

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    col_name = get_collection_name(tenant_id)
    try:
        collection = chroma_client.get_collection(name=col_name)
    except Exception:
        # Fallback to legacy collection if tenant collection doesn't exist yet
        try:
            collection = chroma_client.get_collection(name=COLLECTION_NAME)
        except Exception:
            return []

    n_candidates = min(n_candidates, collection.count())
    if n_candidates == 0:
        return []

    # Build session_id filter
    if session_id == "global":
        session_where = {"session_id": {"$eq": "global"}}
    else:
        session_where = {"$or": [
            {"session_id": {"$eq": "global"}},
            {"session_id": {"$eq": session_id}},
        ]}

    # Build access_level filter (only when user_group is explicitly set)
    access_where: dict | None = None
    if user_group is not None:
        allowed = _allowed_levels(user_group)
        if len(allowed) == 1:
            access_where = {"access_level": {"$eq": allowed[0]}}
        else:
            access_where = {"access_level": {"$in": allowed}}

    # Restrict to app overview doc when query is about the app itself
    source_where: dict | None = None
    if is_app_question(query):
        source_where = {"source": {"$eq": APP_OVERVIEW_SOURCE}}

    # Combine filters
    filters = [session_where]
    if access_where is not None:
        filters.append(access_where)
    if source_where is not None:
        filters.append(source_where)

    where: dict = {"$and": filters} if len(filters) > 1 else filters[0]

    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": n_candidates,
        "where": where,
    }

    results = collection.query(**query_kwargs)

    hits = [
        {
            "chunk_id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]

    if use_hybrid:
        bm25_hits = bm25_search(
            query,
            top_k=top_k * 3,
            session_filter=None if session_id == "global" else session_id,
        )
        ordered_ids = reciprocal_rank_fusion(hits, bm25_hits)

        hit_map = {h["chunk_id"]: h for h in hits}
        bm25_map = {h["id"]: h for h in bm25_hits}

        final_hits = []
        for cid in ordered_ids[:top_k]:
            if cid in hit_map:
                final_hits.append(hit_map[cid])
            elif cid in bm25_map:
                bh = bm25_map[cid]
                final_hits.append({
                    "chunk_id": cid,
                    "content": bh["content"],
                    "metadata": bh["metadata"],
                    "distance": 0.5,
                })
        return final_hits[:top_k]

    if not use_reranking:
        return hits[:top_k]

    # --- Cohere rerank ---
    try:
        key = COHERE_API_KEY or os.getenv("COHERE_API_KEY", "")
        co = _cohere_module.Client(api_key=key)
        docs = [h["content"] for h in hits]
        rerank_result = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=docs,
            top_n=min(top_k, len(docs)),
        )
        reranked = [hits[r.index] for r in rerank_result.results]
        logger.info(f"Reranked {len(hits)} candidates -> {len(reranked)} results")
        return reranked
    except Exception as exc:
        logger.warning(f"Cohere reranking failed ({exc}), falling back to vector scores")
        return hits[:top_k]


if __name__ == "__main__":
    query = "What information must a CSR verify before releasing beneficiary information?"
    print(f"Query: {query}\n")
    print(f"Reranking enabled: {_reranking_enabled()}\n")

    hits = retrieve(query, top_k=3)
    for i, hit in enumerate(hits, start=1):
        print(f"--- Result {i} (distance: {hit['distance']:.4f}) ---")
        print(f"ID: {hit['chunk_id']}")
        print(f"Page: {hit['metadata']['page_number']} | Type: {hit['metadata']['chunk_type']}")
        print(hit["content"][:300])
        print()
