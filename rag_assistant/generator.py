import json
import time
import logging
from typing import Generator

from dotenv import load_dotenv
from openai import OpenAI

from .config import CHAT_MODEL, TOP_K
from .retriever import retrieve
from .query_rewriter import rewrite_query_with_history
from .semantic_cache import get_semantic_cache, save_semantic_cache

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about healthcare policy documents. "
    "Answer ONLY using the information in the provided context. "
    "If the context does not contain enough information to answer the question, say so explicitly. "
    "Always cite the source document name and page number(s) you used in your answer."
)

RESEARCH_PREFIX = (
    "You are in RESEARCH MODE. Synthesize information from ALL provided sources "
    "comprehensively. Use headers and bullet points. Cite every claim."
)


def score_faithfulness(question: str, answer: str, context: str) -> int:
    """Score how grounded the answer is in the context. Returns 0-100."""
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a faithfulness evaluator for a RAG system. "
                    "Given a question, an answer, and the source context used to generate it, "
                    "score how well the answer is grounded in the context.\n\n"
                    "Score 0-100 where:\n"
                    "- 90-100: Every claim in the answer is directly supported by the context\n"
                    "- 70-89: Most claims supported, minor extrapolations\n"
                    "- 50-69: Some claims supported but notable unsupported additions\n"
                    "- 0-49: Answer contains significant claims not in the context (hallucination)\n\n"
                    'Return ONLY a JSON object: {"score": N, "flagged_claims": ["claim1", "claim2"]}\n'
                    "flagged_claims should list any specific claims NOT found in context (max 3). "
                    "If score >= 70, flagged_claims can be empty."
                )},
                {"role": "user", "content": (
                    f"Question: {question}\n\n"
                    f"Context:\n{context[:3000]}\n\n"
                    f"Answer:\n{answer}"
                )},
            ],
            temperature=0,
            max_tokens=200,
        )
        result = json.loads(response.choices[0].message.content)
        return max(0, min(100, int(result.get("score", 70))))
    except Exception:
        return 70  # neutral fallback on error


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        page = hit["metadata"]["page_number"]
        source = hit["metadata"]["source"]
        blocks.append(f"[Source {i} | {source} | page {page}]\n{hit['content']}")
    return "\n\n".join(blocks)


_STYLE_INSTRUCTIONS = {
    "concise": "Answer in 2-3 sentences maximum. Be direct.",
    "bullets": "Answer using bullet points only. Each point max 1 sentence.",
    "detailed": "Answer thoroughly with full explanation.",
}


def _build_messages(
    context: str,
    question: str,
    history: list[dict] | None,
    system_prompt: str = SYSTEM_PROMPT,
    answer_style: str = "detailed",
) -> list[dict]:
    """Build the OpenAI messages list with optional conversation history."""
    msgs = [{"role": "system", "content": system_prompt}]
    for msg in (history or [])[-6:]:   # last 3 turns (6 messages)
        if msg["role"] in ("user", "assistant"):
            msgs.append({"role": msg["role"], "content": msg["content"]})
    style_instruction = _STYLE_INSTRUCTIONS.get(answer_style, _STYLE_INSTRUCTIONS["detailed"])
    msgs.append({"role": "user", "content": (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Format: {style_instruction}"
    )})
    return msgs


def answer_question(
    question: str,
    top_k: int = TOP_K,
    use_cache: bool = True,
    user_group: str | None = None,
    session_id: str = "global",
    history: list[dict] | None = None,
    answer_style: str = "detailed",
    tenant_id: str = "default",
) -> dict:
    """Run the full RAG pipeline with query rewriting and semantic caching.

    Args:
        question:   The user's natural-language question.
        top_k:      Number of chunks to retrieve.
        use_cache:  Whether to read/write the semantic cache.
        user_group: Access-control group forwarded to retriever.
        session_id: Retrieval scope — "global" or a specific chat id.
        history:    Prior conversation messages for multi-turn context.

    Returned dict keys:
        question, answer, sources, from_cache, latency_s, token_count
    """
    if use_cache:
        cached = get_semantic_cache(question)
        if cached:
            logger.info("Semantic cache hit for question")
            cached["from_cache"] = True
            cached.setdefault("latency_s", None)
            cached.setdefault("token_count", None)
            return cached

    rewritten = rewrite_query_with_history(question, history or [])
    if rewritten != question:
        logger.info("Query rewritten: %r → %r", question, rewritten)

    hits = retrieve(rewritten, top_k=top_k, user_group=user_group, session_id=session_id, tenant_id=tenant_id)
    context = build_context(hits)

    client = OpenAI()
    t0 = time.time()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=_build_messages(context, question, history, answer_style=answer_style),
        temperature=0,
    )
    latency_s = round(time.time() - t0, 3)

    result = {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": [
            {
                "source": h["metadata"]["source"],
                "page": h["metadata"]["page_number"],
                "chunk_id": h["chunk_id"],
            }
            for h in hits
        ],
        "from_cache": False,
        "latency_s": latency_s,
        "token_count": response.usage.total_tokens if response.usage else None,
    }

    if use_cache:
        save_semantic_cache(question, result)

    return result


def stream_answer(
    question: str,
    top_k: int = TOP_K,
    user_group: str | None = None,
    session_id: str = "global",
    history: list[dict] | None = None,
    mode: str = "chat",
    answer_style: str = "detailed",
    tenant_id: str = "default",
) -> Generator[str, None, None]:
    """Yield SSE-formatted chunks for streaming responses."""

    # 1. Semantic cache check — yield full answer as single chunk and return
    cached = get_semantic_cache(question)
    if cached:
        yield f"data: {json.dumps({'type': 'answer_chunk', 'content': cached['answer']})}\n\n"
        yield f"data: {json.dumps({'type': 'sources', 'sources': cached['sources']})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'from_cache': True})}\n\n"
        return

    # 2. Rewrite query with conversation context
    rewritten = rewrite_query_with_history(question, history or [])
    if rewritten != question:
        logger.info("Stream query rewritten: %r → %r", question, rewritten)

    # 3. Retrieve — research mode uses 3× chunks up to 20
    effective_top_k = min(top_k * 3, 20) if mode == "research" else top_k
    hits = retrieve(rewritten, top_k=effective_top_k, user_group=user_group, session_id=session_id, tenant_id=tenant_id)
    context = build_context(hits)

    # 4. Stream from OpenAI
    system_prompt = (RESEARCH_PREFIX + " " + SYSTEM_PROMPT) if mode == "research" else SYSTEM_PROMPT
    client = OpenAI()
    full_answer = ""
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=_build_messages(context, question, history, system_prompt=system_prompt, answer_style=answer_style),
        temperature=0,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            full_answer += delta
            yield f"data: {json.dumps({'type': 'answer_chunk', 'content': delta})}\n\n"

    # 5. Send sources
    sources = [
        {
            "source": h["metadata"]["source"],
            "page": h["metadata"]["page_number"],
            "chunk_id": h["chunk_id"],
        }
        for h in hits
    ]
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

    # 6. Faithfulness score (LLM-evaluated)
    faithfulness = score_faithfulness(question, full_answer, context)
    yield f"data: {json.dumps({'type': 'faithfulness', 'score': faithfulness})}\n\n"
    yield f"data: {json.dumps({'type': 'done', 'from_cache': False, 'faithfulness': faithfulness})}\n\n"

    # 7. Save to semantic cache
    save_semantic_cache(question, {
        "question": question,
        "answer": full_answer,
        "sources": sources,
        "from_cache": False,
    })
