import json
import time
import logging
import re
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
    "You are Sourcebound, a helpful assistant that answers questions about uploaded documents and the Sourcebound app. "
    "Answer ONLY using the information in the provided context. "
    "If the context does not contain enough information to answer the question, say so explicitly. "
    "When you use a source, reference it inline as (Source N) where N is the source number. "
    "Do NOT include the filename or page number inline — those will be shown separately."
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

_CONCISE_RE = re.compile(
    r"\b("
    r"brief|briefly|short|shorter|summarize|summary|sum up|tldr|tl;dr|"
    r"concise|quick|simple|simplify|in\s+\d+\s+(?:sentences?|lines?|words?)|"
    r"one\s+(?:sentence|paragraph)|few\s+(?:sentences|lines|words)"
    r")\b",
    re.IGNORECASE,
)
_BULLETS_RE = re.compile(r"\b(bullets?|bullet points?|list|checklist)\b", re.IGNORECASE)
_DETAILED_RE = re.compile(
    r"\b(detailed|detail|thorough|comprehensive|deep dive|explain fully|more context)\b",
    re.IGNORECASE,
)
_NO_SOURCE_ANSWER_RE = re.compile(
    r"(provided context|context|sources?)\s+"
    r"(does not|doesn't|do not|don't)\s+contain|"
    r"not enough information|insufficient information|"
    r"cannot provide an answer|cannot answer|can't answer|"
    r"no information (?:about|on|regarding)",
    re.IGNORECASE,
)

# Matches inline citations like (Source 1), (Source 1 | file.md | page 2), [Source 1 | ...]
_INLINE_CITATION_RE = re.compile(
    r"[\[\(]Source\s+\d+[^\]\)]*[\]\)]",
    re.IGNORECASE,
)
_CITED_INDEX_RE = re.compile(r"[\[\(]Source\s+(\d+)", re.IGNORECASE)


def extract_cited_indices(answer: str) -> set[int]:
    return {int(m) for m in _CITED_INDEX_RE.findall(answer)}


def strip_inline_citations(answer: str) -> str:
    cleaned = _INLINE_CITATION_RE.sub("", answer)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.])", r"\1", cleaned)
    return cleaned.strip()


def resolve_answer_style(question: str, requested_style: str = "detailed") -> str:
    """Infer answer style from the user's latest wording, overriding UI defaults."""
    question = question or ""
    requested_style = requested_style if requested_style in _STYLE_INSTRUCTIONS else "detailed"
    if _BULLETS_RE.search(question):
        return "bullets"
    if _CONCISE_RE.search(question):
        return "concise"
    if _DETAILED_RE.search(question):
        return "detailed"
    return requested_style


def answer_uses_sources(answer: str) -> bool:
    """Return False when the model explicitly says the context cannot answer."""
    return not _NO_SOURCE_ANSWER_RE.search(answer or "")


def build_sources(hits: list[dict], answer: str) -> list[dict]:
    if not answer_uses_sources(answer):
        return []
    cited = extract_cited_indices(answer)
    seen: set[str] = set()
    result = []
    for i, h in enumerate(hits, start=1):
        if cited and i not in cited:
            continue
        src = h["metadata"]["source"]
        if src in seen:
            continue
        seen.add(src)
        result.append({
            "source": src,
            "page": h["metadata"]["page_number"],
            "chunk_id": h["chunk_id"],
        })
    return result


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
        f"Format: {style_instruction}\n"
        "If the user's latest message asks for a shorter, simpler, summarized, or reformatted "
        "version of the previous answer, follow that instruction even when prior answers were long."
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
    effective_style = resolve_answer_style(question, answer_style)
    if use_cache:
        cached = get_semantic_cache(
            question,
            answer_style=effective_style,
            tenant_id=tenant_id,
            session_id=session_id,
        )
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
        messages=_build_messages(context, question, history, answer_style=effective_style),
        temperature=0,
    )
    latency_s = round(time.time() - t0, 3)

    raw_answer = response.choices[0].message.content
    sources = build_sources(hits, raw_answer)
    clean_answer = strip_inline_citations(raw_answer)
    result = {
        "question": question,
        "answer": clean_answer,
        "sources": sources,
        "from_cache": False,
        "answer_style": effective_style,
        "tenant_id": tenant_id,
        "session_id": session_id,
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
    use_cache: bool = True,
) -> Generator[str, None, None]:
    """Yield SSE-formatted chunks for streaming responses."""

    effective_style = resolve_answer_style(question, answer_style)

    # 1. Semantic cache check — yield full answer as single chunk and return
    cached = get_semantic_cache(
        question,
        answer_style=effective_style,
        tenant_id=tenant_id,
        session_id=session_id,
    ) if use_cache else None
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
        messages=_build_messages(context, question, history, system_prompt=system_prompt, answer_style=effective_style),
        temperature=0,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            full_answer += delta
            yield f"data: {json.dumps({'type': 'answer_chunk', 'content': delta})}\n\n"

    # 5. Filter sources to only those cited, strip inline citations from displayed answer
    sources = build_sources(hits, full_answer)
    clean_answer = strip_inline_citations(full_answer)
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'clean_answer': clean_answer})}\n\n"

    # 6. Faithfulness score (LLM-evaluated)
    faithfulness = score_faithfulness(question, full_answer, context)
    yield f"data: {json.dumps({'type': 'faithfulness', 'score': faithfulness})}\n\n"
    yield f"data: {json.dumps({'type': 'done', 'from_cache': False, 'faithfulness': faithfulness})}\n\n"

    # 7. Save to semantic cache (store clean answer)
    if use_cache:
        save_semantic_cache(question, {
            "question": question,
            "answer": clean_answer,
            "sources": sources,
            "from_cache": False,
            "answer_style": effective_style,
            "tenant_id": tenant_id,
            "session_id": session_id,
        })
