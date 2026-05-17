"""FastAPI service for the Healthcare RAG Assistant."""

import json
import logging
import os
import random
import threading

import chromadb
from fastapi import FastAPI, HTTPException, Security, Depends, UploadFile, File, Query, Header
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.status import HTTP_401_UNAUTHORIZED

from rag_assistant.generator import answer_question, stream_answer, build_context
from rag_assistant.config import API_KEYS, CHROMA_DIR, COLLECTION_NAME, DATA_DIR, STORAGE_DIR
from rag_assistant.cache import cache_backend
from rag_assistant.db import (
    init_db, create_chat, list_chats, get_chat, delete_chat, delete_all_chats,
    add_message, get_messages, get_conn,
    save_feedback, save_query_log, purge_old_logs,
    create_user, get_user_by_username, get_user_by_email_or_username,
    get_user_by_id, update_last_login, update_user_display_name,
    update_user_password, list_users, get_analytics,
)
from rag_assistant.retriever import retrieve
from rag_assistant.query_rewriter import rewrite_query
from rag_assistant.auth import verify_password, hash_password, create_token, decode_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Fetch AI")
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET", "fallback-dev-secret-only"),
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _startup() -> None:
    init_db()

    try:
        deleted = purge_old_logs(days=7)
        logger.info("Purged %d old query log entries", deleted)
    except Exception as exc:
        logger.warning("Log purge failed: %s", exc)

    try:
        from rag_assistant.vector_store import migrate_existing_chunks
        migrate_existing_chunks()
    except Exception as exc:
        logger.warning("Chunk migration failed: %s", exc)

    try:
        from rag_assistant.hybrid_retriever import build_bm25_index
        n = build_bm25_index()
        logger.info("BM25 index built: %d chunks", n)
    except Exception as exc:
        logger.warning("BM25 index build failed: %s", exc)

    try:
        if not get_user_by_username("admin"):
            create_user("admin", "admin123", display_name="Administrator", role="admin")
            logger.info("Created default admin user (admin/admin123)")
    except Exception as exc:
        logger.warning("Default admin user creation failed: %s", exc)

    try:
        from rag_assistant.google_oauth import is_google_oauth_configured
        if is_google_oauth_configured():
            logger.info("Google OAuth: configured")
        else:
            logger.info("Google OAuth: not configured (set GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET to enable)")
    except Exception:
        pass

    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        if collection.count() == 0:
            pdf_files = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
            if pdf_files:
                logger.info("Vector store empty but %d PDF(s) found — auto-indexing...", len(pdf_files))
                from rag_assistant.vector_store import index_all_pdfs
                def _bg_index():
                    try:
                        result = index_all_pdfs()
                        count = result.count() if result is not None else 0
                        logger.info("Auto-indexing complete: %d chunks indexed.", count)
                    except Exception as e:
                        logger.error("Auto-indexing failed: %s", e)
                threading.Thread(target=_bg_index, daemon=True).start()
                logger.info("Auto-indexing started in background.")
            else:
                logger.warning(
                    "No documents indexed. "
                    "Call POST /upload to add PDFs or POST /chats/{id}/upload for per-chat docs."
                )
        else:
            logger.info("Vector store ready: %d chunks in '%s'.", collection.count(), COLLECTION_NAME)
    except Exception as exc:
        logger.warning("Could not check vector store at startup: %s", exc)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_api_key(key: str | None = Security(_api_key_header)) -> str:
    if not key or key not in API_KEYS:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )
    return key


get_api_key = _require_api_key


def _require_auth(
    api_key: str | None = Security(_api_key_header),
    authorization: str | None = Header(default=None),
) -> dict:
    """Accept either X-API-Key or Bearer JWT."""
    if api_key and api_key in API_KEYS:
        return {"user_id": "api", "username": "api", "role": "admin"}
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        payload = decode_token(token)
        if payload:
            return payload
    raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid or missing credentials")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryDebugRequest(BaseModel):
    question: str
    top_k: int = 5
    session_id: str = "global"


class SuggestFollowupsRequest(BaseModel):
    question: str
    answer: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_cache: bool = True
    user_group: str | None = None
    chat_id: str | None = None
    mode: str = "chat"
    answer_style: str = "detailed"


class SourceInfo(BaseModel):
    source: str
    page: int
    chunk_id: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceInfo]
    from_cache: bool
    latency_s: float | None = None
    token_count: int | None = None
    chat_id: str | None = None


class FeedbackRequest(BaseModel):
    chat_id: str
    message_id: str | None = None
    question: str
    answer: str
    rating: int


class RegisterRequest(BaseModel):
    display_name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "member"
    email: str | None = None
    display_name: str | None = None


class UpdateProfileRequest(BaseModel):
    display_name: str


class UpdatePasswordRequest(BaseModel):
    current_password: str
    new_password: str


# ---------------------------------------------------------------------------
# Auth endpoints (public)
# ---------------------------------------------------------------------------

@app.get("/login")
def login_page():
    return RedirectResponse(url="/login.html")


@app.get("/settings")
def settings_page():
    return RedirectResponse(url="/settings.html")


@app.post("/auth/register")
def auth_register(request: RegisterRequest):
    # Auto-generate username from email
    base = request.email.split("@")[0].lower().replace(".", "_")
    suffix = str(random.randint(1000, 9999))
    username = f"{base}{suffix}"
    try:
        user = create_user(
            username=username,
            password=request.password,
            display_name=request.display_name,
            email=request.email,
            role="member",
        )
        token = create_token(user["id"], user["username"], user["role"])
        return {
            "token": token,
            "user_id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
            "role": user["role"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/auth/login")
def auth_login(request: LoginRequest):
    user = get_user_by_email_or_username(request.email)
    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    update_last_login(user["id"])
    token = create_token(user["id"], user["username"], user["role"])
    return {
        "token": token,
        "username": user["username"],
        "display_name": user.get("display_name") or user["username"],
        "role": user["role"],
        "user_id": user["id"],
        "email": user.get("email"),
    }


@app.get("/auth/me")
def auth_me(user: dict = Depends(_require_auth)):
    payload = {k: v for k, v in user.items() if k != "exp"}
    # Enrich with display_name from DB if we have a real user_id
    uid = user.get("sub") or user.get("user_id")
    if uid and uid != "api":
        db_user = get_user_by_id(uid)
        if db_user:
            payload["display_name"] = db_user.get("display_name") or db_user["username"]
            payload["email"] = db_user.get("email")
    return payload


@app.put("/auth/profile")
def auth_update_profile(request: UpdateProfileRequest, user: dict = Depends(_require_auth)):
    uid = user.get("sub") or user.get("user_id")
    if not uid or uid == "api":
        raise HTTPException(status_code=400, detail="Profile update requires JWT login")
    update_user_display_name(uid, request.display_name)
    db_user = get_user_by_id(uid)
    return {"display_name": request.display_name, "username": db_user["username"] if db_user else ""}


@app.put("/auth/password")
def auth_update_password(request: UpdatePasswordRequest, user: dict = Depends(_require_auth)):
    uid = user.get("sub") or user.get("user_id")
    if not uid or uid == "api":
        raise HTTPException(status_code=400, detail="Password update requires JWT login")
    db_user = get_user_by_id(uid)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_password(request.current_password, db_user["password_hash"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    update_user_password(uid, hash_password(request.new_password))
    return {"success": True}


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------

@app.get("/chats", dependencies=[Depends(_require_auth)])
def get_chats_list(user: dict = Depends(_require_auth)):
    uid = user.get("sub") or user.get("user_id")
    return list_chats(user_id=uid if uid != "api" else None)


@app.post("/chats")
def new_chat(user: dict = Depends(_require_auth)):
    uid = user.get("sub") or user.get("user_id")
    return create_chat(user_id=uid if uid != "api" else None)


@app.get("/chats/{chat_id}", dependencies=[Depends(_require_auth)])
def get_chat_detail(chat_id: str):
    chat = get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {**chat, "messages": get_messages(chat_id)}


@app.delete("/chats/all")
def delete_all_user_chats(user: dict = Depends(_require_auth)):
    uid = user.get("sub") or user.get("user_id")
    if not uid or uid == "api":
        raise HTTPException(status_code=400, detail="Requires JWT login")
    # Gather chat ids before deleting so we can remove vectors
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id FROM chats WHERE user_id=? OR user_id IS NULL", (uid,)
        ).fetchall()
    chat_ids = [r[0] for r in rows]

    deleted_chunks = 0
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        for cid in chat_ids:
            res = collection.get(where={"session_id": {"$eq": cid}}, include=[])
            if res["ids"]:
                collection.delete(ids=res["ids"])
                deleted_chunks += len(res["ids"])
    except Exception as e:
        logger.warning("Could not delete session vectors during delete_all: %s", e)

    deleted_chats = delete_all_chats(uid)
    return {"deleted_chats": deleted_chats, "deleted_chunks": deleted_chunks}


@app.delete("/chats/{chat_id}", dependencies=[Depends(_require_auth)])
def remove_chat(chat_id: str):
    if not get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    delete_chat(chat_id)
    def _bg_delete():
        try:
            from rag_assistant.vector_store import delete_session_chunks
            delete_session_chunks(chat_id)
        except Exception as e:
            logger.error("Failed to delete session chunks for %s: %s", chat_id, e)
    threading.Thread(target=_bg_delete, daemon=True).start()
    return {"status": "deleted"}


@app.post("/chats/{chat_id}/upload", dependencies=[Depends(_require_auth)])
async def upload_to_chat(chat_id: str, file: UploadFile = File(...)):
    if not get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    session_dir = STORAGE_DIR / "sessions" / chat_id
    session_dir.mkdir(parents=True, exist_ok=True)
    dest = session_dir / file.filename

    try:
        contents = await file.read()
        dest.write_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    filename = file.filename
    def _bg_index():
        try:
            from rag_assistant.vector_store import index_pdf_for_session
            index_pdf_for_session(dest, session_id=chat_id)
            logger.info("Session upload indexed: %s for chat %s", filename, chat_id)
            _fire_webhook("document_indexed", filename, None, chat_id)
        except Exception as e:
            logger.error("Session upload indexing failed: %s", e)
    threading.Thread(target=_bg_index, daemon=True).start()
    return {"status": "upload_received_indexing_started", "filename": filename}


@app.get("/chats/{chat_id}/documents", dependencies=[Depends(_require_auth)])
def get_chat_documents(chat_id: str):
    if not get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        results = collection.get(
            where={"session_id": {"$eq": chat_id}},
            include=["metadatas"],
        )
        counts: dict[str, int] = {}
        for meta in results["metadatas"]:
            src = meta.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return [{"source": src, "chunk_count": cnt} for src, cnt in sorted(counts.items())]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Core query endpoints
# ---------------------------------------------------------------------------

@app.post("/index", dependencies=[Depends(_require_auth)])
def index():
    try:
        from rag_assistant.vector_store import index_all_pdfs
        result = index_all_pdfs()
        count = result.count() if result is not None else 0
        return {"status": "complete", "chunks": count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/upload")
async def upload(file: UploadFile = File(...), user: dict = Depends(_require_auth)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can upload to the Knowledge Base")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest = DATA_DIR / file.filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        contents = await file.read()
        dest.write_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    filename = file.filename
    def _bg_index():
        try:
            from rag_assistant.vector_store import index_single_pdf
            collection = index_single_pdf(dest)
            count = collection.count() if collection is not None else 0
            logger.info("Background upload index complete: %d chunks", count)
            _fire_webhook("document_indexed", filename, count, "global")
        except Exception as e:
            logger.error("Background upload index failed: %s", e)
    threading.Thread(target=_bg_index, daemon=True).start()
    return {
        "status": "upload_received_indexing_started",
        "filename": filename,
        "message": "Indexing running in background. Watch deploy logs.",
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, user: dict = Depends(_require_auth)):
    active_chat_id = request.chat_id
    if active_chat_id == "new":
        uid = user.get("sub") or user.get("user_id")
        chat = create_chat(user_id=uid if uid != "api" else None)
        active_chat_id = chat["id"]

    session_id = active_chat_id if active_chat_id else "global"

    try:
        result = answer_question(
            question=request.question,
            top_k=request.top_k,
            use_cache=request.use_cache,
            user_group=request.user_group,
            session_id=session_id,
            answer_style=request.answer_style,
        )
        if active_chat_id:
            add_message(active_chat_id, "user", request.question)
            add_message(active_chat_id, "assistant", result["answer"],
                        json.dumps(result["sources"]))
            result["chat_id"] = active_chat_id
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query/suggest-followups", dependencies=[Depends(_require_auth)])
def suggest_followups(request: SuggestFollowupsRequest):
    from openai import OpenAI
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate exactly 3 short follow-up questions a user might ask "
                        "after receiving this answer about clinical/healthcare documents. "
                        "Return ONLY a JSON array of 3 strings, nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {request.question}\n\nAnswer: {request.answer}",
                },
            ],
            temperature=0,
            max_tokens=150,
        )
        raw = resp.choices[0].message.content.strip()
        questions = json.loads(raw)
        if not isinstance(questions, list):
            questions = []
    except Exception:
        questions = []
    return {"questions": questions[:3]}


@app.post("/query/stream")
async def query_stream(request: QueryRequest, user: dict = Depends(_require_auth)):
    active_chat_id = request.chat_id
    if active_chat_id == "new":
        uid = user.get("sub") or user.get("user_id")
        chat = create_chat(user_id=uid if uid != "api" else None)
        active_chat_id = chat["id"]

    session_id = active_chat_id if active_chat_id else "global"

    history: list[dict] = []
    if active_chat_id:
        history = get_messages(active_chat_id)

    collected: dict = {"answer": "", "sources": [], "faithfulness": None, "from_cache": False}

    def _event_stream():
        for chunk in stream_answer(
            question=request.question,
            top_k=request.top_k,
            user_group=request.user_group,
            session_id=session_id,
            history=history,
            mode=request.mode,
            answer_style=request.answer_style,
        ):
            try:
                raw = chunk.removeprefix("data: ").strip()
                payload = json.loads(raw)
                t = payload.get("type")
                if t == "answer_chunk":
                    collected["answer"] += payload.get("content", "")
                elif t == "sources":
                    collected["sources"] = payload.get("sources", [])
                elif t == "faithfulness":
                    collected["faithfulness"] = payload.get("score")
                elif t == "done":
                    collected["from_cache"] = payload.get("from_cache", False)
            except Exception:
                pass
            yield chunk

        if active_chat_id:
            add_message(active_chat_id, "user", request.question)
            add_message(active_chat_id, "assistant", collected["answer"],
                        json.dumps(collected["sources"]))
            yield f"data: {json.dumps({'type': 'chat_id', 'chat_id': active_chat_id})}\n\n"

        try:
            save_query_log(
                chat_id=active_chat_id or "global",
                question=request.question,
                rewritten=request.question,
                answer_preview=collected["answer"][:200],
                sources_count=len(collected["sources"]),
                latency_ms=None,
                from_cache=collected["from_cache"],
                faithfulness_score=collected["faithfulness"],
            )
        except Exception as exc:
            logger.warning("save_query_log failed: %s", exc)

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

@app.post("/feedback", dependencies=[Depends(_require_auth)])
def submit_feedback(request: FeedbackRequest):
    if request.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="rating must be 1 or -1")
    try:
        return save_feedback(request.chat_id, request.message_id,
                             request.question, request.answer, request.rating)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------

@app.get("/admin/documents", dependencies=[Depends(_require_auth)])
def list_global_documents():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        if collection.count() == 0:
            return []
        results = collection.get(
            where={"session_id": {"$eq": "global"}},
            include=["metadatas"],
        )
        counts: dict[str, int] = {}
        for meta in results["metadatas"]:
            src = meta.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return [{"source": src, "chunk_count": cnt} for src, cnt in sorted(counts.items())]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/documents")
def documents():
    """Public — no auth required."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        if collection.count() == 0:
            return []
        results = collection.get(
            where={"session_id": {"$eq": "global"}},
            include=["metadatas"],
        )
        counts: dict[str, int] = {}
        for meta in results["metadatas"]:
            src = meta.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return [{"source": src, "chunk_count": cnt} for src, cnt in sorted(counts.items())]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/admin/stats", dependencies=[Depends(_require_auth)])
def admin_stats():
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        total_chunks = collection.count()

        global_results = collection.get(
            where={"session_id": {"$eq": "global"}},
            include=["metadatas"],
        )
        global_chunks = len(global_results["ids"])
        counts: dict[str, int] = {}
        for meta in global_results["metadatas"]:
            src = meta.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        docs = [{"source": s, "chunk_count": c} for s, c in sorted(counts.items())]

        with get_conn() as conn:
            total_chats = conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
            total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

        return {
            "total_chunks": total_chunks,
            "global_chunks": global_chunks,
            "session_chunks": total_chunks - global_chunks,
            "total_chats": total_chats,
            "total_messages": total_messages,
            "documents": docs,
            "cache_backend": cache_backend(),
            "storage_dir": str(STORAGE_DIR),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/admin/documents", dependencies=[Depends(_require_auth)])
def delete_global_document_by_param(filename: str = Query(...)):
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        results = collection.get(where={"source": {"$eq": filename}}, include=[])
        if not results["ids"]:
            raise HTTPException(status_code=404, detail=f"{filename} not found")
        collection.delete(ids=results["ids"])
        return {"deleted": filename, "chunks_removed": len(results["ids"])}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/reindex", dependencies=[Depends(_require_auth)])
def admin_reindex():
    pdf_files = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    filenames = [f.name for f in pdf_files]
    def _bg():
        try:
            from rag_assistant.vector_store import index_all_pdfs
            result = index_all_pdfs()
            count = result.count() if result is not None else 0
            logger.info("Admin reindex complete: %d chunks", count)
        except Exception as e:
            logger.error("Admin reindex failed: %s", e)
    threading.Thread(target=_bg, daemon=True).start()
    return {"status": "reindexing_started", "files": filenames}


@app.post("/admin/query-debug", dependencies=[Depends(_require_auth)])
def admin_query_debug(request: QueryDebugRequest):
    try:
        rewritten = rewrite_query(request.question)
        hits = retrieve(rewritten, top_k=request.top_k, session_id=request.session_id)
        context = build_context(hits)
        return {
            "original_question": request.question,
            "rewritten_question": rewritten,
            "chunks_retrieved": [
                {
                    "chunk_id": h["chunk_id"],
                    "source": h["metadata"].get("source", ""),
                    "page": h["metadata"].get("page_number", 0),
                    "chunk_type": h["metadata"].get("chunk_type", "text"),
                    "session_id": h["metadata"].get("session_id", "global"),
                    "content_preview": h["content"][:200],
                    "distance": round(h.get("distance", 0.0), 4),
                }
                for h in hits
            ],
            "context_sent_to_llm": context,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/admin/sessions", dependencies=[Depends(_require_auth)])
def admin_sessions():
    try:
        with get_conn() as conn:
            rows = conn.execute("""
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       COUNT(m.id) as message_count
                FROM chats c
                LEFT JOIN messages m ON m.chat_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
            """).fetchall()
        chats = [dict(r) for r in rows]

        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        for chat in chats:
            try:
                res = collection.get(
                    where={"session_id": {"$eq": chat["id"]}},
                    include=["metadatas"],
                )
                sources = {m.get("source") for m in res["metadatas"]}
                chat["document_count"] = len(sources)
            except Exception:
                chat["document_count"] = 0

        return chats
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/admin/sessions/{chat_id}", dependencies=[Depends(_require_auth)])
def admin_delete_session(chat_id: str):
    if not get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    chunks_removed = 0
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        res = collection.get(where={"session_id": {"$eq": chat_id}}, include=[])
        if res["ids"]:
            collection.delete(ids=res["ids"])
        chunks_removed = len(res["ids"])
    except Exception as e:
        logger.warning("Failed to remove session chunks for %s: %s", chat_id, e)
    delete_chat(chat_id)
    return {"deleted": chat_id, "chunks_removed": chunks_removed}


@app.delete("/admin/documents/{filename}", dependencies=[Depends(_require_auth)])
def delete_global_document(filename: str):
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        results = collection.get(where={"source": {"$eq": filename}}, include=[])
        if not results["ids"]:
            raise HTTPException(status_code=404, detail=f"{filename} not found")
        collection.delete(ids=results["ids"])
        return {"deleted": filename, "chunks_removed": len(results["ids"])}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/admin/logs", dependencies=[Depends(_require_auth)])
def admin_logs(days: int = 7, limit: int = 100):
    try:
        cutoff = (__import__("datetime").datetime.utcnow()
                  - __import__("datetime").timedelta(days=days)).isoformat()
        with get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM query_logs
                   WHERE created_at >= ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (cutoff, limit),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/admin/feedback", dependencies=[Depends(_require_auth)])
def admin_feedback():
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC"
            ).fetchall()
        items = [dict(r) for r in rows]
        total = len(items)
        good = sum(1 for r in items if r["rating"] == 1)
        bad = total - good
        return {
            "summary": {
                "total": total,
                "good": good,
                "bad": bad,
                "good_pct": round(good / total * 100) if total else 0,
            },
            "items": items,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/admin/analytics", dependencies=[Depends(_require_auth)])
def admin_analytics():
    try:
        return get_analytics()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/admin/users")
def admin_users(user: dict = Depends(_require_auth)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    try:
        return list_users()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/users")
def admin_create_user(request: CreateUserRequest, user: dict = Depends(_require_auth)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    try:
        return create_user(
            request.username, request.password,
            display_name=request.display_name,
            email=request.email, role=request.role,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health():
    return {"status": "ok", "cache_backend": cache_backend()}


# ---------------------------------------------------------------------------
# Google OAuth (stubs, activated when env vars are set)
# ---------------------------------------------------------------------------

@app.get("/auth/google")
async def google_auth_start(request: Request):
    from rag_assistant.google_oauth import is_google_oauth_configured
    if not is_google_oauth_configured():
        return {"error": "Google OAuth not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET env vars."}
    try:
        from rag_assistant.google_oauth import oauth
        redirect_uri = str(request.url_for("google_auth_callback"))
        return await oauth.google.authorize_redirect(request, redirect_uri)
    except Exception as exc:
        logger.error("Google OAuth redirect error: %s", exc)
        return {"error": "Google OAuth redirect failed. Ensure GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET are valid.", "detail": str(exc)}


@app.get("/auth/google/callback", name="google_auth_callback")
async def google_auth_callback(request: Request):
    from rag_assistant.google_oauth import oauth, is_google_oauth_configured
    if not is_google_oauth_configured():
        return RedirectResponse(url="/login?error=oauth_not_configured")
    try:
        token_data = await oauth.google.authorize_access_token(request)
        user_info = token_data.get("userinfo") or {}
        email = user_info.get("email")
        name = user_info.get("name", email)
        if not email:
            return RedirectResponse(url="/login?error=oauth_no_email")
        # Find or create user
        db_user = get_user_by_email_or_username(email)
        if not db_user:
            import secrets
            db_user = create_user(
                username=email.split("@")[0] + str(random.randint(1000, 9999)),
                password=secrets.token_hex(32),
                display_name=name,
                email=email,
                role="member",
            )
            db_user = get_user_by_email_or_username(email)
        update_last_login(db_user["id"])
        jwt_token = create_token(db_user["id"], db_user["username"], db_user["role"])
        return RedirectResponse(url=f"/#token={jwt_token}")
    except Exception as exc:
        logger.error("Google OAuth callback error: %s", exc)
        return RedirectResponse(url="/login?error=oauth_error")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fire_webhook(event: str, filename: str, chunks: int | None, session_id: str):
    import os, requests as req_lib
    webhook_url = os.environ.get("WEBHOOK_URL", "")
    if not webhook_url:
        return
    try:
        req_lib.post(webhook_url, json={
            "event": event,
            "filename": filename,
            "chunks": chunks,
            "session_id": session_id,
        }, timeout=5)
    except Exception:
        pass


# Mount frontend AFTER all API routes so API paths take priority
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
