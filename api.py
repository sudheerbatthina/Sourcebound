"""FastAPI service for the Fetch AI RAG Assistant."""

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
from rag_assistant.config import API_KEYS, CHROMA_DIR, COLLECTION_NAME, DATA_DIR, STORAGE_DIR, get_collection_name
from rag_assistant.cache import cache_backend
from rag_assistant.db import (
    init_db, create_chat, list_chats, get_chat, delete_chat, delete_all_chats,
    add_message, get_messages, get_conn,
    save_feedback, save_query_log, purge_old_logs,
    create_user, get_user_by_username, get_user_by_email_or_username,
    get_user_by_id, update_last_login, update_user_display_name,
    update_user_password, list_users, get_analytics,
    create_tenant, get_tenant, get_tenant_by_slug, list_tenant_users,
    create_invite, get_invite_by_token, accept_invite, update_tenant_user_tenant,
    upsert_connector, get_connector, get_connector_by_id, list_connectors,
    update_connector_status, delete_connector,
    block_user, is_user_blocked, log_audit, get_audit_log,
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

def _migrate_to_tenant():
    """One-time migration: assign existing data to 'default' tenant."""
    default = get_tenant_by_slug("default")
    if not default:
        default = create_tenant("Default Workspace", slug="default")
        logger.info("Created default tenant: %s", default['id'])

    tid = default['id']

    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET tenant_id=? WHERE tenant_id IS NULL",
            (tid,)
        )
        conn.execute(
            "UPDATE chats SET tenant_id=? WHERE tenant_id IS NULL",
            (tid,)
        )
        conn.execute(
            "UPDATE query_logs SET tenant_id=? WHERE tenant_id IS NULL",
            (tid,)
        )

    logger.info("Migration: existing data assigned to default tenant '%s'", tid)


@app.on_event("startup")
def _startup() -> None:
    init_db()
    _migrate_to_tenant()

    try:
        from rag_assistant.sync_engine import start_scheduler
        start_scheduler()
    except Exception as exc:
        logger.warning("Connector scheduler failed to start: %s", exc)

    try:
        deleted = purge_old_logs(days=7)
        logger.info("Purged %d old query log entries", deleted)
    except Exception as exc:
        logger.warning("Log purge failed: %s", exc)

    try:
        from rag_assistant.vector_store import migrate_existing_chunks
        default = get_tenant_by_slug("default")
        tid = default['id'] if default else "default"
        migrate_existing_chunks(tenant_id=tid)
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
            default = get_tenant_by_slug("default")
            tid = default['id'] if default else None
            create_user("admin", "admin123", display_name="Administrator", role="admin", tenant_id=tid)
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
        default = get_tenant_by_slug("default")
        tid = default['id'] if default else "default"
        col_name = get_collection_name(tid)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=col_name)
        if collection.count() == 0:
            pdf_files = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
            if pdf_files:
                logger.info("Vector store empty but %d PDF(s) found — auto-indexing...", len(pdf_files))
                from rag_assistant.vector_store import index_all_pdfs
                def _bg_index():
                    try:
                        result = index_all_pdfs(tenant_id=tid)
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
            logger.info("Vector store ready: %d chunks in '%s'.", collection.count(), col_name)
    except Exception as exc:
        logger.warning("Could not check vector store at startup: %s", exc)


# ---------------------------------------------------------------------------
# Auth helpers
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
    """Accept either X-API-Key or Bearer JWT. Returns normalized user dict."""
    if api_key and api_key in API_KEYS:
        default = get_tenant_by_slug("default")
        tid = default['id'] if default else "default"
        return {"user_id": "api", "username": "api", "role": "admin", "tenant_id": tid, "display_name": "API"}
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        payload = decode_token(token)
        if payload:
            uid = payload.get("sub")
            if uid and is_user_blocked(uid):
                raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Account deleted")
            return {
                "user_id": uid,
                "username": payload.get("username"),
                "role": payload.get("role"),
                "tenant_id": payload.get("tenant_id", "default"),
                "display_name": payload.get("display_name", ""),
            }
    raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid or missing credentials")


def _get_tenant_id(user: dict) -> str:
    return user.get("tenant_id") or "default"


def _get_uid(user: dict) -> str | None:
    uid = user.get("user_id")
    return None if uid == "api" else uid


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


class CreateOrgRequest(BaseModel):
    org_name: str


class AcceptInviteRequest(BaseModel):
    token: str


class InviteRequest(BaseModel):
    email: str
    role: str = "member"


class ConnectorConfig(BaseModel):
    name: str
    config: dict
    sync_interval_minutes: int = 60


# ---------------------------------------------------------------------------
# Auth endpoints (public)
# ---------------------------------------------------------------------------

@app.get("/login")
def login_page():
    return RedirectResponse(url="/login.html")


@app.get("/settings")
def settings_page():
    return RedirectResponse(url="/settings.html")


@app.get("/accept-invite")
def accept_invite_page():
    return RedirectResponse(url="/invite.html")


@app.post("/auth/register")
def auth_register(request: RegisterRequest):
    base = request.email.split("@")[0].lower().replace(".", "_")
    suffix = str(random.randint(1000, 9999))
    username = f"{base}{suffix}"
    try:
        # Register without tenant — onboarding step 2 assigns tenant
        user = create_user(
            username=username,
            password=request.password,
            display_name=request.display_name,
            email=request.email,
            role="member",
            tenant_id=None,
        )
        # No tenant_id yet — frontend will show org creation step
        token = create_token(
            user["id"], user["username"], user["role"],
            tenant_id="", display_name=request.display_name
        )
        return {
            "token": token,
            "user_id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
            "role": user["role"],
            "tenant_id": None,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/auth/login")
def auth_login(login_req: LoginRequest, request: Request):
    user = get_user_by_email_or_username(login_req.email)
    if not user or not verify_password(login_req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    update_last_login(user["id"])
    tenant_id = user.get("tenant_id") or ""
    token = create_token(
        user["id"], user["username"], user["role"],
        tenant_id=tenant_id,
        display_name=user.get("display_name") or user["username"],
    )
    tenant_name = None
    if tenant_id:
        t = get_tenant(tenant_id)
        if t:
            tenant_name = t["name"]
    ip = request.client.host if request.client else None
    try:
        log_audit(tenant_id or "default", user["id"], "login",
                  resource_type="user", resource_id=user["id"], ip_address=ip)
    except Exception:
        pass
    return {
        "token": token,
        "username": user["username"],
        "display_name": user.get("display_name") or user["username"],
        "role": user["role"],
        "user_id": user["id"],
        "email": user.get("email"),
        "tenant_id": tenant_id,
        "tenant_name": tenant_name,
    }


@app.get("/auth/me")
def auth_me(user: dict = Depends(_require_auth)):
    uid = _get_uid(user)
    result = {k: v for k, v in user.items()}
    if uid:
        db_user = get_user_by_id(uid)
        if db_user:
            result["display_name"] = db_user.get("display_name") or db_user["username"]
            result["email"] = db_user.get("email")
            result["tenant_id"] = db_user.get("tenant_id") or user.get("tenant_id")
    tenant_id = result.get("tenant_id")
    if tenant_id:
        t = get_tenant(tenant_id)
        result["tenant_name"] = t["name"] if t else None
    return result


@app.post("/auth/create-org")
def auth_create_org(request: CreateOrgRequest, user: dict = Depends(_require_auth)):
    uid = _get_uid(user)
    if not uid:
        raise HTTPException(status_code=400, detail="Requires JWT login")
    try:
        tenant = create_tenant(request.org_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Assign user as admin of new tenant
    update_tenant_user_tenant(uid, tenant["id"], role="admin")

    db_user = get_user_by_id(uid)
    display_name = db_user.get("display_name") or db_user["username"] if db_user else ""
    token = create_token(
        uid, user["username"], "admin",
        tenant_id=tenant["id"],
        display_name=display_name,
    )
    return {
        "token": token,
        "tenant_id": tenant["id"],
        "tenant_name": tenant["name"],
        "tenant_slug": tenant["slug"],
        "role": "admin",
    }


@app.get("/auth/invite/{token}")
def get_invite_info(token: str):
    invite = get_invite_by_token(token)
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found or expired")
    from datetime import datetime
    if invite["expires_at"] < datetime.utcnow().isoformat():
        raise HTTPException(status_code=410, detail="Invite has expired")
    if invite.get("accepted_at"):
        raise HTTPException(status_code=409, detail="Invite already accepted")
    tenant = get_tenant(invite["tenant_id"])
    return {
        "email": invite["email"],
        "role": invite["role"],
        "org_name": tenant["name"] if tenant else "Unknown",
        "expires_at": invite["expires_at"],
    }


@app.post("/auth/accept-invite")
def auth_accept_invite(request: AcceptInviteRequest, user: dict = Depends(_require_auth)):
    uid = _get_uid(user)
    if not uid:
        raise HTTPException(status_code=400, detail="Requires JWT login")
    invite = get_invite_by_token(request.token)
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")
    ok = accept_invite(request.token, uid)
    if not ok:
        raise HTTPException(status_code=400, detail="Invite expired or already accepted")
    tenant = get_tenant(invite["tenant_id"])
    db_user = get_user_by_id(uid)
    display_name = db_user.get("display_name") or db_user["username"] if db_user else ""
    token = create_token(
        uid, user["username"], invite["role"],
        tenant_id=invite["tenant_id"],
        display_name=display_name,
    )
    return {
        "token": token,
        "tenant_name": tenant["name"] if tenant else None,
        "role": invite["role"],
    }


@app.delete("/account")
def delete_account(request: Request, user: dict = Depends(_require_auth)):
    """GDPR right to erasure — deletes all data for the current user."""
    uid = _get_uid(user)
    if not uid:
        raise HTTPException(status_code=400, detail="Requires JWT login")

    tid = _get_tenant_id(user)
    ip = request.client.host if request.client else None

    # 1. Collect chat IDs before deleting
    with get_conn() as conn:
        chat_rows = conn.execute(
            "SELECT id FROM chats WHERE user_id=?", (uid,)
        ).fetchall()
    chat_ids = [r[0] for r in chat_rows]

    # 2. Delete session chunks from Chroma
    try:
        col_name = get_collection_name(tid)
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(name=col_name)
        for cid in chat_ids:
            res = collection.get(where={"session_id": {"$eq": cid}}, include=[])
            if res["ids"]:
                collection.delete(ids=res["ids"])
    except Exception as exc:
        logger.warning("GDPR: could not delete Chroma chunks for user %s: %s", uid, exc)

    # 3. Log audit before user record is gone
    try:
        log_audit(tid, uid, "account_delete", resource_type="user", resource_id=uid,
                  ip_address=ip, details={"chat_count": len(chat_ids)})
    except Exception:
        pass

    # 4. Delete chats + messages (cascades via FK) and user record
    with get_conn() as conn:
        conn.execute("DELETE FROM chats WHERE user_id=?", (uid,))
        # Delete query logs associated with user's chats
        if chat_ids:
            placeholders = ",".join("?" * len(chat_ids))
            conn.execute(f"DELETE FROM query_logs WHERE chat_id IN ({placeholders})", chat_ids)
        conn.execute("DELETE FROM users WHERE id=?", (uid,))

    # 5. Block the JWT so subsequent requests fail immediately
    block_user(uid)

    return {"deleted": True, "user_id": uid}


@app.put("/auth/profile")
def auth_update_profile(request: UpdateProfileRequest, user: dict = Depends(_require_auth)):
    uid = _get_uid(user)
    if not uid:
        raise HTTPException(status_code=400, detail="Profile update requires JWT login")
    update_user_display_name(uid, request.display_name)
    db_user = get_user_by_id(uid)
    return {"display_name": request.display_name, "username": db_user["username"] if db_user else ""}


@app.put("/auth/password")
def auth_update_password(request: UpdatePasswordRequest, user: dict = Depends(_require_auth)):
    uid = _get_uid(user)
    if not uid:
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
    uid = _get_uid(user)
    tid = _get_tenant_id(user)
    return list_chats(user_id=uid, tenant_id=tid if tid else None)


@app.post("/chats")
def new_chat(user: dict = Depends(_require_auth)):
    uid = _get_uid(user)
    tid = _get_tenant_id(user)
    return create_chat(user_id=uid, tenant_id=tid if tid else None)


@app.get("/chats/{chat_id}", dependencies=[Depends(_require_auth)])
def get_chat_detail(chat_id: str):
    chat = get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {**chat, "messages": get_messages(chat_id)}


@app.delete("/chats/all")
def delete_all_user_chats(user: dict = Depends(_require_auth)):
    uid = _get_uid(user)
    if not uid:
        raise HTTPException(status_code=400, detail="Requires JWT login")
    tid = _get_tenant_id(user)
    col_name = get_collection_name(tid)

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id FROM chats WHERE user_id=? OR user_id IS NULL", (uid,)
        ).fetchall()
    chat_ids = [r[0] for r in rows]

    deleted_chunks = 0
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(name=col_name)
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
def remove_chat(chat_id: str, user: dict = Depends(_require_auth)):
    if not get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    tid = _get_tenant_id(user)
    delete_chat(chat_id)
    def _bg_delete():
        try:
            from rag_assistant.vector_store import delete_session_chunks
            delete_session_chunks(chat_id, tenant_id=tid)
        except Exception as e:
            logger.error("Failed to delete session chunks for %s: %s", chat_id, e)
    threading.Thread(target=_bg_delete, daemon=True).start()
    return {"status": "deleted"}


@app.post("/chats/{chat_id}/upload", dependencies=[Depends(_require_auth)])
async def upload_to_chat(chat_id: str, file: UploadFile = File(...),
                         user: dict = Depends(_require_auth)):
    if not get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    tid = _get_tenant_id(user)
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
            index_pdf_for_session(dest, session_id=chat_id, tenant_id=tid)
            logger.info("Session upload indexed: %s for chat %s", filename, chat_id)
            _fire_webhook("document_indexed", filename, None, chat_id)
        except Exception as e:
            logger.error("Session upload indexing failed: %s", e)
    threading.Thread(target=_bg_index, daemon=True).start()
    return {"status": "upload_received_indexing_started", "filename": filename}


@app.get("/chats/{chat_id}/documents", dependencies=[Depends(_require_auth)])
def get_chat_documents(chat_id: str, user: dict = Depends(_require_auth)):
    if not get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    tid = _get_tenant_id(user)
    col_name = get_collection_name(tid)
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=col_name)
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
def index(user: dict = Depends(_require_auth)):
    tid = _get_tenant_id(user)
    try:
        from rag_assistant.vector_store import index_all_pdfs
        result = index_all_pdfs(tenant_id=tid)
        count = result.count() if result is not None else 0
        return {"status": "complete", "chunks": count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/upload")
async def upload(file: UploadFile = File(...), request: Request = None,
                 user: dict = Depends(_require_auth)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can upload to the Knowledge Base")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    tid = _get_tenant_id(user)
    uid = _get_uid(user)
    ip = request.client.host if request and request.client else None
    try:
        log_audit(tid, uid or "api", "document_upload", resource_type="document",
                  resource_id=file.filename, ip_address=ip,
                  details={"filename": file.filename})
    except Exception:
        pass
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
            collection = index_single_pdf(dest, tenant_id=tid)
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
def query(request: QueryRequest, http_req: Request, user: dict = Depends(_require_auth)):
    uid = _get_uid(user)
    tid = _get_tenant_id(user)
    ip = http_req.client.host if http_req.client else None
    try:
        log_audit(tid, uid or "anon", "query", resource_type="knowledge_base",
                  ip_address=ip, details={"question_preview": request.question[:80]})
    except Exception:
        pass
    active_chat_id = request.chat_id
    if active_chat_id == "new":
        chat = create_chat(user_id=uid, tenant_id=tid if tid else None)
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
            tenant_id=tid,
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
                        "after receiving this answer about documents. "
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
    uid = _get_uid(user)
    tid = _get_tenant_id(user)
    active_chat_id = request.chat_id
    if active_chat_id == "new":
        chat = create_chat(user_id=uid, tenant_id=tid if tid else None)
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
            tenant_id=tid,
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
def list_global_documents(user: dict = Depends(_require_auth)):
    tid = _get_tenant_id(user)
    col_name = get_collection_name(tid)
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=col_name)
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
def documents(user: dict = Depends(_require_auth)):
    tid = _get_tenant_id(user)
    col_name = get_collection_name(tid)
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=col_name)
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
def admin_stats(user: dict = Depends(_require_auth)):
    tid = _get_tenant_id(user)
    col_name = get_collection_name(tid)
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(name=col_name)
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
def delete_global_document_by_param(filename: str = Query(...), request: Request = None,
                                     user: dict = Depends(_require_auth)):
    tid = _get_tenant_id(user)
    uid = _get_uid(user)
    ip = request.client.host if request and request.client else None
    col_name = get_collection_name(tid)
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(name=col_name)
        results = collection.get(where={"source": {"$eq": filename}}, include=[])
        if not results["ids"]:
            raise HTTPException(status_code=404, detail=f"{filename} not found")
        collection.delete(ids=results["ids"])
        try:
            log_audit(tid, uid or "api", "document_delete", resource_type="document",
                      resource_id=filename, ip_address=ip,
                      details={"chunks_removed": len(results["ids"])})
        except Exception:
            pass
        return {"deleted": filename, "chunks_removed": len(results["ids"])}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/reindex", dependencies=[Depends(_require_auth)])
def admin_reindex(user: dict = Depends(_require_auth)):
    tid = _get_tenant_id(user)
    pdf_files = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    filenames = [f.name for f in pdf_files]
    def _bg():
        try:
            from rag_assistant.vector_store import index_all_pdfs
            result = index_all_pdfs(tenant_id=tid)
            count = result.count() if result is not None else 0
            logger.info("Admin reindex complete: %d chunks", count)
        except Exception as e:
            logger.error("Admin reindex failed: %s", e)
    threading.Thread(target=_bg, daemon=True).start()
    return {"status": "reindexing_started", "files": filenames}


@app.post("/admin/query-debug", dependencies=[Depends(_require_auth)])
def admin_query_debug(request: QueryDebugRequest, user: dict = Depends(_require_auth)):
    tid = _get_tenant_id(user)
    try:
        rewritten = rewrite_query(request.question)
        hits = retrieve(rewritten, top_k=request.top_k, session_id=request.session_id, tenant_id=tid)
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
def admin_sessions(user: dict = Depends(_require_auth)):
    tid = _get_tenant_id(user)
    col_name = get_collection_name(tid)
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
        collection = chroma_client.get_or_create_collection(name=col_name)
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
def admin_delete_session(chat_id: str, user: dict = Depends(_require_auth)):
    if not get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    tid = _get_tenant_id(user)
    col_name = get_collection_name(tid)
    chunks_removed = 0
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(name=col_name)
        res = collection.get(where={"session_id": {"$eq": chat_id}}, include=[])
        if res["ids"]:
            collection.delete(ids=res["ids"])
        chunks_removed = len(res["ids"])
    except Exception as e:
        logger.warning("Failed to remove session chunks for %s: %s", chat_id, e)
    delete_chat(chat_id)
    return {"deleted": chat_id, "chunks_removed": chunks_removed}


@app.delete("/admin/documents/{filename}", dependencies=[Depends(_require_auth)])
def delete_global_document(filename: str, user: dict = Depends(_require_auth)):
    tid = _get_tenant_id(user)
    col_name = get_collection_name(tid)
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=col_name)
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


@app.get("/admin/audit-log")
def admin_audit_log(limit: int = 100, user: dict = Depends(_require_auth)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    tid = _get_tenant_id(user)
    return get_audit_log(tid, limit=limit)


@app.get("/admin/analytics", dependencies=[Depends(_require_auth)])
def admin_analytics(user: dict = Depends(_require_auth)):
    try:
        tid = _get_tenant_id(user)
        return get_analytics(tenant_id=tid if tid else None)
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
    tid = _get_tenant_id(user)
    try:
        return create_user(
            request.username, request.password,
            display_name=request.display_name,
            email=request.email,
            role=request.role,
            tenant_id=tid if tid else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/invites")
def admin_send_invite(request: InviteRequest, user: dict = Depends(_require_auth)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    uid = _get_uid(user)
    tid = _get_tenant_id(user)
    if not tid or tid == "default":
        raise HTTPException(status_code=400, detail="No tenant configured")
    try:
        invite = create_invite(
            tenant_id=tid,
            email=request.email,
            role=request.role,
            invited_by=uid or "admin",
        )
        # In production, send email. For now, return the invite link.
        base_url = os.environ.get("APP_BASE_URL", "http://localhost:8000")
        invite_link = f"{base_url}/accept-invite?token={invite['token']}"
        return {
            "invite_link": invite_link,
            "token": invite["token"],
            "email": invite["email"],
            "expires_at": invite["expires_at"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/admin/team")
def admin_team(user: dict = Depends(_require_auth)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    tid = _get_tenant_id(user)
    if not tid:
        raise HTTPException(status_code=400, detail="No tenant configured")
    try:
        return list_tenant_users(tid)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cache_backend": cache_backend(),
        "mcp_endpoint": "/mcp",
        "mcp_sse": "/mcp/sse",
    }


# ---------------------------------------------------------------------------
# Google OAuth
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
        return {"error": "Google OAuth redirect failed.", "detail": str(exc)}


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
        db_user = get_user_by_email_or_username(email)
        if not db_user:
            import secrets as _secrets
            db_user = create_user(
                username=email.split("@")[0] + str(random.randint(1000, 9999)),
                password=_secrets.token_hex(32),
                display_name=name,
                email=email,
                role="member",
            )
            db_user = get_user_by_email_or_username(email)
        update_last_login(db_user["id"])
        tenant_id = db_user.get("tenant_id") or ""
        jwt_token = create_token(
            db_user["id"], db_user["username"], db_user["role"],
            tenant_id=tenant_id,
            display_name=db_user.get("display_name") or db_user["username"],
        )
        return RedirectResponse(url=f"/#token={jwt_token}")
    except Exception as exc:
        logger.error("Google OAuth callback error: %s", exc)
        return RedirectResponse(url="/login?error=oauth_error")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fire_webhook(event: str, filename: str, chunks: int | None, session_id: str):
    import requests as req_lib
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


# ---------------------------------------------------------------------------
# Connector endpoints
# ---------------------------------------------------------------------------

@app.get("/connectors/metadata")
def connectors_metadata(_user: dict = Depends(_require_auth)):
    """Return connector type metadata (for building config modals in the UI)."""
    from rag_assistant.connectors.registry import CONNECTOR_METADATA
    return CONNECTOR_METADATA


@app.get("/admin/connectors")
def admin_list_connectors(user: dict = Depends(_require_auth)):
    tid = _get_tenant_id(user)
    rows = list_connectors(tenant_id=tid)
    # Mask secret fields in config before returning
    result = []
    for row in rows:
        r = dict(row)
        try:
            import json as _json
            cfg = _json.loads(r.get("config") or "{}")
            # Redact any key with "token", "secret", "key", "json", "password"
            for k in list(cfg.keys()):
                kl = k.lower()
                if any(s in kl for s in ("token", "secret", "key", "json", "password")):
                    cfg[k] = "••••••••"
            r["config"] = cfg
        except Exception:
            r["config"] = {}
        result.append(r)
    return result


@app.post("/admin/connectors/{connector_type}")
def admin_upsert_connector(
    connector_type: str,
    request: ConnectorConfig,
    user: dict = Depends(_require_auth),
):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    from rag_assistant.connectors.registry import CONNECTOR_REGISTRY
    if connector_type not in CONNECTOR_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown connector type: {connector_type}")

    tid = _get_tenant_id(user)
    import json as _json
    config_str = _json.dumps(request.config)
    conn = upsert_connector(
        tenant_id=tid,
        connector_type=connector_type,
        name=request.name,
        config=config_str,
        sync_interval_minutes=request.sync_interval_minutes,
    )
    uid = _get_uid(user)
    try:
        log_audit(tid, uid or "api", "connector_configure", resource_type="connector",
                  resource_id=connector_type, details={"name": request.name})
    except Exception:
        pass
    return {"status": "saved", "id": conn["id"]}


@app.post("/admin/connectors/{connector_type}/test")
def admin_test_connector(
    connector_type: str,
    request: ConnectorConfig,
    user: dict = Depends(_require_auth),
):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    from rag_assistant.connectors.registry import get_connector_instance
    tid = _get_tenant_id(user)
    try:
        instance = get_connector_instance(connector_type, request.config, tid)
        result = instance.test_connection()
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


@app.post("/admin/connectors/{connector_type}/sync")
def admin_sync_connector(connector_type: str, user: dict = Depends(_require_auth)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    tid = _get_tenant_id(user)
    conn = get_connector(tid, connector_type)
    if not conn:
        raise HTTPException(status_code=404, detail="Connector not configured")

    def _bg():
        from rag_assistant.sync_engine import trigger_sync
        trigger_sync(conn["id"])

    threading.Thread(target=_bg, daemon=True).start()
    return {"status": "sync_started", "connector": conn["name"]}


@app.delete("/admin/connectors/{connector_type}")
def admin_delete_connector(connector_type: str, user: dict = Depends(_require_auth)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    tid = _get_tenant_id(user)
    conn = get_connector(tid, connector_type)
    if not conn:
        raise HTTPException(status_code=404, detail="Connector not found")
    delete_connector(conn["id"])
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# MCP server (model context protocol)
# ---------------------------------------------------------------------------

try:
    from fastapi_mcp import FastApiMCP
    mcp = FastApiMCP(app)
    mcp.mount()
    logger.info("MCP server mounted at /mcp")
except Exception as _mcp_exc:
    logger.warning("fastapi-mcp not available, MCP endpoint disabled: %s", _mcp_exc)


# Mount frontend AFTER all API routes so API paths take priority
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
