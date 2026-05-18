import re
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from .config import STORAGE_DIR

DB_PATH = Path(STORAGE_DIR) / "db.sqlite"


def get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tenants (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                slug TEXT UNIQUE NOT NULL,
                plan TEXT NOT NULL DEFAULT 'free',
                max_users INTEGER NOT NULL DEFAULT 3,
                max_chunks INTEGER NOT NULL DEFAULT 500,
                storage_bytes_used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tenant_invites (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                email TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'member',
                token TEXT UNIQUE NOT NULL,
                invited_by TEXT,
                expires_at TEXT NOT NULL,
                accepted_at TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                chat_id TEXT,
                message_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                rating INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS query_logs (
                id TEXT PRIMARY KEY,
                chat_id TEXT,
                question TEXT NOT NULL,
                rewritten_question TEXT,
                answer_preview TEXT,
                sources_count INTEGER,
                latency_ms INTEGER,
                from_cache INTEGER DEFAULT 0,
                faithfulness_score INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hallucination_flags (
                id TEXT PRIMARY KEY,
                chat_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                faithfulness_score INTEGER NOT NULL,
                flagged_claims TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'member',
                display_name TEXT,
                created_at TEXT NOT NULL,
                last_login TEXT
            );
            CREATE TABLE IF NOT EXISTS token_blocklist (
                user_id TEXT PRIMARY KEY,
                blocked_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                tenant_id TEXT,
                user_id TEXT,
                action TEXT NOT NULL,
                resource_type TEXT,
                resource_id TEXT,
                ip_address TEXT,
                details TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS connectors (
                id TEXT PRIMARY KEY,
                tenant_id TEXT,
                connector_type TEXT NOT NULL DEFAULT '',
                name TEXT NOT NULL,
                config TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'idle',
                last_sync TEXT,
                last_error TEXT,
                sync_interval_minutes INTEGER NOT NULL DEFAULT 60,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT ''
            );
        """)
    migrate_add_display_name()
    migrate_add_user_id_to_chats()
    migrate_add_tenant_id()
    migrate_connectors_full_schema()


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------

def migrate_add_display_name():
    with get_conn() as conn:
        try:
            conn.execute("ALTER TABLE users ADD COLUMN display_name TEXT")
        except Exception:
            pass


def migrate_add_user_id_to_chats():
    with get_conn() as conn:
        try:
            conn.execute("ALTER TABLE chats ADD COLUMN user_id TEXT")
        except Exception:
            pass


def migrate_add_tenant_id():
    """Add tenant_id to all tables that need it."""
    tables = ['users', 'chats', 'query_logs', 'feedback', 'connectors']
    with get_conn() as conn:
        for table in tables:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN tenant_id TEXT")
            except Exception:
                pass  # already exists


def migrate_connectors_full_schema():
    """Add missing columns to the connectors table for existing installs."""
    new_cols = [
        ("connector_type", "TEXT NOT NULL DEFAULT ''"),
        ("config", "TEXT NOT NULL DEFAULT '{}'"),
        ("status", "TEXT NOT NULL DEFAULT 'idle'"),
        ("last_sync", "TEXT"),
        ("last_error", "TEXT"),
        ("sync_interval_minutes", "INTEGER NOT NULL DEFAULT 60"),
        ("enabled", "INTEGER NOT NULL DEFAULT 1"),
        ("updated_at", "TEXT NOT NULL DEFAULT ''"),
    ]
    with get_conn() as conn:
        for col, defn in new_cols:
            try:
                conn.execute(f"ALTER TABLE connectors ADD COLUMN {col} {defn}")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tenant functions
# ---------------------------------------------------------------------------

def create_tenant(name: str, slug: str = None, plan: str = 'free') -> dict:
    now = datetime.utcnow().isoformat()
    tid = str(uuid.uuid4())
    if not slug:
        slug = re.sub(r'[^a-z0-9]', '-', name.lower()).strip('-')
        slug = re.sub(r'-+', '-', slug)
    with get_conn() as conn:
        try:
            conn.execute(
                "INSERT INTO tenants VALUES (?,?,?,?,?,?,?,?,?)",
                (tid, name, slug, plan, 3, 500, 0, now, now)
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"Slug '{slug}' already taken")
    return get_tenant(tid)


def get_tenant(tenant_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM tenants WHERE id=?", (tenant_id,)
        ).fetchone()
    return dict(row) if row else None


def get_tenant_by_slug(slug: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM tenants WHERE slug=?", (slug,)
        ).fetchone()
    return dict(row) if row else None


def update_tenant_user_tenant(user_id: str, tenant_id: str, role: str = "admin"):
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET tenant_id=?, role=? WHERE id=?",
            (tenant_id, role, user_id)
        )


def list_tenant_users(tenant_id: str) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, username, display_name, email, role,
               created_at, last_login
               FROM users WHERE tenant_id=?
               ORDER BY created_at ASC""",
            (tenant_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def create_invite(tenant_id: str, email: str, role: str, invited_by: str) -> dict:
    now = datetime.utcnow().isoformat()
    expires = (datetime.utcnow() + timedelta(days=7)).isoformat()
    iid = str(uuid.uuid4())
    token = secrets.token_urlsafe(32)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO tenant_invites VALUES (?,?,?,?,?,?,?,?,?)",
            (iid, tenant_id, email, role, token, invited_by, expires, None, now)
        )
    return {"id": iid, "token": token, "email": email, "expires_at": expires}


def get_invite_by_token(token: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM tenant_invites WHERE token=?", (token,)
        ).fetchone()
    return dict(row) if row else None


def accept_invite(token: str, user_id: str) -> bool:
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        invite = conn.execute(
            "SELECT * FROM tenant_invites WHERE token=? AND accepted_at IS NULL",
            (token,)
        ).fetchone()
        if not invite:
            return False
        if invite['expires_at'] < now:
            return False
        conn.execute(
            "UPDATE tenant_invites SET accepted_at=? WHERE token=?",
            (now, token)
        )
        conn.execute(
            "UPDATE users SET tenant_id=?, role=? WHERE id=?",
            (invite['tenant_id'], invite['role'], user_id)
        )
    return True


# ---------------------------------------------------------------------------
# Chat functions
# ---------------------------------------------------------------------------

def create_chat(title="New chat", user_id: str = None, tenant_id: str = None) -> dict:
    now = datetime.utcnow().isoformat()
    chat_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chats (id, user_id, tenant_id, title, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (chat_id, user_id, tenant_id, title, now, now),
        )
    return {"id": chat_id, "title": title, "created_at": now, "user_id": user_id, "tenant_id": tenant_id}


def list_chats(user_id: str = None, tenant_id: str = None) -> list:
    with get_conn() as conn:
        if user_id and tenant_id:
            rows = conn.execute(
                "SELECT * FROM chats WHERE user_id=? AND (tenant_id=? OR tenant_id IS NULL) ORDER BY updated_at DESC",
                (user_id, tenant_id),
            ).fetchall()
        elif user_id:
            rows = conn.execute(
                "SELECT * FROM chats WHERE user_id=? OR user_id IS NULL ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM chats ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_chat(chat_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
    return dict(row) if row else None


def delete_chat(chat_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))


def delete_all_chats(user_id: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM chats WHERE user_id=? OR user_id IS NULL", (user_id,)
        )
    return cur.rowcount


def add_message(chat_id: str, role: str, content: str, sources: str = None) -> dict:
    now = datetime.utcnow().isoformat()
    msg_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "UPDATE chats SET updated_at=?, title=CASE WHEN title='New chat' AND ?='user' THEN substr(?,1,40) ELSE title END WHERE id=?",
            (now, role, content, chat_id)
        )
        conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)",
                     (msg_id, chat_id, role, content, sources, now))
    return {"id": msg_id, "chat_id": chat_id, "role": role, "content": content}


def get_messages(chat_id: str) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE chat_id=? ORDER BY created_at ASC", (chat_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Feedback / logs
# ---------------------------------------------------------------------------

def save_feedback(chat_id: str, message_id: str, question: str,
                  answer: str, rating: int) -> dict:
    now = datetime.utcnow().isoformat()
    fid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback VALUES (?,?,?,?,?,?,?)",
            (fid, chat_id, message_id, question, answer, rating, now)
        )
    return {"id": fid, "rating": rating}


def save_query_log(chat_id: str, question: str, rewritten: str,
                   answer_preview: str, sources_count: int,
                   latency_ms: int, from_cache: bool,
                   faithfulness_score: int | None) -> None:
    now = datetime.utcnow().isoformat()
    lid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO query_logs VALUES (?,?,?,?,?,?,?,?,?,?)",
            (lid, chat_id, question, rewritten, answer_preview,
             sources_count, latency_ms, int(from_cache),
             faithfulness_score, now)
        )


# ---------------------------------------------------------------------------
# User functions
# ---------------------------------------------------------------------------

def create_user(username: str, password: str,
                display_name: str = None,
                email: str = None,
                role: str = "member",
                tenant_id: str = None) -> dict:
    from .auth import hash_password
    now = datetime.utcnow().isoformat()
    uid = str(uuid.uuid4())
    with get_conn() as conn:
        try:
            conn.execute(
                "INSERT INTO users (id, username, email, password_hash, role, display_name, tenant_id, created_at, last_login) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (uid, username, email, hash_password(password), role, display_name, tenant_id, now, None),
            )
        except sqlite3.IntegrityError:
            raise ValueError("Username or email already exists")
    return {"id": uid, "username": username, "role": role,
            "display_name": display_name, "email": email, "tenant_id": tenant_id}


def get_user_by_username(username: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_email_or_username(identifier: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email=? OR username=?", (identifier, identifier)
        ).fetchone()
    return dict(row) if row else None


def update_last_login(user_id: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_login=? WHERE id=?",
            (datetime.utcnow().isoformat(), user_id),
        )


def update_user_display_name(user_id: str, display_name: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET display_name=? WHERE id=?",
            (display_name, user_id),
        )


def update_user_password(user_id: str, new_password_hash: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET password_hash=? WHERE id=?",
            (new_password_hash, user_id),
        )


def get_user_by_id(user_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id=?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def list_users() -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, username, email, role, display_name, tenant_id, created_at, last_login "
            "FROM users ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_analytics(tenant_id: str = None) -> dict:
    """Return usage analytics, optionally filtered by tenant_id."""
    where = "WHERE tenant_id=?" if tenant_id else ""
    and_ = "AND" if tenant_id else "WHERE"
    params = (tenant_id,) if tenant_id else ()

    with get_conn() as conn:
        total_queries = conn.execute(
            f"SELECT COUNT(*) FROM query_logs {where}", params
        ).fetchone()[0]

        cache_hits = conn.execute(
            f"SELECT COUNT(*) FROM query_logs {where} {and_} from_cache=1", params
        ).fetchone()[0]

        avg_latency = conn.execute(
            f"SELECT AVG(latency_ms) FROM query_logs {where} {and_} from_cache=0", params
        ).fetchone()[0]

        avg_faithfulness = conn.execute(
            f"SELECT AVG(faithfulness_score) FROM query_logs {where} "
            f"{and_} faithfulness_score IS NOT NULL", params
        ).fetchone()[0]

        queries_by_day = conn.execute(
            f"""SELECT substr(created_at,1,10) as day, COUNT(*) as count
               FROM query_logs {where} {and_} created_at >= datetime('now','-7 days')
               GROUP BY day ORDER BY day ASC""",
            params
        ).fetchall()

        top_questions = conn.execute(
            f"""SELECT question, COUNT(*) as count FROM query_logs {where}
               GROUP BY question ORDER BY count DESC LIMIT 10""",
            params
        ).fetchall()

        feedback_good = conn.execute(
            f"SELECT COUNT(*) FROM feedback {where} {and_} rating=1", params
        ).fetchone()[0]

        feedback_bad = conn.execute(
            f"SELECT COUNT(*) FROM feedback {where} {and_} rating=-1", params
        ).fetchone()[0]

    total_fb = feedback_good + feedback_bad
    return {
        "total_queries": total_queries,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / total_queries * 100, 1) if total_queries else 0,
        "avg_latency_ms": round(avg_latency or 0),
        "avg_faithfulness": round(avg_faithfulness or 0, 1),
        "queries_by_day": [{"day": r[0], "count": r[1]} for r in queries_by_day],
        "top_questions": [{"question": r[0][:60], "count": r[1]} for r in top_questions],
        "feedback_good": feedback_good,
        "feedback_bad": feedback_bad,
        "satisfaction_rate": round(feedback_good / total_fb * 100) if total_fb > 0 else None,
    }


# ---------------------------------------------------------------------------
# Token blocklist (GDPR erasure)
# ---------------------------------------------------------------------------

def block_user(user_id: str) -> None:
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO token_blocklist VALUES (?,?)",
            (user_id, now)
        )


def is_user_blocked(user_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM token_blocklist WHERE user_id=?", (user_id,)
        ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def log_audit(tenant_id: str, user_id: str, action: str,
              resource_type: str = None, resource_id: str = None,
              ip_address: str = None, details: dict = None) -> None:
    import json as _json
    now = datetime.utcnow().isoformat()
    aid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO audit_log VALUES (?,?,?,?,?,?,?,?,?)",
            (aid, tenant_id, user_id, action, resource_type,
             resource_id, ip_address,
             _json.dumps(details) if details else None, now)
        )


def get_audit_log(tenant_id: str, limit: int = 100) -> list:
    import json as _json
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM audit_log WHERE tenant_id=?
               ORDER BY created_at DESC LIMIT ?""",
            (tenant_id, limit)
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        if d.get("details"):
            try:
                d["details"] = _json.loads(d["details"])
            except Exception:
                pass
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# Connector functions
# ---------------------------------------------------------------------------

def upsert_connector(tenant_id: str, connector_type: str, name: str,
                     config: str, sync_interval_minutes: int = 60) -> dict:
    """Insert or update a connector for a tenant+type pair."""
    import json as _json
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM connectors WHERE tenant_id=? AND connector_type=?",
            (tenant_id, connector_type)
        ).fetchone()
        if existing:
            cid = existing["id"]
            conn.execute(
                "UPDATE connectors SET name=?, config=?, sync_interval_minutes=?, "
                "enabled=1, updated_at=? WHERE id=?",
                (name, config, sync_interval_minutes, now, cid)
            )
        else:
            cid = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO connectors (id, tenant_id, connector_type, name, config, "
                "status, sync_interval_minutes, enabled, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (cid, tenant_id, connector_type, name, config,
                 "idle", sync_interval_minutes, 1, now, now)
            )
    return get_connector_by_id(cid)


def get_connector_by_id(connector_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM connectors WHERE id=?", (connector_id,)
        ).fetchone()
    return dict(row) if row else None


def get_connector(tenant_id: str, connector_type: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM connectors WHERE tenant_id=? AND connector_type=?",
            (tenant_id, connector_type)
        ).fetchone()
    return dict(row) if row else None


def list_connectors(tenant_id: str = None) -> list:
    with get_conn() as conn:
        if tenant_id:
            rows = conn.execute(
                "SELECT * FROM connectors WHERE tenant_id=? ORDER BY created_at ASC",
                (tenant_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM connectors ORDER BY created_at ASC"
            ).fetchall()
    return [dict(r) for r in rows]


def update_connector_status(connector_id: str, status: str,
                             last_sync: str = None, last_error: str = None):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute(
            "UPDATE connectors SET status=?, last_sync=COALESCE(?,last_sync), "
            "last_error=COALESCE(?,last_error), updated_at=? WHERE id=?",
            (status, last_sync, last_error, now, connector_id)
        )


def delete_connector(connector_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM connectors WHERE id=?", (connector_id,))


def purge_old_logs(days: int = 7) -> int:
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM query_logs WHERE created_at < ?", (cutoff,)
        )
        deleted = cur.rowcount
        conn.execute(
            "DELETE FROM hallucination_flags WHERE created_at < ?", (cutoff,)
        )
    return deleted
