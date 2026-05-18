"""Background scheduler that periodically syncs all enabled connectors."""

import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

_scheduler = None


def start_scheduler():
    """Start the APScheduler background job. Safe to call multiple times."""
    global _scheduler
    if _scheduler is not None:
        return
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        _scheduler = BackgroundScheduler(timezone="UTC")
        _scheduler.add_job(
            _sync_all_connectors,
            trigger="interval",
            minutes=5,
            id="connector_sync",
            replace_existing=True,
        )
        _scheduler.start()
        logger.info("Connector sync scheduler started (interval: 5 min)")
    except Exception as exc:
        logger.warning("Could not start connector scheduler: %s", exc)


def stop_scheduler():
    """Gracefully shut down the scheduler."""
    global _scheduler
    if _scheduler is not None:
        try:
            _scheduler.shutdown(wait=False)
        except Exception:
            pass
        _scheduler = None


def _sync_all_connectors():
    """Sync every enabled connector whose next sync time has come."""
    try:
        from rag_assistant.db import list_connectors, update_connector_status
        from rag_assistant.connectors.registry import get_connector_instance

        connectors = list_connectors()
        now = datetime.utcnow()

        for conn in connectors:
            if not conn.get("enabled"):
                continue
            last_sync = conn.get("last_sync")
            interval = int(conn.get("sync_interval_minutes", 60))
            if last_sync:
                from datetime import timedelta
                last_dt = datetime.fromisoformat(last_sync)
                if (now - last_dt).total_seconds() < interval * 60:
                    continue

            _sync_one(conn)
    except Exception as exc:
        logger.warning("Connector sync cycle error: %s", exc)


def _sync_one(conn: dict):
    from rag_assistant.db import update_connector_status
    from rag_assistant.connectors.registry import get_connector_instance

    cid = conn["id"]
    ctype = conn["connector_type"]
    tenant_id = conn.get("tenant_id", "default")
    try:
        config = json.loads(conn.get("config") or "{}")
    except Exception:
        config = {}

    update_connector_status(cid, "syncing")
    try:
        instance = get_connector_instance(ctype, config, tenant_id)
        count = instance.sync()
        now = datetime.utcnow().isoformat()
        update_connector_status(cid, "idle", last_sync=now)
        logger.info("Connector %s (%s) synced: %d chunks", conn.get("name"), ctype, count)
    except Exception as exc:
        update_connector_status(cid, "error", last_error=str(exc))
        logger.error("Connector %s sync failed: %s", conn.get("name"), exc)


def trigger_sync(connector_id: str):
    """Manually trigger a sync for a specific connector (runs in calling thread)."""
    from rag_assistant.db import get_connector_by_id
    conn = get_connector_by_id(connector_id)
    if conn:
        _sync_one(conn)
