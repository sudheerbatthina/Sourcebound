"""Notion connector — reads pages from a database via the Notion API."""

import logging
import requests
from .base import BaseConnector

logger = logging.getLogger(__name__)

NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


class NotionConnector(BaseConnector):
    """Fetches page content from a Notion database."""

    def _headers(self) -> dict:
        token = self.config.get("integration_token", "")
        return {
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

    def fetch_data(self) -> list[dict]:
        token = self.config.get("integration_token", "")
        database_id = self.config.get("database_id", "")

        if not token or not database_id:
            raise ValueError("integration_token and database_id are required")

        hdrs = self._headers()
        # Query database for all pages
        pages = []
        cursor = None
        while True:
            body: dict = {"page_size": 100}
            if cursor:
                body["start_cursor"] = cursor
            resp = requests.post(
                f"{NOTION_API}/databases/{database_id}/query",
                headers=hdrs,
                json=body,
                timeout=30,
            )
            if not resp.ok:
                raise RuntimeError(f"Notion API error {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            pages.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

        docs = []
        for page in pages:
            pid = page["id"]
            title = self._extract_title(page)
            content = self._fetch_page_content(pid, hdrs)
            if content.strip():
                docs.append({
                    "content": content,
                    "source": f"notion://{pid}",
                    "metadata": {"page_id": pid, "title": title},
                })
        return docs

    def _extract_title(self, page: dict) -> str:
        props = page.get("properties", {})
        for prop in props.values():
            if prop.get("type") == "title":
                rich = prop.get("title", [])
                if rich:
                    return rich[0].get("plain_text", "Untitled")
        return "Untitled"

    def _fetch_page_content(self, page_id: str, hdrs: dict) -> str:
        resp = requests.get(f"{NOTION_API}/blocks/{page_id}/children?page_size=100",
                            headers=hdrs, timeout=30)
        if not resp.ok:
            return ""
        blocks = resp.json().get("results", [])
        lines = []
        for block in blocks:
            btype = block.get("type", "")
            bdata = block.get(btype, {})
            rich = bdata.get("rich_text", [])
            text = " ".join(r.get("plain_text", "") for r in rich)
            if text:
                lines.append(text)
        return "\n".join(lines)
