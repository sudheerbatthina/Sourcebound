"""Confluence connector — reads pages from a space via REST API."""

import logging
import requests
from .base import BaseConnector

logger = logging.getLogger(__name__)


class ConfluenceConnector(BaseConnector):
    """Fetches page body text from a Confluence space."""

    def fetch_data(self) -> list[dict]:
        base_url = self.config.get("base_url", "").rstrip("/")
        username = self.config.get("username", "")
        api_token = self.config.get("api_token", "")
        space_key = self.config.get("space_key", "")

        if not all([base_url, username, api_token, space_key]):
            raise ValueError("base_url, username, api_token, and space_key are all required")

        auth = (username, api_token)
        start = 0
        limit = 50
        docs = []

        while True:
            resp = requests.get(
                f"{base_url}/rest/api/content",
                params={
                    "spaceKey": space_key,
                    "type": "page",
                    "expand": "body.storage,title",
                    "start": start,
                    "limit": limit,
                },
                auth=auth,
                timeout=30,
            )
            if not resp.ok:
                raise RuntimeError(f"Confluence API error {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            results = data.get("results", [])
            for page in results:
                pid = page["id"]
                title = page.get("title", "Untitled")
                body = page.get("body", {}).get("storage", {}).get("value", "")
                # Strip HTML tags for plain text
                text = _strip_html(body)
                if text.strip():
                    docs.append({
                        "content": text,
                        "source": f"confluence://{space_key}/{pid}",
                        "metadata": {"page_id": pid, "title": title, "space": space_key},
                    })

            total = data.get("size", 0)
            start += limit
            if start >= total or not results:
                break

        return docs


def _strip_html(html: str) -> str:
    """Very lightweight HTML stripper — removes tags, decodes common entities."""
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&quot;", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()
