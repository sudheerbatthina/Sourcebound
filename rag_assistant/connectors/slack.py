"""Slack connector — reads messages from a channel via Bot Token."""

import logging
from .base import BaseConnector

logger = logging.getLogger(__name__)


class SlackConnector(BaseConnector):
    """Fetches recent messages from a Slack channel using the Web API."""

    def fetch_data(self) -> list[dict]:
        try:
            from slack_sdk import WebClient
            from slack_sdk.errors import SlackApiError
        except ImportError:
            raise RuntimeError("slack-sdk is not installed. Run: pip install slack-sdk")

        token = self.config.get("bot_token", "")
        channel = self.config.get("channel_id", "")
        limit = int(self.config.get("message_limit", 200))

        if not token or not channel:
            raise ValueError("bot_token and channel_id are required for Slack connector")

        client = WebClient(token=token)
        try:
            resp = client.conversations_history(channel=channel, limit=limit)
        except SlackApiError as exc:
            raise RuntimeError(f"Slack API error: {exc.response['error']}")

        messages = resp.get("messages", [])
        docs = []
        for msg in messages:
            text = msg.get("text", "").strip()
            if not text:
                continue
            ts = msg.get("ts", "0")
            user = msg.get("user", "unknown")
            docs.append({
                "content": text,
                "source": f"slack://{channel}/{ts}",
                "metadata": {"channel": channel, "user": user, "ts": ts},
            })
        return docs
