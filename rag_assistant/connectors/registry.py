"""Connector registry — maps type strings to classes and field metadata."""

from .slack import SlackConnector
from .gdrive import GoogleDriveConnector
from .notion import NotionConnector
from .confluence import ConfluenceConnector

# type → connector class
CONNECTOR_REGISTRY = {
    "slack": SlackConnector,
    "gdrive": GoogleDriveConnector,
    "notion": NotionConnector,
    "confluence": ConfluenceConnector,
}

# Metadata used by the admin UI to build config modals dynamically
CONNECTOR_METADATA = {
    "slack": {
        "label": "Slack",
        "icon": "💬",
        "description": "Import messages from a Slack channel",
        "fields": [
            {"key": "bot_token", "label": "Bot Token", "type": "password",
             "placeholder": "xoxb-...", "required": True},
            {"key": "channel_id", "label": "Channel ID", "type": "text",
             "placeholder": "C0123456789", "required": True},
            {"key": "message_limit", "label": "Message Limit", "type": "number",
             "placeholder": "200", "required": False},
        ],
    },
    "gdrive": {
        "label": "Google Drive",
        "icon": "📁",
        "description": "Import docs and text files from a Drive folder",
        "fields": [
            {"key": "service_account_json", "label": "Service Account JSON", "type": "textarea",
             "placeholder": '{"type":"service_account",...}', "required": True},
            {"key": "folder_id", "label": "Folder ID", "type": "text",
             "placeholder": "1a2b3c4d...", "required": True},
        ],
    },
    "notion": {
        "label": "Notion",
        "icon": "📝",
        "description": "Import pages from a Notion database",
        "fields": [
            {"key": "integration_token", "label": "Integration Token", "type": "password",
             "placeholder": "secret_...", "required": True},
            {"key": "database_id", "label": "Database ID", "type": "text",
             "placeholder": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", "required": True},
        ],
    },
    "confluence": {
        "label": "Confluence",
        "icon": "🌐",
        "description": "Import pages from a Confluence space",
        "fields": [
            {"key": "base_url", "label": "Base URL", "type": "text",
             "placeholder": "https://yourcompany.atlassian.net/wiki", "required": True},
            {"key": "username", "label": "Username / Email", "type": "text",
             "placeholder": "you@company.com", "required": True},
            {"key": "api_token", "label": "API Token", "type": "password",
             "placeholder": "ATATT...", "required": True},
            {"key": "space_key", "label": "Space Key", "type": "text",
             "placeholder": "MYSPACE", "required": True},
        ],
    },
}


def get_connector_instance(connector_type: str, config: dict, tenant_id: str):
    """Instantiate a connector by type string."""
    cls = CONNECTOR_REGISTRY.get(connector_type)
    if cls is None:
        raise ValueError(f"Unknown connector type: {connector_type}")
    return cls(config=config, tenant_id=tenant_id)
