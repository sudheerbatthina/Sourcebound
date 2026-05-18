"""Google Drive connector — reads text files and exports Google Docs."""

import io
import json
import logging
from .base import BaseConnector

logger = logging.getLogger(__name__)


class GoogleDriveConnector(BaseConnector):
    """Fetches text content from a Google Drive folder using a service account."""

    def fetch_data(self) -> list[dict]:
        try:
            from google.oauth2.service_account import Credentials
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError:
            raise RuntimeError(
                "google-api-python-client and google-auth are not installed. "
                "Run: pip install google-api-python-client google-auth"
            )

        creds_json = self.config.get("service_account_json", "")
        folder_id = self.config.get("folder_id", "")

        if not creds_json or not folder_id:
            raise ValueError("service_account_json and folder_id are required")

        try:
            creds_info = json.loads(creds_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid service_account_json: {exc}")

        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        service = build("drive", "v3", credentials=creds, cache_discovery=False)

        # List supported files in folder
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            fields="files(id,name,mimeType)",
            pageSize=100,
        ).execute()
        files = results.get("files", [])

        GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
        TEXT_MIMES = {"text/plain", "text/markdown"}

        docs = []
        for f in files:
            mime = f["mimeType"]
            name = f["name"]
            fid = f["id"]
            try:
                if mime == GOOGLE_DOC_MIME:
                    export = service.files().export(fileId=fid, mimeType="text/plain").execute()
                    text = export.decode("utf-8") if isinstance(export, bytes) else export
                elif mime in TEXT_MIMES:
                    req = service.files().get_media(fileId=fid)
                    buf = io.BytesIO()
                    dl = MediaIoBaseDownload(buf, req)
                    done = False
                    while not done:
                        _, done = dl.next_chunk()
                    text = buf.getvalue().decode("utf-8", errors="replace")
                else:
                    continue

                if text.strip():
                    docs.append({
                        "content": text,
                        "source": f"gdrive://{fid}/{name}",
                        "metadata": {"drive_file_id": fid, "filename": name},
                    })
            except Exception as exc:
                logger.warning("GDrive: could not read %s (%s): %s", name, fid, exc)

        return docs
