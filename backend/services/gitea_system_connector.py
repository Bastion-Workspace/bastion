"""
Gitea REST connector for ExecuteGitHubEndpoint (shared RPC).

Uses PAT stored as access_token; authorization_scheme "token" per Gitea API.
Endpoint paths mirror the GitHub system connector where Gitea is compatible.
"""

import copy
from typing import Any, Dict
from urllib.parse import urlparse

from services.github_system_connector import GITHUB_SYSTEM_CONNECTOR_DEFINITION


def normalize_gitea_api_base_url(url: str) -> str:
    """
    Normalize user input to Gitea API root (…/api/v1, no trailing slash).
    Accepts origin only (https://git.example.com) or full API base.
    """
    s = (url or "").strip().rstrip("/")
    if not s:
        return ""
    if not s.startswith(("http://", "https://")):
        s = "https://" + s
    low = s.lower()
    if "/api/v1" not in low:
        s = f"{s}/api/v1"
    return s.rstrip("/")


def build_gitea_connector_definition(api_base_url: str) -> Dict[str, Any]:
    """Build connector definition for connections-service. api_base_url must be normalized."""
    definition = copy.deepcopy(GITHUB_SYSTEM_CONNECTOR_DEFINITION)
    definition["base_url"] = api_base_url.rstrip("/")
    definition["auth"] = {
        "type": "oauth_connection",
        "authorization_scheme": "token",
    }
    definition["headers"] = {"Accept": "application/json"}
    return definition


def api_base_host_label(api_base_url: str) -> str:
    """Short host for account_identifier uniqueness (e.g. git.example.com)."""
    try:
        p = urlparse(api_base_url if "://" in api_base_url else f"https://{api_base_url}")
        return (p.hostname or "").lower() or "gitea"
    except Exception:
        return "gitea"
