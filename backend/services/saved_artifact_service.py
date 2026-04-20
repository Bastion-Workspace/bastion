"""
CRUD and export for user-saved chat artifacts (PostgreSQL `saved_artifacts`).
"""

from __future__ import annotations

import json
import logging
import re
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from models.saved_artifact_models import (
    MAX_SAVED_ARTIFACT_CODE_BYTES,
    SavedArtifactCreate,
    SavedArtifactListResponse,
    SavedArtifactResponse,
    SavedArtifactShareResponse,
    SavedArtifactSummary,
    SavedArtifactUpdate,
    PublicArtifactResponse,
)
from services.database_manager.database_helpers import execute, fetch_all, fetch_one

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid_str(val: Any) -> str:
    if val is None:
        return ""
    return str(val)


def _normalize_react_artifact_user_code(code: str) -> str:
    """
    Standalone HTML uses new Function(), not an ES module. Babel preset-react leaves export syntax;
    strip/rewrite common patterns so the runner matches the in-app iframe (ArtifactRenderer).
    """
    if not (code or "").strip():
        return code or ""
    s = (code or "").replace("\r\n", "\n")
    s = re.sub(
        r"^\s*export\s+default\s+async\s+function\s+",
        "async function ",
        s,
        count=1,
        flags=re.MULTILINE,
    )
    s = re.sub(
        r"^\s*export\s+default\s+function\s+",
        "function ",
        s,
        count=1,
        flags=re.MULTILINE,
    )
    s = re.sub(
        r"^\s*export\s+default\s+class\s+",
        "class ",
        s,
        count=1,
        flags=re.MULTILINE,
    )
    s = re.sub(
        r"^(\s*)export\s+default\s+([A-Za-z_$][\w$]*)\s*;?\s*$",
        r"\1var App = \2",
        s,
        flags=re.MULTILINE,
    )
    if re.search(r"^\s*export\s+default\s+", s, flags=re.MULTILINE):
        s = re.sub(
            r"^\s*export\s+default\s+",
            "const App = ",
            s,
            count=1,
            flags=re.MULTILINE,
        )
    s = re.sub(
        r"^(\s*)export\s+(?!default\b)",
        r"\1",
        s,
        flags=re.MULTILINE,
    )
    return s


def _validate_code_size(code: str) -> None:
    raw = (code or "").encode("utf-8")
    if len(raw) > MAX_SAVED_ARTIFACT_CODE_BYTES:
        raise ValueError(
            f"Artifact code exceeds maximum size ({MAX_SAVED_ARTIFACT_CODE_BYTES} bytes)."
        )


def _row_to_response(row: Dict[str, Any]) -> SavedArtifactResponse:
    return SavedArtifactResponse(
        id=_uuid_str(row["id"]),
        user_id=str(row["user_id"]),
        title=str(row["title"]),
        artifact_type=str(row["artifact_type"]),
        code=str(row["code"] or ""),
        language=row.get("language"),
        share_token=row.get("share_token"),
        is_public=bool(row.get("is_public")),
        source_conversation_id=row.get("source_conversation_id"),
        source_message_id=row.get("source_message_id"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_summary(row: Dict[str, Any]) -> SavedArtifactSummary:
    return SavedArtifactSummary(
        id=_uuid_str(row["id"]),
        title=str(row["title"]),
        artifact_type=str(row["artifact_type"]),
        is_public=bool(row.get("is_public")),
        created_at=row["created_at"],
    )


def build_standalone_export_html(
    artifact_type: str,
    title: str,
    code: str,
) -> str:
    """
    Self-contained HTML suitable for download or offline viewing.
    Mirrors frontend ArtifactRenderer behavior (React via CDN + Babel, etc.).
    """
    at = (artifact_type or "").lower().strip()
    safe_title = (title or "Artifact").replace("<", "").replace(">", "")[:200]
    c = code or ""

    if at == "react":
        user_code_json = json.dumps(_normalize_react_artifact_user_code(c))
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <style>
    *,*::before,*::after{{box-sizing:border-box}}
    body{{margin:0;padding:16px;font-family:system-ui,-apple-system,sans-serif;font-size:14px;background:#fff;color:#1a1a1a}}
    #root{{min-height:40px}}
    pre.err{{white-space:pre-wrap;word-break:break-word;color:#c62828;background:#ffebee;padding:12px;border-radius:4px;font-size:12px;margin:0}}
  </style>
</head>
<body>
  <div id="root"></div>
  <script>
    (function () {{
      var USER_CODE = {user_code_json};
      function showError(msg) {{
        var el = document.getElementById('root');
        var t = String(msg == null ? '' : msg);
        el.innerHTML = '<pre class="err">' + t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</pre>';
      }}
      window.onerror = function (message, source, lineno, colno, error) {{
        showError(String(message) + (error && error.stack ? String.fromCharCode(10) + error.stack : ''));
        return true;
      }};
      if (!USER_CODE || !String(USER_CODE).trim()) {{
        showError('Empty React artifact');
        return;
      }}
      try {{
        if (typeof Babel === 'undefined' || typeof React === 'undefined' || typeof ReactDOM === 'undefined') {{
          showError('Failed to load React or Babel from CDN');
          return;
        }}
        var transformed = Babel.transform(USER_CODE, {{ presets: ['react'], filename: 'artifact.jsx' }}).code;
        var tail = [
          '',
          'var _C = typeof App !== "undefined" ? App : null;',
          'if (!_C && typeof exports !== "undefined" && exports.default) _C = exports.default;',
          'return _C;',
        ].join(String.fromCharCode(10));
        var runner = new Function('React', 'ReactDOM', 'exports', transformed + tail);
        var exportsObj = {{}};
        var Component = runner(React, ReactDOM, exportsObj);
        if (!Component) {{
          showError('Define App or export default (function App, const App, class App, or export default App)');
          return;
        }}
        var mount = document.getElementById('root');
        mount.innerHTML = '';
        var root = ReactDOM.createRoot(mount);
        root.render(React.createElement(Component));
      }} catch (e) {{
        showError((e && e.message ? e.message : String(e)) + (e && e.stack ? String.fromCharCode(10) + e.stack : ''));
      }}
    }})();
  </script>
</body>
</html>"""

    if at in ("html", "chart"):
        low = c.strip().lower()
        if low.startswith("<!doctype") or low.startswith("<html"):
            return c
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
</head>
<body>
{c}
</body>
</html>"""

    if at == "mermaid":
        diagram_json = json.dumps(c)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: false, securityLevel: 'strict', theme: 'neutral' }});
    const diagram = {diagram_json};
    const id = 'mermaid-export';
    const el = document.getElementById('out');
    try {{
      const {{ svg }} = await mermaid.render(id, diagram);
      el.innerHTML = svg;
    }} catch (e) {{
      el.textContent = String(e && e.message ? e.message : e);
    }}
  </script>
</head>
<body>
  <div id="out"></div>
</body>
</html>"""

    if at == "svg":
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>body{{margin:0;padding:16px}} svg{{max-width:100%;height:auto}}</style>
</head>
<body>
{c}
</body>
</html>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{safe_title}</title></head>
<body><pre>Unsupported type: {at}</pre></body></html>"""


async def create_saved_artifact(user_id: str, data: SavedArtifactCreate) -> SavedArtifactResponse:
    _validate_code_size(data.code)
    row = await fetch_one(
        """
        INSERT INTO saved_artifacts (
            user_id, title, artifact_type, code, language,
            source_conversation_id, source_message_id, updated_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        RETURNING *
        """,
        user_id,
        data.title[:255],
        data.artifact_type,
        data.code,
        data.language,
        data.source_conversation_id,
        data.source_message_id,
    )
    if not row:
        raise RuntimeError("Failed to create saved artifact")
    return _row_to_response(row)


async def list_saved_artifacts(user_id: str) -> SavedArtifactListResponse:
    rows = await fetch_all(
        """
        SELECT id, title, artifact_type, is_public, created_at
        FROM saved_artifacts
        WHERE user_id = $1
        ORDER BY created_at DESC
        """,
        user_id,
    )
    return SavedArtifactListResponse(artifacts=[_row_to_summary(r) for r in rows])


async def get_saved_artifact(user_id: str, artifact_id: str) -> Optional[SavedArtifactResponse]:
    row = await fetch_one(
        """
        SELECT * FROM saved_artifacts
        WHERE id = $1::uuid AND user_id = $2
        """,
        artifact_id,
        user_id,
    )
    return _row_to_response(row) if row else None


async def update_saved_artifact(
    user_id: str, artifact_id: str, data: SavedArtifactUpdate
) -> Optional[SavedArtifactResponse]:
    existing = await fetch_one(
        "SELECT * FROM saved_artifacts WHERE id = $1::uuid AND user_id = $2",
        artifact_id,
        user_id,
    )
    if not existing:
        return None
    title = data.title if data.title is not None else existing["title"]
    is_public = bool(existing["is_public"])
    share_token = existing.get("share_token")
    if data.is_public is not None:
        is_public = bool(data.is_public)
        if not is_public:
            share_token = None
    await execute(
        """
        UPDATE saved_artifacts
        SET title = $3, is_public = $4, share_token = $5, updated_at = NOW()
        WHERE id = $1::uuid AND user_id = $2
        """,
        artifact_id,
        user_id,
        title[:255],
        is_public,
        share_token,
    )
    row = await fetch_one(
        "SELECT * FROM saved_artifacts WHERE id = $1::uuid AND user_id = $2",
        artifact_id,
        user_id,
    )
    return _row_to_response(row) if row else None


async def delete_saved_artifact(user_id: str, artifact_id: str) -> bool:
    row = await fetch_one(
        "DELETE FROM saved_artifacts WHERE id = $1::uuid AND user_id = $2 RETURNING id",
        artifact_id,
        user_id,
    )
    return row is not None


def _build_share_urls(base_url: str, share_token: str) -> Tuple[str, str, str]:
    base = (base_url or "").rstrip("/")
    public_path = f"/shared/artifact/{share_token}"
    public_url = f"{base}{public_path}"
    embed_url = f"{public_url}?embed=1"
    api_url = f"{base}/api/public/artifacts/{share_token}"
    return public_url, embed_url, api_url


async def generate_share_token(
    user_id: str, artifact_id: str, base_url: str
) -> Optional[SavedArtifactShareResponse]:
    row = await fetch_one(
        "SELECT * FROM saved_artifacts WHERE id = $1::uuid AND user_id = $2",
        artifact_id,
        user_id,
    )
    if not row:
        return None
    existing = (row.get("share_token") or "").strip()
    if existing and row.get("is_public"):
        token = existing
    else:
        token = secrets.token_hex(32)
        await execute(
            """
            UPDATE saved_artifacts
            SET share_token = $3, is_public = TRUE, updated_at = NOW()
            WHERE id = $1::uuid AND user_id = $2
            """,
            artifact_id,
            user_id,
            token,
        )
    public_url, embed_url, api_url = _build_share_urls(base_url, token)
    return SavedArtifactShareResponse(
        share_token=token,
        public_url=public_url,
        embed_url=embed_url,
        api_url=api_url,
    )


async def revoke_share_token(user_id: str, artifact_id: str) -> bool:
    row = await fetch_one(
        "SELECT id FROM saved_artifacts WHERE id = $1::uuid AND user_id = $2",
        artifact_id,
        user_id,
    )
    if not row:
        return False
    await execute(
        """
        UPDATE saved_artifacts
        SET share_token = NULL, is_public = FALSE, updated_at = NOW()
        WHERE id = $1::uuid AND user_id = $2
        """,
        artifact_id,
        user_id,
    )
    return True


async def get_artifact_by_share_token(token: str) -> Optional[PublicArtifactResponse]:
    if not token or len(token) > 64:
        return None
    row = await fetch_one(
        """
        SELECT title, artifact_type, code, language
        FROM saved_artifacts
        WHERE share_token = $1 AND is_public = TRUE
        """,
        token.strip(),
    )
    if not row:
        return None
    return PublicArtifactResponse(
        title=str(row["title"]),
        artifact_type=str(row["artifact_type"]),
        code=str(row["code"] or ""),
        language=row.get("language"),
    )


async def user_owns_artifact(user_id: str, artifact_id: str) -> bool:
    row = await fetch_one(
        "SELECT 1 FROM saved_artifacts WHERE id = $1::uuid AND user_id = $2",
        artifact_id,
        user_id,
    )
    return row is not None


async def get_export_html(user_id: str, artifact_id: str) -> Optional[Tuple[str, str]]:
    row = await fetch_one(
        """
        SELECT title, artifact_type, code FROM saved_artifacts
        WHERE id = $1::uuid AND user_id = $2
        """,
        artifact_id,
        user_id,
    )
    if not row:
        return None
    title = str(row["title"])
    at = str(row["artifact_type"])
    code = str(row["code"] or "")
    html = build_standalone_export_html(at, title, code)
    safe = "".join(ch for ch in title if ch.isalnum() or ch in (" ", "-", "_")).strip() or "artifact"
    safe = safe.replace(" ", "-")[:80]
    filename = f"{safe}.html"
    return html, filename
