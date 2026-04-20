"""
Chat artifact tool: structured HTML, Mermaid, chart, SVG, or React/JSX payloads for the UI drawer.
Zone 1 (orchestrator-local): no gRPC or external I/O.
"""

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)

_MAX_CODE_BYTES = 512_000
_ALLOWED_TYPES: frozenset = frozenset({"html", "mermaid", "chart", "svg", "react"})


class CreateArtifactInputs(BaseModel):
    artifact_type: str = Field(
        description=(
            "One of: html, mermaid, chart, svg, react. "
            "chart = Plotly-style HTML. "
            "react = single-file JSX/JS in a sandboxed iframe: React/ReactDOM globals only, no import/require; "
            "root component named App or one export default (see artifact_react skill). "
            "For html, chart, and react: live JSON is available via window.bastion.query(path, queryParams) — "
            "parent-proxied allowlisted GET only; do not fetch() /api/* from the iframe. Full path list and limits in artifact_react skill. "
            "The user can later save the artifact and embed it as a dashboard widget or control pane; design for variable sizes."
        )
    )
    title: str = Field(description="Short title shown in chat and the artifact panel")
    code: str = Field(
        description=(
            "Full source: HTML, Mermaid text, SVG, or react artifact (one file, no imports; "
            "App or single export default for react). "
            "HTML/chart may call window.bastion.query for allowlisted GET data same as react (see artifact_react skill). "
            "For stateful artifacts (timers, counters, toggles): use bastion.setState(key, value), "
            "bastion.getState(key), bastion.onStateChange(cb) to keep state synced across dashboard and "
            "control pane instances. See artifact_react skill for full pattern."
        )
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional hint for code view: html, javascript, jsx, mermaid, svg",
    )


class CreateArtifactOutputs(BaseModel):
    artifact: Dict[str, Any] = Field(description="Validated payload for the chat UI")
    formatted: str = Field(description="Human-readable summary for the LLM")


def create_artifact_tool(
    artifact_type: str,
    title: str,
    code: str,
    language: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Register a visual artifact for display in the chat sidebar drawer.
    Returns dict with artifact payload and formatted summary.
    """
    _ = user_id
    at = (artifact_type or "").strip().lower()
    if at not in _ALLOWED_TYPES:
        msg = (
            f"Invalid artifact_type '{artifact_type}'. "
            f"Must be one of: {', '.join(sorted(_ALLOWED_TYPES))}."
        )
        return {
            "artifact": {},
            "formatted": msg,
        }

    raw = code or ""
    if len(raw.encode("utf-8")) > _MAX_CODE_BYTES:
        msg = f"Artifact code exceeds maximum size ({_MAX_CODE_BYTES} bytes). Shorten the content."
        logger.warning("create_artifact rejected: code too large")
        return {"artifact": {}, "formatted": msg}

    t = (title or "").strip() or "Untitled artifact"
    lang = (language or "").strip() or None
    payload: Dict[str, Any] = {
        "artifact_type": at,
        "title": t,
        "code": raw,
    }
    if lang:
        payload["language"] = lang

    formatted = f"Created artifact: {t} ({at})"
    return {"artifact": payload, "formatted": formatted}


register_action(
    name="create_artifact",
    category="output",
    description=(
        "Create a rich visual artifact (HTML, chart, Mermaid diagram, SVG, or live React/JSX preview) for the chat artifact panel. "
        "HTML/chart/react previews may use window.bastion.query for allowlisted authenticated GET JSON (see artifact_react skill); "
        "no raw fetch to private APIs from the iframe."
    ),
    inputs_model=CreateArtifactInputs,
    outputs_model=CreateArtifactOutputs,
    tool_function=create_artifact_tool,
)
