"""
Build optional DELIVERABLE CONTRACT appendix from heartbeat_config.delivery for CEO/committee prompts.
"""

from typing import Any, Dict, List, Optional


def _delivery_dict(heartbeat_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not heartbeat_config or not isinstance(heartbeat_config, dict):
        return {}
    d = heartbeat_config.get("delivery")
    return d if isinstance(d, dict) else {}


def format_delivery_contract_appendix(heartbeat_config: Optional[Dict[str, Any]]) -> str:
    """Markdown-oriented instructions appended to heartbeat queries."""
    d = _delivery_dict(heartbeat_config)
    sections = d.get("output_sections") or []
    if isinstance(sections, str):
        sections = [s.strip() for s in sections.replace(",", "\n").split("\n") if s.strip()]
    if not isinstance(sections, list):
        sections = []
    disclaimer = (d.get("disclaimer_block") or "").strip()
    extra = (d.get("extra_instructions") or "").strip()

    parts: List[str] = []
    if sections:
        parts.append("\n\nDELIVERABLE CONTRACT (apply to your main response and timeline summary):\n")
        for title in sections[:40]:
            if not isinstance(title, str) or not title.strip():
                continue
            t = title.strip()[:500]
            parts.append(f"- Include a clear section with heading ## {t}\n")
    if disclaimer:
        parts.append("\nRequired disclaimer (append verbatim at the end of the user-visible summary):\n\n")
        parts.append(disclaimer[:12000])
        parts.append("\n")
    if extra:
        parts.append("\nAdditional delivery instructions:\n\n")
        parts.append(extra[:8000])
        parts.append("\n")
    return "".join(parts)
