"""
Minimal YAML-like frontmatter parser for Markdown files.

This utility mirrors the lightweight parsing used in the frontend
and supports simple key: value lines. It ignores complex YAML constructs
to avoid extra dependencies.
"""

from typing import Dict, Tuple


def parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """
    Parse a simple YAML-like frontmatter block from the top of the text.

    Returns (data, body). Tolerant of \r\n or \n newlines and optional BOM.
    """
    try:
        if not text:
            return {}, ""

        import re

        trimmed = text[1:] if text.startswith("\ufeff") else text
        m = re.match(r"^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n?", trimmed)
        if not m:
            return {}, text

        yaml_block = m.group(1)
        body = trimmed[m.end():]

        data: Dict[str, str] = {}
        for line in re.split(r"\r?\n", yaml_block):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip()

        return data, body
    except Exception:
        return {}, text


def build_frontmatter(data: Dict[str, str]) -> str:
    lines = []
    for k, v in data.items():
        if v is None:
            continue
        vs = str(v)
        if len(vs) == 0:
            continue
        lines.append(f"{k}: {vs}")
    if not lines:
        return ""
    return "---\n" + "\n".join(lines) + "\n---\n"




