"""
Org file header (in-buffer settings) parser.

Parses the block of lines before the first * heading in an org file.
Used by archive, todo create, and refile to respect #+ARCHIVE:, #+CATEGORY:, #+FILETAGS:, #+PRIORITY:.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class OrgFileHeader:
    """Parsed in-buffer settings from an org file (lines before first heading)."""

    archive: Optional[str] = None   # e.g. ./Archive/%s_archive.org
    category: Optional[str] = None  # e.g. work
    filetags: Optional[str] = None  # raw value e.g. :work:org: or work org
    priority: Optional[str] = None  # A, B, or C


# Regexes: optional leading whitespace, then #+KEY: value. Stop at first heading.
_RE_ARCHIVE = re.compile(r"^\s*#\+ARCHIVE:\s+(.+?)(?:::|$)", re.IGNORECASE)
_RE_CATEGORY = re.compile(r"^\s*#\+CATEGORY:\s+(.+)$", re.IGNORECASE)
_RE_FILETAGS = re.compile(r"^\s*#\+FILETAGS:\s+(.+)$", re.IGNORECASE)
_RE_PRIORITY = re.compile(r"^\s*#\+PRIORITY:\s+([ABC])", re.IGNORECASE)
_RE_FIRST_HEADING = re.compile(r"^\*+\s+")


def parse_org_file_header(content: str) -> OrgFileHeader:
    """
    Parse in-buffer settings from org file content.

    Only considers lines before the first line that looks like a heading (starts with * and space).
    Strips BOM from content. Returns an OrgFileHeader with any found directives.
    """
    if not content:
        return OrgFileHeader()
    content = content.lstrip("\ufeff")
    lines = content.split("\n")
    archive: Optional[str] = None
    category: Optional[str] = None
    filetags: Optional[str] = None
    priority: Optional[str] = None

    for line in lines:
        if _RE_FIRST_HEADING.match(line):
            break
        if not line.strip():
            continue
        if archive is None:
            m = _RE_ARCHIVE.match(line)
            if m:
                archive = m.group(1).strip()
                continue
        if category is None:
            m = _RE_CATEGORY.match(line)
            if m:
                category = m.group(1).strip()
                continue
        if filetags is None:
            m = _RE_FILETAGS.match(line)
            if m:
                filetags = m.group(1).strip()
                continue
        if priority is None:
            m = _RE_PRIORITY.match(line)
            if m:
                priority = m.group(1).upper()
                continue

    return OrgFileHeader(
        archive=archive,
        category=category,
        filetags=filetags,
        priority=priority,
    )


def parse_org_file_header_from_path(path: Path) -> OrgFileHeader:
    """Read file with BOM-safe encoding and parse header."""
    if not path.exists():
        return OrgFileHeader()
    try:
        content = path.read_text(encoding="utf-8-sig")
        return parse_org_file_header(content)
    except Exception:
        return OrgFileHeader()


def filetags_to_list(filetags_raw: Optional[str]) -> List[str]:
    """
    Convert #+FILETAGS: value to a list of tag strings.

    Accepts :tag1:tag2: or tag1 tag2 or tag1  tag2. Returns empty list if None or empty.
    """
    if not filetags_raw or not filetags_raw.strip():
        return []
    s = filetags_raw.strip()
    if ":" in s:
        parts = [p for p in s.split(":") if p.strip()]
        return parts
    return [p for p in s.split() if p.strip()]
