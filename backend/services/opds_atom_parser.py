"""Parse OPDS 1.x Atom acquisition/navigation feeds into JSON-friendly structures."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ATOM = "{http://www.w3.org/2005/Atom}"
_DC = "{http://purl.org/dc/elements/1.1/}"
_OPDS = "{http://opds-spec.org/2010/catalog}"


def _text(el: Optional[ET.Element]) -> Optional[str]:
    if el is None or el.text is None:
        return None
    t = el.text.strip()
    return t if t else None


def _find_links(entry: ET.Element) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for link in entry.findall(f"{_ATOM}link"):
        href = link.get("href")
        if not href:
            continue
        out.append(
            {
                "href": href,
                "rel": link.get("rel") or "",
                "type": link.get("type") or "",
                "title": link.get("title") or "",
            }
        )
    return out


def _is_opds_search_link_rel(rel: str) -> bool:
    """True if Atom link rel identifies an OpenSearch / catalog search link."""
    if not rel or not rel.strip():
        return False
    parts = rel.lower().split()
    if "search" in parts:
        return True
    joined = " ".join(parts)
    if "opensearch.org/specs/opensearch/1.1" in joined:
        return True
    if "a9.com/-/spec/opensearch" in joined:
        return True
    return False


def parse_opensearch_description_template(xml_bytes: bytes) -> Optional[str]:
    """
    Extract the best Atom/XML Url@template from an OpenSearch Description document.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    scored: List[tuple[int, str]] = []
    for el in root.iter():
        local = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if local != "Url":
            continue
        template = el.get("template")
        if not template or not str(template).strip():
            continue
        typ = (el.get("type") or "").lower()
        score = 0
        if "application/atom+xml" in typ and ("opds" in typ or "catalog" in typ):
            score = 5
        elif "application/atom+xml" in typ:
            score = 4
        elif "atom" in typ and "xml" in typ:
            score = 3
        elif "opds" in typ:
            score = 3
        elif "xml" in typ:
            score = 2
        elif "rss" in typ:
            score = 1
        scored.append((score, str(template).strip()))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def parse_feed_atom_search_link(feed_el: ET.Element) -> tuple[Optional[str], Optional[str]]:
    """First feed-level search link: (href, type) or (None, None)."""
    for link in feed_el.findall(f"{_ATOM}link"):
        rel = link.get("rel") or ""
        if not _is_opds_search_link_rel(rel):
            continue
        href = link.get("href")
        if not href:
            continue
        typ = link.get("type") or ""
        return str(href).strip(), typ.strip() or None
    return None, None


def parse_opds_atom(xml_bytes: bytes, base_url: str) -> Dict[str, Any]:
    """
    Return { title, id, entries, links, search_template }.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        logger.warning("OPDS XML parse error: %s", e)
        raise ValueError("Invalid OPDS or Atom XML") from e

    feed_el = root if root.tag.endswith("feed") else root.find(f"{_ATOM}feed")
    if feed_el is None:
        raise ValueError("Not an Atom feed")

    title = _text(feed_el.find(f"{_ATOM}title"))
    feed_id = _text(feed_el.find(f"{_ATOM}id"))

    entries: List[Dict[str, Any]] = []
    for entry in feed_el.findall(f"{_ATOM}entry"):
        eid = _text(entry.find(f"{_ATOM}id"))
        etitle = _text(entry.find(f"{_ATOM}title"))
        summary = _text(entry.find(f"{_ATOM}summary")) or _text(entry.find(f"{_ATOM}content"))
        updated = _text(entry.find(f"{_ATOM}updated"))
        author_el = entry.find(f"{_ATOM}author")
        author_name = _text(author_el.find(f"{_ATOM}name")) if author_el is not None else None
        dc_lang = _text(entry.find(f"{_DC}language"))
        dc_issued = _text(entry.find(f"{_DC}issued"))
        opds_count = None
        cnt = entry.find(f"{_OPDS}numberOfItems")
        if cnt is not None and cnt.text:
            try:
                opds_count = int(cnt.text.strip())
            except ValueError:
                opds_count = None

        links = _find_links(entry)
        acquisition: Optional[str] = None
        acquisition_type: Optional[str] = None
        nav_links: List[Dict[str, Any]] = []
        seen_nav_href: set[str] = set()
        for lk in links:
            href = lk.get("href")
            if not href:
                continue
            rel = lk.get("rel") or ""
            rel_l = rel.lower()
            typ = (lk.get("type") or "").lower()
            href_s = str(href).strip()
            is_opds_acquisition = "opds-spec.org/acquisition" in rel_l or (
                "acquisition" in rel_l and "opds" in rel_l
            )
            if is_opds_acquisition:
                href_l = href_s.lower()
                is_epub = "epub" in typ or href_l.endswith(".epub")
                is_pdf = "application/pdf" in typ or href_l.endswith(".pdf")
                if is_epub:
                    acquisition = href_s
                    acquisition_type = "epub"
                elif is_pdf and acquisition_type != "epub":
                    acquisition = href_s
                    acquisition_type = "pdf"
                continue
            is_nav = (
                "subsection" in rel_l
                or "http://opds-spec.org/catalog" in rel_l
                or "opds-spec.org/group" in rel_l
                or ("opds-catalog" in typ and "application/atom" in typ)
                or ("profile=opds-catalog" in typ and "atom" in typ)
            )
            if is_nav and href_s not in seen_nav_href:
                if "thumbnail" in rel_l or "opds-spec.org/image" in rel_l:
                    continue
                nav_links.append(lk)
                seen_nav_href.add(href_s)

        entries.append(
            {
                "id": eid,
                "title": etitle,
                "summary": summary,
                "updated": updated,
                "author": author_name,
                "language": dc_lang,
                "issued": dc_issued,
                "number_of_items": opds_count,
                "links": links,
                "acquisition_href": acquisition,
                "acquisition_type": acquisition_type,
                "navigation_links": nav_links,
            }
        )

    feed_links = _find_links(feed_el)
    search_href, search_link_type = parse_feed_atom_search_link(feed_el)

    return {
        "feed_title": title,
        "feed_id": feed_id,
        "base_url": base_url,
        "search_template": search_href,
        "search_link_type": search_link_type,
        "feed_links": feed_links,
        "entries": entries,
    }
