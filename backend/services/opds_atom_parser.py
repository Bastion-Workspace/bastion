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


def _parse_search_template(feed_el: ET.Element) -> Optional[str]:
    for link in feed_el.findall(f"{_ATOM}link"):
        rel = (link.get("rel") or "").lower()
        if "search" in rel:
            href = link.get("href")
            if href:
                return href
    return None


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
        acquisition = None
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
                if "epub" in typ or href_s.lower().endswith(".epub"):
                    acquisition = href_s
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
                "navigation_links": nav_links,
            }
        )

    feed_links = _find_links(feed_el)
    search_template = _parse_search_template(feed_el)

    return {
        "feed_title": title,
        "feed_id": feed_id,
        "base_url": base_url,
        "search_template": search_template,
        "feed_links": feed_links,
        "entries": entries,
    }
