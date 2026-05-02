"""
Extract plain-text chapters from EPUB bytes for terminal reading.

Uses stdlib zipfile + ElementTree and html2text (no DRM, no images in output).
KoSync document id matches KOReader / web reader (partial MD5 of file bytes).
"""

from __future__ import annotations

import hashlib
import logging
import zipfile
from io import BytesIO
from typing import List, Optional, Tuple
from urllib.parse import quote, unquote
import xml.etree.ElementTree as ET

import html2text

logger = logging.getLogger(__name__)

# Parity with backend.services.koreader_partial_md5 and frontend koreaderPartialMd5.
_KOREADER_PARTIAL_MD5_GOLDEN = bytes((i % 256) for i in range(10240))
_KOREADER_PARTIAL_MD5_REF = "a52dff8366b1473d2e13edd2415def67"


def koreader_partial_md5(data: bytes) -> str:
    """
    Sample bytes at exponentially spaced offsets; same digest as KOReader KoSync document id.
    """
    h = hashlib.md5()
    step = 1024
    size = 1024
    for i in range(-1, 11):
        shift = 2 * i
        if shift >= 0:
            offset = step << shift
        else:
            offset = step >> (-shift)
        if offset < 0:
            offset = 0
        chunk = data[offset : offset + size]
        if not chunk:
            break
        h.update(chunk)
    return h.hexdigest()


def _verify_koreader_partial_md5() -> None:
    got = koreader_partial_md5(_KOREADER_PARTIAL_MD5_GOLDEN)
    if got != _KOREADER_PARTIAL_MD5_REF:
        logger.warning("koreader_partial_md5 self-check failed: got %s expected %s", got, _KOREADER_PARTIAL_MD5_REF)


_verify_koreader_partial_md5()

_h2t = html2text.HTML2Text()
_h2t.ignore_links = True
_h2t.ignore_images = True
_h2t.body_width = 0


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _find_first_child(parent: ET.Element, local: str) -> Optional[ET.Element]:
    for c in parent:
        if _local_name(c.tag) == local:
            return c
    return None


def _find_all_children(parent: ET.Element, local: str) -> List[ET.Element]:
    return [c for c in parent if _local_name(c.tag) == local]


def _xhtml_to_plain(xhtml_bytes: bytes) -> str:
    try:
        raw = xhtml_bytes.decode("utf-8", errors="replace")
    except Exception:
        return ""
    try:
        return _h2t.handle(raw).strip()
    except Exception as e:
        logger.debug("html2text failed: %s", e)
        return ""


def extract_epub_chapters(epub_bytes: bytes) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Returns (koreader_document_digest, [(chapter_title, plain_text), ...]).
    digest is 32-char hex partial MD5 for KoSync / KOReader parity.
    """
    digest = koreader_partial_md5(epub_bytes)
    chapters: List[Tuple[str, str]] = []

    try:
        zf = zipfile.ZipFile(BytesIO(epub_bytes), "r")
    except zipfile.BadZipFile as e:
        logger.warning("Bad EPUB zip: %s", e)
        return digest, [("Error", "This file is not a valid EPUB (bad zip).")]

    try:
        if "META-INF/container.xml" not in zf.namelist():
            return digest, [("Error", "EPUB missing META-INF/container.xml")]
        container_xml = zf.read("META-INF/container.xml")
        root = ET.fromstring(container_xml)
        rootfiles = _find_first_child(root, "rootfiles")
        if rootfiles is None:
            for c in root:
                if _local_name(c.tag) == "rootfiles":
                    rootfiles = c
                    break
        if rootfiles is None:
            return digest, [("Error", "Invalid container.xml (no rootfiles).")]

        opf_path: Optional[str] = None
        for rf in _find_all_children(rootfiles, "rootfile"):
            fp = rf.get("full-path") or rf.get("fullPath")
            if fp:
                opf_path = fp.replace("\\", "/")
                break
        if not opf_path:
            return digest, [("Error", "No rootfile full-path in container.xml")]

        opf_bytes = zf.read(opf_path)
        opf_root = ET.fromstring(opf_bytes)
        opf_dir = ""
        if "/" in opf_path:
            opf_dir = opf_path.rsplit("/", 1)[0] + "/"

        manifest: dict[str, dict[str, str]] = {}
        man_el = _find_first_child(opf_root, "manifest")
        if man_el is not None:
            for item in _find_all_children(man_el, "item"):
                iid = item.get("id")
                href = item.get("href")
                media = (item.get("media-type") or "").lower()
                if iid and href:
                    manifest[iid] = {"href": href, "media-type": media}

        spine_ids: List[str] = []
        spine_el = _find_first_child(opf_root, "spine")
        if spine_el is not None:
            for itemref in _find_all_children(spine_el, "itemref"):
                idref = itemref.get("idref")
                if idref:
                    spine_ids.append(idref)

        if not spine_ids:
            return digest, [("Error", "EPUB spine is empty.")]

        for idx, sid in enumerate(spine_ids):
            info = manifest.get(sid)
            if not info:
                continue
            href = info["href"]
            media = info["media-type"]
            if "html" not in media and not href.lower().endswith((".xhtml", ".html", ".htm")):
                continue
            inner_path = (opf_dir + unquote(href)).replace("\\", "/")
            try:
                raw = zf.read(inner_path)
            except KeyError:
                logger.debug("Missing spine item: %s", inner_path)
                continue
            plain = _xhtml_to_plain(raw)
            if not plain.strip():
                continue
            first_line = plain.split("\n", 1)[0].strip()
            while first_line.startswith("#"):
                first_line = first_line.lstrip("#").strip()
            title = (first_line[:70] or f"Part {idx + 1}").strip() or f"Part {idx + 1}"
            chapters.append((title, plain))

        if not chapters:
            return digest, [("Error", "No readable XHTML chapters found in this EPUB.")]

        return digest, chapters
    except Exception as e:
        logger.exception("EPUB parse failed: %s", e)
        return digest, [("Error", f"Failed to read EPUB: {e}")]


def opensearch_apply_template(template: str, query: str) -> str:
    """Replace {searchTerms} (OpenSearch) with URL-encoded query."""
    q = quote(query, safe="")
    return template.replace("{searchTerms}", q).replace("%7BsearchTerms%7D", q)
