"""Fetch OPDS / EPUB bytes with redirect and size limits."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urljoin

import httpx

from services.opds_atom_parser import (
    parse_opds_atom,
    parse_opensearch_description_template,
)
from services.opds_url_validator import is_fetch_url_allowed

logger = logging.getLogger(__name__)

MAX_OPDS_XML_BYTES = 8 * 1024 * 1024
MAX_EPUB_BYTES = 80 * 1024 * 1024
MAX_OPENSEARCH_DESC_BYTES = 512 * 1024
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=15.0)


def _looks_like_opensearch_url_template(href: str) -> bool:
    h = href.strip()
    if "{" in h and "}" in h:
        return True
    return "searchterms" in h.lower()


async def _fetch_xml_body_limited(
    *,
    catalog_root: str,
    url: str,
    http_basic: str | None,
    verify_ssl: bool,
    max_bytes: int,
    accept: str,
) -> Tuple[bytes, str]:
    if not is_fetch_url_allowed(catalog_root=catalog_root, target_url=url):
        raise ValueError("URL is not allowed for this catalog")
    headers = {
        "User-Agent": "BastionOPDS/1.0",
        "Accept": accept,
    }
    if http_basic:
        headers["Authorization"] = f"Basic {http_basic}"
    async with httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT,
        follow_redirects=True,
        max_redirects=6,
        verify=verify_ssl,
    ) as client:
        async with client.stream("GET", url, headers=headers) as resp:
            resp.raise_for_status()
            final_url = str(resp.url)
            if not is_fetch_url_allowed(catalog_root=catalog_root, target_url=final_url):
                raise ValueError("Redirect target is not allowed for this catalog")
            total = 0
            chunks: list[bytes] = []
            async for part in resp.aiter_bytes():
                if not part:
                    continue
                total += len(part)
                if total > max_bytes:
                    raise ValueError("Response exceeds maximum allowed size")
                chunks.append(part)
            return b"".join(chunks), final_url


async def _resolve_feed_search_template(
    *,
    catalog_root: str,
    feed_base_url: str,
    search_href: Optional[str],
    search_link_type: Optional[str],
    http_basic: str | None,
    verify_ssl: bool,
) -> Optional[str]:
    """
    If the Atom search link points to an OpenSearch Description document, fetch it
    and return the Url@template; otherwise return href when it already looks like a template.
    """
    if not search_href or not str(search_href).strip():
        return None
    href = urljoin(feed_base_url, str(search_href).strip())
    lt = (search_link_type or "").lower()

    if _looks_like_opensearch_url_template(href):
        return href

    hl = href.lower()
    treat_as_descriptor = "opensearchdescription" in lt or ("opensearch" in hl and hl.endswith(".xml"))

    if not treat_as_descriptor:
        return href

    try:
        raw, _desc_final = await _fetch_xml_body_limited(
            catalog_root=catalog_root,
            url=href,
            http_basic=http_basic,
            verify_ssl=verify_ssl,
            max_bytes=MAX_OPENSEARCH_DESC_BYTES,
            accept="application/opensearchdescription+xml, application/xml, text/xml;q=0.9, */*;q=0.1",
        )
    except Exception as e:
        logger.warning("OpenSearch descriptor fetch failed for %s: %s", href, e)
        return None

    resolved = parse_opensearch_description_template(raw)
    if resolved:
        return resolved.strip()
    logger.warning("OpenSearch descriptor at %s contained no Url template", href)
    return None


async def fetch_opds_resource(
    *,
    catalog_root: str,
    url: str,
    want: str,
    http_basic: str | None = None,
    verify_ssl: bool = True,
) -> Tuple[str, Union[Dict[str, Any], Tuple[bytes, str, str]]]:
    """
    Returns (mode, payload): mode ``json`` + dict for Atom/XML, or mode ``octet`` +
    ``(raw_bytes, final_url, content_type)`` for binary bodies.
    """
    if not is_fetch_url_allowed(catalog_root=catalog_root, target_url=url):
        raise ValueError("URL is not allowed for this catalog")

    cap = MAX_EPUB_BYTES if want == "binary" else MAX_OPDS_XML_BYTES
    accept = "application/atom+xml, application/xml;q=0.9, */*;q=0.1"
    if want == "binary":
        headers = {
            "User-Agent": "BastionOPDS/1.0",
            "Accept": "application/epub+zip, application/octet-stream;q=0.9, */*;q=0.1",
        }
        if http_basic:
            headers["Authorization"] = f"Basic {http_basic}"
        async with httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT,
            follow_redirects=True,
            max_redirects=6,
            verify=verify_ssl,
        ) as client:
            async with client.stream("GET", url, headers=headers) as resp:
                resp.raise_for_status()
                final_url = str(resp.url)
                if not is_fetch_url_allowed(catalog_root=catalog_root, target_url=final_url):
                    raise ValueError("Redirect target is not allowed for this catalog")
                ctype = (resp.headers.get("content-type") or "").lower()
                total = 0
                chunks: list[bytes] = []
                async for part in resp.aiter_bytes():
                    if not part:
                        continue
                    total += len(part)
                    if total > cap:
                        raise ValueError("Response exceeds maximum allowed size")
                    chunks.append(part)
                raw = b"".join(chunks)
    else:
        raw, final_url = await _fetch_xml_body_limited(
            catalog_root=catalog_root,
            url=url,
            http_basic=http_basic,
            verify_ssl=verify_ssl,
            max_bytes=cap,
            accept=accept,
        )
        ctype = ""

    if want == "binary":
        media = ctype.split(";")[0].strip() if ctype else "application/octet-stream"
        return "octet", (raw, final_url, media)

    if "xml" not in ctype and not raw.lstrip().startswith(b"<"):
        raise ValueError("Expected XML for atom mode")
    parsed = parse_opds_atom(raw, base_url=final_url)
    search_lt = parsed.pop("search_link_type", None)
    raw_search = parsed.get("search_template")
    resolved_search = await _resolve_feed_search_template(
        catalog_root=catalog_root,
        feed_base_url=final_url,
        search_href=raw_search if isinstance(raw_search, str) else None,
        search_link_type=search_lt if isinstance(search_lt, str) else None,
        http_basic=http_basic,
        verify_ssl=verify_ssl,
    )
    if resolved_search:
        parsed["search_template"] = resolved_search
    elif raw_search and not _looks_like_opensearch_url_template(str(raw_search)):
        parsed["search_template"] = None

    return "json", {"feed": parsed, "fetched_url": final_url}
