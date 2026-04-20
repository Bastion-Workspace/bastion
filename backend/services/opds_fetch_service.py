"""Fetch OPDS / EPUB bytes with redirect and size limits."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple, Union

import httpx

from services.opds_atom_parser import parse_opds_atom
from services.opds_url_validator import is_fetch_url_allowed

logger = logging.getLogger(__name__)

MAX_OPDS_XML_BYTES = 8 * 1024 * 1024
MAX_EPUB_BYTES = 80 * 1024 * 1024
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=15.0)


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

    headers = {
        "User-Agent": "BastionOPDS/1.0",
        "Accept": "application/atom+xml, application/xml;q=0.9, */*;q=0.1",
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
            cap = MAX_EPUB_BYTES if want == "binary" else MAX_OPDS_XML_BYTES
            async for part in resp.aiter_bytes():
                if not part:
                    continue
                total += len(part)
                if total > cap:
                    raise ValueError("Response exceeds maximum allowed size")
                chunks.append(part)
            raw = b"".join(chunks)

    if want == "binary":
        media = ctype.split(";")[0].strip() if ctype else "application/octet-stream"
        return "octet", (raw, final_url, media)

    if "xml" not in ctype and not raw.lstrip().startswith(b"<"):
        raise ValueError("Expected XML for atom mode")
    parsed = parse_opds_atom(raw, base_url=final_url)
    return "json", {"feed": parsed, "fetched_url": final_url}
