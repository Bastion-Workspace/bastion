"""URL checks for OPDS proxy catalog registration and fetches.

RFC1918, Tailscale (100.64/10), and other ``is_private`` addresses are allowed so homelab
OPDS works. Loopback, link-local (including cloud metadata), multicast, and unspecified
addresses stay blocked.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Union
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)


def _ip_blocked_for_opds(ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]) -> bool:
    """True if this resolved address must not be used as an OPDS catalog origin."""
    if getattr(ip, "is_unspecified", False):
        return True
    if ip.is_loopback or ip.is_link_local or ip.is_multicast:
        return True
    return False


def _host_is_blocked(host: str) -> bool:
    if not host:
        return True
    h = host.strip().lower()
    if h in ("localhost",):
        return True
    try:
        ip = ipaddress.ip_address(h)
        return _ip_blocked_for_opds(ip)
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        for info in infos:
            addr = info[4][0]
            try:
                ip = ipaddress.ip_address(addr)
                if _ip_blocked_for_opds(ip):
                    return True
            except ValueError:
                continue
    except OSError as e:
        # Do not block catalog save when DNS is temporarily unavailable (e.g. container startup).
        logger.warning("OPDS URL host resolution inconclusive for %s: %s (allowing hostname)", host, e)
        return False
    return False


def normalize_catalog_root_url(url: str) -> str:
    """
    Strip and ensure a usable http(s) URL with a host.
    Accepts host-only or path-first strings by prepending https://.
    """
    u = (url or "").strip()
    if not u:
        return u
    p = urlparse(u)
    sch = (p.scheme or "").lower()
    if sch and sch not in ("http", "https"):
        return u
    if not p.scheme and p.netloc:
        u = f"https://{u}" if not u.startswith("//") else f"https:{u}"
    elif not p.scheme and ("/" in u or "." in u):
        if not u.startswith("//"):
            u = f"https://{u}"
    return u.strip()


def assert_http_catalog_url(url: str) -> None:
    u = normalize_catalog_root_url(url)
    p = urlparse(u)
    if (p.scheme or "").lower() not in ("http", "https"):
        raise ValueError("Catalog URL must use http or https")
    if not p.netloc:
        raise ValueError("Catalog URL must include a hostname (e.g. https://example.com/catalog.atom)")
    host = p.hostname
    if host and _host_is_blocked(host):
        raise ValueError(
            "Catalog host is not allowed: it resolves to loopback, link-local, multicast, "
            "or an unspecified address from the server."
        )


def normalize_url_base(url: str) -> str:
    u = url.strip()
    if not u.endswith("/"):
        u = u + "/"
    return u


def strip_url_fragment(url: str) -> str:
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, p.query, ""))


def catalog_fetch_prefix(catalog_root: str) -> str:
    """
    Build URL prefix for allowed fetches: same origin as catalog, path is directory
    containing the catalog file (or catalog path with trailing slash).
    """
    r = strip_url_fragment(catalog_root.strip())
    p = urlparse(r)
    path = p.path or "/"
    pl = path.lower()
    if pl.endswith(".xml") or pl.endswith(".atom") or pl.endswith(".opds"):
        if "/" in path:
            path = path.rsplit("/", 1)[0] + "/"
        else:
            path = "/"
    elif not path.endswith("/"):
        path = path + "/"
    base = urlunparse((p.scheme, p.netloc, path, "", "", ""))
    if not base.endswith("/"):
        base = base + "/"
    return base


def is_fetch_url_allowed(*, catalog_root: str, target_url: str) -> bool:
    """Allow target only if same host/scheme as catalog and path is under catalog directory prefix."""
    prefix_url = catalog_fetch_prefix(catalog_root)
    target = strip_url_fragment(target_url.strip())
    pu, tu = urlparse(prefix_url), urlparse(target)
    if tu.scheme not in ("http", "https"):
        return False
    host = tu.hostname
    if host and _host_is_blocked(host):
        return False
    if pu.scheme.lower() != tu.scheme.lower() or pu.netloc.lower() != tu.netloc.lower():
        return False
    pp = pu.path or "/"
    tp = tu.path or "/"
    if not pp.endswith("/"):
        pp = pp + "/"
    return tp == pp.rstrip("/") or tp.startswith(pp)
