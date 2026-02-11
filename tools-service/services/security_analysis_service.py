"""
Security Analysis Service
Passive security scanning for exposed files, headers, and information disclosure.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp

logger = logging.getLogger(__name__)

SECURITY_DISCLAIMER = (
    "This security scan performs passive reconnaissance only. You must have "
    "authorization to scan the target website. Unauthorized security scanning "
    "may be illegal in your jurisdiction. Use this tool only on websites you "
    "own or have explicit permission to test. By proceeding, you confirm that "
    "you have proper authorization to scan this website."
)

USER_AGENT = "BastionSecurityScanner/1.0 (Passive reconnaissance; authorized use only)"
REQUEST_DELAY_SECONDS = 0.6
REQUEST_TIMEOUT_SECONDS = 15

EXPOSED_FILE_CHECKS = {
    "critical": [
        ".env",
        ".git/config",
        "config.php",
        "database.sql",
        ".env.local",
        ".env.production",
    ],
    "high": [
        ".env.backup",
        ".git/HEAD",
        "backup.sql",
        "backup.zip",
        "config.php.bak",
        "web.config.bak",
        ".htpasswd",
        "phpinfo.php",
    ],
    "medium": [
        "robots.txt",
        ".htaccess",
        "web.config",
        "sitemap.xml",
        "crossdomain.xml",
        "clientaccesspolicy.xml",
    ],
    "low": [
        "readme.md",
        "README.md",
        "changelog.txt",
        "CHANGELOG.md",
        "license.txt",
    ],
}

ADMIN_PATHS = [
    "admin/",
    "wp-admin/",
    "administrator/",
    "phpmyadmin/",
    "cpanel/",
    "webmail/",
    "admin.php",
    "login/",
    "dashboard/",
]

CRITICAL_HEADERS = [
    "Content-Security-Policy",
    "X-Frame-Options",
    "Strict-Transport-Security",
    "X-Content-Type-Options",
    "Referrer-Policy",
    "Permissions-Policy",
]


async def analyze_website(
    url: str,
    user_id: str = "system",
    scan_depth: str = "comprehensive",
) -> Dict[str, Any]:
    """
    Perform passive security analysis of a website.

    Args:
        url: Target URL (must include scheme, e.g. https://example.com)
        user_id: User ID for audit logging
        scan_depth: "basic", "intermediate", or "comprehensive"

    Returns:
        Dict with findings, technology_stack, risk_score, summary, disclaimer
    """
    logger.info("Security analysis started for %s (user=%s)", url, user_id)
    scan_timestamp = datetime.now(timezone.utc).isoformat()

    if not url or not url.strip():
        return _error_response(url, scan_timestamp, "Target URL is required")

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return _error_response(url, scan_timestamp, "Invalid URL: no host")
    except Exception as e:
        return _error_response(url, scan_timestamp, f"Invalid URL: {e}")

    findings: List[Dict[str, Any]] = []
    technology_stack: Dict[str, str] = {}
    security_headers: Dict[str, Any] = {"present": [], "missing": [], "raw": {}}

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    headers = {"User-Agent": USER_AGENT}

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        base_url = url.rstrip("/")

        if scan_depth in ("intermediate", "comprehensive"):
            exposed = await _check_exposed_files(session, base_url, scan_depth)
            findings.extend(exposed)
            await asyncio.sleep(REQUEST_DELAY_SECONDS)

        header_findings, tech, header_map = await _check_security_headers(session, base_url)
        findings.extend(header_findings)
        technology_stack.update(tech)
        security_headers["present"] = list(header_map.get("present", []))
        security_headers["missing"] = list(header_map.get("missing", []))
        security_headers["raw"] = header_map.get("raw", {})
        await asyncio.sleep(REQUEST_DELAY_SECONDS)

        if scan_depth == "comprehensive":
            robots_findings = await _analyze_robots_sitemap(session, base_url)
            findings.extend(robots_findings)
            await asyncio.sleep(REQUEST_DELAY_SECONDS)

            admin_findings = await _check_admin_paths(session, base_url)
            findings.extend(admin_findings)
            await asyncio.sleep(REQUEST_DELAY_SECONDS)

            info_findings = await _check_information_disclosure(session, base_url)
            findings.extend(info_findings)

    risk_score = _calculate_risk_score(findings)
    summary = _build_summary(findings, risk_score, technology_stack)

    return {
        "success": True,
        "target_url": url,
        "scan_timestamp": scan_timestamp,
        "findings": findings,
        "technology_stack": technology_stack,
        "security_headers": security_headers,
        "risk_score": round(risk_score, 1),
        "summary": summary,
        "disclaimer": SECURITY_DISCLAIMER,
        "error": None,
    }


def _error_response(
    target_url: str,
    scan_timestamp: str,
    error: str,
) -> Dict[str, Any]:
    return {
        "success": False,
        "target_url": target_url or "",
        "scan_timestamp": scan_timestamp,
        "findings": [],
        "technology_stack": {},
        "security_headers": {"present": [], "missing": [], "raw": {}},
        "risk_score": 0.0,
        "summary": "",
        "disclaimer": SECURITY_DISCLAIMER,
        "error": error,
    }


async def _check_exposed_files(
    session: aiohttp.ClientSession,
    base_url: str,
    scan_depth: str,
) -> List[Dict[str, Any]]:
    findings = []
    severities_to_check = ["critical", "high"]
    if scan_depth == "comprehensive":
        severities_to_check.extend(["medium", "low"])

    for severity in severities_to_check:
        paths = EXPOSED_FILE_CHECKS.get(severity, [])
        for path in paths:
            full_url = urljoin(base_url + "/", path.lstrip("/"))
            try:
                async with session.head(full_url) as resp:
                    await asyncio.sleep(REQUEST_DELAY_SECONDS)
                    status = resp.status
                    if status == 200:
                        content_length = resp.headers.get("Content-Length", "unknown")
                        findings.append({
                            "category": "exposed_files",
                            "severity": severity,
                            "title": f"Exposed file or path: {path}",
                            "description": f"The path {path} is publicly accessible (HTTP 200).",
                            "url": full_url,
                            "evidence": f"HTTP 200, Content-Length: {content_length}",
                            "remediation": (
                                "Remove sensitive files from the web root, or configure "
                                "the server to deny access (e.g. .htaccess, nginx location block)."
                            ),
                        })
                    elif status == 403:
                        findings.append({
                            "category": "exposed_files",
                            "severity": "info",
                            "title": f"Path exists but forbidden: {path}",
                            "description": f"The path {path} exists and returns 403 Forbidden.",
                            "url": full_url,
                            "evidence": "HTTP 403",
                            "remediation": "Access is restricted; consider removing or moving outside web root.",
                        })
            except asyncio.TimeoutError:
                pass
            except aiohttp.ClientError:
                pass
            except Exception as e:
                logger.debug("Check %s failed: %s", full_url, e)

    return findings


async def _check_security_headers(
    session: aiohttp.ClientSession,
    base_url: str,
) -> tuple:
    findings = []
    tech = {}
    present = []
    missing = []
    raw = {}

    try:
        async with session.get(base_url) as resp:
            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            headers = {k.lower(): v for k, v in resp.headers.items()}
            raw = dict(resp.headers)

            server = resp.headers.get("Server") or resp.headers.get("X-Powered-By")
            if server:
                tech["web_server"] = server.strip()
                if "version" in server.lower() or re.search(r"/[\d.]+", server):
                    findings.append({
                        "category": "security_headers",
                        "severity": "low",
                        "title": "Server version disclosure",
                        "description": f"Server or runtime version is exposed: {server}",
                        "url": base_url,
                        "evidence": f"Header: {server}",
                        "remediation": "Disable server version disclosure in server configuration.",
                    })

            for name in CRITICAL_HEADERS:
                key = name.lower()
                if key in headers:
                    present.append(name)
                else:
                    missing.append(name)

            if missing:
                findings.append({
                    "category": "security_headers",
                    "severity": "medium",
                    "title": "Missing security headers",
                    "description": f"Missing recommended headers: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}",
                    "url": base_url,
                    "evidence": f"Missing: {', '.join(missing)}",
                    "remediation": "Add Content-Security-Policy, X-Frame-Options, Strict-Transport-Security, X-Content-Type-Options.",
                })

            cors = resp.headers.get("Access-Control-Allow-Origin")
            if cors == "*":
                findings.append({
                    "category": "security_headers",
                    "severity": "medium",
                    "title": "Permissive CORS policy",
                    "description": "Access-Control-Allow-Origin is * (all origins allowed).",
                    "url": base_url,
                    "evidence": "Access-Control-Allow-Origin: *",
                    "remediation": "Restrict CORS to specific origins where possible.",
                })

    except asyncio.TimeoutError:
        findings.append({
            "category": "security_headers",
            "severity": "info",
            "title": "Timeout fetching base URL",
            "description": "The request to the base URL timed out; header analysis skipped.",
            "url": base_url,
            "evidence": "Request timeout",
            "remediation": "Check site availability and firewall rules.",
        })
    except Exception as e:
        logger.warning("Security headers check failed: %s", e)
        findings.append({
            "category": "security_headers",
            "severity": "info",
            "title": "Could not fetch base URL",
            "description": str(e)[:200],
            "url": base_url,
            "evidence": str(e),
            "remediation": "Verify URL and network access.",
        })

    header_map = {"present": present, "missing": missing, "raw": raw}
    return findings, tech, header_map


async def _analyze_robots_sitemap(
    session: aiohttp.ClientSession,
    base_url: str,
) -> List[Dict[str, Any]]:
    findings = []
    try:
        robots_url = urljoin(base_url + "/", "robots.txt")
        async with session.get(robots_url) as resp:
            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            if resp.status == 200:
                text = await resp.text()
                if "Disallow:" in text and ("/admin" in text or "/wp-admin" in text):
                    findings.append({
                        "category": "info_disclosure",
                        "severity": "info",
                        "title": "robots.txt reveals disallowed paths",
                        "description": "robots.txt lists Disallow rules that may indicate admin or sensitive paths.",
                        "url": robots_url,
                        "evidence": "robots.txt accessible with Disallow rules",
                        "remediation": "Ensure disallowed paths are not guessable; consider minimal robots.txt.",
                    })
    except Exception:
        pass

    try:
        sitemap_url = urljoin(base_url + "/", "sitemap.xml")
        async with session.get(sitemap_url) as resp:
            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            if resp.status == 200:
                findings.append({
                    "category": "info_disclosure",
                    "severity": "info",
                    "title": "sitemap.xml accessible",
                    "description": "sitemap.xml is publicly accessible; reveals site structure.",
                    "url": sitemap_url,
                    "evidence": "HTTP 200",
                    "remediation": "Normal for SEO; ensure no sensitive URLs are listed.",
                })
    except Exception:
        pass

    return findings


async def _check_admin_paths(
    session: aiohttp.ClientSession,
    base_url: str,
) -> List[Dict[str, Any]]:
    findings = []
    for path in ADMIN_PATHS[:10]:
        full_url = urljoin(base_url + "/", path)
        try:
            async with session.head(full_url) as resp:
                await asyncio.sleep(REQUEST_DELAY_SECONDS)
                if resp.status == 200:
                    findings.append({
                        "category": "exposed_files",
                        "severity": "high",
                        "title": f"Admin or login path accessible: {path}",
                        "description": f"The path {path} returns HTTP 200; may be an admin or login panel.",
                        "url": full_url,
                        "evidence": "HTTP 200",
                        "remediation": "Restrict access with authentication and IP allowlist; use strong passwords.",
                    })
        except Exception:
            pass
    return findings


async def _check_information_disclosure(
    session: aiohttp.ClientSession,
    base_url: str,
) -> List[Dict[str, Any]]:
    findings = []
    try:
        async with session.get(base_url) as resp:
            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            if resp.status != 200:
                return findings
            html = await resp.text()
            if "<!--" in html and ("password" in html.lower() or "secret" in html.lower() or "key" in html.lower()):
                findings.append({
                    "category": "info_disclosure",
                    "severity": "medium",
                    "title": "HTML may contain sensitive comments",
                    "description": "Page contains HTML comments and words like password/secret/key.",
                    "url": base_url,
                    "evidence": "HTML comments present with sensitive keywords",
                    "remediation": "Remove sensitive data from HTML comments and source.",
                })
            if "phpinfo" in html.lower() or "PHP Version" in html:
                findings.append({
                    "category": "info_disclosure",
                    "severity": "high",
                    "title": "PHPInfo or version info in response",
                    "description": "Page may expose PHP or server configuration.",
                    "url": base_url,
                    "evidence": "phpinfo or PHP Version detected in content",
                    "remediation": "Remove or protect phpinfo and version disclosure pages.",
                })
    except Exception:
        pass
    return findings


def _calculate_risk_score(findings: List[Dict[str, Any]]) -> float:
    weights = {"critical": 2.5, "high": 1.5, "medium": 0.8, "low": 0.3, "info": 0.1}
    score = 0.0
    for f in findings:
        sev = f.get("severity", "info")
        score += weights.get(sev, 0.1)
    return min(10.0, round(score, 1))


def _build_summary(
    findings: List[Dict[str, Any]],
    risk_score: float,
    technology_stack: Dict[str, str],
) -> str:
    critical_count = sum(1 for f in findings if f.get("severity") == "critical")
    high_count = sum(1 for f in findings if f.get("severity") == "high")
    if risk_score >= 7:
        level = "HIGH"
    elif risk_score >= 4:
        level = "MEDIUM"
    else:
        level = "LOW"
    parts = [
        f"Risk score: {risk_score}/10 ({level}).",
        f"Findings: {len(findings)} total ({critical_count} critical, {high_count} high).",
    ]
    if technology_stack:
        parts.append(f"Technology: {', '.join(f'{k}={v}' for k, v in list(technology_stack.items())[:3])}.")
    if critical_count or high_count:
        parts.append("Immediate remediation recommended for critical and high findings.")
    return " ".join(parts)
