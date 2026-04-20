"""Shared GitHub tool helpers: backend dispatch, list preview, and formatters."""

import logging
from typing import Any, Callable, Dict, List, Optional

from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


def _err(formatted: str) -> Dict[str, Any]:
    return {"records": [], "count": 0, "formatted": formatted, "error": formatted}


async def github_execute(
    user_id: str,
    connection_id: int,
    endpoint_id: str,
    params: Optional[Dict[str, Any]] = None,
    max_pages: int = 5,
) -> Dict[str, Any]:
    if not connection_id:
        return _err("GitHub: connection_id is required (bind a GitHub account or use a scoped tool from the github tool pack).")
    client = await get_backend_tool_client()
    result = await client.execute_github_endpoint(
        user_id=user_id,
        connection_id=int(connection_id),
        endpoint_id=endpoint_id,
        params=params or {},
        max_pages=max_pages,
    )
    if result is None:
        return _err("GitHub request failed (no response from backend).")
    if result.get("error"):
        return {
            "records": result.get("records") or [],
            "count": result.get("count") or 0,
            "formatted": result.get("formatted") or result.get("error"),
            "error": result.get("error"),
        }
    records = result.get("records") or []
    if not isinstance(records, list):
        records = [records] if records else []
    formatted = result.get("formatted") or ""
    if not formatted.strip():
        formatted = f"GitHub returned {len(records)} record(s) for {endpoint_id}."
    return {
        "records": records,
        "count": len(records),
        "formatted": formatted,
    }


def _preview_list(
    items: List[Dict[str, Any]],
    title: str,
    line_fn: Callable[[Dict[str, Any], int], str],
    limit: int = 15,
    grounding: Optional[str] = None,
) -> str:
    if not items:
        return f"No {title} found."
    parts: List[str] = []
    if grounding:
        parts.append(grounding)
        parts.append(f"count: {len(items)}")
        parts.append("")
    for i, it in enumerate(items[:limit]):
        parts.append(line_fn(it, i))
    more = len(items) - limit
    if more > 0:
        parts.append(f"... and {more} more.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Formatters — used by github_tools_impl.py
# ---------------------------------------------------------------------------

def fmt_repos(records: List[Dict[str, Any]]) -> str:
    def _line(r: Dict[str, Any], _: int) -> str:
        name = r.get("full_name") or r.get("name") or "?"
        vis = "private" if r.get("private") else "public"
        head = f"- **{name}** ({vis})"
        extras = []
        ts = r.get("updated_at") or r.get("pushed_at")
        if ts:
            extras.append(f"updated {str(ts)[:10]}")
        desc = (r.get("description") or "").strip()
        if desc:
            extras.append((desc[:160] + "…") if len(desc) > 160 else desc)
        if extras:
            return head + " — " + " — ".join(extras)
        return head

    return _preview_list(
        records, "repositories", _line,
        grounding="GitHub repositories. Use only these exact names when referencing repos.",
    )


def fmt_repo_detail(r: Dict[str, Any]) -> str:
    return (
        f"GitHub repo detail. Quote the full name exactly when referencing.\n\n"
        f"**{r.get('full_name', '')}** — {r.get('description') or 'no description'}\n"
        f"Default branch: {r.get('default_branch')} · Stars: {r.get('stargazers_count')} · Open issues: {r.get('open_issues_count')}"
    )


def fmt_issues(recs: List[Dict[str, Any]], owner: str, repo: str) -> str:
    return _preview_list(
        recs, "issues",
        lambda i, _: f"- #{i.get('number')} [{i.get('state')}] {i.get('title', '')[:120]}",
        grounding=f"GitHub issues in {owner}/{repo}. Use only these exact numbers and titles in your reply.",
    )


def fmt_issue_detail(i: Dict[str, Any]) -> str:
    body = (i.get("body") or "")[:2000]
    return (
        f"GitHub issue detail. Quote the title exactly when referencing.\n\n"
        f"#{i.get('number')} **{i.get('title')}** ({i.get('state')})\n{i.get('html_url')}\n\n{body}"
    )


def fmt_comments(recs: List[Dict[str, Any]], context: str) -> str:
    return _preview_list(
        recs, "comments",
        lambda c, _: f"- @{c.get('user', {}).get('login', '?')}: {(c.get('body') or '')[:200]}",
        grounding=f"GitHub comments on {context}. Quote comment text verbatim when referencing.",
    )


def fmt_pulls(recs: List[Dict[str, Any]], owner: str, repo: str) -> str:
    return _preview_list(
        recs, "pull requests",
        lambda p, _: f"- PR #{p.get('number')}: {p.get('title', '')[:100]} ({p.get('head', {}).get('ref')} → {p.get('base', {}).get('ref')})",
        grounding=f"GitHub pull requests in {owner}/{repo}. Use only these exact PR numbers and titles.",
    )


def fmt_pull_detail(p: Dict[str, Any]) -> str:
    return (
        f"GitHub PR detail. Quote the title exactly when referencing.\n\n"
        f"PR #{p.get('number')} **{p.get('title')}** — {p.get('state')}\n"
        f"{p.get('html_url')}\n{p.get('body', '')[:1500] or ''}"
    )


def fmt_pull_diff(recs: List[Dict[str, Any]]) -> str:
    lines = ["GitHub PR diff. Reference filenames exactly as shown."]
    for f in recs[:30]:
        patch = f.get("patch") or ""
        pprev = patch[:1200] + ("…" if len(patch) > 1200 else "")
        lines.append(
            f"\n### {f.get('filename')} (+{f.get('additions', 0)}/-{f.get('deletions', 0)})\n{pprev or '(binary or large diff omitted)'}"
        )
    return "\n".join(lines) if len(lines) > 1 else "No files or empty diff."


def fmt_reviews(recs: List[Dict[str, Any]]) -> str:
    return _preview_list(
        recs, "reviews",
        lambda r, _: f"- {r.get('state', r.get('user', {}).get('login'))}: {r.get('body', '')[:150]}",
        grounding="GitHub PR reviews.",
    )


def fmt_pr_comments(recs: List[Dict[str, Any]]) -> str:
    return _preview_list(
        recs, "PR comments",
        lambda c, _: f"- {c.get('path')}:{c.get('line') or c.get('original_line')} @{c.get('user', {}).get('login')}: {(c.get('body') or '')[:120]}",
        grounding="GitHub PR line-level review comments.",
    )


def fmt_commits(recs: List[Dict[str, Any]], owner: str, repo: str) -> str:
    return _preview_list(
        recs, "commits",
        lambda c, _: f"- `{c.get('sha', '')[:7]}` {c.get('commit', {}).get('message', '').split(chr(10))[0][:100]}",
        grounding=f"GitHub commits in {owner}/{repo}. Use the short SHA when referencing.",
    )


def fmt_commit_detail(c: Dict[str, Any]) -> str:
    files = c.get("files") or []
    return (
        f"GitHub commit detail.\n\n"
        f"`{c.get('sha', '')[:7]}` {c.get('commit', {}).get('message', '')[:500]}\n"
        f"Files ({len(files)}): " + ", ".join(f.get("filename", "") for f in files[:20])
    )


def fmt_compare(data: Dict[str, Any], basehead: str) -> str:
    files = data.get("files") or []
    return (
        f"GitHub comparison. Reference filenames exactly.\n\n"
        f"Compare {basehead}: {data.get('ahead_by', '?')} ahead, {data.get('behind_by', '?')} behind\n"
        f"Files: {len(files)}\n"
        + "\n".join(f"- {f.get('filename')} +{f.get('additions')}/-{f.get('deletions')}" for f in files[:25])
    )


def fmt_branches(recs: List[Dict[str, Any]], owner: str, repo: str) -> str:
    return _preview_list(
        recs, "branches",
        lambda b, _: f"- {b.get('name')}",
        grounding=f"GitHub branches in {owner}/{repo}. Use exact branch names.",
    )


def fmt_search_code(recs: List[Dict[str, Any]], query: str) -> str:
    return _preview_list(
        recs, "code results",
        lambda it, _: f"- `{it.get('path')}` in {it.get('repository', {}).get('full_name', '?')}",
        grounding=f'GitHub code search results for "{query}".',
    )


def fmt_mutation(operation: str, success: bool, url: str, error: str) -> str:
    if not success:
        return f"GitHub {operation} failed: {error or 'unknown error'}"
    parts = [f"GitHub {operation} succeeded."]
    if url:
        parts.append(f"  url: {url}")
    return "\n".join(parts)
