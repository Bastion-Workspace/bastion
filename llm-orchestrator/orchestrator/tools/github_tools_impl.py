"""GitHub tool function implementations."""

from typing import Any, Dict
from urllib.parse import quote

from orchestrator.tools.github_tools_common import (
    fmt_branches,
    fmt_comments,
    fmt_commit_detail,
    fmt_commits,
    fmt_compare,
    fmt_issue_detail,
    fmt_issues,
    fmt_mutation,
    fmt_pr_comments,
    fmt_pull_detail,
    fmt_pull_diff,
    fmt_pulls,
    fmt_repos,
    fmt_repo_detail,
    fmt_reviews,
    fmt_search_code,
    github_execute,
)


async def github_list_repos(
    connection_id: int = 0,
    user_id: str = "system",
    visibility: str = "all",
    sort: str = "updated",
    per_page: int = 30,
) -> Dict[str, Any]:
    """List repositories for the authenticated GitHub user."""
    out = await github_execute(
        user_id, connection_id, "user_repos",
        {"visibility": visibility, "sort": sort, "per_page": min(per_page, 100)},
    )
    recs = out.get("records") or []
    err = out.get("error")
    body = fmt_repos(recs) if recs else (err or "No repositories found.")
    return {
        "repos": recs,
        "count": len(recs),
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def github_get_repo(
    owner: str,
    repo: str,
    connection_id: int = 0,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Get repository metadata."""
    out = await github_execute(user_id, connection_id, "repo_info", {"owner": owner, "repo": repo})
    recs = out.get("records") or []
    r = recs[0] if recs else {}
    err = out.get("error")
    body = fmt_repo_detail(r) if r else (err or "Repository not found.")
    return {
        "repo": r,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def github_list_issues(
    owner: str,
    repo: str,
    connection_id: int = 0,
    user_id: str = "system",
    state: str = "open",
    per_page: int = 30,
) -> Dict[str, Any]:
    """List issues in a repository (excludes pull requests)."""
    out = await github_execute(
        user_id, connection_id, "repo_issues",
        {"owner": owner, "repo": repo, "state": state, "per_page": min(per_page, 100)},
    )
    recs = [x for x in (out.get("records") or []) if not x.get("pull_request")]
    err = out.get("error")
    body = fmt_issues(recs, owner, repo)
    return {
        "issues": recs,
        "count": len(recs),
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def github_get_issue(
    owner: str,
    repo: str,
    issue_number: int,
    connection_id: int = 0,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Get a single issue by number."""
    out = await github_execute(
        user_id, connection_id, "issue_detail",
        {"owner": owner, "repo": repo, "issue_number": issue_number},
    )
    recs = out.get("records") or []
    i = recs[0] if recs else {}
    err = out.get("error")
    body = fmt_issue_detail(i) if i else (err or "Issue not found.")
    return {
        "issue": i,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def github_list_issue_comments(
    owner: str,
    repo: str,
    issue_number: int,
    connection_id: int = 0,
    user_id: str = "system",
    per_page: int = 50,
) -> Dict[str, Any]:
    """List comments on an issue or pull request."""
    out = await github_execute(
        user_id, connection_id, "issue_comments",
        {"owner": owner, "repo": repo, "issue_number": issue_number, "per_page": min(per_page, 100)},
    )
    recs = out.get("records") or []
    err = out.get("error")
    body = fmt_comments(recs, f"{owner}/{repo}#{issue_number}")
    return {
        "comments": recs,
        "count": len(recs),
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def github_list_pulls(
    owner: str,
    repo: str,
    connection_id: int = 0,
    user_id: str = "system",
    state: str = "open",
    per_page: int = 30,
) -> Dict[str, Any]:
    """List pull requests."""
    out = await github_execute(
        user_id, connection_id, "repo_pulls",
        {"owner": owner, "repo": repo, "state": state, "per_page": min(per_page, 100)},
    )
    recs = out.get("records") or []
    err = out.get("error")
    body = fmt_pulls(recs, owner, repo)
    return {
        "pull_requests": recs,
        "count": len(recs),
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def github_get_pull(
    owner: str,
    repo: str,
    pull_number: int,
    connection_id: int = 0,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Get pull request details."""
    out = await github_execute(
        user_id, connection_id, "pull_detail",
        {"owner": owner, "repo": repo, "pull_number": pull_number},
    )
    recs = out.get("records") or []
    p = recs[0] if recs else {}
    err = out.get("error")
    body = fmt_pull_detail(p) if p else (err or "PR not found.")
    return {
        "pull_request": p,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def github_get_pull_diff(
    owner: str,
    repo: str,
    pull_number: int,
    connection_id: int = 0,
    user_id: str = "system",
) -> Dict[str, Any]:
    """List files changed in a PR with patches (per-file unified diff snippets)."""
    out = await github_execute(
        user_id, connection_id, "pull_files",
        {"owner": owner, "repo": repo, "pull_number": pull_number, "per_page": 100},
        max_pages=5,
    )
    recs = out.get("records") or []
    err = out.get("error")
    body = fmt_pull_diff(recs)
    return {
        "files": recs,
        "count": len(recs),
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def github_list_pull_reviews(
    owner: str,
    repo: str,
    pull_number: int,
    connection_id: int = 0,
    user_id: str = "system",
) -> Dict[str, Any]:
    """List reviews on a pull request."""
    out = await github_execute(
        user_id, connection_id, "pull_reviews",
        {"owner": owner, "repo": repo, "pull_number": pull_number},
    )
    recs = out.get("records") or []
    err = out.get("error")
    body = fmt_reviews(recs)
    return {
        "reviews": recs,
        "count": len(recs),
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def github_list_pull_comments(
    owner: str,
    repo: str,
    pull_number: int,
    connection_id: int = 0,
    user_id: str = "system",
) -> Dict[str, Any]:
    """List review comments on a pull request (line-level)."""
    out = await github_execute(
        user_id, connection_id, "pull_comments",
        {"owner": owner, "repo": repo, "pull_number": pull_number, "per_page": 100},
        max_pages=3,
    )
    recs = out.get("records") or []
    err = out.get("error")
    body = fmt_pr_comments(recs)
    return {
        "comments": recs,
        "count": len(recs),
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def github_list_commits(
    owner: str,
    repo: str,
    connection_id: int = 0,
    user_id: str = "system",
    sha: str = "",
    path: str = "",
    author: str = "",
    since: str = "",
    until: str = "",
    per_page: int = 30,
) -> Dict[str, Any]:
    """List commits on a repository."""
    params: Dict[str, Any] = {"owner": owner, "repo": repo, "per_page": min(per_page, 100)}
    if sha:
        params["sha"] = sha
    if path:
        params["path"] = path
    if author:
        params["author"] = author
    if since:
        params["since"] = since
    if until:
        params["until"] = until
    out = await github_execute(user_id, connection_id, "repo_commits", params, max_pages=5)
    recs = out.get("records") or []
    err = out.get("error")
    body = fmt_commits(recs, owner, repo)
    return {
        "commits": recs,
        "count": len(recs),
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def github_get_commit(
    owner: str,
    repo: str,
    sha: str,
    connection_id: int = 0,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Get a single commit with file list."""
    out = await github_execute(
        user_id, connection_id, "commit_detail",
        {"owner": owner, "repo": repo, "sha": sha},
    )
    recs = out.get("records") or []
    c = recs[0] if recs else {}
    err = out.get("error")
    body = fmt_commit_detail(c) if c else (err or "Commit not found.")
    return {
        "commit": c,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def github_compare_refs(
    owner: str,
    repo: str,
    base_ref: str,
    head_ref: str,
    connection_id: int = 0,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Compare two branches, tags, or SHAs (base...head)."""
    basehead = f"{base_ref}...{head_ref}"
    out = await github_execute(
        user_id, connection_id, "compare",
        {"owner": owner, "repo": repo, "basehead": basehead},
    )
    recs = out.get("records") or []
    data = recs[0] if recs else {}
    err = out.get("error")
    files = data.get("files") or [] if data else []
    body = fmt_compare(data, basehead) if data else (err or "Comparison not found.")
    return {
        "comparison": data,
        "files": files,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def github_get_file_content(
    owner: str,
    repo: str,
    path: str,
    connection_id: int = 0,
    user_id: str = "system",
    ref: str = "",
) -> Dict[str, Any]:
    """Get file contents from a repository (decodes base64 for files)."""
    filepath = quote(path.strip().lstrip("/"), safe="/")
    params: Dict[str, Any] = {"owner": owner, "repo": repo, "filepath": filepath}
    if ref:
        params["ref"] = ref
    out = await github_execute(user_id, connection_id, "file_content", params)
    recs = out.get("records") or []
    err = out.get("error")
    content = ""

    if len(recs) == 1 and isinstance(recs[0], dict) and recs[0].get("type") == "file" and recs[0].get("content"):
        import base64
        data = recs[0]
        raw = base64.b64decode(data["content"].replace("\n", "")).decode("utf-8", errors="replace")
        content = raw
        preview = raw[:8000]
        body = f"GitHub file content.\n\n`{path}` ({len(raw)} chars)\n```\n{preview}\n```"
    elif recs and all(isinstance(x, dict) and x.get("name") for x in recs):
        body = "GitHub directory listing.\n\n" + "\n".join(
            f"- {x.get('name')} ({x.get('type', 'item')})" for x in recs[:100]
        )
    else:
        body = err or "File not found."

    return {
        "content": content,
        "records": recs,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def github_list_branches(
    owner: str,
    repo: str,
    connection_id: int = 0,
    user_id: str = "system",
    per_page: int = 50,
) -> Dict[str, Any]:
    """List branches in a repository."""
    out = await github_execute(
        user_id, connection_id, "repo_branches",
        {"owner": owner, "repo": repo, "per_page": min(per_page, 100)},
        max_pages=5,
    )
    recs = out.get("records") or []
    err = out.get("error")
    body = fmt_branches(recs, owner, repo)
    return {
        "branches": recs,
        "count": len(recs),
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def github_search_code(
    q: str,
    connection_id: int = 0,
    user_id: str = "system",
    per_page: int = 30,
) -> Dict[str, Any]:
    """Search code across GitHub (uses GitHub search query syntax)."""
    out = await github_execute(
        user_id, connection_id, "search_code",
        {"q": q, "per_page": min(per_page, 100)},
        max_pages=3,
    )
    recs = out.get("records") or []
    err = out.get("error")
    body = fmt_search_code(recs, q)
    return {
        "results": recs,
        "count": len(recs),
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def github_create_issue(
    owner: str,
    repo: str,
    title: str,
    connection_id: int = 0,
    user_id: str = "system",
    body: str = "",
) -> Dict[str, Any]:
    """Create an issue in a repository."""
    out = await github_execute(
        user_id, connection_id, "create_issue",
        {"owner": owner, "repo": repo, "title": title, "body": body or ""},
    )
    recs = out.get("records") or []
    i = recs[0] if recs else {}
    ok = bool(i.get("number"))
    url = i.get("html_url") or ""
    err = out.get("error") or ""
    if ok:
        fmt = f'GitHub create_issue succeeded.\n  issue: #{i.get("number")} "{title}"\n  url: {url}'
    else:
        fmt = fmt_mutation("create_issue", False, "", err)
    return {"success": ok, "url": url, "formatted": fmt}


async def github_create_issue_comment(
    owner: str,
    repo: str,
    issue_number: int,
    body: str,
    connection_id: int = 0,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Add a comment to an issue or pull request."""
    out = await github_execute(
        user_id, connection_id, "create_issue_comment",
        {"owner": owner, "repo": repo, "issue_number": issue_number, "body": body},
    )
    recs = out.get("records") or []
    c = recs[0] if recs else {}
    ok = bool(c.get("id"))
    url = c.get("html_url") or ""
    err = out.get("error") or ""
    return {
        "success": ok,
        "url": url,
        "formatted": fmt_mutation("create_comment", ok, url, err),
    }


async def github_create_pr_review(
    owner: str,
    repo: str,
    pull_number: int,
    event: str,
    connection_id: int = 0,
    user_id: str = "system",
    body: str = "",
) -> Dict[str, Any]:
    """Submit a PR review. event: COMMENT, APPROVE, or REQUEST_CHANGES."""
    out = await github_execute(
        user_id, connection_id, "create_pr_review",
        {"owner": owner, "repo": repo, "pull_number": pull_number, "event": event.upper(), "body": body or ""},
    )
    recs = out.get("records") or []
    r = recs[0] if recs else {}
    ok = bool(r.get("id"))
    url = r.get("html_url") or ""
    err = out.get("error") or ""
    if ok:
        fmt = f"GitHub PR review submitted: {r.get('state', event)}\n  url: {url}"
    else:
        fmt = fmt_mutation("create_pr_review", False, "", err)
    return {"success": ok, "url": url, "formatted": fmt}
