"""
System GitHub REST connector definition for ExecuteGitHubEndpoint.

Uses oauth_connection auth; the backend passes oauth_token from external_connections.
"""

from typing import Any, Dict

GITHUB_SYSTEM_CONNECTOR_DEFINITION: Dict[str, Any] = {
    "base_url": "https://api.github.com",
    "connector_type": "rest",
    "auth": {"type": "oauth_connection"},
    "headers": {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    },
    "endpoints": {
        "user_repos": {
            "path": "/user/repos",
            "method": "GET",
            "params": [
                {"name": "visibility", "in": "query", "default": "all"},
                {"name": "sort", "in": "query", "default": "updated"},
                {"name": "per_page", "in": "query", "default": 30},
                {"name": "page", "in": "query", "default": 1},
            ],
            "response_list_path": ".",
            "description": "List repositories for the authenticated user",
        },
        "repo_info": {
            "path": "/repos/{owner}/{repo}",
            "method": "GET",
            "params": [
                {"name": "owner", "in": "path", "required": True},
                {"name": "repo", "in": "path", "required": True},
            ],
            "response_list_path": ".",
            "description": "Repository metadata",
        },
        "repo_issues": {
            "path": "/repos/{owner}/{repo}/issues",
            "method": "GET",
            "params": [
                {"name": "owner", "in": "path", "required": True},
                {"name": "repo", "in": "path", "required": True},
                {"name": "state", "in": "query", "default": "open"},
                {"name": "per_page", "in": "query", "default": 30},
                {"name": "page", "in": "query", "default": 1},
            ],
            "response_list_path": ".",
            "description": "List issues",
        },
        "issue_detail": {
            "path": "/repos/{owner}/{repo}/issues/{issue_number}",
            "method": "GET",
            "params": [
                {"name": "owner", "in": "path", "required": True},
                {"name": "repo", "in": "path", "required": True},
                {"name": "issue_number", "in": "path", "required": True},
            ],
            "response_list_path": ".",
            "description": "Single issue",
        },
        "issue_comments": {
            "path": "/repos/{owner}/{repo}/issues/{issue_number}/comments",
            "method": "GET",
            "params": [
                {"name": "owner", "in": "path", "required": True},
                {"name": "repo", "in": "path", "required": True},
                {"name": "issue_number", "in": "path", "required": True},
                {"name": "per_page", "in": "query", "default": 30},
            ],
            "response_list_path": ".",
            "description": "Issue comments",
        },
        "repo_pulls": {
            "path": "/repos/{owner}/{repo}/pulls",
            "method": "GET",
            "params": [
                {"name": "owner", "in": "path", "required": True},
                {"name": "repo", "in": "path", "required": True},
                {"name": "state", "in": "query", "default": "open"},
                {"name": "per_page", "in": "query", "default": 30},
            ],
            "response_list_path": ".",
            "description": "List pull requests",
        },
        "pull_detail": {
            "path": "/repos/{owner}/{repo}/pulls/{pull_number}",
            "method": "GET",
            "params": [
                {"name": "owner", "in": "path", "required": True},
                {"name": "repo", "in": "path", "required": True},
                {"name": "pull_number", "in": "path", "required": True},
            ],
            "response_list_path": ".",
            "description": "Pull request detail",
        },
        "pull_files": {
            "path": "/repos/{owner}/{repo}/pulls/{pull_number}/files",
            "method": "GET",
            "params": [
                {"name": "owner", "in": "path", "required": True},
                {"name": "repo", "in": "path", "required": True},
                {"name": "pull_number", "in": "path", "required": True},
                {"name": "per_page", "in": "query", "default": 100},
            ],
            "response_list_path": ".",
            "description": "PR changed files with patches",
        },
        "pull_reviews": {
            "path": "/repos/{owner}/{repo}/pulls/{pull_number}/reviews",
            "method": "GET",
            "params": [
                {"name": "owner", "in": "path", "required": True},
                {"name": "repo", "in": "path", "required": True},
                {"name": "pull_number", "in": "path", "required": True},
            ],
            "response_list_path": ".",
            "description": "PR reviews",
        },
        "pull_comments": {
            "path": "/repos/{owner}/{repo}/pulls/{pull_number}/comments",
            "method": "GET",
            "params": [
                {"name": "owner", "in": "path", "required": True},
                {"name": "repo", "in": "path", "required": True},
                {"name": "pull_number", "in": "path", "required": True},
                {"name": "per_page", "in": "query", "default": 100},
            ],
            "response_list_path": ".",
            "description": "PR review line comments",
        },
        "repo_commits": {
            "path": "/repos/{owner}/{repo}/commits",
            "method": "GET",
            "params": [
                {"name": "owner", "in": "path", "required": True},
                {"name": "repo", "in": "path", "required": True},
                {"name": "sha", "in": "query"},
                {"name": "path", "in": "query"},
                {"name": "author", "in": "query"},
                {"name": "since", "in": "query"},
                {"name": "until", "in": "query"},
                {"name": "per_page", "in": "query", "default": 30},
            ],
            "response_list_path": ".",
            "description": "Commit history",
        },
        "commit_detail": {
            "path": "/repos/{owner}/{repo}/commits/{sha}",
            "method": "GET",
            "params": [
                {"name": "owner", "in": "path", "required": True},
                {"name": "repo", "in": "path", "required": True},
                {"name": "sha", "in": "path", "required": True},
            ],
            "response_list_path": ".",
            "description": "Single commit",
        },
        "compare": {
            "path": "/repos/{owner}/{repo}/compare/{basehead}",
            "method": "GET",
            "params": [
                {"name": "owner", "in": "path", "required": True},
                {"name": "repo", "in": "path", "required": True},
                {"name": "basehead", "in": "path", "required": True},
            ],
            "response_list_path": ".",
            "description": "Compare two refs (basehead like main...feature)",
        },
        "file_content": {
            "path": "/repos/{owner}/{repo}/contents/{filepath}",
            "method": "GET",
            "params": [
                {"name": "owner", "in": "path", "required": True},
                {"name": "repo", "in": "path", "required": True},
                {"name": "filepath", "in": "path", "required": True},
                {"name": "ref", "in": "query"},
            ],
            "response_list_path": ".",
            "description": "File or directory contents",
        },
        "repo_branches": {
            "path": "/repos/{owner}/{repo}/branches",
            "method": "GET",
            "params": [
                {"name": "owner", "in": "path", "required": True},
                {"name": "repo", "in": "path", "required": True},
                {"name": "per_page", "in": "query", "default": 30},
            ],
            "response_list_path": ".",
            "description": "List branches",
        },
        "search_code": {
            "path": "/search/code",
            "method": "GET",
            "params": [
                {"name": "q", "in": "query", "required": True},
                {"name": "per_page", "in": "query", "default": 30},
                {"name": "page", "in": "query", "default": 1},
            ],
            "response_list_path": "items",
            "description": "Search code",
        },
        "create_issue": {
            "path": "/repos/{owner}/{repo}/issues",
            "method": "POST",
            "params": [
                {"name": "owner", "in": "path", "required": True},
                {"name": "repo", "in": "path", "required": True},
                {"name": "title", "in": "body", "required": True},
                {"name": "body", "in": "body", "default": ""},
            ],
            "response_list_path": ".",
            "description": "Create issue",
        },
        "create_issue_comment": {
            "path": "/repos/{owner}/{repo}/issues/{issue_number}/comments",
            "method": "POST",
            "params": [
                {"name": "owner", "in": "path", "required": True},
                {"name": "repo", "in": "path", "required": True},
                {"name": "issue_number", "in": "path", "required": True},
                {"name": "body", "in": "body", "required": True},
            ],
            "response_list_path": ".",
            "description": "Comment on issue or PR",
        },
        "create_pr_review": {
            "path": "/repos/{owner}/{repo}/pulls/{pull_number}/reviews",
            "method": "POST",
            "params": [
                {"name": "owner", "in": "path", "required": True},
                {"name": "repo", "in": "path", "required": True},
                {"name": "pull_number", "in": "path", "required": True},
                {"name": "event", "in": "body", "required": True},
                {"name": "body", "in": "body", "default": ""},
            ],
            "response_list_path": ".",
            "description": "Submit PR review",
        },
    },
}
