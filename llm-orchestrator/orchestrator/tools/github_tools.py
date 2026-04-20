"""
GitHub REST tools — Pydantic models and action registry.

Tool implementations live in github_tools_impl.py.
Use with github:{connection_id}:{tool_name} or gitea:{connection_id}:{tool_name}.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from orchestrator.tools.github_tools_impl import (
    github_compare_refs,
    github_create_issue,
    github_create_issue_comment,
    github_create_pr_review,
    github_get_commit,
    github_get_file_content,
    github_get_issue,
    github_get_pull,
    github_get_pull_diff,
    github_get_repo,
    github_list_branches,
    github_list_commits,
    github_list_issue_comments,
    github_list_pull_comments,
    github_list_pull_reviews,
    github_list_pulls,
    github_list_repos,
    github_list_issues,
    github_search_code,
)
from orchestrator.utils.action_io_registry import register_action

# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class GHListReposIn(BaseModel):
    connection_id: int = Field(default=0, description="GitHub connection id (auto-injected from tool pack binding when 0)")
    visibility: str = Field(default="all", description="Filter: all, public, or private")
    sort: str = Field(default="updated", description="Sort by: created, updated, pushed, full_name")
    per_page: int = Field(default=30, ge=1, le=100, description="Maximum results per page (1-100)")


class GHOwnerRepoIn(BaseModel):
    owner: str = Field(..., description="Repository owner (e.g. 'microsoft' from 'microsoft/vscode')")
    repo: str = Field(..., description="Repository name (e.g. 'vscode' from 'microsoft/vscode')")
    connection_id: int = Field(default=0, description="GitHub connection id (auto-injected from tool pack binding when 0)")


class GHListIssuesIn(GHOwnerRepoIn):
    state: str = Field(default="open", description="Filter: open, closed, or all")
    per_page: int = Field(default=30, ge=1, le=100, description="Maximum results per page (1-100)")


class GHGetIssueIn(GHOwnerRepoIn):
    issue_number: int = Field(..., description="Issue number from github_list_issues (the # number, not the title)")


class GHIssueCommentsIn(GHOwnerRepoIn):
    issue_number: int = Field(..., description="Issue or PR number whose comments to list")
    per_page: int = Field(default=50, ge=1, le=100, description="Maximum results per page (1-100)")


class GHListPullsIn(GHOwnerRepoIn):
    state: str = Field(default="open", description="Filter: open, closed, or all")
    per_page: int = Field(default=30, ge=1, le=100, description="Maximum results per page (1-100)")


class GHGetPullIn(GHOwnerRepoIn):
    pull_number: int = Field(..., description="PR number from github_list_pulls (the # number, not the title)")


class GHListCommitsIn(GHOwnerRepoIn):
    sha: str = Field(default="", description="Branch name or commit SHA to list from (default: repo default branch)")
    path: str = Field(default="", description="Only commits touching this file path")
    author: str = Field(default="", description="Only commits from this author (GitHub username or email)")
    since: str = Field(default="", description="ISO 8601 date — only commits after this date")
    until: str = Field(default="", description="ISO 8601 date — only commits before this date")
    per_page: int = Field(default=30, ge=1, le=100, description="Maximum results per page (1-100)")


class GHGetCommitIn(GHOwnerRepoIn):
    sha: str = Field(..., description="Full or abbreviated commit SHA from github_list_commits")


class GHCompareIn(GHOwnerRepoIn):
    base_ref: str = Field(..., description="Base branch, tag, or SHA for the comparison")
    head_ref: str = Field(..., description="Head branch, tag, or SHA for the comparison")


class GHFilePathIn(GHOwnerRepoIn):
    path: str = Field(..., description="File or directory path within the repo (e.g. 'src/main.py')")
    ref: str = Field(default="", description="Branch name, tag, or SHA (default: repo default branch)")


class GHBranchesIn(GHOwnerRepoIn):
    per_page: int = Field(default=50, ge=1, le=100, description="Maximum results per page (1-100)")


class GHSearchCodeIn(BaseModel):
    q: str = Field(..., description="GitHub code search query (e.g. 'addClass repo:jquery/jquery')")
    connection_id: int = Field(default=0, description="GitHub connection id (auto-injected from tool pack binding when 0)")
    per_page: int = Field(default=30, ge=1, le=100, description="Maximum results per page (1-100)")


class GHCreateIssueIn(GHOwnerRepoIn):
    title: str = Field(..., description="Title for the new issue")
    body: str = Field(default="", description="Markdown body for the new issue")


class GHCreateCommentIn(GHOwnerRepoIn):
    issue_number: int = Field(..., description="Issue or PR number to comment on")
    body: str = Field(..., description="Markdown comment body")


class GHCreateReviewIn(GHOwnerRepoIn):
    pull_number: int = Field(..., description="PR number to submit review on")
    event: str = Field(..., description="Review action: COMMENT, APPROVE, or REQUEST_CHANGES")
    body: str = Field(default="", description="Review body text (required for REQUEST_CHANGES)")


# ---------------------------------------------------------------------------
# Output models — one per tool pattern
# ---------------------------------------------------------------------------

class GHRepoListOut(BaseModel):
    repos: List[Dict[str, Any]] = Field(default_factory=list, description="Repository objects")
    count: int = Field(default=0, description="Number of repositories returned")
    formatted: str = Field(default="", description="Human-readable summary for LLM/chat display")


class GHRepoDetailOut(BaseModel):
    repo: Dict[str, Any] = Field(default_factory=dict, description="Repository metadata object")
    formatted: str = Field(default="", description="Human-readable repo detail")


class GHIssueListOut(BaseModel):
    issues: List[Dict[str, Any]] = Field(default_factory=list, description="Issue objects")
    count: int = Field(default=0, description="Number of issues returned")
    formatted: str = Field(default="", description="Human-readable issue list")


class GHIssueDetailOut(BaseModel):
    issue: Dict[str, Any] = Field(default_factory=dict, description="Issue detail object")
    formatted: str = Field(default="", description="Human-readable issue detail")


class GHCommentListOut(BaseModel):
    comments: List[Dict[str, Any]] = Field(default_factory=list, description="Comment objects")
    count: int = Field(default=0, description="Number of comments returned")
    formatted: str = Field(default="", description="Human-readable comment list")


class GHPullListOut(BaseModel):
    pull_requests: List[Dict[str, Any]] = Field(default_factory=list, description="Pull request objects")
    count: int = Field(default=0, description="Number of PRs returned")
    formatted: str = Field(default="", description="Human-readable PR list")


class GHPullDetailOut(BaseModel):
    pull_request: Dict[str, Any] = Field(default_factory=dict, description="Pull request detail object")
    formatted: str = Field(default="", description="Human-readable PR detail")


class GHPullDiffOut(BaseModel):
    files: List[Dict[str, Any]] = Field(default_factory=list, description="Changed file objects with patches")
    count: int = Field(default=0, description="Number of changed files")
    formatted: str = Field(default="", description="Human-readable diff summary")


class GHReviewListOut(BaseModel):
    reviews: List[Dict[str, Any]] = Field(default_factory=list, description="Review objects")
    count: int = Field(default=0, description="Number of reviews")
    formatted: str = Field(default="", description="Human-readable review list")


class GHCommitListOut(BaseModel):
    commits: List[Dict[str, Any]] = Field(default_factory=list, description="Commit objects")
    count: int = Field(default=0, description="Number of commits returned")
    formatted: str = Field(default="", description="Human-readable commit list")


class GHCommitDetailOut(BaseModel):
    commit: Dict[str, Any] = Field(default_factory=dict, description="Commit detail object")
    formatted: str = Field(default="", description="Human-readable commit detail")


class GHCompareOut(BaseModel):
    comparison: Dict[str, Any] = Field(default_factory=dict, description="Comparison metadata")
    files: List[Dict[str, Any]] = Field(default_factory=list, description="Changed file objects")
    formatted: str = Field(default="", description="Human-readable comparison")


class GHFileContentOut(BaseModel):
    content: str = Field(default="", description="Decoded file content (UTF-8 text)")
    records: List[Dict[str, Any]] = Field(default_factory=list, description="Raw API records (file or directory entries)")
    formatted: str = Field(default="", description="Human-readable file content or directory listing")


class GHBranchListOut(BaseModel):
    branches: List[Dict[str, Any]] = Field(default_factory=list, description="Branch objects")
    count: int = Field(default=0, description="Number of branches")
    formatted: str = Field(default="", description="Human-readable branch list")


class GHSearchCodeOut(BaseModel):
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Code search result objects")
    count: int = Field(default=0, description="Number of search results")
    formatted: str = Field(default="", description="Human-readable search results")


class GHMutationOut(BaseModel):
    success: bool = Field(default=False, description="Whether the mutation succeeded")
    url: str = Field(default="", description="URL of the created resource")
    formatted: str = Field(default="", description="Human-readable success/failure message")


# Backward-compat alias
GitHubRecordsOut = GHRepoListOut


# ---------------------------------------------------------------------------
# Action registry
# ---------------------------------------------------------------------------
_CI = "github"

register_action(
    name="github_list_repos", category=_CI,
    description="List GitHub repositories for the authenticated user",
    inputs_model=GHListReposIn, outputs_model=GHRepoListOut, tool_function=github_list_repos,
)
register_action(
    name="github_get_repo", category=_CI,
    description="Get GitHub repository metadata",
    inputs_model=GHOwnerRepoIn, outputs_model=GHRepoDetailOut, tool_function=github_get_repo,
)
register_action(
    name="github_list_issues", category=_CI,
    description="List issues in a GitHub repository",
    inputs_model=GHListIssuesIn, outputs_model=GHIssueListOut, tool_function=github_list_issues,
)
register_action(
    name="github_get_issue", category=_CI,
    description="Get one GitHub issue by number",
    inputs_model=GHGetIssueIn, outputs_model=GHIssueDetailOut, tool_function=github_get_issue,
)
register_action(
    name="github_list_issue_comments", category=_CI,
    description="List comments on an issue or PR",
    inputs_model=GHIssueCommentsIn, outputs_model=GHCommentListOut, tool_function=github_list_issue_comments,
)
register_action(
    name="github_list_pulls", category=_CI,
    description="List pull requests",
    inputs_model=GHListPullsIn, outputs_model=GHPullListOut, tool_function=github_list_pulls,
)
register_action(
    name="github_get_pull", category=_CI,
    description="Get pull request details",
    inputs_model=GHGetPullIn, outputs_model=GHPullDetailOut, tool_function=github_get_pull,
)
register_action(
    name="github_get_pull_diff", category=_CI,
    description="List files changed in a PR with patch snippets",
    inputs_model=GHGetPullIn, outputs_model=GHPullDiffOut, tool_function=github_get_pull_diff,
)
register_action(
    name="github_list_pull_reviews", category=_CI,
    description="List PR reviews",
    inputs_model=GHGetPullIn, outputs_model=GHReviewListOut, tool_function=github_list_pull_reviews,
)
register_action(
    name="github_list_pull_comments", category=_CI,
    description="List line-level PR review comments",
    inputs_model=GHGetPullIn, outputs_model=GHCommentListOut, tool_function=github_list_pull_comments,
)
register_action(
    name="github_list_commits", category=_CI,
    description="List commits on a repository",
    inputs_model=GHListCommitsIn, outputs_model=GHCommitListOut, tool_function=github_list_commits,
)
register_action(
    name="github_get_commit", category=_CI,
    description="Get one commit with changed files",
    inputs_model=GHGetCommitIn, outputs_model=GHCommitDetailOut, tool_function=github_get_commit,
)
register_action(
    name="github_compare_refs", category=_CI,
    description="Compare two refs (branches/tags/SHAs)",
    inputs_model=GHCompareIn, outputs_model=GHCompareOut, tool_function=github_compare_refs,
)
register_action(
    name="github_get_file_content", category=_CI,
    description="Read a file or list a directory from a repo",
    inputs_model=GHFilePathIn, outputs_model=GHFileContentOut, tool_function=github_get_file_content,
)
register_action(
    name="github_list_branches", category=_CI,
    description="List branches",
    inputs_model=GHBranchesIn, outputs_model=GHBranchListOut, tool_function=github_list_branches,
)
register_action(
    name="github_search_code", category=_CI,
    description="Search code (GitHub search syntax)",
    inputs_model=GHSearchCodeIn, outputs_model=GHSearchCodeOut, tool_function=github_search_code,
)
register_action(
    name="github_create_issue", category=_CI,
    description="Create a GitHub issue",
    inputs_model=GHCreateIssueIn, outputs_model=GHMutationOut, tool_function=github_create_issue,
)
register_action(
    name="github_create_issue_comment", category=_CI,
    description="Comment on an issue or PR",
    inputs_model=GHCreateCommentIn, outputs_model=GHMutationOut, tool_function=github_create_issue_comment,
)
register_action(
    name="github_create_pr_review", category=_CI,
    description="Submit a PR review (COMMENT/APPROVE/REQUEST_CHANGES)",
    inputs_model=GHCreateReviewIn, outputs_model=GHMutationOut, tool_function=github_create_pr_review,
)
