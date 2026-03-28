"""
GitHub plugin - list repos, issues, PRs and create issues for Agent Factory (Zone 4).

Uses GitHub REST API. Requires a personal access token (or OAuth token) in connection config.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.plugins.base_plugin import BasePlugin, PluginToolSpec


class ListReposInputs(BaseModel):
    """Inputs for listing GitHub repos."""

    visibility: str = Field(default="all", description="all, public, or private")
    sort: str = Field(default="updated", description="created, updated, pushed, full_name")
    limit: int = Field(default=20, description="Max repos to return")


class RepoRef(BaseModel):
    """Reference to a GitHub repo."""

    id: str = Field(description="Repo ID")
    name: str = Field(description="Repo name")
    full_name: str = Field(description="owner/name")
    url: str = Field(description="HTML URL")
    private: bool = Field(default=False)
    description: Optional[str] = Field(default=None)


class ListReposOutputs(BaseModel):
    """Outputs for list GitHub repos tool."""

    repos: List[RepoRef] = Field(description="List of repos")
    count: int = Field(description="Number of repos")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ListIssuesInputs(BaseModel):
    """Inputs for listing GitHub issues."""

    owner: str = Field(description="Repo owner")
    repo: str = Field(description="Repo name")
    state: str = Field(default="open", description="open, closed, or all")
    limit: int = Field(default=20, description="Max issues to return")


class IssueRef(BaseModel):
    """Reference to a GitHub issue."""

    number: int = Field(description="Issue number")
    title: str = Field(description="Title")
    state: str = Field(description="open or closed")
    url: str = Field(description="HTML URL")
    body_preview: Optional[str] = Field(default=None, description="First 200 chars of body")


class ListIssuesOutputs(BaseModel):
    """Outputs for list GitHub issues tool."""

    issues: List[IssueRef] = Field(description="List of issues")
    count: int = Field(description="Number of issues")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class CreateIssueInputs(BaseModel):
    """Inputs for creating a GitHub issue."""

    owner: str = Field(description="Repo owner")
    repo: str = Field(description="Repo name")
    title: str = Field(description="Issue title")
    body: str = Field(default="", description="Issue body (markdown)")


class CreateIssueOutputs(BaseModel):
    """Outputs for create GitHub issue tool."""

    issue_number: int = Field(description="Created issue number")
    issue_url: str = Field(description="Issue URL")
    success: bool = Field(description="Whether creation succeeded")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ListPullRequestsInputs(BaseModel):
    """Inputs for listing GitHub pull requests."""

    owner: str = Field(description="Repo owner")
    repo: str = Field(description="Repo name")
    state: str = Field(default="open", description="open, closed, or all")
    limit: int = Field(default=20, description="Max PRs to return")


class PullRequestRef(BaseModel):
    """Reference to a GitHub pull request."""

    number: int = Field(description="PR number")
    title: str = Field(description="Title")
    state: str = Field(description="open or closed")
    url: str = Field(description="HTML URL")
    head_ref: str = Field(description="Head branch")
    base_ref: str = Field(description="Base branch")


class ListPullRequestsOutputs(BaseModel):
    """Outputs for list GitHub pull requests tool."""

    pull_requests: List[PullRequestRef] = Field(description="List of PRs")
    count: int = Field(description="Number of PRs")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class GitHubPlugin(BasePlugin):
    """GitHub integration - list repos, issues, PRs; create issues."""

    @property
    def plugin_name(self) -> str:
        return "github"

    @property
    def plugin_version(self) -> str:
        return "0.1.0"

    def get_connection_requirements(self) -> Dict[str, str]:
        return {
            "token": "GitHub Personal Access Token (or OAuth token)",
        }

    def get_tools(self) -> List[PluginToolSpec]:
        return [
            PluginToolSpec(
                name="github_list_repos",
                category="plugin:github",
                description="List GitHub repositories for the authenticated user",
                inputs_model=ListReposInputs,
                outputs_model=ListReposOutputs,
                tool_function=self._list_repos,
            ),
            PluginToolSpec(
                name="github_list_issues",
                category="plugin:github",
                description="List issues for a GitHub repository",
                inputs_model=ListIssuesInputs,
                outputs_model=ListIssuesOutputs,
                tool_function=self._list_issues,
            ),
            PluginToolSpec(
                name="github_list_pull_requests",
                category="plugin:github",
                description="List pull requests for a GitHub repository",
                inputs_model=ListPullRequestsInputs,
                outputs_model=ListPullRequestsOutputs,
                tool_function=self._list_pull_requests,
            ),
            PluginToolSpec(
                name="github_create_issue",
                category="plugin:github",
                description="Create an issue in a GitHub repository",
                inputs_model=CreateIssueInputs,
                outputs_model=CreateIssueOutputs,
                tool_function=self._create_issue,
            ),
        ]

    def _headers(self) -> Dict[str, str]:
        config = getattr(self, "_config", None) or {}
        token = config.get("token", "")
        return {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {token}" if token else "",
        }

    async def _list_repos(
        self,
        visibility: str = "all",
        sort: str = "updated",
        limit: int = 20,
    ) -> Dict[str, Any]:
        """List GitHub repos for the authenticated user."""
        config = getattr(self, "_config", None) or {}
        if not config.get("token"):
            return {"repos": [], "count": 0, "formatted": "GitHub plugin: configure token to list repos."}
        try:
            import aiohttp
            url = "https://api.github.com/user/repos"
            params = {"visibility": visibility, "sort": sort, "per_page": min(limit, 100)}
            out = []
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self._headers()) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return {"repos": [], "count": 0, "formatted": f"GitHub API error ({resp.status}): {text[:200]}"}
                    data = await resp.json()
            for r in data[:limit]:
                out.append(RepoRef(
                    id=str(r.get("id", "")),
                    name=r.get("name", ""),
                    full_name=r.get("full_name", ""),
                    url=r.get("html_url", ""),
                    private=r.get("private", False),
                    description=r.get("description"),
                ))
            formatted = f"Found {len(out)} repo(s): " + ", ".join(r.full_name for r in out) if out else "No repos found."
            return {"repos": [r.model_dump() for r in out], "count": len(out), "formatted": formatted}
        except ImportError:
            return {"repos": [], "count": 0, "formatted": "GitHub plugin: aiohttp not installed."}
        except Exception as e:
            return {"repos": [], "count": 0, "formatted": f"GitHub list repos failed: {e}"}

    async def _list_issues(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        limit: int = 20,
    ) -> Dict[str, Any]:
        """List issues for a repo."""
        if not getattr(self, "_config", None) or not (getattr(self, "_config", None) or {}).get("token"):
            return {"issues": [], "count": 0, "formatted": "GitHub plugin: configure token."}
        try:
            import aiohttp
            url = f"https://api.github.com/repos/{owner}/{repo}/issues"
            params = {"state": state, "per_page": min(limit, 100)}
            out = []
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self._headers()) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return {"issues": [], "count": 0, "formatted": f"GitHub API error ({resp.status}): {text[:200]}"}
                    data = await resp.json()
            for i in data:
                if i.get("pull_request"):
                    continue
                body = i.get("body") or ""
                out.append(IssueRef(
                    number=i.get("number", 0),
                    title=i.get("title", ""),
                    state=i.get("state", "open"),
                    url=i.get("html_url", ""),
                    body_preview=body[:200] if body else None,
                ))
                if len(out) >= limit:
                    break
            formatted = f"Found {len(out)} issue(s) in {owner}/{repo}." if out else "No issues found."
            return {"issues": [i.model_dump() for i in out], "count": len(out), "formatted": formatted}
        except ImportError:
            return {"issues": [], "count": 0, "formatted": "GitHub plugin: aiohttp not installed."}
        except Exception as e:
            return {"issues": [], "count": 0, "formatted": f"GitHub list issues failed: {e}"}

    async def _list_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        limit: int = 20,
    ) -> Dict[str, Any]:
        """List pull requests for a repo."""
        if not getattr(self, "_config", None) or not (getattr(self, "_config", None) or {}).get("token"):
            return {"pull_requests": [], "count": 0, "formatted": "GitHub plugin: configure token."}
        try:
            import aiohttp
            url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
            params = {"state": state, "per_page": min(limit, 100)}
            out = []
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self._headers()) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return {"pull_requests": [], "count": 0, "formatted": f"GitHub API error ({resp.status}): {text[:200]}"}
                    data = await resp.json()
            for p in data[:limit]:
                out.append(PullRequestRef(
                    number=p.get("number", 0),
                    title=p.get("title", ""),
                    state=p.get("state", "open"),
                    url=p.get("html_url", ""),
                    head_ref=p.get("head", {}).get("ref", ""),
                    base_ref=p.get("base", {}).get("ref", ""),
                ))
            formatted = f"Found {len(out)} PR(s) in {owner}/{repo}." if out else "No pull requests found."
            return {"pull_requests": [p.model_dump() for p in out], "count": len(out), "formatted": formatted}
        except ImportError:
            return {"pull_requests": [], "count": 0, "formatted": "GitHub plugin: aiohttp not installed."}
        except Exception as e:
            return {"pull_requests": [], "count": 0, "formatted": f"GitHub list pull requests failed: {e}"}

    async def _create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str = "",
    ) -> Dict[str, Any]:
        """Create an issue in a repo."""
        if not getattr(self, "_config", None) or not (getattr(self, "_config", None) or {}).get("token"):
            return {"issue_number": 0, "issue_url": "", "success": False, "formatted": "GitHub plugin: configure token."}
        try:
            import aiohttp
            url = f"https://api.github.com/repos/{owner}/{repo}/issues"
            payload = {"title": title, "body": body}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=self._headers()) as resp:
                    if resp.status not in (200, 201):
                        text = await resp.text()
                        return {"issue_number": 0, "issue_url": "", "success": False, "formatted": f"GitHub API error ({resp.status}): {text[:200]}"}
                    data = await resp.json()
            return {
                "issue_number": data.get("number", 0),
                "issue_url": data.get("html_url", ""),
                "success": True,
                "formatted": f"Created issue #{data.get('number')}: {title} ({data.get('html_url', '')})",
            }
        except ImportError:
            return {"issue_number": 0, "issue_url": "", "success": False, "formatted": "GitHub plugin: aiohttp not installed."}
        except Exception as e:
            return {"issue_number": 0, "issue_url": "", "success": False, "formatted": f"GitHub create issue failed: {e}"}
