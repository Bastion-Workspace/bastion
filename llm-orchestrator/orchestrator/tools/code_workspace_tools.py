"""
Code workspace tools - higher-level coding tools built on top of the local proxy device.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


# ── code_open_workspace ─────────────────────────────────────────────────────


class CodeOpenWorkspaceInputs(BaseModel):
    workspace_root: str = Field(description="Absolute path to the project root on the device")
    device_id: Optional[str] = Field(default=None, description="Optional device id to target")
    timeout_seconds: int = Field(default=30, description="Timeout in seconds")


class CodeOpenWorkspaceOutputs(BaseModel):
    workspace_root: str = Field(description="Workspace root path set on the device")
    file_count: int = Field(description="Quick file count under workspace")
    git_detected: bool = Field(description="Whether .git was detected")
    formatted: str = Field(description="Human-readable summary")


async def code_open_workspace_tool(
    workspace_root: str,
    device_id: Optional[str] = None,
    timeout_seconds: int = 30,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Set the device-side active workspace root for relative path resolution."""
    try:
        client = await get_backend_tool_client()
        result = await client.set_device_workspace(
            user_id=user_id,
            device_id=device_id or "",
            workspace_root=workspace_root,
            timeout=timeout_seconds,
        )
        if not result.get("success"):
            err = result.get("error", "Failed to set workspace")
            return {
                "workspace_root": workspace_root,
                "file_count": 0,
                "git_detected": False,
                "formatted": err,
            }
        data = result.get("result") or {}
        return {
            "workspace_root": data.get("workspace_root", workspace_root),
            "file_count": int(data.get("file_count", 0) or 0),
            "git_detected": bool(data.get("git_detected", False)),
            "formatted": result.get("formatted", "Workspace set"),
        }
    except Exception as e:
        logger.error("code_open_workspace_tool error: %s", e)
        return {
            "workspace_root": workspace_root,
            "file_count": 0,
            "git_detected": False,
            "formatted": str(e),
        }


register_action(
    name="code_open_workspace",
    category="code_workspace",
    description="Set the device-side active workspace root for relative path resolution.",
    inputs_model=CodeOpenWorkspaceInputs,
    params_model=None,
    outputs_model=CodeOpenWorkspaceOutputs,
    tool_function=code_open_workspace_tool,
)


# ── code_file_tree ──────────────────────────────────────────────────────────


class CodeFileTreeInputs(BaseModel):
    path: str = Field(description="Directory path (absolute or relative to opened workspace)")
    max_depth: int = Field(default=10, description="Maximum recursion depth")
    ignore_patterns: Optional[List[str]] = Field(default=None, description="Directory names to ignore")
    include_hidden: bool = Field(default=False, description="Include dotfiles and dotfolders")
    device_id: Optional[str] = Field(default=None, description="Optional device id to target")
    timeout_seconds: int = Field(default=60, description="Timeout in seconds")


class CodeFileTreeOutputs(BaseModel):
    tree: List[Dict[str, Any]] = Field(description="Tree nodes")
    total_files: int = Field(description="Total files counted")
    total_dirs: int = Field(description="Total dirs counted")
    truncated: bool = Field(description="Whether results were truncated")
    formatted: str = Field(description="Human-readable summary")


async def code_file_tree_tool(
    path: str,
    max_depth: int = 10,
    ignore_patterns: Optional[List[str]] = None,
    include_hidden: bool = False,
    device_id: Optional[str] = None,
    timeout_seconds: int = 60,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Get a recursive file tree from the device."""
    try:
        client = await get_backend_tool_client()
        args: Dict[str, Any] = {
            "path": path,
            "max_depth": int(max_depth),
            "include_hidden": bool(include_hidden),
        }
        if ignore_patterns is not None:
            args["ignore_patterns"] = ignore_patterns
        result = await client.invoke_device_tool(
            user_id=user_id,
            device_id=device_id or "",
            tool="file_tree",
            args=args,
            timeout=timeout_seconds,
        )
        if not result.get("success"):
            err = result.get("error", "file_tree failed")
            return {"tree": [], "total_files": 0, "total_dirs": 0, "truncated": False, "formatted": err}
        data = result.get("result") or {}
        return {
            "tree": data.get("tree", []) or [],
            "total_files": int(data.get("total_files", 0) or 0),
            "total_dirs": int(data.get("total_dirs", 0) or 0),
            "truncated": bool(data.get("truncated", False)),
            "formatted": result.get("formatted", "File tree retrieved"),
        }
    except Exception as e:
        logger.error("code_file_tree_tool error: %s", e)
        return {"tree": [], "total_files": 0, "total_dirs": 0, "truncated": False, "formatted": str(e)}


register_action(
    name="code_file_tree",
    category="code_workspace",
    description="Get a recursive file tree from the device (local proxy).",
    inputs_model=CodeFileTreeInputs,
    params_model=None,
    outputs_model=CodeFileTreeOutputs,
    tool_function=code_file_tree_tool,
)


# ── code_search_files ───────────────────────────────────────────────────────


class CodeSearchFilesInputs(BaseModel):
    pattern: str = Field(description="Regex pattern to search for")
    path: str = Field(description="Directory root to search (absolute or relative to opened workspace)")
    glob: Optional[str] = Field(default=None, description="Optional glob filter (minimal support: '*.ext')")
    max_results: int = Field(default=100, description="Maximum results")
    context_lines: int = Field(default=2, description="Context lines before and after match")
    case_insensitive: bool = Field(default=False, description="Case-insensitive matching")
    include_hidden: bool = Field(default=False, description="Include dotfiles and dotfolders")
    device_id: Optional[str] = Field(default=None, description="Optional device id to target")
    timeout_seconds: int = Field(default=60, description="Timeout in seconds")


class CodeSearchFilesOutputs(BaseModel):
    matches: List[Dict[str, Any]] = Field(description="Match records")
    total_matches: int = Field(description="Number of matches returned")
    files_searched: int = Field(description="Number of files searched")
    truncated: bool = Field(description="Whether results were truncated")
    formatted: str = Field(description="Human-readable summary")


async def code_search_files_tool(
    pattern: str,
    path: str,
    glob: Optional[str] = None,
    max_results: int = 100,
    context_lines: int = 2,
    case_insensitive: bool = False,
    include_hidden: bool = False,
    device_id: Optional[str] = None,
    timeout_seconds: int = 60,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Search file contents on the device."""
    try:
        client = await get_backend_tool_client()
        args: Dict[str, Any] = {
            "pattern": pattern,
            "path": path,
            "max_results": int(max_results),
            "context_lines": int(context_lines),
            "case_insensitive": bool(case_insensitive),
            "include_hidden": bool(include_hidden),
        }
        if glob:
            args["glob"] = glob
        result = await client.invoke_device_tool(
            user_id=user_id,
            device_id=device_id or "",
            tool="search_files",
            args=args,
            timeout=timeout_seconds,
        )
        if not result.get("success"):
            err = result.get("error", "search_files failed")
            return {"matches": [], "total_matches": 0, "files_searched": 0, "truncated": False, "formatted": err}
        data = result.get("result") or {}
        matches = data.get("matches", []) or []
        return {
            "matches": matches,
            "total_matches": int(data.get("total_matches", len(matches)) or 0),
            "files_searched": int(data.get("files_searched", 0) or 0),
            "truncated": bool(data.get("truncated", False)),
            "formatted": result.get("formatted", "Search complete"),
        }
    except Exception as e:
        logger.error("code_search_files_tool error: %s", e)
        return {"matches": [], "total_matches": 0, "files_searched": 0, "truncated": False, "formatted": str(e)}


register_action(
    name="code_search_files",
    category="code_workspace",
    description="Search for a regex pattern in files under a directory via the local proxy.",
    inputs_model=CodeSearchFilesInputs,
    params_model=None,
    outputs_model=CodeSearchFilesOutputs,
    tool_function=code_search_files_tool,
)


# ── code_git_info ───────────────────────────────────────────────────────────


class CodeGitInfoInputs(BaseModel):
    path: str = Field(description="Repo path (absolute or relative to opened workspace)")
    operation: str = Field(description="One of: status|diff|log|branch|show")
    file: Optional[str] = Field(default=None, description="For diff: optional file path")
    limit: Optional[int] = Field(default=None, description="For log: commit limit")
    commit: Optional[str] = Field(default=None, description="For show: commit SHA")
    device_id: Optional[str] = Field(default=None, description="Optional device id to target")
    timeout_seconds: int = Field(default=60, description="Timeout in seconds")


class CodeGitInfoOutputs(BaseModel):
    result: Dict[str, Any] = Field(description="Operation-specific result")
    formatted: str = Field(description="Human-readable summary")


async def code_git_info_tool(
    path: str,
    operation: str,
    file: Optional[str] = None,
    limit: Optional[int] = None,
    commit: Optional[str] = None,
    device_id: Optional[str] = None,
    timeout_seconds: int = 60,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Run a read-only git operation on the device via the local proxy."""
    try:
        client = await get_backend_tool_client()
        args: Dict[str, Any] = {"path": path, "operation": operation}
        if file:
            args["file"] = file
        if limit is not None:
            args["limit"] = int(limit)
        if commit:
            args["commit"] = commit
        result = await client.invoke_device_tool(
            user_id=user_id,
            device_id=device_id or "",
            tool="git_info",
            args=args,
            timeout=timeout_seconds,
        )
        if not result.get("success"):
            err = result.get("error", "git_info failed")
            return {"result": {}, "formatted": err}
        data = result.get("result") or {}
        return {"result": data, "formatted": result.get("formatted", "Git info retrieved")}
    except Exception as e:
        logger.error("code_git_info_tool error: %s", e)
        return {"result": {}, "formatted": str(e)}


register_action(
    name="code_git_info",
    category="code_workspace",
    description="Read-only git operations (status/diff/log/branch/show) via the local proxy.",
    inputs_model=CodeGitInfoInputs,
    params_model=None,
    outputs_model=CodeGitInfoOutputs,
    tool_function=code_git_info_tool,
)

