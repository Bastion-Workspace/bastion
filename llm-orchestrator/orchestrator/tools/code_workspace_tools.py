"""
Code workspace tools - higher-level coding tools built on top of the local proxy device.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.tools.local_proxy_tools import resolve_local_proxy_device_id
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
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Set the device-side active workspace root for relative path resolution."""
    try:
        client = await get_backend_tool_client()
        target_dev = resolve_local_proxy_device_id(device_id, _pipeline_metadata)
        result = await client.set_device_workspace(
            user_id=user_id,
            device_id=target_dev or "",
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
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get a recursive file tree from the device."""
    try:
        client = await get_backend_tool_client()
        target_dev = resolve_local_proxy_device_id(device_id, _pipeline_metadata)
        args: Dict[str, Any] = {
            "path": path,
            "max_depth": int(max_depth),
            "include_hidden": bool(include_hidden),
        }
        if ignore_patterns is not None:
            args["ignore_patterns"] = ignore_patterns
        result = await client.invoke_device_tool(
            user_id=user_id,
            device_id=target_dev or "",
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
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Search file contents on the device."""
    try:
        client = await get_backend_tool_client()
        target_dev = resolve_local_proxy_device_id(device_id, _pipeline_metadata)
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
            device_id=target_dev or "",
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
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a read-only git operation on the device via the local proxy."""
    try:
        client = await get_backend_tool_client()
        target_dev = resolve_local_proxy_device_id(device_id, _pipeline_metadata)
        args: Dict[str, Any] = {"path": path, "operation": operation}
        if file:
            args["file"] = file
        if limit is not None:
            args["limit"] = int(limit)
        if commit:
            args["commit"] = commit
        result = await client.invoke_device_tool(
            user_id=user_id,
            device_id=target_dev or "",
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


# ── code_list_workspaces ───────────────────────────────────────────────────


class CodeListWorkspacesInputs(BaseModel):
    pass


class CodeListWorkspacesOutputs(BaseModel):
    workspaces: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)
    formatted: str = Field(default="")


async def code_list_workspaces_tool(user_id: str = "system") -> Dict[str, Any]:
    """List saved code workspaces (name, path, device) from Bastion."""
    try:
        client = await get_backend_tool_client()
        res = await client.list_code_workspaces(user_id=user_id)
        if res.get("error"):
            return {
                "workspaces": [],
                "total": 0,
                "formatted": res.get("error", "list failed"),
            }
        rows = res.get("workspaces") or []
        lines = [f"Found {len(rows)} code workspace(s):"]
        for w in rows:
            lines.append(
                f"- {w.get('name') or '?'}  id={w.get('workspace_id')}  path={w.get('workspace_path')}  device={w.get('device_id')}"
            )
        return {"workspaces": rows, "total": int(res.get("total") or len(rows)), "formatted": "\n".join(lines)}
    except Exception as e:
        logger.error("code_list_workspaces_tool error: %s", e)
        return {"workspaces": [], "total": 0, "formatted": str(e)}


register_action(
    name="code_list_workspaces",
    category="code_workspace",
    description="List the user's saved local code workspaces (id, name, path, device_id).",
    inputs_model=CodeListWorkspacesInputs,
    params_model=None,
    outputs_model=CodeListWorkspacesOutputs,
    tool_function=code_list_workspaces_tool,
)


# ── code_get_workspace ─────────────────────────────────────────────────────


class CodeGetWorkspaceInputs(BaseModel):
    workspace_id: str = Field(description="UUID of the saved code workspace")


class CodeGetWorkspaceOutputs(BaseModel):
    success: bool = Field(default=False)
    workspace: Dict[str, Any] = Field(default_factory=dict)
    formatted: str = Field(default="")


async def code_get_workspace_tool(workspace_id: str, user_id: str = "system") -> Dict[str, Any]:
    """Get details for one saved code workspace."""
    try:
        client = await get_backend_tool_client()
        res = await client.get_code_workspace(workspace_id=workspace_id, user_id=user_id)
        if not res.get("success"):
            err = res.get("error", "not found")
            return {"success": False, "workspace": {}, "formatted": err}
        w = res.get("workspace") or {}
        return {
            "success": True,
            "workspace": w,
            "formatted": f"{w.get('name')}: {w.get('workspace_path')} (device {w.get('device_id')})",
        }
    except Exception as e:
        logger.error("code_get_workspace_tool error: %s", e)
        return {"success": False, "workspace": {}, "formatted": str(e)}


register_action(
    name="code_get_workspace",
    category="code_workspace",
    description="Get one saved code workspace by id (path, device_id, settings).",
    inputs_model=CodeGetWorkspaceInputs,
    params_model=None,
    outputs_model=CodeGetWorkspaceOutputs,
    tool_function=code_get_workspace_tool,
)


# ── code_index_workspace ────────────────────────────────────────────────────


class CodeIndexWorkspaceInputs(BaseModel):
    workspace_id: str = Field(description="UUID of the saved code workspace to index")
    replace_index: bool = Field(
        default=True,
        description="If true, clears existing indexed chunks for this workspace before the first batch",
    )
    max_batches: int = Field(default=30, description="Safety cap on index_workspace round-trips")
    device_id: Optional[str] = Field(default=None, description="Override device id (default from workspace record)")
    timeout_seconds: int = Field(default=120, description="Timeout per device index_workspace call")


class CodeIndexWorkspaceOutputs(BaseModel):
    success: bool = Field(default=False)
    batches: int = Field(default=0)
    total_chunks_upserted: int = Field(default=0)
    total_embedded: int = Field(default=0)
    formatted: str = Field(default="")


async def code_index_workspace_tool(
    workspace_id: str,
    replace_index: bool = True,
    max_batches: int = 30,
    device_id: Optional[str] = None,
    timeout_seconds: int = 120,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Walk the device workspace, chunk source files via local proxy, upsert into Bastion for semantic search.
    """
    try:
        client = await get_backend_tool_client()
        meta = await client.get_code_workspace(workspace_id=workspace_id, user_id=user_id)
        if not meta.get("success"):
            return {
                "success": False,
                "batches": 0,
                "total_chunks_upserted": 0,
                "total_embedded": 0,
                "formatted": meta.get("error", "workspace not found"),
            }
        w = meta.get("workspace") or {}
        root = (w.get("workspace_path") or "").strip()
        dev = (device_id or w.get("device_id") or "").strip()
        if not root or not dev:
            return {
                "success": False,
                "batches": 0,
                "total_chunks_upserted": 0,
                "total_embedded": 0,
                "formatted": "workspace_path or device_id missing on record",
            }

        batches = 0
        total_ins = 0
        total_emb = 0
        resume_from: Optional[str] = None
        resume_line = 1
        first_replace = bool(replace_index)

        while batches < int(max_batches):
            batches += 1
            args: Dict[str, Any] = {"path": root, "max_chunks": 200, "max_files": 120}
            if resume_from:
                args["resume_from_path"] = resume_from
                args["resume_start_line"] = int(resume_line)
            inv = await client.invoke_device_tool(
                user_id=user_id,
                device_id=dev,
                tool="index_workspace",
                args=args,
                timeout=int(timeout_seconds),
            )
            if not inv.get("success"):
                return {
                    "success": False,
                    "batches": batches,
                    "total_chunks_upserted": total_ins,
                    "total_embedded": total_emb,
                    "formatted": inv.get("error") or inv.get("formatted") or "index_workspace failed",
                }
            data = inv.get("result") or {}
            raw_chunks = data.get("chunks") or []
            chunks: List[Dict[str, Any]] = []
            for c in raw_chunks:
                if isinstance(c, dict):
                    chunks.append(
                        {
                            "file_path": c.get("file_path", ""),
                            "chunk_index": int(c.get("chunk_index", 0)),
                            "start_line": int(c.get("start_line", 1)),
                            "end_line": int(c.get("end_line", 1)),
                            "content": c.get("content", "") or "",
                            "language": c.get("language", "") or "",
                            "git_sha": c.get("git_sha", "") or "",
                        }
                    )
            up = await client.upsert_code_workspace_chunks(
                user_id=user_id,
                workspace_id=workspace_id,
                replace_workspace=first_replace,
                chunks=chunks,
            )
            first_replace = False
            if up.get("success"):
                total_ins += int(up.get("inserted") or 0)
                total_emb += int(up.get("embedded") or 0)
            else:
                return {
                    "success": False,
                    "batches": batches,
                    "total_chunks_upserted": total_ins,
                    "total_embedded": total_emb,
                    "formatted": up.get("error") or "upsert failed",
                }

            if not data.get("truncated"):
                break
            nxt = data.get("next_resume_path")
            resume_from = nxt if nxt else None
            rl = data.get("next_resume_line")
            if rl is not None:
                try:
                    resume_line = int(rl)
                except (TypeError, ValueError):
                    resume_line = 1
            else:
                resume_line = 1
            if not resume_from:
                break

        return {
            "success": True,
            "batches": batches,
            "total_chunks_upserted": total_ins,
            "total_embedded": total_emb,
            "formatted": f"Indexed in {batches} batch(es): {total_ins} chunk row(s), {total_emb} embedded",
        }
    except Exception as e:
        logger.error("code_index_workspace_tool error: %s", e)
        return {
            "success": False,
            "batches": 0,
            "total_chunks_upserted": 0,
            "total_embedded": 0,
            "formatted": str(e),
        }


register_action(
    name="code_index_workspace",
    category="code_workspace",
    description="Index a saved code workspace for semantic search (local proxy walk + server-side vectors).",
    inputs_model=CodeIndexWorkspaceInputs,
    params_model=None,
    outputs_model=CodeIndexWorkspaceOutputs,
    tool_function=code_index_workspace_tool,
)


# ── code_semantic_search ────────────────────────────────────────────────────


class CodeSemanticSearchInputs(BaseModel):
    workspace_id: str = Field(description="UUID of the saved code workspace")
    query: str = Field(description="Natural language or keyword query")
    limit: int = Field(default=20, ge=1, le=100)
    file_glob: Optional[str] = Field(default=None, description="Optional path substring or glob-style filter")


class CodeSemanticSearchOutputs(BaseModel):
    success: bool = Field(default=False)
    hits: List[Dict[str, Any]] = Field(default_factory=list)
    formatted: str = Field(default="")


async def code_semantic_search_tool(
    workspace_id: str,
    query: str,
    limit: int = 20,
    file_glob: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Hybrid semantic + full-text search over indexed chunks for a workspace."""
    try:
        client = await get_backend_tool_client()
        res = await client.code_semantic_search(
            user_id=user_id,
            workspace_id=workspace_id,
            query=query,
            limit=limit,
            file_glob=file_glob or "",
        )
        if not res.get("success"):
            return {
                "success": False,
                "hits": [],
                "formatted": res.get("error", "search failed"),
            }
        hits = res.get("hits") or []
        lines = [f"{len(hits)} hit(s):"]
        for h in hits[:30]:
            lines.append(
                f"- {h.get('file_path')}:{h.get('start_line')}-{h.get('end_line')}  score={h.get('score')}"
            )
        return {"success": True, "hits": hits, "formatted": "\n".join(lines)}
    except Exception as e:
        logger.error("code_semantic_search_tool error: %s", e)
        return {"success": False, "hits": [], "formatted": str(e)}


register_action(
    name="code_semantic_search",
    category="code_workspace",
    description="Search indexed code in a workspace (semantic + keyword hybrid).",
    inputs_model=CodeSemanticSearchInputs,
    params_model=None,
    outputs_model=CodeSemanticSearchOutputs,
    tool_function=code_semantic_search_tool,
)


CODE_WORKSPACE_TOOLS = {
    "code_open_workspace": code_open_workspace_tool,
    "code_file_tree": code_file_tree_tool,
    "code_search_files": code_search_files_tool,
    "code_git_info": code_git_info_tool,
    "code_list_workspaces": code_list_workspaces_tool,
    "code_get_workspace": code_get_workspace_tool,
    "code_index_workspace": code_index_workspace_tool,
    "code_semantic_search": code_semantic_search_tool,
}
