"""
Local proxy tools - Invoke capabilities on the user's connected Bastion Local Proxy daemon.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from orchestrator.utils.shell_policy import evaluate_shell_policy

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


def resolve_local_proxy_device_id(
    explicit_device_id: Optional[str],
    pipeline_metadata: Optional[Dict[str, Any]],
) -> Optional[str]:
    """
    Target device for local proxy tools. Prefer an explicit tool argument, then the Code Space
    device from request metadata (set when code_workspace_id is sent to the orchestrator).
    """
    d = (explicit_device_id or "").strip()
    if d:
        return d
    meta = pipeline_metadata or {}
    cw = (meta.get("code_workspace_device_id") or "").strip()
    return cw or None


# Map orchestrator tool name -> daemon capability name (used for invocation and filtering)
TOOL_TO_CAPABILITY = {
    "local_screenshot": "screenshot",
    "local_clipboard_read": "clipboard_read",
    "local_clipboard_write": "clipboard_write",
    "local_system_info": "system_info",
    "local_desktop_notify": "desktop_notify",
    "local_shell_execute": "shell_execute",
    "local_read_file": "read_file",
    "local_list_directory": "list_directory",
    "local_write_file": "write_file",
    "local_patch_file": "patch_file",
    "local_list_processes": "list_processes",
    "local_open_url": "open_url",
}


async def get_available_local_proxy_tools(user_id: str = "system") -> List[str]:
    """Query device capabilities and return orchestrator tool names that are available (e.g. local_screenshot_tool)."""
    try:
        client = await get_backend_tool_client()
        caps = await client.get_device_capabilities(user_id)
        return [f"{tool}_tool" for tool, cap in TOOL_TO_CAPABILITY.items() if cap in caps]
    except Exception as e:
        logger.warning("get_available_local_proxy_tools failed: %s", e)
        return []


# ── local_screenshot ───────────────────────────────────────────────────────

class LocalScreenshotInputs(BaseModel):
    pass


class LocalScreenshotOutputs(BaseModel):
    image_data_uri: str = Field(description="Data URI of the screenshot image (data:image/png;base64,...)")
    width: int = Field(description="Width in pixels")
    height: int = Field(description="Height in pixels")
    formatted: str = Field(description="Human-readable summary and markdown image for chat")


async def local_screenshot_tool(user_id: str = "system") -> Dict[str, Any]:
    """Capture a screenshot from the user's local machine via the Bastion Local Proxy daemon."""
    try:
        client = await get_backend_tool_client()
        result = await client.invoke_device_tool(user_id=user_id, tool="screenshot", args={})
        if not result.get("success"):
            err = result.get("error", "No device connected or capability denied")
            return {
                "image_data_uri": "",
                "width": 0,
                "height": 0,
                "formatted": f"Screenshot failed: {err}",
            }
        data = result.get("result") or {}
        image_base64 = data.get("image_base64", "")
        width = data.get("width", 0)
        height = data.get("height", 0)
        data_uri = f"data:image/png;base64,{image_base64}" if image_base64 else ""
        formatted = result.get("formatted", f"Screenshot captured ({width}x{height})")
        if data_uri:
            formatted = f"![Screenshot]({data_uri})\n\n{formatted}"
        return {
            "image_data_uri": data_uri,
            "width": width,
            "height": height,
            "formatted": formatted,
        }
    except Exception as e:
        logger.error("local_screenshot_tool error: %s", e)
        return {
            "image_data_uri": "",
            "width": 0,
            "height": 0,
            "formatted": f"Error: {str(e)}",
        }


register_action(
    name="local_screenshot",
    category="local_proxy",
    description="Capture a screenshot from the user's local machine. Requires the Bastion Local Proxy daemon to be running and connected.",
    inputs_model=LocalScreenshotInputs,
    params_model=None,
    outputs_model=LocalScreenshotOutputs,
    tool_function=local_screenshot_tool,
)


# ── local_clipboard_read ────────────────────────────────────────────────────

class LocalClipboardReadInputs(BaseModel):
    pass


class LocalClipboardReadOutputs(BaseModel):
    content: str = Field(description="Text content from the clipboard")
    length: int = Field(description="Length in bytes")
    formatted: str = Field(description="Human-readable summary")


async def local_clipboard_read_tool(user_id: str = "system") -> Dict[str, Any]:
    """Read text from the system clipboard on the user's local machine."""
    try:
        client = await get_backend_tool_client()
        result = await client.invoke_device_tool(user_id=user_id, tool="clipboard_read", args={})
        if not result.get("success"):
            return {"content": "", "length": 0, "formatted": result.get("error", "Clipboard read failed")}
        data = result.get("result") or {}
        content = data.get("content", "")
        length = data.get("length", 0)
        # Include actual content in formatted so the agent (and user) see what was read
        if content:
            max_formatted = 8000
            if len(content) <= max_formatted:
                formatted = f"Clipboard content ({length} bytes):\n\n{content}"
            else:
                formatted = f"Clipboard content ({length} bytes, showing first {max_formatted}):\n\n{content[:max_formatted]}\n\n... (truncated)"
        else:
            formatted = "Clipboard is empty." if length == 0 else result.get("formatted", f"Clipboard read ({length} bytes)")
        return {"content": content, "length": length, "formatted": formatted}
    except Exception as e:
        logger.error("local_clipboard_read_tool error: %s", e)
        return {"content": "", "length": 0, "formatted": str(e)}


register_action(
    name="local_clipboard_read",
    category="local_proxy",
    description="Read text from the system clipboard on the user's local machine.",
    inputs_model=LocalClipboardReadInputs,
    params_model=None,
    outputs_model=LocalClipboardReadOutputs,
    tool_function=local_clipboard_read_tool,
)


# ── local_clipboard_write ───────────────────────────────────────────────────

class LocalClipboardWriteInputs(BaseModel):
    content: str = Field(description="Text to write to the clipboard")


class LocalClipboardWriteOutputs(BaseModel):
    success: bool = Field(description="Whether the write succeeded")
    formatted: str = Field(description="Human-readable summary")


async def local_clipboard_write_tool(content: str, user_id: str = "system") -> Dict[str, Any]:
    """Write text to the system clipboard on the user's local machine."""
    try:
        client = await get_backend_tool_client()
        result = await client.invoke_device_tool(user_id=user_id, tool="clipboard_write", args={"content": content or ""})
        if not result.get("success"):
            return {"success": False, "formatted": result.get("error", "Clipboard write failed")}
        return {"success": True, "formatted": result.get("formatted", "Clipboard written")}
    except Exception as e:
        logger.error("local_clipboard_write_tool error: %s", e)
        return {"success": False, "formatted": str(e)}


register_action(
    name="local_clipboard_write",
    category="local_proxy",
    description="Write text to the system clipboard on the user's local machine.",
    inputs_model=LocalClipboardWriteInputs,
    params_model=None,
    outputs_model=LocalClipboardWriteOutputs,
    tool_function=local_clipboard_write_tool,
)


# ── local_system_info ──────────────────────────────────────────────────────

class LocalSystemInfoInputs(BaseModel):
    pass


class LocalSystemInfoOutputs(BaseModel):
    os: str = Field(description="OS name")
    hostname: str = Field(description="Hostname")
    cpu_count: int = Field(description="Number of CPUs")
    total_memory_mb: int = Field(description="Total RAM in MB")
    used_memory_mb: int = Field(description="Used RAM in MB")
    formatted: str = Field(description="Human-readable summary")


async def local_system_info_tool(user_id: str = "system") -> Dict[str, Any]:
    """Get OS, hostname, CPU count, memory, and disk usage from the user's local machine."""
    try:
        client = await get_backend_tool_client()
        result = await client.invoke_device_tool(user_id=user_id, tool="system_info", args={})
        if not result.get("success"):
            return {"os": "", "hostname": "", "cpu_count": 0, "total_memory_mb": 0, "used_memory_mb": 0, "formatted": result.get("error", "Failed")}
        data = result.get("result") or {}
        return {
            "os": data.get("os", ""),
            "hostname": data.get("hostname", ""),
            "cpu_count": data.get("cpu_count", 0),
            "total_memory_mb": data.get("total_memory_mb", 0),
            "used_memory_mb": data.get("used_memory_mb", 0),
            "formatted": result.get("formatted", "System info retrieved"),
        }
    except Exception as e:
        logger.error("local_system_info_tool error: %s", e)
        return {"os": "", "hostname": "", "cpu_count": 0, "total_memory_mb": 0, "used_memory_mb": 0, "formatted": str(e)}


register_action(
    name="local_system_info",
    category="local_proxy",
    description="Get OS, hostname, CPU count, memory, and disk usage from the user's local machine.",
    inputs_model=LocalSystemInfoInputs,
    params_model=None,
    outputs_model=LocalSystemInfoOutputs,
    tool_function=local_system_info_tool,
)


# ── local_desktop_notify ───────────────────────────────────────────────────

class LocalDesktopNotifyInputs(BaseModel):
    title: str = Field(description="Notification title")
    body: str = Field(default="", description="Notification body text")


class LocalDesktopNotifyOutputs(BaseModel):
    success: bool = Field(description="Whether the notification was shown")
    formatted: str = Field(description="Human-readable summary")


async def local_desktop_notify_tool(title: str, body: str = "", user_id: str = "system") -> Dict[str, Any]:
    """Show a desktop notification on the user's local machine."""
    try:
        client = await get_backend_tool_client()
        result = await client.invoke_device_tool(user_id=user_id, tool="desktop_notify", args={"title": title, "body": body or ""})
        success = result.get("success", False)
        return {"success": success, "formatted": result.get("formatted", "Notification sent" if success else result.get("error", "Failed"))}
    except Exception as e:
        logger.error("local_desktop_notify_tool error: %s", e)
        return {"success": False, "formatted": str(e)}


register_action(
    name="local_desktop_notify",
    category="local_proxy",
    description="Show a desktop notification on the user's local machine.",
    inputs_model=LocalDesktopNotifyInputs,
    params_model=None,
    outputs_model=LocalDesktopNotifyOutputs,
    tool_function=local_desktop_notify_tool,
)


# ── local_shell_execute ────────────────────────────────────────────────────

class LocalShellExecuteInputs(BaseModel):
    command: str = Field(description="Shell command to run")
    timeout_seconds: Optional[int] = Field(default=60, description="Timeout in seconds")
    cwd: Optional[str] = Field(default=None, description="Working directory (optional)")
    device_id: Optional[str] = Field(
        default=None,
        description="Local proxy device_id when multiple proxies are connected (else use Code Space context)",
    )


class LocalShellExecuteOutputs(BaseModel):
    stdout: str = Field(description="Standard output")
    stderr: str = Field(description="Standard error")
    exit_code: int = Field(description="Process exit code")
    formatted: str = Field(description="Human-readable summary")


async def local_shell_execute_tool(
    command: str,
    timeout_seconds: int = 60,
    cwd: Optional[str] = None,
    device_id: Optional[str] = None,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a shell command on the user's local machine (daemon policy + optional user shell_command_policy)."""
    try:
        meta = _pipeline_metadata or {}
        workspace_id = meta.get("workspace_id") or meta.get("code_workspace_id")
        raw_rules = meta.get("shell_command_policy")
        if isinstance(raw_rules, str):
            try:
                rules = json.loads(raw_rules or "[]")
            except (json.JSONDecodeError, TypeError):
                rules = []
        elif isinstance(raw_rules, list):
            rules = raw_rules
        else:
            rules = []

        action, label = evaluate_shell_policy(
            rules if isinstance(rules, list) else [], command or "", str(workspace_id).strip() if workspace_id else None
        )

        if action == "deny":
            reason = label or "command denied by policy"
            return {
                "stdout": "",
                "stderr": f"Blocked: {reason}",
                "exit_code": -1,
                "formatted": f"Command blocked by policy: {reason}",
            }

        client = await get_backend_tool_client()

        if action == "require_approval":
            gres = await client.grant_and_consume_shell_approval(
                user_id,
                approval_id="",
                command=command or "",
                consume=True,
            )
            if not (gres.get("success") and gres.get("granted_or_consumed")):
                _ex = meta.get("execution_id")
                approval_id = await client.park_approval(
                    user_id=user_id,
                    agent_profile_id=str(meta.get("agent_profile_id") or ""),
                    execution_id=str(_ex) if _ex else None,
                    step_name="shell_command_approval",
                    prompt=(f"Allow agent to run: `{command}`" + (f"\n({label})" if label else "")),
                    preview_data={"command": command, "label": label},
                    thread_id=str(meta.get("thread_id") or meta.get("langgraph_thread_id") or "")[:500],
                    checkpoint_ns=str(meta.get("checkpoint_ns") or "")[:255],
                    playbook_config=meta.get("playbook_config") if isinstance(meta.get("playbook_config"), dict) else None,
                    governance_type="shell_command_approval",
                )
                return {
                    "_needs_human_interaction": True,
                    "interaction_type": "shell_command_approval",
                    "interaction_data": {
                        "command": command,
                        "label": label,
                        "approval_id": approval_id or "",
                    },
                    "session_id": approval_id or "",
                    "formatted": f"Approval required to run: `{command}`. Request sent.",
                }

        args = {"command": command, "timeout_seconds": timeout_seconds}
        if cwd:
            args["cwd"] = cwd
        target_dev = resolve_local_proxy_device_id(device_id, meta)
        result = await client.invoke_device_tool(
            user_id=user_id,
            tool="shell_execute",
            args=args,
            device_id=target_dev or "",
            timeout=timeout_seconds + 5,
        )
        if not result.get("success"):
            return {"stdout": "", "stderr": "", "exit_code": -1, "formatted": result.get("error", "Command failed")}
        data = result.get("result") or {}
        return {
            "stdout": data.get("stdout", ""),
            "stderr": data.get("stderr", ""),
            "exit_code": data.get("exit_code", -1),
            "formatted": result.get("formatted", f"Exit code {data.get('exit_code', -1)}"),
        }
    except Exception as e:
        logger.error("local_shell_execute_tool error: %s", e)
        return {"stdout": "", "stderr": str(e), "exit_code": -1, "formatted": str(e)}


register_action(
    name="local_shell_execute",
    category="local_proxy",
    description="Run a shell command on the user's local machine. Subject to daemon policy (allowed_commands, denied_patterns) and optional user shell_command_policy (allow/deny/require_approval).",
    inputs_model=LocalShellExecuteInputs,
    params_model=None,
    outputs_model=LocalShellExecuteOutputs,
    tool_function=local_shell_execute_tool,
)


# ── local_read_file ────────────────────────────────────────────────────────

class LocalReadFileInputs(BaseModel):
    path: str = Field(description="Absolute or relative path to the file")
    device_id: Optional[str] = Field(
        default=None,
        description="Local proxy device_id when multiple proxies are connected (else use Code Space context)",
    )


class LocalReadFileOutputs(BaseModel):
    content: str = Field(description="File contents")
    size_bytes: int = Field(description="Size in bytes")
    formatted: str = Field(description="Human-readable summary")


async def local_read_file_tool(
    path: str,
    device_id: Optional[str] = None,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Read a file from the user's local machine (path must be allowed by daemon policy)."""
    try:
        client = await get_backend_tool_client()
        target_dev = resolve_local_proxy_device_id(device_id, _pipeline_metadata)
        result = await client.invoke_device_tool(
            user_id=user_id,
            tool="read_file",
            args={"path": path},
            device_id=target_dev or "",
        )
        if not result.get("success"):
            return {"content": "", "size_bytes": 0, "formatted": result.get("error", "Read failed")}
        data = result.get("result") or {}
        return {"content": data.get("content", ""), "size_bytes": data.get("size_bytes", 0), "formatted": result.get("formatted", "File read")}
    except Exception as e:
        logger.error("local_read_file_tool error: %s", e)
        return {"content": "", "size_bytes": 0, "formatted": str(e)}


register_action(
    name="local_read_file",
    category="local_proxy",
    description="Read a file from the user's local machine. Path must be allowed by daemon policy.",
    inputs_model=LocalReadFileInputs,
    params_model=None,
    outputs_model=LocalReadFileOutputs,
    tool_function=local_read_file_tool,
)


# ── local_list_directory ───────────────────────────────────────────────────

class LocalListDirectoryInputs(BaseModel):
    path: str = Field(description="Directory path to list")
    recursive: bool = Field(default=False, description="Include subdirectories (one level)")
    device_id: Optional[str] = Field(
        default=None,
        description="Local proxy device_id when multiple proxies are connected (else use Code Space context)",
    )


class LocalListDirectoryOutputs(BaseModel):
    entries: List[Dict[str, Any]] = Field(description="List of entries (name, is_dir, size_bytes, modified)")
    count: int = Field(description="Number of entries")
    formatted: str = Field(description="Human-readable summary")


async def local_list_directory_tool(
    path: str,
    recursive: bool = False,
    device_id: Optional[str] = None,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """List directory contents on the user's local machine (path must be allowed by daemon policy)."""
    try:
        client = await get_backend_tool_client()
        target_dev = resolve_local_proxy_device_id(device_id, _pipeline_metadata)
        result = await client.invoke_device_tool(
            user_id=user_id,
            tool="list_directory",
            args={"path": path, "recursive": recursive},
            device_id=target_dev or "",
        )
        if not result.get("success"):
            return {"entries": [], "count": 0, "formatted": result.get("error", "List failed")}
        data = result.get("result") or {}
        entries = data.get("entries", [])
        return {"entries": entries, "count": len(entries), "formatted": result.get("formatted", f"Listed {len(entries)} entries")}
    except Exception as e:
        logger.error("local_list_directory_tool error: %s", e)
        return {"entries": [], "count": 0, "formatted": str(e)}


register_action(
    name="local_list_directory",
    category="local_proxy",
    description="List directory contents on the user's local machine. Path must be allowed by daemon policy.",
    inputs_model=LocalListDirectoryInputs,
    params_model=None,
    outputs_model=LocalListDirectoryOutputs,
    tool_function=local_list_directory_tool,
)


# ── local_write_file ────────────────────────────────────────────────────────

class LocalWriteFileInputs(BaseModel):
    path: str = Field(description="File path to write")
    content: str = Field(description="Content to write")
    append: bool = Field(default=False, description="Append instead of overwrite")
    device_id: Optional[str] = Field(
        default=None,
        description="Local proxy device_id when multiple proxies are connected (else use Code Space context)",
    )


class LocalWriteFileOutputs(BaseModel):
    success: bool = Field(description="Whether the write succeeded")
    bytes_written: int = Field(description="Number of bytes written")
    formatted: str = Field(description="Human-readable summary")


async def local_write_file_tool(
    path: str,
    content: str,
    append: bool = False,
    device_id: Optional[str] = None,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write content to a file on the user's local machine (path must be allowed by daemon policy)."""
    try:
        client = await get_backend_tool_client()
        target_dev = resolve_local_proxy_device_id(device_id, _pipeline_metadata)
        result = await client.invoke_device_tool(
            user_id=user_id,
            tool="write_file",
            args={"path": path, "content": content, "append": append},
            device_id=target_dev or "",
        )
        if not result.get("success"):
            return {"success": False, "bytes_written": 0, "formatted": result.get("error", "Write failed")}
        data = result.get("result") or {}
        return {"success": True, "bytes_written": data.get("bytes_written", 0), "formatted": result.get("formatted", "File written")}
    except Exception as e:
        logger.error("local_write_file_tool error: %s", e)
        return {"success": False, "bytes_written": 0, "formatted": str(e)}


register_action(
    name="local_write_file",
    category="local_proxy",
    description="Write content to a file on the user's local machine. Path must be allowed by daemon policy.",
    inputs_model=LocalWriteFileInputs,
    params_model=None,
    outputs_model=LocalWriteFileOutputs,
    tool_function=local_write_file_tool,
)


# ── local_patch_file ───────────────────────────────────────────────────────

class LocalPatchFileInputs(BaseModel):
    path: str = Field(description="File path on the device")
    old_string: str = Field(description="Exact substring to replace (must be unique unless replace_all is true)")
    new_string: str = Field(default="", description="Replacement text")
    replace_all: bool = Field(default=False, description="If true, replace every occurrence of old_string")
    device_id: Optional[str] = Field(
        default=None,
        description="Local proxy device_id when multiple proxies are connected (else use Code Space context)",
    )


class LocalPatchFileOutputs(BaseModel):
    success: bool = Field(description="Whether the patch was applied")
    replacements: int = Field(default=0, description="Number of replacements made")
    bytes_written: int = Field(default=0, description="Size of file after write")
    formatted: str = Field(description="Human-readable summary")


async def local_patch_file_tool(
    path: str,
    old_string: str,
    new_string: str = "",
    replace_all: bool = False,
    device_id: Optional[str] = None,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply a search-and-replace patch on the user's local machine (same path policy as write_file)."""
    try:
        client = await get_backend_tool_client()
        target_dev = resolve_local_proxy_device_id(device_id, _pipeline_metadata)
        result = await client.invoke_device_tool(
            user_id=user_id,
            tool="patch_file",
            args={
                "path": path,
                "old_string": old_string,
                "new_string": new_string,
                "replace_all": replace_all,
            },
            device_id=target_dev or "",
        )
        if not result.get("success"):
            return {
                "success": False,
                "replacements": 0,
                "bytes_written": 0,
                "formatted": result.get("error", "Patch failed"),
            }
        data = result.get("result") or {}
        return {
            "success": bool(data.get("success", True)),
            "replacements": int(data.get("replacements", 0) or 0),
            "bytes_written": int(data.get("bytes_written", 0) or 0),
            "formatted": result.get("formatted", "Patched"),
        }
    except Exception as e:
        logger.error("local_patch_file_tool error: %s", e)
        return {"success": False, "replacements": 0, "bytes_written": 0, "formatted": str(e)}


register_action(
    name="local_patch_file",
    category="local_proxy",
    description=(
        "Replace exact text in a local file via the proxy (unique match, or replace_all). "
        "Prefer this over rewriting entire files for small edits. Same path allowlist as read/write."
    ),
    inputs_model=LocalPatchFileInputs,
    params_model=None,
    outputs_model=LocalPatchFileOutputs,
    tool_function=local_patch_file_tool,
)


# ── local_list_processes ────────────────────────────────────────────────────

class LocalListProcessesInputs(BaseModel):
    sort_by: str = Field(default="cpu", description="Sort by 'cpu' or 'memory'")
    limit: int = Field(default=50, description="Maximum number of processes to return")


class LocalListProcessesOutputs(BaseModel):
    processes: List[Dict[str, Any]] = Field(description="List of processes (pid, name, cpu_percent, memory_mb, status)")
    count: int = Field(description="Number of processes returned")
    formatted: str = Field(description="Human-readable summary")


async def local_list_processes_tool(sort_by: str = "cpu", limit: int = 50, user_id: str = "system") -> Dict[str, Any]:
    """List running processes on the user's local machine, sorted by CPU or memory."""
    try:
        client = await get_backend_tool_client()
        result = await client.invoke_device_tool(user_id=user_id, tool="list_processes", args={"sort_by": sort_by, "limit": limit})
        if not result.get("success"):
            return {"processes": [], "count": 0, "formatted": result.get("error", "List failed")}
        data = result.get("result") or {}
        processes = data.get("processes", [])
        return {"processes": processes, "count": len(processes), "formatted": result.get("formatted", f"Listed {len(processes)} processes")}
    except Exception as e:
        logger.error("local_list_processes_tool error: %s", e)
        return {"processes": [], "count": 0, "formatted": str(e)}


register_action(
    name="local_list_processes",
    category="local_proxy",
    description="List running processes on the user's local machine, sorted by CPU or memory.",
    inputs_model=LocalListProcessesInputs,
    params_model=None,
    outputs_model=LocalListProcessesOutputs,
    tool_function=local_list_processes_tool,
)


# ── local_open_url ─────────────────────────────────────────────────────────

class LocalOpenUrlInputs(BaseModel):
    url: str = Field(description="URL to open in the default browser")


class LocalOpenUrlOutputs(BaseModel):
    success: bool = Field(description="Whether the URL was opened")
    formatted: str = Field(description="Human-readable summary")


async def local_open_url_tool(url: str, user_id: str = "system") -> Dict[str, Any]:
    """Open a URL in the default browser on the user's local machine."""
    try:
        client = await get_backend_tool_client()
        result = await client.invoke_device_tool(user_id=user_id, tool="open_url", args={"url": url})
        success = result.get("success", False)
        return {"success": success, "formatted": result.get("formatted", "URL opened" if success else result.get("error", "Failed"))}
    except Exception as e:
        logger.error("local_open_url_tool error: %s", e)
        return {"success": False, "formatted": str(e)}


register_action(
    name="local_open_url",
    category="local_proxy",
    description="Open a URL in the default browser on the user's local machine.",
    inputs_model=LocalOpenUrlInputs,
    params_model=None,
    outputs_model=LocalOpenUrlOutputs,
    tool_function=local_open_url_tool,
)
