"""
Local proxy tools - Invoke capabilities on the user's connected Bastion Local Proxy daemon.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)

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


class LocalShellExecuteOutputs(BaseModel):
    stdout: str = Field(description="Standard output")
    stderr: str = Field(description="Standard error")
    exit_code: int = Field(description="Process exit code")
    formatted: str = Field(description="Human-readable summary")


async def local_shell_execute_tool(command: str, timeout_seconds: int = 60, cwd: Optional[str] = None, user_id: str = "system") -> Dict[str, Any]:
    """Run a shell command on the user's local machine (policy may restrict allowed commands)."""
    try:
        client = await get_backend_tool_client()
        args = {"command": command, "timeout_seconds": timeout_seconds}
        if cwd:
            args["cwd"] = cwd
        result = await client.invoke_device_tool(user_id=user_id, tool="shell_execute", args=args, timeout=timeout_seconds + 5)
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
    description="Run a shell command on the user's local machine. Subject to daemon policy (allowed_commands, denied_patterns).",
    inputs_model=LocalShellExecuteInputs,
    params_model=None,
    outputs_model=LocalShellExecuteOutputs,
    tool_function=local_shell_execute_tool,
)


# ── local_read_file ────────────────────────────────────────────────────────

class LocalReadFileInputs(BaseModel):
    path: str = Field(description="Absolute or relative path to the file")


class LocalReadFileOutputs(BaseModel):
    content: str = Field(description="File contents")
    size_bytes: int = Field(description="Size in bytes")
    formatted: str = Field(description="Human-readable summary")


async def local_read_file_tool(path: str, user_id: str = "system") -> Dict[str, Any]:
    """Read a file from the user's local machine (path must be allowed by daemon policy)."""
    try:
        client = await get_backend_tool_client()
        result = await client.invoke_device_tool(user_id=user_id, tool="read_file", args={"path": path})
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


class LocalListDirectoryOutputs(BaseModel):
    entries: List[Dict[str, Any]] = Field(description="List of entries (name, is_dir, size_bytes, modified)")
    count: int = Field(description="Number of entries")
    formatted: str = Field(description="Human-readable summary")


async def local_list_directory_tool(path: str, recursive: bool = False, user_id: str = "system") -> Dict[str, Any]:
    """List directory contents on the user's local machine (path must be allowed by daemon policy)."""
    try:
        client = await get_backend_tool_client()
        result = await client.invoke_device_tool(user_id=user_id, tool="list_directory", args={"path": path, "recursive": recursive})
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


class LocalWriteFileOutputs(BaseModel):
    success: bool = Field(description="Whether the write succeeded")
    bytes_written: int = Field(description="Number of bytes written")
    formatted: str = Field(description="Human-readable summary")


async def local_write_file_tool(path: str, content: str, append: bool = False, user_id: str = "system") -> Dict[str, Any]:
    """Write content to a file on the user's local machine (path must be allowed by daemon policy)."""
    try:
        client = await get_backend_tool_client()
        result = await client.invoke_device_tool(user_id=user_id, tool="write_file", args={"path": path, "content": content, "append": append})
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
