"""
MCP client: connect to user-configured Model Context Protocol servers (stdio, SSE, streamable HTTP).
Used for discovery (tools/list) and execution (tools/call) from the backend and gRPC tool service.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

# Budget for MCP initialize + single operation (tools/list, tools/call, etc.)
MCP_SESSION_TIMEOUT_SEC = 60.0

T = TypeVar("T")

ClientSession: Any = None
stdio_client: Any = None
StdioServerParameters: Any = None
sse_client: Any = None
streamablehttp_client: Any = None
_MCP_CORE = False
_HAS_STREAMABLE = False

try:
    from mcp import ClientSession as _ClientSession
    from mcp.client.stdio import StdioServerParameters as _StdioServerParameters, stdio_client as _stdio_client
    from mcp.client.sse import sse_client as _sse_client

    ClientSession = _ClientSession
    stdio_client = _stdio_client
    StdioServerParameters = _StdioServerParameters
    sse_client = _sse_client
    _MCP_CORE = True
except ImportError:
    pass

try:
    from mcp.client.streamable_http import streamablehttp_client as _streamablehttp_client

    streamablehttp_client = _streamablehttp_client
    _HAS_STREAMABLE = True
except ImportError:
    pass


def _row_headers(row: Dict[str, Any]) -> Dict[str, str]:
    raw = row.get("headers") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in raw.items():
        if v is not None:
            out[str(k)] = str(v)
    return out


def _row_args(row: Dict[str, Any]) -> List[str]:
    a = row.get("args") or []
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except json.JSONDecodeError:
            return []
    if not isinstance(a, list):
        return []
    return [str(x) for x in a]


def _row_env(row: Dict[str, Any]) -> Dict[str, str]:
    e = row.get("env") or {}
    if isinstance(e, str):
        try:
            e = json.loads(e)
        except json.JSONDecodeError:
            return {}
    if not isinstance(e, dict):
        return {}
    merged = dict(os.environ)
    for k, v in e.items():
        merged[str(k)] = str(v)
    return merged


async def _mcp_initialize_and_run(
    session: Any,
    transport: str,
    fn: Callable[[Any], Awaitable[Any]],
) -> Any:
    """
    Run session.initialize() then fn(session), with a single wall-clock budget.
    Logs phase boundaries for diagnosing hangs (e.g. subprocess never completes handshake).
    """

    async def _body() -> Any:
        logger.info("MCP [%s] session.initialize starting", transport)
        await session.initialize()
        logger.info("MCP [%s] session.initialize complete; running operation", transport)
        result = await fn(session)
        logger.info("MCP [%s] operation complete", transport)
        return result

    try:
        return await asyncio.wait_for(_body(), timeout=MCP_SESSION_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        logger.error(
            "MCP [%s] timed out after %ss (initialize + operation)",
            transport,
            int(MCP_SESSION_TIMEOUT_SEC),
        )
        raise RuntimeError(
            f"MCP server did not complete within {int(MCP_SESSION_TIMEOUT_SEC)}s"
        ) from None


async def _run_with_session(
    row: Dict[str, Any],
    fn: Callable[[Any], Awaitable[Any]],
) -> Any:
    if not _MCP_CORE:
        raise RuntimeError("mcp package is not installed; add mcp to backend requirements.txt")

    transport = (row.get("transport") or "").strip().lower()
    headers = _row_headers(row)
    logger.info("MCP _run_with_session transport=%s", transport)

    if transport == "stdio":
        cmd = (row.get("command") or "").strip()
        if not cmd:
            raise ValueError("stdio transport requires command")
        args = _row_args(row)
        env = _row_env(row)
        params = StdioServerParameters(command=cmd, args=args, env=env)
        logger.info(
            "MCP [stdio] opening stdio_client command=%s arg_count=%s",
            cmd,
            len(args),
        )
        async with stdio_client(params) as streams:
            read, write = streams
            logger.info("MCP [stdio] stdio streams ready")
            async with ClientSession(read, write) as session:
                return await _mcp_initialize_and_run(session, "stdio", fn)

    if transport == "sse":
        url = (row.get("url") or "").strip()
        if not url:
            raise ValueError("sse transport requires url")
        logger.info("MCP [sse] connecting url=%s", url)
        async with sse_client(url, headers=headers if headers else None) as streams:
            read, write = streams
            async with ClientSession(read, write) as session:
                return await _mcp_initialize_and_run(session, "sse", fn)

    if transport in ("streamable_http", "streamablehttp", "http"):
        url = (row.get("url") or "").strip()
        if not url:
            raise ValueError("streamable_http transport requires url")
        if not _HAS_STREAMABLE or streamablehttp_client is None:
            raise ValueError("streamable_http transport requires a newer mcp package with streamablehttp_client")
        logger.info("MCP [streamable_http] connecting url=%s", url)
        async with streamablehttp_client(url, headers=headers if headers else None) as streams:
            # mcp>=1.4 yields (read_stream, write_stream, get_session_id_callback)
            read, write, _get_session_id = streams
            async with ClientSession(read, write) as session:
                return await _mcp_initialize_and_run(session, "streamable_http", fn)

    raise ValueError(f"Unsupported MCP transport: {transport}")


def _tool_to_discovery_entry(tool: Any) -> Optional[Dict[str, Any]]:
    """Normalize a tool from tools/list (SDK may return pydantic models or plain dicts)."""
    if isinstance(tool, dict):
        name = tool.get("name")
        name_s = str(name).strip() if name is not None else ""
        if not name_s:
            return None
        desc = tool.get("description")
        entry: Dict[str, Any] = {"name": name_s, "description": str(desc) if desc is not None else ""}
        schema = tool.get("inputSchema")
        if schema is None:
            schema = tool.get("input_schema")
        if schema is not None:
            entry["input_schema"] = schema
        return entry

    name = getattr(tool, "name", None)
    name_s = str(name).strip() if name is not None else ""
    if not name_s:
        return None
    desc = getattr(tool, "description", None)
    entry = {"name": name_s, "description": str(desc) if desc is not None else ""}
    try:
        sch = getattr(tool, "inputSchema", None)
        if sch is not None:
            entry["input_schema"] = sch
    except Exception:
        pass
    return entry


async def discover_tools(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Call tools/list on the server. Returns list of dicts: name, description, input_schema (optional).
    """

    async def _list(session: ClientSession) -> List[Dict[str, Any]]:
        logger.info("MCP discover_tools: calling list_tools")
        result = await session.list_tools()
        tools = getattr(result, "tools", None) or []
        logger.info("MCP discover_tools: list_tools returned %s raw tool(s)", len(tools))
        out: List[Dict[str, Any]] = []
        for t in tools:
            entry = _tool_to_discovery_entry(t)
            if entry:
                out.append(entry)
        if tools and not out:
            logger.warning(
                "MCP tools/list returned %s tool(s) but none had a usable name (check dict vs model shape)",
                len(tools),
            )
        return out

    return await _run_with_session(row, _list)


async def call_tool(
    row: Dict[str, Any],
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, str]:
    """
    Execute tools/call. Returns (success, result_json, formatted_or_error).
    """
    args = arguments if isinstance(arguments, dict) else {}

    async def _call(session: ClientSession) -> Tuple[bool, str, str]:
        logger.info("MCP call_tool: invoking tool_name=%s", tool_name)
        result = await session.call_tool(tool_name, args)
        texts: List[str] = []
        structured: Any = None
        if hasattr(result, "content") and result.content:
            for block in result.content:
                t = getattr(block, "text", None)
                if t:
                    texts.append(t)
        is_err = bool(getattr(result, "isError", False))
        payload = {
            "isError": is_err,
            "content": texts,
        }
        if getattr(result, "structuredContent", None) is not None:
            try:
                structured = result.structuredContent
                payload["structuredContent"] = structured
            except Exception:
                pass
        result_json = json.dumps(payload, default=str)
        formatted = "\n".join(texts) if texts else result_json
        return (not is_err, result_json, formatted if not is_err else formatted)

    try:
        return await _run_with_session(row, _call)
    except Exception as e:
        logger.exception("MCP call_tool failed: %s", e)
        err = str(e)
        return False, json.dumps({"error": err}), err
