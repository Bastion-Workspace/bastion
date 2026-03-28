"""
Sandboxed subprocess execution for CLI tools.
Runs commands with wall-clock timeout and output size limits. No shell.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    """Result of a sandboxed subprocess run."""
    success: bool
    exit_code: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    error_message: Optional[str] = None


async def run_sandboxed(
    cmd: list[str],
    cwd: str,
    timeout_seconds: int = 120,
    max_stdout_bytes: int = 100 * 1024 * 1024,
    max_stderr_bytes: int = 1024 * 1024,
) -> SandboxResult:
    """
    Run a command in a directory with timeout and output limits.
    Uses asyncio.create_subprocess_exec (no shell). Stderr and stdout are
    captured and truncated if they exceed limits.
    """
    if not cmd:
        return SandboxResult(
            success=False,
            exit_code=-1,
            stdout=b"",
            stderr=b"",
            timed_out=False,
            error_message="empty command",
        )
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            return SandboxResult(
                success=False,
                exit_code=-1,
                stdout=b"",
                stderr=b"",
                timed_out=True,
                error_message=f"command timed out after {timeout_seconds}s",
            )
        if stdout_bytes is None:
            stdout_bytes = b""
        if stderr_bytes is None:
            stderr_bytes = b""
        if len(stdout_bytes) > max_stdout_bytes:
            stdout_bytes = stdout_bytes[:max_stdout_bytes] + b"\n... (truncated)"
        if len(stderr_bytes) > max_stderr_bytes:
            stderr_bytes = stderr_bytes[:max_stderr_bytes] + b"\n... (truncated)"
        return SandboxResult(
            success=proc.returncode == 0 if proc.returncode is not None else False,
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout_bytes,
            stderr=stderr_bytes,
            timed_out=False,
            error_message=None if (proc.returncode == 0) else (stderr_bytes.decode("utf-8", errors="replace") or f"exit code {proc.returncode}"),
        )
    except FileNotFoundError as e:
        logger.warning("Command binary not found: %s", e)
        return SandboxResult(
            success=False,
            exit_code=-1,
            stdout=b"",
            stderr=b"",
            timed_out=False,
            error_message=f"binary not found: {e}",
        )
    except Exception as e:
        logger.exception("Sandbox execution failed")
        return SandboxResult(
            success=False,
            exit_code=-1,
            stdout=b"",
            stderr=b"",
            timed_out=False,
            error_message=str(e),
        )
