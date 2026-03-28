"""
Bastion document/knowledge MCP server (tools for external MCP clients).

Renamed from `mcp` to avoid shadowing the PyPI `mcp` SDK package on `sys.path`
when the backend app root is `/app`.
"""

from version import __version__ 