"""gRPC handlers for Connector and MCP operations."""

import json
import logging
from typing import List

import grpc
from protos import tool_service_pb2

logger = logging.getLogger(__name__)


class ConnectorMcpHandlersMixin:
    """Mixin providing Connector and MCP gRPC handlers.

    Mixed into ToolServiceImplementation; provides handlers for connector execution,
    GitHub endpoint execution, MCP tool operations, and MCP server discovery.
    """

    async def ExecuteConnector(
        self,
        request: tool_service_pb2.ExecuteConnectorRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ExecuteConnectorResponse:
        """Execute a connector endpoint; load definition and credentials from DB."""
        try:
            from services.database_manager.database_helpers import fetch_one
            from clients.connections_service_client import get_connections_service_client

            user_id = request.user_id or "system"
            profile_id = request.profile_id or None
            connector_id = request.connector_id or None
            endpoint_id = request.endpoint_id or None
            if not profile_id or not connector_id or not endpoint_id:
                return tool_service_pb2.ExecuteConnectorResponse(
                    success=False, result_json="", error="profile_id, connector_id, endpoint_id required"
                )
            params = {}
            if request.params_json:
                try:
                    params = json.loads(request.params_json)
                except json.JSONDecodeError:
                    return tool_service_pb2.ExecuteConnectorResponse(
                        success=False, result_json="", error="Invalid params_json"
                    )
            connector = await fetch_one(
                "SELECT id, definition, connector_type FROM data_source_connectors WHERE id = $1",
                connector_id,
            )
            if not connector:
                return tool_service_pb2.ExecuteConnectorResponse(
                    success=False, result_json="", error="Connector not found"
                )
            source = await fetch_one(
                "SELECT credentials_encrypted, config_overrides FROM agent_data_sources "
                "WHERE agent_profile_id = $1 AND connector_id = $2 AND is_enabled = true",
                profile_id,
                connector_id,
            )
            credentials = {}
            if source:
                creds = source.get("credentials_encrypted")
                if isinstance(creds, dict):
                    credentials = creds
                overrides = source.get("config_overrides") or {}
                if isinstance(overrides, dict) and overrides.get("api_key"):
                    credentials.setdefault("api_key", overrides["api_key"])
            definition = connector.get("definition") or {}
            if isinstance(definition, str):
                definition = json.loads(definition) if definition else {}
            client = await get_connections_service_client()
            result = await client.execute_connector_endpoint(
                definition=definition,
                credentials=credentials,
                endpoint_id=endpoint_id,
                params=params,
                connector_type=connector.get("connector_type"),
            )
            return tool_service_pb2.ExecuteConnectorResponse(
                success=True,
                result_json=json.dumps(result),
            )
        except Exception as e:
            logger.exception("ExecuteConnector failed")
            return tool_service_pb2.ExecuteConnectorResponse(
                success=False, result_json="", error=str(e)
            )

    async def ExecuteGitHubEndpoint(
        self,
        request: tool_service_pb2.ExecuteGitHubEndpointRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ExecuteGitHubEndpointResponse:
        """Execute a system GitHub REST endpoint using OAuth from external_connections."""
        try:
            from services.database_manager.database_helpers import fetch_one
            from clients.connections_service_client import get_connections_service_client
            from services.external_connections_service import external_connections_service
            from services.github_system_connector import GITHUB_SYSTEM_CONNECTOR_DEFINITION

            user_id = request.user_id or "system"
            connection_id = int(request.connection_id) if request.connection_id else 0
            endpoint_id = (request.endpoint_id or "").strip()
            if not connection_id or not endpoint_id:
                return tool_service_pb2.ExecuteGitHubEndpointResponse(
                    success=False, result_json="", error="connection_id and endpoint_id required"
                )
            row = await fetch_one(
                """
                SELECT id, user_id, provider, connection_type
                FROM external_connections
                WHERE id = $1 AND user_id = $2 AND is_active = true
                """,
                connection_id,
                user_id,
            )
            if not row:
                return tool_service_pb2.ExecuteGitHubEndpointResponse(
                    success=False, result_json="", error="GitHub connection not found"
                )
            if row.get("provider") != "github" or row.get("connection_type") != "code_platform":
                return tool_service_pb2.ExecuteGitHubEndpointResponse(
                    success=False, result_json="", error="Not a GitHub code_platform connection"
                )
            oauth_token = await external_connections_service.get_valid_access_token(connection_id)
            if not oauth_token:
                return tool_service_pb2.ExecuteGitHubEndpointResponse(
                    success=False, result_json="", error="Could not resolve GitHub access token"
                )
            params = {}
            if request.params_json:
                try:
                    params = json.loads(request.params_json)
                except json.JSONDecodeError:
                    return tool_service_pb2.ExecuteGitHubEndpointResponse(
                        success=False, result_json="", error="Invalid params_json"
                    )
            if isinstance(params, dict):
                params = {k: v for k, v in params.items() if v is not None and v != ""}
            max_pages = int(request.max_pages) if request.max_pages else 5
            if max_pages < 1:
                max_pages = 5
            if max_pages > 20:
                max_pages = 20
            client = await get_connections_service_client()
            result = await client.execute_connector_endpoint(
                definition=GITHUB_SYSTEM_CONNECTOR_DEFINITION,
                credentials={},
                endpoint_id=endpoint_id,
                params=params,
                oauth_token=oauth_token,
                max_pages=max_pages,
                connector_type="rest",
            )
            return tool_service_pb2.ExecuteGitHubEndpointResponse(
                success=True,
                result_json=json.dumps(result),
            )
        except Exception as e:
            logger.exception("ExecuteGitHubEndpoint failed")
            return tool_service_pb2.ExecuteGitHubEndpointResponse(
                success=False, result_json="", error=str(e)
            )

    async def ExecuteMcpTool(
        self,
        request: tool_service_pb2.ExecuteMcpToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ExecuteMcpToolResponse:
        """Call tools/call on a user-configured MCP server."""
        try:
            from services.database_manager.database_helpers import fetch_one
            from services.mcp_client_service import call_tool

            user_id = request.user_id or "system"
            server_id = int(request.server_id) if request.server_id else 0
            tool_name = (request.tool_name or "").strip()
            if not server_id or not tool_name:
                return tool_service_pb2.ExecuteMcpToolResponse(
                    success=False, result_json="", formatted="", error="server_id and tool_name required"
                )
            raw_args = request.arguments_json or "{}"
            try:
                args = json.loads(raw_args) if raw_args else {}
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                return tool_service_pb2.ExecuteMcpToolResponse(
                    success=False, result_json="", formatted="", error="Invalid arguments_json"
                )

            row = await fetch_one(
                """
                SELECT id, user_id, name, description, transport, url, command, args, env, headers, is_active
                FROM mcp_servers
                WHERE id = $1 AND user_id = $2 AND is_active = true
                """,
                server_id,
                user_id,
            )
            if not row:
                return tool_service_pb2.ExecuteMcpToolResponse(
                    success=False, result_json="", formatted="", error="MCP server not found"
                )
            cfg = dict(row)
            for key in ("args", "env", "headers"):
                v = cfg.get(key)
                if isinstance(v, str):
                    try:
                        cfg[key] = json.loads(v) if v else ([] if key == "args" else {})
                    except json.JSONDecodeError:
                        cfg[key] = [] if key == "args" else {}

            ok, result_json, formatted = await call_tool(cfg, tool_name, args)
            return tool_service_pb2.ExecuteMcpToolResponse(
                success=bool(ok),
                result_json=result_json or "",
                formatted=formatted or "",
                error="" if ok else (formatted or result_json or "MCP tool error"),
            )
        except Exception as e:
            logger.exception("ExecuteMcpTool failed")
            return tool_service_pb2.ExecuteMcpToolResponse(
                success=False, result_json="", formatted="", error=str(e)
            )

    async def GetMcpServerTools(
        self,
        request: tool_service_pb2.GetMcpServerToolsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetMcpServerToolsResponse:
        """Return tool names from cached discovered_tools for an MCP server."""
        try:
            from services.database_manager.database_helpers import fetch_one

            user_id = request.user_id or "system"
            server_id = int(request.server_id) if request.server_id else 0
            if not server_id:
                return tool_service_pb2.GetMcpServerToolsResponse(
                    success=False, tool_names=[], error="server_id required"
                )
            row = await fetch_one(
                """
                SELECT discovered_tools FROM mcp_servers
                WHERE id = $1 AND user_id = $2 AND is_active = true
                """,
                server_id,
                user_id,
            )
            if not row:
                return tool_service_pb2.GetMcpServerToolsResponse(
                    success=False, tool_names=[], error="MCP server not found"
                )
            raw = row.get("discovered_tools")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw) if raw else []
                except json.JSONDecodeError:
                    raw = []
            names: List[str] = []
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and item.get("name"):
                        names.append(str(item["name"]))
                    elif isinstance(item, str):
                        names.append(item)
            return tool_service_pb2.GetMcpServerToolsResponse(
                success=True,
                tool_names=names,
                error="",
            )
        except Exception as e:
            logger.exception("GetMcpServerTools failed")
            return tool_service_pb2.GetMcpServerToolsResponse(
                success=False, tool_names=[], error=str(e)
            )

    async def DiscoverMcpServer(
        self,
        request: tool_service_pb2.DiscoverMcpServerRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.DiscoverMcpServerResponse:
        """Run tools/list for a user MCP server (same runtime as ExecuteMcpTool: stdio uses uvx/npx here)."""
        try:
            from services.database_manager.database_helpers import fetch_one
            from services.mcp_client_service import discover_tools

            user_id = request.user_id or "system"
            server_id = int(request.server_id) if request.server_id else 0
            if not server_id:
                return tool_service_pb2.DiscoverMcpServerResponse(
                    success=False, tools_json="[]", error="server_id required"
                )
            row = await fetch_one(
                """
                SELECT id, user_id, name, description, transport, url, command, args, env, headers, is_active
                FROM mcp_servers
                WHERE id = $1 AND user_id = $2
                """,
                server_id,
                user_id,
            )
            if not row:
                return tool_service_pb2.DiscoverMcpServerResponse(
                    success=False, tools_json="[]", error="MCP server not found"
                )
            cfg = dict(row)
            for key in ("args", "env", "headers"):
                v = cfg.get(key)
                if isinstance(v, str):
                    try:
                        cfg[key] = json.loads(v) if v else ([] if key == "args" else {})
                    except json.JSONDecodeError:
                        cfg[key] = [] if key == "args" else {}

            tools = await discover_tools(cfg)
            return tool_service_pb2.DiscoverMcpServerResponse(
                success=True,
                tools_json=json.dumps(tools, default=str),
                error="",
            )
        except Exception as e:
            logger.exception("DiscoverMcpServer failed")
            return tool_service_pb2.DiscoverMcpServerResponse(
                success=False, tools_json="[]", error=str(e)
            )
