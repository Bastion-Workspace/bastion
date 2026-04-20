"""gRPC handlers for Data Connector Builder operations."""

import json
import logging

import grpc
from protos import tool_service_pb2
from services.grpc_handlers._utils import jsonb_list

logger = logging.getLogger(__name__)


class DataConnectorBuilderHandlersMixin:
    """Mixin providing Data Connector Builder gRPC handlers.

    Mixed into ToolServiceImplementation; provides handlers for API endpoint
    probing, connector testing/creation, bulk scraping, and connector CRUD.
    """

    # ===== Data Connection Builder =====

    async def ProbeApiEndpoint(
        self,
        request: tool_service_pb2.ProbeApiEndpointRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ProbeApiEndpointResponse:
        """Raw HTTP request for API discovery; delegates to connections-service."""
        try:
            from clients.connections_service_client import get_connections_service_client
            user_id = request.user_id or "system"
            headers = {}
            if request.headers_json:
                try:
                    headers = json.loads(request.headers_json)
                except json.JSONDecodeError:
                    return tool_service_pb2.ProbeApiEndpointResponse(
                        success=False, error="Invalid headers_json"
                    )
            body = None
            if request.body_json:
                try:
                    body = json.loads(request.body_json)
                except json.JSONDecodeError:
                    return tool_service_pb2.ProbeApiEndpointResponse(
                        success=False, error="Invalid body_json"
                    )
            params = None
            if request.params_json:
                try:
                    params = json.loads(request.params_json)
                except json.JSONDecodeError:
                    return tool_service_pb2.ProbeApiEndpointResponse(
                        success=False, error="Invalid params_json"
                    )
            client = await get_connections_service_client()
            result = await client.probe_api_endpoint(
                url=request.url or "",
                method=request.method or "GET",
                headers=headers,
                body=body,
                params=params,
            )
            if not result.get("success"):
                return tool_service_pb2.ProbeApiEndpointResponse(
                    success=False,
                    error=result.get("error", "Probe failed"),
                )
            return tool_service_pb2.ProbeApiEndpointResponse(
                success=True,
                status_code=result.get("status_code", 0),
                response_headers_json=json.dumps(result.get("response_headers", {})),
                response_body=result.get("response_body", ""),
                content_type=result.get("content_type", ""),
            )
        except Exception as e:
            logger.exception("ProbeApiEndpoint failed")
            return tool_service_pb2.ProbeApiEndpointResponse(success=False, error=str(e))

    async def TestConnectorEndpoint(
        self,
        request: tool_service_pb2.TestConnectorEndpointRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.TestConnectorEndpointResponse:
        """Test a connector definition against the live API (no save required)."""
        try:
            from clients.connections_service_client import get_connections_service_client
            definition = {}
            if request.definition_json:
                definition = json.loads(request.definition_json)
            params = {}
            if request.params_json:
                params = json.loads(request.params_json)
            credentials = {}
            if request.credentials_json:
                credentials = json.loads(request.credentials_json)
            client = await get_connections_service_client()
            result = await client.execute_connector_endpoint(
                definition=definition,
                credentials=credentials,
                endpoint_id=request.endpoint_id or "",
                params=params,
                raw_response=True,
            )
            records = result.get("records", [])
            raw_response = result.get("raw_response")
            formatted = result.get("formatted", "")
            if result.get("error"):
                return tool_service_pb2.TestConnectorEndpointResponse(
                    success=False,
                    records_json="[]",
                    count=0,
                    raw_response_json="",
                    formatted=result.get("error", ""),
                    error=result.get("error"),
                )
            return tool_service_pb2.TestConnectorEndpointResponse(
                success=True,
                records_json=json.dumps(records),
                count=len(records),
                raw_response_json=json.dumps(raw_response) if raw_response is not None else "{}",
                formatted=formatted,
            )
        except json.JSONDecodeError as e:
            return tool_service_pb2.TestConnectorEndpointResponse(
                success=False, error=f"Invalid JSON: {e}"
            )
        except Exception as e:
            logger.exception("TestConnectorEndpoint failed")
            return tool_service_pb2.TestConnectorEndpointResponse(success=False, error=str(e))

    async def CreateDataConnector(
        self,
        request: tool_service_pb2.CreateDataConnectorRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateDataConnectorResponse:
        """Save a connector definition to the database."""
        try:
            from services.database_manager.database_helpers import execute, fetch_one
            user_id = request.user_id or "system"
            name = request.name or "Unnamed Connector"
            definition = {}
            if request.definition_json:
                definition = json.loads(request.definition_json)
            auth_fields = []
            if request.auth_fields_json:
                try:
                    auth_fields = json.loads(request.auth_fields_json)
                except json.JSONDecodeError:
                    pass
            await execute(
                """
                INSERT INTO data_source_connectors (
                    user_id, name, description, connector_type, version, definition,
                    is_template, requires_auth, auth_fields, icon, category, tags
                ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, false, $7, $8::jsonb, $9, $10, $11)
                """,
                user_id,
                name,
                request.description or "",
                "rest",
                "1.0",
                json.dumps(definition),
                request.requires_auth,
                json.dumps(auth_fields),
                None,
                request.category or None,
                [],
            )
            row = await fetch_one(
                "SELECT id, name FROM data_source_connectors WHERE user_id = $1 AND name = $2 ORDER BY created_at DESC LIMIT 1",
                user_id,
                name,
            )
            if not row:
                return tool_service_pb2.CreateDataConnectorResponse(
                    success=False, error="Failed to create connector"
                )
            connector_id = str(row["id"])
            formatted = f"Created data connector: {row.get('name', name)} (ID: {connector_id})"
            return tool_service_pb2.CreateDataConnectorResponse(
                success=True,
                connector_id=connector_id,
                name=row.get("name", name),
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("CreateDataConnector failed")
            return tool_service_pb2.CreateDataConnectorResponse(success=False, error=str(e))

    async def BulkScrapeUrls(
        self,
        request: tool_service_pb2.BulkScrapeUrlsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.BulkScrapeUrlsResponse:
        """Scrape URLs for content and optionally images. Inline for <20 URLs, Celery for 20+."""
        try:
            urls = []
            if request.urls_json:
                urls = json.loads(request.urls_json)
            if not isinstance(urls, list):
                urls = []
            urls = [u for u in urls if isinstance(u, str) and u.strip()]
            user_id = request.user_id or "system"
            extract_images = request.extract_images
            download_images = request.download_images
            max_concurrent = request.max_concurrent if request.max_concurrent > 0 else 10
            rate_limit_seconds = request.rate_limit_seconds if request.rate_limit_seconds > 0 else 1.0
            folder_id = request.folder_id or ""

            if len(urls) >= 20:
                from services.celery_tasks.scraper_tasks import batch_url_scrape_task
                task = batch_url_scrape_task.delay(
                    urls=urls,
                    user_id=user_id,
                    config={
                        "extract_images": extract_images,
                        "download_images": download_images,
                        "image_output_folder": request.image_output_folder or "",
                        "metadata_fields_json": request.metadata_fields_json or "[]",
                        "max_concurrent": max_concurrent,
                        "rate_limit_seconds": rate_limit_seconds,
                        "folder_id": folder_id,
                    },
                )
                return tool_service_pb2.BulkScrapeUrlsResponse(
                    success=True,
                    task_id=task.id,
                    results_json="[]",
                    count=0,
                    images_found=0,
                    images_downloaded=0,
                    formatted=f"Bulk scrape started for {len(urls)} URLs. Task ID: {task.id}. Use get_bulk_scrape_status to check progress.",
                )
            else:
                from clients.crawl_service_client import get_crawl_service_client
                client = await get_crawl_service_client()
                response = await client.crawl_many(
                    urls=urls[:20],
                    max_concurrent=max_concurrent,
                    rate_limit_seconds=rate_limit_seconds,
                    include_metadata=True,
                )
                results = response.get("results", [])
                images_found = 0
                images_downloaded = 0
                for r in results:
                    images_found += len(r.get("images", []))
                formatted = f"Crawled {len(results)} URL(s). Images found: {images_found}."
                return tool_service_pb2.BulkScrapeUrlsResponse(
                    success=True,
                    task_id="",
                    results_json=json.dumps(results),
                    count=len(results),
                    images_found=images_found,
                    images_downloaded=images_downloaded,
                    formatted=formatted,
                )
        except Exception as e:
            logger.exception("BulkScrapeUrls failed")
            return tool_service_pb2.BulkScrapeUrlsResponse(
                success=False, error=str(e)
            )

    async def GetBulkScrapeStatus(
        self,
        request: tool_service_pb2.GetBulkScrapeStatusRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetBulkScrapeStatusResponse:
        """Get status and optional results of a bulk scrape Celery task."""
        try:
            from celery.result import AsyncResult
            from services.celery_app import celery_app
            task_id = request.task_id or ""
            if not task_id:
                return tool_service_pb2.GetBulkScrapeStatusResponse(
                    success=False, error="task_id required"
                )
            result = AsyncResult(task_id, app=celery_app)
            state = result.state or "PENDING"
            progress_current = 0
            progress_total = 0
            progress_message = ""
            results_json = "[]"
            if state == "SUCCESS" and result.result:
                res = result.result
                if isinstance(res, dict):
                    progress_current = res.get("progress_current", 0)
                    progress_total = res.get("progress_total", 0)
                    progress_message = res.get("progress_message", "")
                    results_json = json.dumps(res.get("results", []))
            elif state == "PROGRESS" and result.info:
                info = result.info if isinstance(result.info, dict) else {}
                progress_current = info.get("current", 0)
                progress_total = info.get("total", 0)
                progress_message = info.get("message", "")
                results_json = json.dumps(info.get("results", []))
            formatted = f"Task {task_id}: {state}. {progress_message or state}"
            return tool_service_pb2.GetBulkScrapeStatusResponse(
                success=True,
                status=state,
                progress_current=progress_current,
                progress_total=progress_total,
                progress_message=progress_message,
                results_json=results_json,
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("GetBulkScrapeStatus failed")
            return tool_service_pb2.GetBulkScrapeStatusResponse(
                success=False, error=str(e)
            )

    async def ListControlPanes(
        self,
        request: tool_service_pb2.ListControlPanesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListControlPanesResponse:
        """List all control panes for the user."""
        try:
            from services.database_manager.database_helpers import fetch_all
            user_id = request.user_id or "system"
            rows = await fetch_all(
                """
                SELECT p.id, p.user_id, p.name, p.icon, p.pane_type, p.connector_id, p.artifact_id,
                       p.artifact_popover_width, p.artifact_popover_height,
                       p.credentials_encrypted, p.connection_id, p.controls, p.is_visible, p.sort_order, p.refresh_interval,
                       p.created_at, p.updated_at, c.name AS connector_name
                FROM user_control_panes p
                LEFT JOIN data_source_connectors c ON c.id = p.connector_id
                WHERE p.user_id = $1
                ORDER BY p.sort_order ASC, p.name ASC
                """,
                user_id,
            )
            result = []
            for r in rows:
                row = dict(r)
                controls = row.get("controls")
                if controls is not None and isinstance(controls, str):
                    try:
                        controls = json.loads(controls)
                    except json.JSONDecodeError:
                        controls = []
                cid = row.get("connector_id")
                aid = row.get("artifact_id")
                result.append({
                    "id": str(row["id"]),
                    "user_id": row.get("user_id"),
                    "name": row.get("name", ""),
                    "icon": row.get("icon", "Tune"),
                    "pane_type": row.get("pane_type") or "connector",
                    "connector_id": str(cid) if cid else None,
                    "artifact_id": str(aid) if aid else None,
                    "artifact_popover_width": row.get("artifact_popover_width"),
                    "artifact_popover_height": row.get("artifact_popover_height"),
                    "connector_name": row.get("connector_name"),
                    "controls": controls or [],
                    "is_visible": row.get("is_visible", True),
                    "sort_order": row.get("sort_order", 0),
                    "refresh_interval": row.get("refresh_interval", 0),
                })
            parts = [f"Found {len(result)} control pane(s):"]
            for p in result:
                name = p.get("name", "(unnamed)")
                pid = p.get("id", "")
                if (p.get("pane_type") or "connector") == "artifact":
                    extra = f"artifact: {p.get('artifact_id', '')}"
                else:
                    extra = p.get("connector_name") or p.get("connector_id") or ""
                parts.append(f"  - {name} (id: {pid}, {extra})")
            formatted = "\n".join(parts) if result else parts[0]
            return tool_service_pb2.ListControlPanesResponse(
                success=True,
                panes_json=json.dumps(result),
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("ListControlPanes failed")
            return tool_service_pb2.ListControlPanesResponse(success=False, error=str(e))

    async def GetConnectorEndpoints(
        self,
        request: tool_service_pb2.GetConnectorEndpointsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetConnectorEndpointsResponse:
        """Return endpoint ids and metadata from a connector definition for control pane mapping."""
        try:
            from services.database_manager.database_helpers import fetch_one
            user_id = request.user_id or "system"
            connector_id = request.connector_id or ""
            if not connector_id:
                return tool_service_pb2.GetConnectorEndpointsResponse(
                    success=False, error="connector_id required"
                )
            row = await fetch_one(
                "SELECT id, definition FROM data_source_connectors WHERE id = $1 AND (user_id = $2 OR is_template = true)",
                connector_id,
                user_id,
            )
            if not row:
                return tool_service_pb2.GetConnectorEndpointsResponse(
                    success=False, error="Connector not found"
                )
            definition = row.get("definition") or {}
            if isinstance(definition, str):
                try:
                    definition = json.loads(definition)
                except json.JSONDecodeError:
                    definition = {}
            endpoints_def = definition.get("endpoints") or {}
            if isinstance(endpoints_def, list):
                endpoints_def = {ep.get("id") or ep.get("name"): ep for ep in endpoints_def if ep.get("id") or ep.get("name")}
            endpoints_list = []
            for eid, ep in (endpoints_def.items() if isinstance(endpoints_def, dict) else []):
                raw_params = ep.get("params") or []
                if isinstance(raw_params, dict):
                    raw_params = [{"name": k, "in": "query", "default": v} for k, v in raw_params.items()]
                param_list = []
                for p in raw_params:
                    name = p.get("name") or p.get("id")
                    if name:
                        param_list.append({
                            "name": name,
                            "in": p.get("in", "query"),
                            "description": p.get("description") or "",
                            "required": p.get("required", False),
                            "default": p.get("default"),
                        })
                endpoints_list.append({
                    "id": eid,
                    "path": ep.get("path", "/"),
                    "method": (ep.get("method") or "GET").upper(),
                    "description": ep.get("description") or "",
                    "params": param_list,
                })
            parts = []
            for e in endpoints_list:
                param_names = [p["name"] for p in e.get("params", [])]
                parts.append(f"  {e['id']} ({e['method']} {e['path']}) params: {param_names or 'none'}")
            formatted = f"Connector has {len(endpoints_list)} endpoint(s):\n" + "\n".join(parts)
            return tool_service_pb2.GetConnectorEndpointsResponse(
                success=True,
                endpoints_json=json.dumps(endpoints_list),
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("GetConnectorEndpoints failed")
            return tool_service_pb2.GetConnectorEndpointsResponse(success=False, error=str(e))

    async def ListDataConnectors(
        self,
        request: tool_service_pb2.ListDataConnectorsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListDataConnectorsResponse:
        """List user-owned data source connectors (non-templates)."""
        try:
            from services.database_manager.database_helpers import fetch_all
            user_id = request.user_id or "system"
            rows = await fetch_all(
                """
                SELECT id, name, description, connector_type, definition, is_locked, category, tags, created_at, updated_at
                FROM data_source_connectors
                WHERE user_id = $1 AND (is_template = false OR is_template IS NULL)
                ORDER BY updated_at DESC NULLS LAST, created_at DESC
                """,
                user_id,
            )
            result = []
            for r in rows:
                definition = r.get("definition") or {}
                if isinstance(definition, str):
                    try:
                        definition = json.loads(definition)
                    except json.JSONDecodeError:
                        definition = {}
                endpoints = definition.get("endpoints") or {}
                endpoint_count = len(endpoints) if isinstance(endpoints, dict) else 0
                result.append({
                    "id": str(r["id"]),
                    "name": r.get("name", ""),
                    "description": r.get("description"),
                    "connector_type": r.get("connector_type", "rest"),
                    "endpoint_count": endpoint_count,
                    "is_locked": r.get("is_locked", False),
                    "category": r.get("category"),
                    "tags": list(r.get("tags") or []),
                    "created_at": r.get("created_at").isoformat() if r.get("created_at") else None,
                    "updated_at": r.get("updated_at").isoformat() if r.get("updated_at") else None,
                })
            parts = [f"Found {len(result)} connector(s):"]
            for c in result:
                name = c.get("name", "(unnamed)")
                cid = c.get("id", "")
                ctype = c.get("connector_type", "rest")
                n_ep = c.get("endpoint_count", 0)
                parts.append(f"  - {name} (id: {cid}, type: {ctype}, {n_ep} endpoint(s))")
            formatted = "\n".join(parts) if result else parts[0]
            return tool_service_pb2.ListDataConnectorsResponse(
                success=True,
                connectors_json=json.dumps(result),
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("ListDataConnectors failed")
            return tool_service_pb2.ListDataConnectorsResponse(success=False, error=str(e))

    async def GetDataConnector(
        self,
        request: tool_service_pb2.GetDataConnectorRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetDataConnectorResponse:
        """Return full connector by ID (definition, endpoints; auth values redacted)."""
        try:
            from services.database_manager.database_helpers import fetch_one
            user_id = request.user_id or "system"
            connector_id = request.connector_id or ""
            if not connector_id:
                return tool_service_pb2.GetDataConnectorResponse(
                    success=False, error="connector_id required"
                )
            row = await fetch_one(
                "SELECT id, name, description, connector_type, definition, requires_auth, auth_fields, "
                "is_locked, category, tags, created_at, updated_at FROM data_source_connectors "
                "WHERE id = $1 AND user_id = $2",
                connector_id,
                user_id,
            )
            if not row:
                return tool_service_pb2.GetDataConnectorResponse(
                    success=False, error="Connector not found"
                )
            definition = row.get("definition") or {}
            if isinstance(definition, str):
                try:
                    definition = json.loads(definition) if definition else {}
                except json.JSONDecodeError:
                    definition = {}
            if not isinstance(definition, dict):
                definition = {}
            auth_fields_raw = row.get("auth_fields") or []
            if isinstance(auth_fields_raw, str):
                try:
                    auth_fields_raw = json.loads(auth_fields_raw) if auth_fields_raw else []
                except json.JSONDecodeError:
                    auth_fields_raw = []
            auth_field_names = []
            if isinstance(auth_fields_raw, list):
                for f in auth_fields_raw:
                    if isinstance(f, dict) and f.get("name"):
                        auth_field_names.append(f["name"])
                    elif isinstance(f, str):
                        auth_field_names.append(f)
            connector = {
                "id": str(row["id"]),
                "name": row.get("name", ""),
                "description": row.get("description"),
                "connector_type": row.get("connector_type", "rest"),
                "definition": definition,
                "requires_auth": row.get("requires_auth", False),
                "auth_field_names": auth_field_names,
                "is_locked": row.get("is_locked", False),
                "category": row.get("category"),
                "tags": jsonb_list(row.get("tags")),
                "endpoint_count": len(definition.get("endpoints") or {}) if isinstance(definition.get("endpoints"), dict) else 0,
                "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
                "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
            }
            parts = [
                f"**{connector['name']}** (ID: {connector['id']})",
                f"Type: {connector['connector_type']}, Endpoints: {connector['endpoint_count']}",
            ]
            if connector.get("requires_auth"):
                parts.append(f"Auth: required (fields: {', '.join(connector.get('auth_field_names', []))})")
            return tool_service_pb2.GetDataConnectorResponse(
                success=True,
                connector_json=json.dumps(connector),
                formatted="\n".join(parts),
            )
        except Exception as e:
            logger.exception("GetDataConnector failed")
            return tool_service_pb2.GetDataConnectorResponse(
                success=False, error=str(e)
            )

    async def UpdateDataConnector(
        self,
        request: tool_service_pb2.UpdateDataConnectorRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateDataConnectorResponse:
        """Update a data connector (partial update)."""
        try:
            from services.database_manager.database_helpers import fetch_one, execute
            user_id = request.user_id or "system"
            connector_id = request.connector_id or ""
            if not connector_id:
                return tool_service_pb2.UpdateDataConnectorResponse(
                    success=False, error="connector_id required"
                )
            row = await fetch_one(
                "SELECT id, is_locked FROM data_source_connectors WHERE id = $1 AND user_id = $2",
                connector_id,
                user_id,
            )
            if not row:
                return tool_service_pb2.UpdateDataConnectorResponse(
                    success=False, error="Connector not found"
                )
            updates = {}
            if request.HasField("name"):
                updates["name"] = request.name
            if request.HasField("description"):
                updates["description"] = request.description
            if request.HasField("connector_type"):
                updates["connector_type"] = request.connector_type
            if request.HasField("definition_json"):
                updates["definition"] = request.definition_json
            if request.HasField("requires_auth"):
                updates["requires_auth"] = request.requires_auth
            if request.HasField("auth_fields_json"):
                updates["auth_fields"] = request.auth_fields_json
            if request.HasField("is_locked"):
                updates["is_locked"] = request.is_locked
            if not updates:
                formatted = "No updates provided."
                return tool_service_pb2.UpdateDataConnectorResponse(
                    success=True,
                    connector_id=connector_id,
                    formatted=formatted,
                )
            if row.get("is_locked") and set(updates.keys()) != {"is_locked"}:
                return tool_service_pb2.UpdateDataConnectorResponse(
                    success=False, error="Connector is locked; only lock toggle is allowed"
                )
            set_clauses = []
            args = []
            idx = 1
            jsonb_fields = ("definition", "auth_fields")
            for k, v in updates.items():
                if k in jsonb_fields:
                    set_clauses.append(f"{k} = ${idx}::jsonb")
                    args.append(v if isinstance(v, str) else json.dumps(v) if v is not None else "{}")
                else:
                    set_clauses.append(f"{k} = ${idx}")
                    args.append(v)
                idx += 1
            set_clauses.append("updated_at = NOW()")
            args.extend([connector_id, user_id])
            await execute(
                f"UPDATE data_source_connectors SET {', '.join(set_clauses)} WHERE id = ${idx} AND user_id = ${idx + 1}",
                *args,
            )
            formatted = f"Updated connector {connector_id}."
            return tool_service_pb2.UpdateDataConnectorResponse(
                success=True,
                connector_id=connector_id,
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("UpdateDataConnector failed")
            return tool_service_pb2.UpdateDataConnectorResponse(success=False, error=str(e))

