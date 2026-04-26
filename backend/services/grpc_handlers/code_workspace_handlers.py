"""gRPC handlers for code workspace registry and indexed chunks."""

import json
import logging
from typing import Any, Dict, List

import grpc
from protos import tool_service_pb2

from services.database_manager.database_helpers import execute, fetch_all, fetch_one

logger = logging.getLogger(__name__)


def _rls(user_id: str) -> Dict[str, str]:
    return {"user_id": user_id, "user_role": "user"}


class CodeWorkspaceHandlersMixin:
    """Mixin: ListCodeWorkspaces, GetCodeWorkspace, UpsertCodeWorkspaceChunks, CodeSemanticSearch."""

    async def ListCodeWorkspaces(
        self,
        request: tool_service_pb2.ListCodeWorkspacesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListCodeWorkspacesResponse:
        try:
            uid = (request.user_id or "").strip()
            if not uid:
                return tool_service_pb2.ListCodeWorkspacesResponse(workspaces=[], total=0)
            rows = await fetch_all(
                """
                SELECT id::text, name, device_id, workspace_path, last_git_branch, updated_at
                FROM code_workspaces
                WHERE user_id = $1
                ORDER BY updated_at DESC NULLS LAST
                """,
                uid,
                rls_context=_rls(uid),
            )
            out: List[tool_service_pb2.CodeWorkspaceSummary] = []
            for r in rows or []:
                ts = r.get("updated_at")
                ts_s = ts.isoformat() if hasattr(ts, "isoformat") else str(ts or "")
                out.append(
                    tool_service_pb2.CodeWorkspaceSummary(
                        workspace_id=str(r.get("id") or ""),
                        name=str(r.get("name") or ""),
                        device_id=str(r.get("device_id") or ""),
                        workspace_path=str(r.get("workspace_path") or ""),
                        last_git_branch=str(r.get("last_git_branch") or ""),
                        updated_at=ts_s,
                    )
                )
            return tool_service_pb2.ListCodeWorkspacesResponse(workspaces=out, total=len(out))
        except Exception as e:
            logger.error("ListCodeWorkspaces failed: %s", e)
            return tool_service_pb2.ListCodeWorkspacesResponse(workspaces=[], total=0)

    async def GetCodeWorkspace(
        self,
        request: tool_service_pb2.GetCodeWorkspaceRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetCodeWorkspaceResponse:
        try:
            uid = (request.user_id or "").strip()
            wid = (request.workspace_id or "").strip()
            if not uid or not wid:
                return tool_service_pb2.GetCodeWorkspaceResponse(
                    success=False, error="user_id and workspace_id required"
                )
            r = await fetch_one(
                """
                SELECT id::text, user_id, name, device_id, device_name, workspace_path,
                       last_git_branch, last_file_tree, settings, conversation_id::text,
                       created_at, updated_at
                FROM code_workspaces
                WHERE id = $1::uuid AND user_id = $2
                """,
                wid,
                uid,
                rls_context=_rls(uid),
            )
            if not r:
                return tool_service_pb2.GetCodeWorkspaceResponse(success=False, error="not_found")
            lft = r.get("last_file_tree")
            lft_json = json.dumps(lft) if lft is not None else ""
            st = r.get("settings")
            st_json = json.dumps(st) if isinstance(st, dict) else (st or "{}")
            ca = r.get("created_at")
            ua = r.get("updated_at")
            detail = tool_service_pb2.CodeWorkspaceDetail(
                workspace_id=str(r.get("id") or ""),
                user_id=str(r.get("user_id") or ""),
                name=str(r.get("name") or ""),
                device_id=str(r.get("device_id") or ""),
                device_name=str(r.get("device_name") or ""),
                workspace_path=str(r.get("workspace_path") or ""),
                last_git_branch=str(r.get("last_git_branch") or ""),
                last_file_tree_json=lft_json,
                settings_json=st_json if isinstance(st_json, str) else "{}",
                conversation_id=str(r.get("conversation_id") or ""),
                created_at=ca.isoformat() if hasattr(ca, "isoformat") else str(ca or ""),
                updated_at=ua.isoformat() if hasattr(ua, "isoformat") else str(ua or ""),
            )
            return tool_service_pb2.GetCodeWorkspaceResponse(success=True, workspace=detail)
        except Exception as e:
            logger.error("GetCodeWorkspace failed: %s", e)
            return tool_service_pb2.GetCodeWorkspaceResponse(success=False, error=str(e))

    async def UpsertCodeWorkspaceChunks(
        self,
        request: tool_service_pb2.UpsertCodeWorkspaceChunksRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpsertCodeWorkspaceChunksResponse:
        from services import code_workspace_index_service as cidx

        try:
            uid = (request.user_id or "").strip()
            wid = (request.workspace_id or "").strip()
            if not uid or not wid:
                return tool_service_pb2.UpsertCodeWorkspaceChunksResponse(
                    success=False, inserted=0, embedded=0, error="user_id and workspace_id required"
                )
            ws = await fetch_one(
                "SELECT id::text FROM code_workspaces WHERE id = $1::uuid AND user_id = $2",
                wid,
                uid,
                rls_context=_rls(uid),
            )
            if not ws:
                return tool_service_pb2.UpsertCodeWorkspaceChunksResponse(
                    success=False, inserted=0, embedded=0, error="workspace_not_found"
                )
            rls = _rls(uid)
            if request.replace_workspace:
                await cidx.delete_workspace_vectors(wid)
                await execute(
                    "DELETE FROM code_chunks WHERE workspace_id = $1::uuid AND user_id = $2",
                    wid,
                    uid,
                    rls_context=rls,
                )

            inserted = 0
            chunk_ids_for_embed: List[Dict[str, Any]] = []
            for ch in request.chunks:
                fp = (ch.file_path or "").strip()
                if not fp:
                    continue
                row = await fetch_one(
                    """
                    INSERT INTO code_chunks (
                        user_id, workspace_id, file_path, chunk_index, start_line, end_line,
                        content, language, git_sha, content_tsv, embedding_pending
                    ) VALUES (
                        $1, $2::uuid, $3, $4, $5, $6, $7, $8, $9,
                        to_tsvector('english', coalesce($7, '')),
                        true
                    )
                    ON CONFLICT (workspace_id, file_path, chunk_index) DO UPDATE SET
                        start_line = EXCLUDED.start_line,
                        end_line = EXCLUDED.end_line,
                        content = EXCLUDED.content,
                        language = EXCLUDED.language,
                        git_sha = EXCLUDED.git_sha,
                        content_tsv = to_tsvector('english', coalesce(EXCLUDED.content, '')),
                        embedding_pending = true,
                        qdrant_point_id = NULL,
                        updated_at = NOW()
                    RETURNING id::text, user_id, workspace_id::text, file_path, start_line, end_line, content
                    """,
                    uid,
                    wid,
                    fp,
                    int(ch.chunk_index),
                    int(ch.start_line or 1),
                    int(ch.end_line or 1),
                    ch.content or "",
                    (ch.language or "").strip() or None,
                    (ch.git_sha or "").strip() or None,
                    rls_context=rls,
                )
                if row:
                    inserted += 1
                    chunk_ids_for_embed.append(
                        {
                            "id": row["id"],
                            "user_id": row["user_id"],
                            "workspace_id": row["workspace_id"],
                            "file_path": row["file_path"],
                            "start_line": row["start_line"],
                            "end_line": row["end_line"],
                            "content": row["content"],
                        }
                    )

            embedded = await cidx.embed_chunks_batch(chunk_ids_for_embed, rls)
            if inserted > 0:
                try:
                    from services.celery_tasks.code_workspace_tasks import (
                        embed_pending_code_chunks_task,
                    )

                    embed_pending_code_chunks_task.delay(wid, uid)
                except Exception as exc:
                    logger.debug("embed_pending schedule skipped: %s", exc)
            return tool_service_pb2.UpsertCodeWorkspaceChunksResponse(
                success=True, inserted=inserted, embedded=embedded
            )
        except Exception as e:
            logger.error("UpsertCodeWorkspaceChunks failed: %s", e)
            return tool_service_pb2.UpsertCodeWorkspaceChunksResponse(
                success=False, inserted=0, embedded=0, error=str(e)
            )

    async def CodeSemanticSearch(
        self,
        request: tool_service_pb2.CodeSemanticSearchRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CodeSemanticSearchResponse:
        from services import code_workspace_index_service as cidx

        try:
            uid = (request.user_id or "").strip()
            wid = (request.workspace_id or "").strip()
            q = (request.query or "").strip()
            lim = int(request.limit or 20)
            if lim < 1:
                lim = 20
            if lim > 100:
                lim = 100
            if not uid or not wid or not q:
                return tool_service_pb2.CodeSemanticSearchResponse(
                    success=False, error="user_id, workspace_id, and query required"
                )
            ws = await fetch_one(
                "SELECT id::text FROM code_workspaces WHERE id = $1::uuid AND user_id = $2",
                wid,
                uid,
                rls_context=_rls(uid),
            )
            if not ws:
                return tool_service_pb2.CodeSemanticSearchResponse(success=False, error="workspace_not_found")

            hits = await cidx.hybrid_code_search(
                user_id=uid,
                workspace_id=wid,
                query=q,
                limit=lim,
                file_glob=request.file_glob or "",
            )
            out = []
            for h in hits:
                out.append(
                    tool_service_pb2.CodeSemanticSearchHit(
                        chunk_id=str(h.get("chunk_id") or ""),
                        file_path=str(h.get("file_path") or ""),
                        start_line=int(h.get("start_line") or 1),
                        end_line=int(h.get("end_line") or 1),
                        snippet=str(h.get("snippet") or ""),
                        score=float(h.get("score") or 0.0),
                    )
                )
            return tool_service_pb2.CodeSemanticSearchResponse(success=True, hits=out)
        except Exception as e:
            logger.error("CodeSemanticSearch failed: %s", e)
            return tool_service_pb2.CodeSemanticSearchResponse(success=False, error=str(e))
