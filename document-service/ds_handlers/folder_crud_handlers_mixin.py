"""Folder CRUD, exemptions, document delete/store via folder service and service container."""

import asyncio
import json
import logging

from protos import document_service_pb2

logger = logging.getLogger(__name__)


class FolderCrudHandlersMixin:
    async def CreateFolder(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_create_folder(request)

    async def UpdateFolder(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_update_folder(request)

    async def DeleteFolder(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_delete_folder(request)

    async def MoveFolder(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_move_folder(request)

    async def GetFolderContents(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_get_folder_contents(request)

    async def GetFolderContentsBatch(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_get_folder_contents_batch(request)

    async def UpdateFolderMetadata(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_update_folder_metadata(request)

    async def ExemptFolder(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_exempt_folder(request)

    async def RemoveFolderExemption(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_remove_folder_exemption(request)

    async def OverrideFolderExemption(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_override_folder_exemption(request)

    async def ExemptDocument(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_exempt_document(request)

    async def RemoveDocumentExemption(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_remove_document_exemption(request)

    async def DeleteDocument(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_delete_document(request)

    async def StoreTextDocument(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_store_text_document(request)

    async def _json_create_folder(self, request: document_service_pb2.JsonToolRequest):
        """Payload mirrors folder_api create (simplified): name, parent_folder_id, collection_type, role, team_id."""
        try:
            p = json.loads(request.payload_json or "{}")
            fs = await self._get_folder_service()
            folder = await fs.create_folder(
                name=p.get("name") or p.get("folder_name"),
                parent_folder_id=p.get("parent_folder_id"),
                user_id=p.get("user_id") or request.user_id,
                collection_type=p.get("collection_type", "user"),
                current_user_role=p.get("current_user_role", "user"),
                admin_user_id=p.get("admin_user_id"),
                team_id=p.get("team_id"),
            )
            return document_service_pb2.JsonToolResponse(
                success=True,
                result_json=json.dumps({"folder_id": folder.folder_id, "name": folder.name}),
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_update_folder(self, request: document_service_pb2.JsonToolRequest):
        from ds_models.api_models import FolderUpdateRequest

        try:
            p = json.loads(request.payload_json or "{}")
            fs = await self._get_folder_service()
            upd = FolderUpdateRequest.model_validate(p.get("body", p))
            folder = await fs.update_folder(
                p["folder_id"], upd, p.get("user_id") or request.user_id, p.get("role", "user")
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=folder.model_dump_json() if folder else "{}"
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_delete_folder(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            fs = await self._get_folder_service()
            ok = await fs.delete_folder(
                p["folder_id"],
                p.get("user_id") or request.user_id,
                p.get("recursive", False),
                p.get("role", "user"),
                allow_team_root=bool(p.get("allow_team_root", False)),
            )
            return document_service_pb2.JsonToolResponse(
                success=ok, result_json=json.dumps({"success": ok})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_move_folder(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            fs = await self._get_folder_service()
            ok = await fs.move_folder(
                p["folder_id"],
                p["new_parent_id"],
                p.get("user_id") or request.user_id,
                p.get("role", "user"),
            )
            return document_service_pb2.JsonToolResponse(
                success=ok, result_json=json.dumps({"success": ok})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_get_folder_contents(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            fs = await self._get_folder_service()
            contents = await fs.get_folder_contents(
                p["folder_id"],
                p.get("user_id") or request.user_id,
                limit=int(p.get("limit", 250)),
                offset=int(p.get("offset", 0)),
            )
            if contents is None:
                return document_service_pb2.JsonToolResponse(
                    success=False, error="not_found_or_access_denied"
                )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=contents.model_dump_json()
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_get_folder_contents_batch(self, request: document_service_pb2.JsonToolRequest):
        """Load multiple folders with bounded concurrency. Payload: folder_ids, optional limit, offset, max_concurrent."""
        max_batch = 100
        default_concurrent = 12
        max_concurrent_cap = 32

        try:
            p = json.loads(request.payload_json or "{}")
            raw_ids = p.get("folder_ids")
            if not raw_ids or not isinstance(raw_ids, list):
                return document_service_pb2.JsonToolResponse(
                    success=False,
                    error="folder_ids is required and must be a non-empty array",
                )

            user_id = p.get("user_id") or request.user_id
            limit = int(p.get("limit", 250))
            offset = int(p.get("offset", 0))
            mc = int(p.get("max_concurrent", default_concurrent))
            mc = max(1, min(mc, max_concurrent_cap))

            folder_ids = []
            seen = set()
            for x in raw_ids:
                if not isinstance(x, str) or not x.strip():
                    continue
                fid = x.strip()
                if fid in seen:
                    continue
                seen.add(fid)
                folder_ids.append(fid)

            if not folder_ids:
                return document_service_pb2.JsonToolResponse(
                    success=False,
                    error="folder_ids contains no valid folder IDs",
                )
            if len(folder_ids) > max_batch:
                return document_service_pb2.JsonToolResponse(
                    success=False,
                    error=f"folder_ids exceeds maximum batch size ({max_batch})",
                )

            fs = await self._get_folder_service()
            sem = asyncio.Semaphore(mc)

            async def fetch_one(folder_id: str):
                async with sem:
                    try:
                        contents = await fs.get_folder_contents(
                            folder_id, user_id, limit=limit, offset=offset
                        )
                        if contents is None:
                            return folder_id, None, "not_found_or_access_denied"
                        return folder_id, json.loads(contents.model_dump_json()), None
                    except Exception as e:
                        logger.warning(
                            "GetFolderContentsBatch failed for folder_id=%s: %s",
                            folder_id,
                            e,
                        )
                        return folder_id, None, str(e)

            results = await asyncio.gather(*[fetch_one(fid) for fid in folder_ids])
            out_contents: dict = {}
            out_errors: dict = {}
            for fid, data, err in results:
                if err:
                    out_errors[fid] = err
                elif data is not None:
                    out_contents[fid] = data
            return document_service_pb2.JsonToolResponse(
                success=True,
                result_json=json.dumps({"contents": out_contents, "errors": out_errors}),
            )
        except Exception as e:
            logger.exception("GetFolderContentsBatch failed")
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_update_folder_metadata(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            fs = await self._get_folder_service()
            cat = p.get("category")
            cat_val = cat if isinstance(cat, str) else getattr(cat, "value", None)
            ok = await fs.update_folder_metadata(
                p["folder_id"],
                category=cat_val,
                tags=p.get("tags"),
                inherit_tags=p.get("inherit_tags"),
            )
            return document_service_pb2.JsonToolResponse(
                success=ok, result_json=json.dumps({"success": ok})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_exempt_folder(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            fs = await self._get_folder_service()
            ok = await fs.exempt_folder_from_vectorization(
                p["folder_id"], p.get("user_id") or request.user_id, p.get("role", "user")
            )
            return document_service_pb2.JsonToolResponse(
                success=ok, result_json=json.dumps({"success": ok})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_remove_folder_exemption(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            fs = await self._get_folder_service()
            ok = await fs.remove_folder_exemption(
                p["folder_id"], p.get("user_id") or request.user_id, p.get("role", "user")
            )
            return document_service_pb2.JsonToolResponse(
                success=ok, result_json=json.dumps({"success": ok})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_override_folder_exemption(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            fs = await self._get_folder_service()
            ok = await fs.override_folder_exemption(
                p["folder_id"], p.get("user_id") or request.user_id, p.get("role", "user")
            )
            return document_service_pb2.JsonToolResponse(
                success=ok, result_json=json.dumps({"success": ok})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_exempt_document(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from shims.services.service_container import get_service_container

            c = await get_service_container()
            ds = c.document_service
            ok = await ds.exempt_document_from_vectorization(
                p["document_id"], p.get("user_id") or request.user_id
            )
            return document_service_pb2.JsonToolResponse(
                success=bool(ok), result_json=json.dumps({"success": bool(ok)})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_remove_document_exemption(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from shims.services.service_container import get_service_container

            c = await get_service_container()
            ds = c.document_service
            ok = await ds.remove_document_exemption(
                p["document_id"],
                p.get("user_id") or request.user_id,
                inherit=bool(p.get("inherit", False)),
            )
            return document_service_pb2.JsonToolResponse(
                success=bool(ok), result_json=json.dumps({"success": bool(ok)})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_delete_document(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from shims.services.service_container import get_service_container

            c = await get_service_container()
            ds = c.document_service
            ok = await ds.delete_document(p["document_id"], p.get("user_id") or request.user_id)
            return document_service_pb2.JsonToolResponse(
                success=bool(ok), result_json=json.dumps({"success": bool(ok)})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_store_text_document(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from shims.services.service_container import get_service_container

            c = await get_service_container()
            ds = c.document_service
            doc_id = p.get("doc_id") or p.get("document_id")
            if not doc_id:
                return document_service_pb2.JsonToolResponse(
                    success=False, error="doc_id or document_id required"
                )
            ok = await ds.store_text_document(
                doc_id,
                p.get("content", ""),
                p.get("metadata") or {},
                filename=p.get("filename"),
                user_id=p.get("user_id") or request.user_id,
                collection_type=p.get("collection_type", "user"),
                folder_id=p.get("folder_id"),
                file_path=p.get("file_path"),
                team_id=p.get("team_id"),
            )
            return document_service_pb2.JsonToolResponse(
                success=bool(ok), result_json=json.dumps({"success": bool(ok), "document_id": doc_id})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))
