"""Phase 2: streaming download, collab flush, JSON-bridged file/folder/version/encryption APIs."""

import asyncio
import json
import logging
import mimetypes
from pathlib import Path
from typing import AsyncIterator

import grpc

from protos import document_service_pb2

logger = logging.getLogger(__name__)


class Phase2HandlersMixin:
    async def DownloadDocument(
        self,
        request: document_service_pb2.DownloadDocumentRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[document_service_pb2.FileDownloadChunk]:
        """Stream file bytes for a document (plaintext / binary)."""
        from ds_config import settings
        from ds_db.document_repository import DocumentRepository
        from ds_services.folder_service import FolderService

        repo = DocumentRepository()
        fs = FolderService()
        await fs.initialize()
        try:
            doc = await repo.get_by_id(request.document_id)
            if not doc:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Document not found")
                return
            filename = getattr(doc, "filename", None)
            user_id = request.user_id or getattr(doc, "user_id", None)
            folder_id = getattr(doc, "folder_id", None)
            collection_type = getattr(doc, "collection_type", None) or "user"
            team_id = getattr(doc, "team_id", None)
            if team_id is not None and not isinstance(team_id, str):
                team_id = str(team_id)
            path_str = await fs.get_document_file_path(
                filename=filename,
                folder_id=folder_id,
                user_id=user_id,
                collection_type=collection_type,
                team_id=team_id,
            )
            fp = Path(path_str)
            if not fp.exists():
                alt = Path(settings.UPLOAD_DIR) / f"{request.document_id}_{filename}"
                fp = alt if alt.exists() else fp
            if not fp.exists():
                await context.abort(grpc.StatusCode.NOT_FOUND, "File not on disk")
                return
            ctype, _ = mimetypes.guess_type(str(fp))
            if not ctype:
                ctype = "application/octet-stream"
            chunk_size = 256 * 1024
            with open(fp, "rb") as f:
                while True:
                    blob = f.read(chunk_size)
                    if not blob:
                        yield document_service_pb2.FileDownloadChunk(
                            data=b"",
                            filename=filename or fp.name,
                            content_type=ctype,
                            done=True,
                        )
                        break
                    yield document_service_pb2.FileDownloadChunk(
                        data=blob,
                        filename=filename or fp.name,
                        content_type=ctype,
                        done=False,
                    )
        except grpc.aio.AioRpcError:
            raise
        except Exception as e:
            logger.exception("DownloadDocument failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GenerateLibraryZip(
        self,
        request: document_service_pb2.GenerateLibraryZipRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[document_service_pb2.FileDownloadChunk]:
        from ds_services.folder_service import FolderService

        fs = FolderService()
        await fs.initialize()
        try:
            data = await fs.build_library_zip(request.user_id)
            yield document_service_pb2.FileDownloadChunk(
                data=data,
                filename="my-library.zip",
                content_type="application/zip",
                done=True,
            )
        except Exception as e:
            logger.exception("GenerateLibraryZip failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def FlushCollaborativeDocument(
        self,
        request: document_service_pb2.FlushCollaborativeDocumentRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.FlushCollaborativeDocumentResponse:
        from shims.services.service_container import (
            get_service_container,
            set_document_services,
        )
        from ds_services.collab_persist import flush_collaborative_document
        from ds_services.parallel_document_service import ParallelDocumentService
        from ds_services.folder_service import FolderService

        try:
            container = await get_service_container()
            try:
                _ = container.document_service
            except RuntimeError:
                pds = ParallelDocumentService()
                await pds.initialize()
                fs = FolderService()
                await fs.initialize()
                set_document_services(pds, fs)
            await flush_collaborative_document(
                request.document_id, request.new_content or ""
            )
            return document_service_pb2.FlushCollaborativeDocumentResponse(
                success=True, message="ok"
            )
        except ValueError as e:
            return document_service_pb2.FlushCollaborativeDocumentResponse(
                success=False, error=str(e), message=str(e)
            )
        except Exception as e:
            logger.exception("FlushCollaborativeDocument failed")
            return document_service_pb2.FlushCollaborativeDocumentResponse(
                success=False, error=str(e), message=str(e)
            )

    async def PlaceFile(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_place_file(request)

    async def MoveFile(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_move_file(request)

    async def DeleteFile(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_delete_file(request)

    async def RenameFile(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_rename_file(request)

    async def CreateFolderStructure(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_create_folder_structure(request)

    async def CreateFolder(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_create_folder(request)

    async def UpdateFolder(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_update_folder(request)

    async def DeleteFolder(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_delete_folder(request)

    async def MoveFolder(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_move_folder(request)

    async def GetFolderContents(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_get_folder_contents(request)

    async def GetFolderContentsBatch(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_get_folder_contents_batch(request)

    async def UpdateFolderMetadata(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_update_folder_metadata(request)

    async def ExemptFolder(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_exempt_folder(request)

    async def RemoveFolderExemption(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_remove_folder_exemption(request)

    async def OverrideFolderExemption(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_override_folder_exemption(request)

    async def ExemptDocument(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_exempt_document(request)

    async def RemoveDocumentExemption(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_remove_document_exemption(request)

    async def DeleteDocument(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_delete_document(request)

    async def StoreTextDocument(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_store_text_document(request)

    async def GetDocumentVersions(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_get_document_versions(request)

    async def GetVersionContent(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_get_version_content(request)

    async def DiffVersions(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_diff_versions(request)

    async def RollbackToVersion(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_rollback_to_version(request)

    async def EncryptDocument(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_encrypt_document(request)

    async def CreateDecryptSession(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_create_decrypt_session(request)

    async def TryDecrypt(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_try_decrypt(request)

    async def EncryptionHeartbeat(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_encryption_heartbeat(request)

    async def EncryptionLock(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_encryption_lock(request)

    async def ChangeEncryptionPassword(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_change_encryption_password(request)

    async def RemoveEncryption(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_remove_encryption(request)

    async def WriteEncryptedContentFromSession(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_write_encrypted_content_from_session(request)

    async def ReadUploadRelativeFile(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_read_upload_relative_file(request)

    async def WriteUploadRelativeFile(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_write_upload_relative_file(request)

    async def ListUploadRelativeDir(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_list_upload_relative_dir(request)

    async def ScanAndRecoverFiles(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_scan_recover(request)

    # --- internals ---

    async def _get_fm(self):
        from ds_services.file_manager_service import get_file_manager

        fm = await get_file_manager()
        if not fm._initialized:
            await fm.initialize()
        return fm

    async def _json_place_file(self, request: document_service_pb2.JsonToolRequest):
        from ds_services.models.file_placement_models import FilePlacementRequest

        try:
            data = json.loads(request.payload_json or "{}")
            req = FilePlacementRequest.model_validate(data)
            fm = await self._get_fm()
            resp = await fm.place_file(req)
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=resp.model_dump_json()
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_move_file(self, request: document_service_pb2.JsonToolRequest):
        from ds_services.models.file_placement_models import FileMoveRequest

        try:
            data = json.loads(request.payload_json or "{}")
            req = FileMoveRequest.model_validate(data)
            fm = await self._get_fm()
            resp = await fm.move_file(req)
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=resp.model_dump_json()
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_delete_file(self, request: document_service_pb2.JsonToolRequest):
        from ds_services.models.file_placement_models import FileDeleteRequest

        try:
            data = json.loads(request.payload_json or "{}")
            req = FileDeleteRequest.model_validate(data)
            fm = await self._get_fm()
            resp = await fm.delete_file(req)
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=resp.model_dump_json()
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_rename_file(self, request: document_service_pb2.JsonToolRequest):
        from ds_services.models.file_placement_models import FileRenameRequest

        try:
            data = json.loads(request.payload_json or "{}")
            req = FileRenameRequest.model_validate(data)
            fm = await self._get_fm()
            resp = await fm.rename_file(req)
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=resp.model_dump_json()
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_create_folder_structure(self, request: document_service_pb2.JsonToolRequest):
        from ds_services.models.file_placement_models import FolderStructureRequest

        try:
            data = json.loads(request.payload_json or "{}")
            req = FolderStructureRequest.model_validate(data)
            fm = await self._get_fm()
            resp = await fm.create_folder_structure(req)
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=resp.model_dump_json()
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_create_folder(self, request: document_service_pb2.JsonToolRequest):
        """Payload mirrors folder_api create (simplified): name, parent_folder_id, collection_type, role, team_id."""
        from ds_services.folder_service import FolderService

        try:
            p = json.loads(request.payload_json or "{}")
            fs = FolderService()
            await fs.initialize()
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
        from ds_services.folder_service import FolderService

        try:
            p = json.loads(request.payload_json or "{}")
            fs = FolderService()
            await fs.initialize()
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
        from ds_services.folder_service import FolderService

        try:
            p = json.loads(request.payload_json or "{}")
            fs = FolderService()
            await fs.initialize()
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
        from ds_services.folder_service import FolderService

        try:
            p = json.loads(request.payload_json or "{}")
            fs = FolderService()
            await fs.initialize()
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
        from ds_services.folder_service import FolderService

        try:
            p = json.loads(request.payload_json or "{}")
            fs = FolderService()
            await fs.initialize()
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
        from ds_services.folder_service import FolderService

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

            fs = FolderService()
            await fs.initialize()
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
        from ds_services.folder_service import FolderService

        try:
            p = json.loads(request.payload_json or "{}")
            fs = FolderService()
            await fs.initialize()
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
        from ds_services.folder_service import FolderService

        try:
            p = json.loads(request.payload_json or "{}")
            fs = FolderService()
            await fs.initialize()
            ok = await fs.exempt_folder_from_vectorization(
                p["folder_id"], p.get("user_id") or request.user_id, p.get("role", "user")
            )
            return document_service_pb2.JsonToolResponse(
                success=ok, result_json=json.dumps({"success": ok})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_remove_folder_exemption(self, request: document_service_pb2.JsonToolRequest):
        from ds_services.folder_service import FolderService

        try:
            p = json.loads(request.payload_json or "{}")
            fs = FolderService()
            await fs.initialize()
            ok = await fs.remove_folder_exemption(
                p["folder_id"], p.get("user_id") or request.user_id, p.get("role", "user")
            )
            return document_service_pb2.JsonToolResponse(
                success=ok, result_json=json.dumps({"success": ok})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_override_folder_exemption(self, request: document_service_pb2.JsonToolRequest):
        from ds_services.folder_service import FolderService

        try:
            p = json.loads(request.payload_json or "{}")
            fs = FolderService()
            await fs.initialize()
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

    async def _json_get_document_versions(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import document_version_service as dvs

            versions = await dvs.list_versions(
                p["document_id"],
                skip=int(p.get("skip", 0)),
                limit=int(p.get("limit", 100)),
                user_id=p.get("user_id") or request.user_id,
                collection_type=p.get("collection_type", "user"),
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"versions": versions})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_get_version_content(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from uuid import UUID
            from ds_services import document_version_service as dvs

            content = await dvs.get_version_content(
                UUID(p["version_id"]),
                user_id=p.get("user_id") or request.user_id,
                collection_type=p.get("collection_type", "user"),
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"content": content})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_diff_versions(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from uuid import UUID
            from ds_services import document_version_service as dvs

            result = await dvs.diff_versions(
                p["document_id"],
                UUID(p["from_version"]),
                UUID(p["to_version"]),
                user_id=p.get("user_id") or request.user_id,
                collection_type=p.get("collection_type", "user"),
            )
            return document_service_pb2.JsonToolResponse(
                success=result is not None, result_json=json.dumps(result or {})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_rollback_to_version(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from uuid import UUID
            from ds_services import document_version_service as dvs

            result = await dvs.rollback_to_version(
                p["document_id"], UUID(p["version_id"]), p.get("user_id") or request.user_id
            )
            return document_service_pb2.JsonToolResponse(
                success=bool(result.get("success")),
                result_json=json.dumps(result),
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_encrypt_document(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import file_encryption_service as enc
            from shims.services.service_container import get_service_container

            c = await get_service_container()
            await enc.encrypt_document(
                c.document_service,
                c.folder_service,
                p["document_id"],
                p["password"],
                p.get("confirm_password") or p["password"],
                p.get("user_id") or request.user_id,
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"ok": True})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_create_decrypt_session(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import file_encryption_service as enc
            from shims.services.service_container import get_service_container

            c = await get_service_container()
            plaintext, session_token = await enc.create_decrypt_session(
                c.document_service,
                c.folder_service,
                p["document_id"],
                p["password"],
                p.get("user_id") or request.user_id,
            )
            return document_service_pb2.JsonToolResponse(
                success=True,
                result_json=json.dumps(
                    {"content": plaintext, "session_token": session_token}
                ),
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_try_decrypt(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import file_encryption_service as enc
            from shims.services.service_container import get_service_container

            c = await get_service_container()
            uid = p.get("user_id") or request.user_id
            r = await enc.try_decrypt_content_with_session(
                c.document_service,
                c.folder_service,
                p["document_id"],
                uid,
                p.get("session_token", ""),
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"content": r})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_encryption_heartbeat(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import file_encryption_service as enc

            uid = p.get("user_id") or request.user_id
            r = await enc.heartbeat_session(
                p["document_id"], uid, p.get("session_token", "")
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"ttl_seconds": r})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_encryption_lock(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import file_encryption_service as enc

            await enc.lock_document(p["document_id"], p.get("user_id") or request.user_id)
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"ok": True})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_change_encryption_password(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import file_encryption_service as enc
            from shims.services.service_container import get_service_container

            c = await get_service_container()
            await enc.change_encryption_password(
                c.document_service,
                c.folder_service,
                p["document_id"],
                p["old_password"],
                p["new_password"],
                p.get("user_id") or request.user_id,
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"ok": True})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_remove_encryption(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import file_encryption_service as enc
            from shims.services.service_container import get_service_container

            c = await get_service_container()
            await enc.remove_encryption(
                c.document_service,
                c.folder_service,
                p["document_id"],
                p["password"],
                p.get("user_id") or request.user_id,
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"ok": True})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_write_encrypted_content_from_session(
        self, request: document_service_pb2.JsonToolRequest
    ):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import file_encryption_service as enc
            from shims.services.service_container import get_service_container

            c = await get_service_container()
            await enc.write_encrypted_content_from_session(
                c.document_service,
                c.folder_service,
                p["document_id"],
                p.get("user_id") or request.user_id,
                p.get("session_token", ""),
                p.get("content", ""),
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"ok": True})
            )
        except PermissionError as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    def _safe_path_under_upload(self, rel_path: str) -> Path:
        from ds_config import settings

        root = Path(settings.UPLOAD_DIR).resolve()
        p = Path(rel_path or "")
        if p.is_absolute():
            raise ValueError("absolute path not allowed")
        full = (root / p).resolve()
        try:
            full.relative_to(root)
        except ValueError as e:
            raise ValueError("path escapes upload root") from e
        return full

    async def _json_read_upload_relative_file(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            fp = self._safe_path_under_upload(p["rel_path"])
            if not fp.is_file():
                return document_service_pb2.JsonToolResponse(success=False, error="not found")
            text = fp.read_text(encoding="utf-8")
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"content": text})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_write_upload_relative_file(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            fp = self._safe_path_under_upload(p["rel_path"])
            if p.get("delete"):
                if not fp.is_file():
                    return document_service_pb2.JsonToolResponse(success=False, error="not found")
                fp.unlink()
                return document_service_pb2.JsonToolResponse(
                    success=True, result_json=json.dumps({"ok": True, "deleted": True})
                )
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(p.get("content", ""), encoding="utf-8")
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"ok": True})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_list_upload_relative_dir(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            dp = self._safe_path_under_upload(p.get("rel_path", ""))
            if not dp.is_dir():
                return document_service_pb2.JsonToolResponse(success=False, error="not a directory")
            names = sorted([x.name for x in dp.iterdir()])
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"entries": names})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_scan_recover(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services.file_recovery_service import FileRecoveryService

            svc = FileRecoveryService()
            result = await svc.scan_and_recover_user_files(
                p.get("user_id") or request.user_id,
                dry_run=p.get("dry_run", False),
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps(result)
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))
