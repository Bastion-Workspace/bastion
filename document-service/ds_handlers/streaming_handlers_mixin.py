"""Streaming download, library zip, collaborative document flush."""

import logging
import mimetypes
from pathlib import Path
from typing import AsyncIterator

import grpc

from protos import document_service_pb2

logger = logging.getLogger(__name__)


class StreamingHandlersMixin:
    async def DownloadDocument(
        self,
        request: document_service_pb2.DownloadDocumentRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[document_service_pb2.FileDownloadChunk]:
        """Stream file bytes for a document (plaintext / binary)."""
        from ds_config import settings

        repo = self._get_document_repo()
        fs = await self._get_folder_service()
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
        fs = await self._get_folder_service()
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

        try:
            container = await get_service_container()
            try:
                _ = container.document_service
            except RuntimeError:
                pds = ParallelDocumentService()
                await pds.initialize()
                fs = await self._get_folder_service()
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
