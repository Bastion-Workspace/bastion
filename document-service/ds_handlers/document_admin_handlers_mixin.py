"""Typed gRPC handlers for document admin / facade operations (former DocumentMirror actions)."""

import json
import logging

import grpc

from protos import document_service_pb2

from ds_handlers.document_admin_ops import (
    op_bulk_categorize_documents,
    op_check_document_exists,
    op_cleanup_orphaned_embeddings,
    op_delete_document_database_only,
    op_delete_zip_with_children,
    op_filter_documents,
    op_get_document,
    op_get_document_categories_overview,
    op_get_documents_stats,
    op_get_documents_with_hierarchy,
    op_get_document_status,
    op_get_duplicate_documents,
    op_get_zip_hierarchy,
    op_import_from_url,
    op_list_documents,
    op_process_url_async,
    op_remove_team_upload_directory,
    op_resume_incomplete_processing,
    op_update_document_metadata,
)

logger = logging.getLogger(__name__)


class DocumentAdminHandlersMixin:
    def _admin_ok(self, result: dict) -> document_service_pb2.JsonToolResponse:
        return document_service_pb2.JsonToolResponse(
            success=True,
            result_json=json.dumps(result, default=str),
        )

    def _admin_err(self, err: str) -> document_service_pb2.JsonToolResponse:
        return document_service_pb2.JsonToolResponse(success=False, error=err)

    async def AdminGetDocument(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_get_document(p, uid))
        except Exception as e:
            logger.exception("AdminGetDocument failed")
            return self._admin_err(str(e))

    async def CheckDocumentExists(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_check_document_exists(p, uid))
        except Exception as e:
            logger.exception("CheckDocumentExists failed")
            return self._admin_err(str(e))

    async def ListDocuments(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_list_documents(p, uid))
        except Exception as e:
            logger.exception("ListDocuments failed")
            return self._admin_err(str(e))

    async def FilterDocuments(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_filter_documents(p, uid))
        except Exception as e:
            logger.exception("FilterDocuments failed")
            return self._admin_err(str(e))

    async def GetDocumentStatus(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_get_document_status(p, uid))
        except Exception as e:
            logger.exception("GetDocumentStatus admin failed")
            return self._admin_err(str(e))

    async def UpdateDocumentMetadataAdmin(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_update_document_metadata(p, uid))
        except Exception as e:
            logger.exception("UpdateDocumentMetadataAdmin failed")
            return self._admin_err(str(e))

    async def BulkCategorizeDocuments(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_bulk_categorize_documents(p, uid))
        except Exception as e:
            logger.exception("BulkCategorizeDocuments failed")
            return self._admin_err(str(e))

    async def DeleteDocumentDatabaseOnly(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_delete_document_database_only(p, uid))
        except Exception as e:
            logger.exception("DeleteDocumentDatabaseOnly failed")
            return self._admin_err(str(e))

    async def CleanupOrphanedEmbeddings(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_cleanup_orphaned_embeddings(p, uid))
        except Exception as e:
            logger.exception("CleanupOrphanedEmbeddings failed")
            return self._admin_err(str(e))

    async def GetDuplicateDocuments(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_get_duplicate_documents(p, uid))
        except Exception as e:
            logger.exception("GetDuplicateDocuments failed")
            return self._admin_err(str(e))

    async def GetDocumentsStats(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_get_documents_stats(p, uid))
        except Exception as e:
            logger.exception("GetDocumentsStats failed")
            return self._admin_err(str(e))

    async def GetDocumentCategoriesOverview(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_get_document_categories_overview(p, uid))
        except Exception as e:
            logger.exception("GetDocumentCategoriesOverview failed")
            return self._admin_err(str(e))

    async def ImportFromUrl(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_import_from_url(p, uid))
        except Exception as e:
            logger.exception("ImportFromUrl failed")
            return self._admin_err(str(e))

    async def GetDocumentsWithHierarchy(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_get_documents_with_hierarchy(p, uid))
        except Exception as e:
            logger.exception("GetDocumentsWithHierarchy failed")
            return self._admin_err(str(e))

    async def GetZipHierarchy(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_get_zip_hierarchy(p, uid))
        except Exception as e:
            logger.exception("GetZipHierarchy failed")
            return self._admin_err(str(e))

    async def DeleteZipWithChildren(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_delete_zip_with_children(p, uid))
        except Exception as e:
            logger.exception("DeleteZipWithChildren failed")
            return self._admin_err(str(e))

    async def ProcessUrlAsync(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_process_url_async(p, uid))
        except Exception as e:
            logger.exception("ProcessUrlAsync failed")
            return self._admin_err(str(e))

    async def ResumeIncompleteProcessing(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            return self._admin_ok(await op_resume_incomplete_processing(p, uid))
        except Exception as e:
            logger.exception("ResumeIncompleteProcessing failed")
            return self._admin_err(str(e))

    async def RemoveTeamUploadDirectory(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        try:
            p = json.loads(request.payload_json or "{}")
            uid = (p.get("user_id") or request.user_id or "").strip()
            fs = await self._get_folder_service()
            return self._admin_ok(await op_remove_team_upload_directory(p, uid, folder_service=fs))
        except ValueError as e:
            return self._admin_err(str(e))
        except Exception as e:
            logger.exception("RemoveTeamUploadDirectory failed")
            return self._admin_err(str(e))
