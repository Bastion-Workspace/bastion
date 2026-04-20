"""Entity-linked document find (subset of former SearchUtilityHandlersMixin)."""

import logging

import grpc
from protos import tool_service_pb2

logger = logging.getLogger(__name__)


class DsEntitySearchMixin:
    async def FindDocumentsByEntities(
        self,
        request: tool_service_pb2.FindDocumentsByEntitiesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.FindDocumentsByEntitiesResponse:
        try:
            logger.info(
                "FindDocumentsByEntities: user=%s, entities=%s",
                request.user_id,
                list(request.entity_names),
            )
            kg_service = None
            if getattr(self, "_pipeline", None) and getattr(self._pipeline, "_parallel", None):
                kg_service = getattr(self._pipeline._parallel, "kg_service", None)
            if not kg_service or not kg_service.is_connected():
                return tool_service_pb2.FindDocumentsByEntitiesResponse(
                    document_ids=[],
                    total_count=0,
                )
            document_ids = await kg_service.find_documents_by_entities(list(request.entity_names))
            doc_repo = self._get_document_repo()
            accessible_doc_ids = []
            for doc_id in document_ids:
                doc = await doc_repo.get_document_by_id(
                    document_id=doc_id, user_id=request.user_id
                )
                if doc:
                    accessible_doc_ids.append(doc_id)
            return tool_service_pb2.FindDocumentsByEntitiesResponse(
                document_ids=accessible_doc_ids,
                total_count=len(accessible_doc_ids),
            )
        except Exception as e:
            logger.error("FindDocumentsByEntities failed: %s", e)
            await context.abort(grpc.StatusCode.INTERNAL, f"Entity search failed: {str(e)}")

    async def FindRelatedDocumentsByEntities(
        self,
        request: tool_service_pb2.FindRelatedDocumentsByEntitiesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.FindRelatedDocumentsByEntitiesResponse:
        try:
            logger.info(
                "FindRelatedDocumentsByEntities: user=%s, entities=%s, hops=%s",
                request.user_id,
                list(request.entity_names),
                request.max_hops,
            )
            kg_service = None
            if getattr(self, "_pipeline", None) and getattr(self._pipeline, "_parallel", None):
                kg_service = getattr(self._pipeline._parallel, "kg_service", None)
            if not kg_service or not kg_service.is_connected():
                return tool_service_pb2.FindRelatedDocumentsByEntitiesResponse(
                    document_ids=[],
                    total_count=0,
                )
            document_ids = await kg_service.find_related_documents_by_entities(
                list(request.entity_names),
                max_hops=request.max_hops or 2,
            )
            doc_repo = self._get_document_repo()
            accessible_doc_ids = []
            for doc_id in document_ids:
                doc = await doc_repo.get_document_by_id(
                    document_id=doc_id, user_id=request.user_id
                )
                if doc:
                    accessible_doc_ids.append(doc_id)
            return tool_service_pb2.FindRelatedDocumentsByEntitiesResponse(
                document_ids=accessible_doc_ids,
                total_count=len(accessible_doc_ids),
            )
        except Exception as e:
            logger.error("FindRelatedDocumentsByEntities failed: %s", e)
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Related entity search failed: {str(e)}"
            )
