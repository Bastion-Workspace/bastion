"""Shared lazy singletons for document gRPC handler mixins (document-service)."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DocumentGrpcHandlerBase:
    """Provides _get_search_service, _get_document_repo, _get_embedding_manager for handler mixins."""

    def __init__(self) -> None:
        self._search_service: Optional[object] = None
        self._document_repo: Optional[object] = None
        self._embedding_manager: Optional[object] = None
        self._folder_service: Optional[object] = None

    async def _get_search_service(self):
        if not self._search_service:
            from ds_services.direct_search_service import DirectSearchService

            self._search_service = DirectSearchService()
        return self._search_service

    def _get_document_repo(self):
        if not self._document_repo:
            from ds_db.document_repository import DocumentRepository

            self._document_repo = DocumentRepository()
        return self._document_repo

    async def _get_embedding_manager(self):
        if not self._embedding_manager:
            from ds_services.embedding_service_wrapper import get_embedding_service

            self._embedding_manager = await get_embedding_service()
        return self._embedding_manager

    async def _get_folder_service(self):
        if not self._folder_service:
            from ds_services.folder_service import FolderService

            self._folder_service = FolderService()
            await self._folder_service.initialize()
        return self._folder_service
