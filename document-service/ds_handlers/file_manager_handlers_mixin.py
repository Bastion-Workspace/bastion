"""File placement and folder structure via file manager service."""

import json

from protos import document_service_pb2


class FileManagerHandlersMixin:
    async def PlaceFile(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_place_file(request)

    async def MoveFile(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_move_file(request)

    async def DeleteFile(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_delete_file(request)

    async def RenameFile(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_rename_file(request)

    async def CreateFolderStructure(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_create_folder_structure(request)

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
