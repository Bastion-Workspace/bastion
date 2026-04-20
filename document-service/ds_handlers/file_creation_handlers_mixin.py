"""gRPC handlers for File Creation operations."""

import logging
import random
from typing import Optional

import grpc
from protos import tool_service_pb2

logger = logging.getLogger(__name__)


class FileCreationHandlersMixin:
    """Mixin providing File Creation gRPC handlers.

    Mixed into ToolServiceImplementation; accesses self._get_search_service(),
    self._get_document_repo(), etc. via standard Python MRO.
    """

    # ===== File Creation Operations =====
    
    async def CreateUserFile(
        self,
        request: tool_service_pb2.CreateUserFileRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.CreateUserFileResponse:
        """Create a file in the user's My Documents section"""
        try:
            logger.info(f"CreateUserFile: user={request.user_id}, filename={request.filename}")
            
            # Import file creation tool
            from ds_langgraph_tools.file_creation_tools import create_user_file
            
            # Execute file creation
            result = await create_user_file(
                filename=request.filename,
                content=request.content,
                folder_id=request.folder_id if request.folder_id else None,
                folder_path=request.folder_path if request.folder_path else None,
                title=request.title if request.title else None,
                tags=list(request.tags) if request.tags else None,
                category=request.category if request.category else None,
                user_id=request.user_id,
                content_bytes=bytes(request.binary_content) if request.binary_content else None,
            )
            
            # Build response
            if result.get("success"):
                response = tool_service_pb2.CreateUserFileResponse(
                    success=True,
                    document_id=result.get("document_id", ""),
                    filename=result.get("filename", request.filename),
                    folder_id=result.get("folder_id", ""),
                    message=result.get("message", "File created successfully")
                )
                logger.info(f"CreateUserFile: Success - {response.document_id}")
            else:
                response = tool_service_pb2.CreateUserFileResponse(
                    success=False,
                    message=result.get("message", "File creation failed"),
                    error=result.get("error", "Unknown error")
                )
                logger.warning(f"CreateUserFile: Failed - {response.error}")
            
            return response
            
        except Exception as e:
            logger.error(f"CreateUserFile error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"File creation failed: {str(e)}")
    
    async def CreateUserFolder(
        self,
        request: tool_service_pb2.CreateUserFolderRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.CreateUserFolderResponse:
        """Create a folder in the user's My Documents section"""
        try:
            logger.info(f"CreateUserFolder: user={request.user_id}, folder_name={request.folder_name}")
            
            # Import folder creation tool
            from ds_langgraph_tools.file_creation_tools import create_user_folder
            
            # Execute folder creation
            result = await create_user_folder(
                folder_name=request.folder_name,
                parent_folder_id=request.parent_folder_id if request.parent_folder_id else None,
                parent_folder_path=request.parent_folder_path if request.parent_folder_path else None,
                user_id=request.user_id
            )
            
            # Build response
            if result.get("success"):
                response = tool_service_pb2.CreateUserFolderResponse(
                    success=True,
                    folder_id=result.get("folder_id", ""),
                    folder_name=result.get("folder_name", request.folder_name),
                    parent_folder_id=result.get("parent_folder_id", ""),
                    message=result.get("message", "Folder created successfully")
                )
                logger.info(f"CreateUserFolder: Success - {response.folder_id}")
            else:
                response = tool_service_pb2.CreateUserFolderResponse(
                    success=False,
                    message=result.get("message", "Folder creation failed"),
                    error=result.get("error", "Unknown error")
                )
                logger.warning(f"CreateUserFolder: Failed - {response.error}")
            
            return response
            
        except Exception as e:
            logger.error(f"CreateUserFolder error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Folder creation failed: {str(e)}")

    def _flatten_folder_tree(self, roots, parent_in_tree: Optional[str] = None) -> list:
        """Flatten hierarchical folder tree into list of FolderInfo for proto.

        First-level folders under virtual roots (my_documents_root, global_documents_root)
        still have parent_folder_id=None in the database. When flattening, their parent in
        the wire format must be the virtual root id so API consumers can rebuild a nested
        tree from parent_folder_id alone.
        """
        flat = []
        for f in roots:
            if parent_in_tree is not None:
                eff_parent = parent_in_tree
            else:
                eff_parent = f.parent_folder_id or ""

            flat.append(tool_service_pb2.FolderInfo(
                folder_id=f.folder_id,
                name=f.name,
                parent_folder_id=eff_parent,
                collection_type=getattr(f, "collection_type", "user") or "user",
                document_count=getattr(f, "document_count", 0) or 0,
            ))
            for child in getattr(f, "children", []) or []:
                flat.extend(self._flatten_folder_tree([child], parent_in_tree=f.folder_id))
        return flat

    async def GetFolderTree(
        self,
        request: tool_service_pb2.GetFolderTreeRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetFolderTreeResponse:
        """Return flat list of folders in the user's document tree."""
        try:
            from shims.services.service_container import get_service_container

            container = await get_service_container()
            folder_service = container.folder_service
            # Full tree for wire export: shallow=True strips grandchildren before flatten,
            # which drops nested folders from the flat list and makes subfolder_count wrong
            # after the API rebuild (only folders with document_count > 0 show expand carets).
            roots = await folder_service.get_folder_tree(
                user_id=request.user_id, shallow=False
            )
            flat = self._flatten_folder_tree(roots)
            return tool_service_pb2.GetFolderTreeResponse(
                folders=flat,
                total_folders=len(flat),
            )
        except Exception as e:
            logger.error(f"GetFolderTree error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Folder tree failed: {str(e)}")

    async def ListFolderDocuments(
        self,
        request: tool_service_pb2.ListFolderDocumentsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListFolderDocumentsResponse:
        """List documents directly in a folder (same access rules as folder contents API)."""
        try:
            from shims.services.service_container import get_service_container

            limit = int(request.limit) if request.limit and request.limit > 0 else 500
            offset = int(request.offset) if request.offset and request.offset >= 0 else 0
            container = await get_service_container()
            folder_service = container.folder_service
            contents = await folder_service.get_folder_contents(
                request.folder_id, request.user_id, limit=limit, offset=offset
            )
            if not contents:
                return tool_service_pb2.ListFolderDocumentsResponse(
                    documents=[],
                    total_count=0,
                    error="Folder not found or access denied",
                )
            entries = []
            for d in contents.documents:
                raw_ct = getattr(d, "collection_type", "") or ""
                ct_str = str(raw_ct.value) if hasattr(raw_ct, "value") else str(raw_ct)
                entries.append(
                    tool_service_pb2.FolderDocumentEntry(
                        document_id=str(getattr(d, "document_id", "") or ""),
                        filename=str(getattr(d, "filename", "") or ""),
                        title=str(getattr(d, "title", "") or ""),
                        collection_type=ct_str,
                    )
                )
            return tool_service_pb2.ListFolderDocumentsResponse(
                documents=entries,
                total_count=int(contents.total_documents or len(entries)),
                error="",
            )
        except Exception as e:
            logger.error("ListFolderDocuments error: %s", e)
            return tool_service_pb2.ListFolderDocumentsResponse(
                documents=[],
                total_count=0,
                error=str(e)[:500],
            )

    async def PickRandomDocumentFromFolder(
        self,
        request: tool_service_pb2.PickRandomDocumentFromFolderRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.PickRandomDocumentFromFolderResponse:
        """Return a randomly chosen document from the given folder. Optional file_extension filter (e.g. png, jpg)."""
        import random
        try:
            from shims.services.service_container import get_service_container

            container = await get_service_container()
            folder_service = container.folder_service
            contents = await folder_service.get_folder_contents(
                request.folder_id, request.user_id, limit=500, offset=0
            )
            if not contents or not contents.documents:
                return tool_service_pb2.PickRandomDocumentFromFolderResponse(
                    found=False,
                    document_id="",
                    filename="",
                    message="No documents in folder or folder not found.",
                )
            docs = list(contents.documents)
            ext = (request.file_extension or "").strip().lower()
            if ext and not ext.startswith("."):
                ext = f".{ext}"
            if ext:
                docs = [d for d in docs if (getattr(d, "filename", "") or "").lower().endswith(ext)]
            if not docs:
                return tool_service_pb2.PickRandomDocumentFromFolderResponse(
                    found=False,
                    document_id="",
                    filename="",
                    message=f"No documents in folder with extension {request.file_extension or ext}.",
                )
            doc = random.choice(docs)
            doc_type = getattr(doc, "doc_type", None)
            doc_type_str = str(doc_type.value) if hasattr(doc_type, "value") else str(doc_type or "")
            return tool_service_pb2.PickRandomDocumentFromFolderResponse(
                found=True,
                document_id=getattr(doc, "document_id", "") or "",
                filename=getattr(doc, "filename", "") or "",
                title=getattr(doc, "title", "") or "",
                folder_id=getattr(doc, "folder_id", "") or "",
                doc_type=doc_type_str or "",
                message=f"Random document: {getattr(doc, 'filename', '')}",
            )
        except Exception as e:
            logger.error(f"PickRandomDocumentFromFolder error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Pick random document failed: {str(e)}")

