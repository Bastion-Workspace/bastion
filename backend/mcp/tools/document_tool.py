"""
Document Tool - MCP Tool for Full Document Retrieval
Allows LLM to retrieve complete document content and metadata with intelligent chunking
"""

import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional

from mcp.schemas.tool_schemas import (
    GetDocumentInput,
    GetDocumentOutput,
    DocumentInfo,
    DocumentType,
    ToolResponse,
    ToolError
)

logger = logging.getLogger(__name__)

# Constants for content management
MAX_CONTENT_LENGTH = 50000  # Maximum characters to return in one call
MAX_CHUNK_PREVIEW = 300     # Maximum characters per chunk in preview mode
OPTIMAL_CONTENT_LENGTH = 25000  # Optimal content length for LLM processing


class DocumentTool:
    """MCP tool for retrieving complete documents with intelligent content management"""
    
    def __init__(self, embedding_manager=None, document_repository=None, chat_service=None):
        """Initialize with required services"""
        self.embedding_manager = embedding_manager
        self.document_repository = document_repository
        self.chat_service = chat_service
        self.name = "get_document"
        self.description = "Retrieve document content and metadata by document ID with intelligent content management"
        
    async def initialize(self):
        """Initialize the document tool"""
        if not self.document_repository:
            raise ValueError("DocumentRepository is required")
        if not self.embedding_manager:
            raise ValueError("EmbeddingManager is required")
        
        logger.info("📄 DocumentTool initialized")
    
    async def execute(self, input_data: GetDocumentInput) -> ToolResponse:
        """Execute document retrieval with intelligent content management"""
        start_time = time.time()
        
        try:
            logger.info(f"📄 Retrieving document: {input_data.document_id}")
            
            # Get document metadata
            document_metadata = await self._get_document_metadata(input_data.document_id)
            if not document_metadata:
                return ToolResponse(
                    success=False,
                    error=ToolError(
                        error_code="DOCUMENT_NOT_FOUND",
                        error_message=f"Document '{input_data.document_id}' not found",
                        details={"document_id": input_data.document_id}
                    ),
                    execution_time=time.time() - start_time
                )
            
            # Get document content with intelligent management
            content = ""
            content_info = {}
            
            if input_data.include_content:
                content, content_info = await self._get_managed_document_content(
                    input_data.document_id, 
                    getattr(input_data, 'max_length', MAX_CONTENT_LENGTH),
                    getattr(input_data, 'preview_mode', False)
                )
            
            # Get document chunks info if requested
            chunk_count = 0
            if input_data.include_metadata:
                chunk_count = await self._get_document_chunk_count(input_data.document_id)
            
            # Create document info
            document_info = DocumentInfo(
                document_id=input_data.document_id,
                filename=document_metadata.get('filename', ''),
                title=document_metadata.get('title'),
                content=content,
                doc_type=DocumentType(document_metadata.get('doc_type', 'text')),
                category=document_metadata.get('category'),
                tags=document_metadata.get('tags', []),
                author=document_metadata.get('author'),
                upload_date=document_metadata['upload_date'],
                file_size=document_metadata.get('file_size', 0),
                page_count=document_metadata.get('page_count'),
                chunk_count=chunk_count,
                metadata={
                    **document_metadata.get('metadata_json', {}),
                    **content_info  # Add content management info
                }
            )
            
            # Create output
            output = GetDocumentOutput(
                document=document_info,
                retrieval_time=time.time() - start_time
            )
            
            logger.info(f"✅ Document retrieved: {len(content)} characters in {output.retrieval_time:.2f}s")
            
            return ToolResponse(
                success=True,
                data=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Document retrieval failed: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="RETRIEVAL_FAILED",
                    error_message=str(e),
                    details={"document_id": input_data.document_id}
                ),
                execution_time=time.time() - start_time
            )
    
    async def _get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata from database"""
        try:
            query = """
                SELECT 
                    document_id, filename, title, doc_type, category, tags, author,
                    upload_date, file_size, metadata_json, processing_status
                FROM document_metadata 
                WHERE document_id = $1
            """
            results = await self.document_repository.execute_query(query, document_id)
            
            if results:
                row = results[0]
                # Parse metadata_json if it's a string
                metadata = {}
                if row['metadata_json']:
                    if isinstance(row['metadata_json'], str):
                        try:
                            metadata = json.loads(row['metadata_json'])
                        except json.JSONDecodeError:
                            logger.warning(f"⚠️ Failed to parse metadata_json for document {row['document_id']}")
                            metadata = {}
                    else:
                        metadata = row['metadata_json']

                return {
                    'document_id': row['document_id'],
                    'filename': row['filename'],
                    'title': row['title'],
                    'doc_type': row['doc_type'],
                    'category': row['category'],
                    'tags': row['tags'] or [],
                    'author': row['author'],
                    'upload_date': row['upload_date'],
                    'file_size': row['file_size'],
                    'page_count': None,  # Page count not available in current schema
                    'metadata_json': metadata,
                    'processing_status': row['processing_status']
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get document metadata: {e}")
            return None
    
    async def _get_document_content(self, document_id: str) -> str:
        """Get full document content from Qdrant vector database"""
        try:
            # Get all chunks for this document from embedding manager
            chunks = await self.embedding_manager.get_all_document_chunks(document_id)
            
            if not chunks:
                logger.warning(f"⚠️ No chunks found for document {document_id}")
                return ""
            
            # Sort chunks by chunk_index and combine content
            chunks.sort(key=lambda x: x.get('chunk_index', 0))
            full_content = []
            for chunk in chunks:
                full_content.append(chunk['content'])
            
            logger.info(f"📄 Retrieved {len(chunks)} chunks for document {document_id}")
            return "\n\n".join(full_content)
            
        except Exception as e:
            logger.error(f"❌ Failed to get document content: {e}")
            return ""
    
    async def _get_document_chunk_count(self, document_id: str) -> int:
        """Get number of chunks for document from Qdrant"""
        try:
            # Get all chunks for this document from embedding manager
            chunks = await self.embedding_manager.get_all_document_chunks(document_id)
            return len(chunks) if chunks else 0
            
        except Exception as e:
            logger.error(f"❌ Failed to get chunk count: {e}")
            return 0
    
    async def _get_managed_document_content(self, document_id: str, max_length: int = MAX_CONTENT_LENGTH, preview_mode: bool = False) -> tuple[str, Dict[str, Any]]:
        """Get document content with intelligent management to prevent prompt overflow"""
        try:
            # Get all chunks for this document from embedding manager
            chunks = await self.embedding_manager.get_all_document_chunks(document_id)
            
            if not chunks:
                logger.warning(f"⚠️ No chunks found for document {document_id}")
                return "", {"total_chunks": 0, "included_chunks": 0, "content_truncated": False}
            
            # Sort chunks by chunk_index
            chunks.sort(key=lambda x: x.get('chunk_index', 0))
            
            total_chunks = len(chunks)
            logger.info(f"📄 Retrieved {total_chunks} chunks for document {document_id}")
            
            if preview_mode:
                # In preview mode, provide summaries of chunks
                return await self._get_preview_content(chunks, max_length)
            
            # Strategy 1: If content is small enough, return all
            total_content = "\n\n".join(chunk['content'] for chunk in chunks)
            
            if len(total_content) <= max_length:
                return total_content, {
                    "total_chunks": total_chunks,
                    "included_chunks": total_chunks,
                    "content_truncated": False,
                    "content_strategy": "full_content"
                }
            
            # Strategy 2: Intelligent truncation with key sections
            return await self._get_truncated_content(chunks, max_length, total_chunks)
            
        except Exception as e:
            logger.error(f"❌ Failed to get managed document content: {e}")
            return "", {"error": str(e)}
    
    async def _get_preview_content(self, chunks: List[Dict], max_length: int) -> tuple[str, Dict[str, Any]]:
        """Generate preview content with chunk summaries"""
        preview_parts = []
        current_length = 0
        included_chunks = 0
        
        for i, chunk in enumerate(chunks):
            chunk_preview = chunk['content'][:MAX_CHUNK_PREVIEW]
            if len(chunk['content']) > MAX_CHUNK_PREVIEW:
                chunk_preview += "..."
            
            chunk_section = f"--- Chunk {i+1} ---\n{chunk_preview}\n"
            
            if current_length + len(chunk_section) > max_length:
                break
                
            preview_parts.append(chunk_section)
            current_length += len(chunk_section)
            included_chunks += 1
        
        preview_content = "\n".join(preview_parts)
        
        if included_chunks < len(chunks):
            preview_content += f"\n\n[{len(chunks) - included_chunks} more chunks available - use full retrieval for complete content]"
        
        return preview_content, {
            "total_chunks": len(chunks),
            "included_chunks": included_chunks,
            "content_truncated": included_chunks < len(chunks),
            "content_strategy": "preview_mode"
        }
    
    async def _get_truncated_content(self, chunks: List[Dict], max_length: int, total_chunks: int) -> tuple[str, Dict[str, Any]]:
        """Get truncated content with intelligent selection"""
        content_parts = []
        current_length = 0
        included_chunks = 0
        
        # Strategy: Include beginning chunks and as many as possible within limit
        for chunk in chunks:
            chunk_content = chunk['content']
            chunk_section = chunk_content + "\n\n"
            
            # Check if adding this chunk would exceed limit
            if current_length + len(chunk_section) > max_length:
                # If we haven't included any chunks yet, truncate this one
                if included_chunks == 0:
                    remaining_space = max_length - current_length - 100  # Reserve space for truncation notice
                    if remaining_space > 500:  # Only truncate if we have reasonable space
                        truncated_content = chunk_content[:remaining_space] + "..."
                        content_parts.append(truncated_content)
                        included_chunks += 1
                break
            
            content_parts.append(chunk_content)
            current_length += len(chunk_section)
            included_chunks += 1
        
        final_content = "\n\n".join(content_parts)
        
        # Add truncation notice if needed
        if included_chunks < total_chunks:
            final_content += f"\n\n[Content truncated - showing {included_chunks}/{total_chunks} chunks. Document has {total_chunks - included_chunks} additional chunks available.]"
        
        return final_content, {
            "total_chunks": total_chunks,
            "included_chunks": included_chunks,
            "content_truncated": included_chunks < total_chunks,
            "content_strategy": "intelligent_truncation",
            "content_length": len(final_content)
        }
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition for registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": GetDocumentInput.schema(),
            "outputSchema": GetDocumentOutput.schema()
        } 