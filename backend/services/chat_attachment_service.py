"""
Chat Attachment Service
Handles file uploads, storage, and cleanup for chat conversation attachments
"""

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from fastapi import UploadFile, HTTPException
from fastapi.responses import FileResponse
import asyncpg
import aiofiles

from config import settings
from utils.shared_db_pool import get_shared_db_pool

logger = logging.getLogger(__name__)


class ChatAttachmentService:
    """Service for managing chat conversation attachments"""
    
    def __init__(self):
        self.db_pool = None
        # Use temporary storage for chat attachments
        self.storage_base = Path("/tmp/chat_attachments")
        self.max_size = 10 * 1024 * 1024  # 10MB per file
        # Allow all common file types
        self.allowed_types = [
            # Images
            "image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp", "image/bmp",
            # Documents
            "application/pdf", "text/plain", "text/markdown", "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            # Audio
            "audio/wav", "audio/wave", "audio/x-wav", "audio/mpeg", "audio/mp3",
            "audio/mp4", "audio/x-m4a", "audio/aac", "audio/ogg", "audio/flac",
        ]
    
    async def initialize(self, shared_db_pool=None):
        """Initialize with database pool"""
        if shared_db_pool:
            self.db_pool = shared_db_pool
        else:
            self.db_pool = await get_shared_db_pool()
        
        # Ensure storage directory exists
        self.storage_base.mkdir(parents=True, exist_ok=True)
        logger.info(f"Chat attachment service initialized. Storage: {self.storage_base}")
    
    async def _ensure_initialized(self):
        """Ensure service is initialized"""
        if not self.db_pool:
            await self.initialize()
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        # Remove path components
        filename = os.path.basename(filename)
        # Remove dangerous characters
        dangerous_chars = ['..', '/', '\\', '\x00']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        return filename
    
    def _validate_file(self, file: UploadFile):
        """Validate file size and type"""
        # Check content type
        if file.content_type and file.content_type not in self.allowed_types:
            # Allow files without explicit content type (browsers sometimes don't send it)
            if file.content_type:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type {file.content_type} not allowed. Allowed types: {', '.join(self.allowed_types[:5])}..."
                )
        
        # Note: File size validation happens during upload since we need to read the file
    
    async def save_attachment(
        self,
        conversation_id: str,
        message_id: str,
        file: UploadFile,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Save an attachment for a chat message
        
        Args:
            conversation_id: Conversation UUID
            message_id: Message UUID
            file: Uploaded file
            user_id: User ID uploading the file
        
        Returns:
            Dict with attachment metadata
        """
        await self._ensure_initialized()
        
        # Validate file type
        self._validate_file(file)
        
        # Verify user has access to conversation
        async with self.db_pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
            
            has_access = await conn.fetchval("""
                SELECT EXISTS(
                    SELECT 1 FROM conversations
                    WHERE conversation_id = $1 AND user_id = $2
                )
            """, conversation_id, user_id)
            
            if not has_access:
                raise HTTPException(status_code=403, detail="Not authorized to add attachments to this conversation")
        
        # Sanitize filename
        sanitized_filename = self._sanitize_filename(file.filename or "attachment")
        
        # Generate unique filename
        timestamp = int(datetime.now().timestamp() * 1000)
        file_ext = Path(sanitized_filename).suffix
        unique_filename = f"{message_id}_{timestamp}{file_ext}"
        
        # Create conversation/message directory structure
        conv_dir = self.storage_base / conversation_id
        msg_dir = conv_dir / message_id
        msg_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = msg_dir / unique_filename
        file_size = 0
        
        try:
            # Read file in chunks to validate size
            async with aiofiles.open(file_path, "wb") as f:
                while True:
                    chunk = await file.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    file_size += len(chunk)
                    if file_size > self.max_size:
                        # Clean up partial file
                        if file_path.exists():
                            file_path.unlink()
                        raise HTTPException(
                            status_code=400,
                            detail=f"File size exceeds maximum of {self.max_size / (1024*1024):.1f}MB"
                        )
                    await f.write(chunk)
            
            # Reset file pointer for potential reuse
            await file.seek(0)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to save attachment file: {e}")
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=500, detail="Failed to save attachment")
        
        # Generate attachment ID
        attachment_id = str(uuid.uuid4())
        content_type = file.content_type or "application/octet-stream"
        is_image = content_type.startswith("image/") if content_type else False

        # Insert attachment record into database
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                await conn.execute(
                    """
                    INSERT INTO conversation_message_attachments (
                        attachment_id, message_id, filename, content_type, file_size,
                        file_path, is_image, metadata_json
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    attachment_id,
                    message_id,
                    sanitized_filename,
                    content_type,
                    file_size,
                    str(file_path),
                    is_image,
                    json.dumps({
                        "original_filename": file.filename,
                        "upload_timestamp": datetime.now().isoformat(),
                    }),
                )
            logger.info(f"Saved attachment record: {attachment_id}")
        except Exception as e:
            logger.warning(f"Failed to save attachment record to database (table may not exist yet): {e}")

        # Create attachment metadata (return shape unchanged for API consumers)
        attachment_metadata = {
            "attachment_id": attachment_id,
            "filename": sanitized_filename,
            "content_type": content_type,
            "size_bytes": file_size,
            "storage_path": str(file_path),
            "uploaded_at": datetime.now().isoformat(),
        }

        logger.info(f"Saved chat attachment: {attachment_id} ({file_size} bytes) for message {message_id}")

        return attachment_metadata

    async def get_attachment_path(
        self,
        conversation_id: str,
        message_id: str,
        attachment_id: str,
        user_id: str
    ) -> Optional[Path]:
        """
        Get file path for an attachment (with access validation)
        
        Returns:
            Path object if found and user has access, None otherwise
        """
        await self._ensure_initialized()
        
        # Verify user has access to conversation
        async with self.db_pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
            
            has_access = await conn.fetchval("""
                SELECT EXISTS(
                    SELECT 1 FROM conversations
                    WHERE conversation_id = $1 AND user_id = $2
                )
            """, conversation_id, user_id)
            
            if not has_access:
                return None
        
        # Search for attachment in message directory
        msg_dir = self.storage_base / conversation_id / message_id
        if not msg_dir.exists():
            return None
        
        # Find file matching attachment_id (stored in metadata, so we need to check message metadata)
        # For now, we'll search by filename pattern - in production, we'd store attachment_id mapping
        # This is a simplified version - in practice, we'd query message metadata_json for attachment_id
        for file_path in msg_dir.iterdir():
            if file_path.is_file():
                # Check if this file's metadata matches attachment_id
                # We'll need to check message metadata in database
                pass
        
        # Fallback: return first file in message directory (simplified)
        # In production, we'd maintain an attachment_id -> filename mapping
        files = list(msg_dir.iterdir())
        if files:
            return files[0]
        
        return None
    
    async def serve_attachment_file(
        self,
        conversation_id: str,
        message_id: str,
        attachment_id: str,
        user_id: str
    ) -> FileResponse:
        """Serve attachment file with proper headers"""
        file_path = await self.get_attachment_path(conversation_id, message_id, attachment_id, user_id)
        
        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="Attachment not found")
        
        # Determine content type from file extension
        content_type = "application/octet-stream"
        if file_path.suffix:
            ext = file_path.suffix.lower()
            content_type_map = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".gif": "image/gif", ".webp": "image/webp",
                ".pdf": "application/pdf", ".txt": "text/plain",
                ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4"
            }
            content_type = content_type_map.get(ext, content_type)
        
        return FileResponse(
            path=str(file_path),
            media_type=content_type,
            filename=file_path.name
        )
    
    async def delete_attachment(
        self,
        conversation_id: str,
        message_id: str,
        attachment_id: str,
        user_id: str
    ) -> bool:
        """Delete an attachment"""
        await self._ensure_initialized()
        
        file_path = await self.get_attachment_path(conversation_id, message_id, attachment_id, user_id)
        
        if not file_path:
            return False
        
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted chat attachment: {attachment_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete attachment {attachment_id}: {e}")
            return False
    
    async def cleanup_conversation_attachments(self, conversation_id: str) -> int:
        """Delete all attachments for a conversation"""
        await self._ensure_initialized()
        
        try:
            conv_dir = self.storage_base / conversation_id
            if not conv_dir.exists():
                return 0
            
            # Count files before deletion
            file_count = sum(1 for _ in conv_dir.rglob("*") if _.is_file())
            
            # Remove entire conversation directory
            shutil.rmtree(conv_dir)
            
            logger.info(f"Cleaned up {file_count} attachment files for conversation {conversation_id}")
            return file_count
        except Exception as e:
            logger.error(f"Failed to cleanup attachments for conversation {conversation_id}: {e}")
            return 0
    
    async def cleanup_old_attachments(self, days: int = 7) -> int:
        """Delete attachments older than specified days"""
        await self._ensure_initialized()
        
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            deleted_count = 0
            
            # Iterate through all conversation directories
            for conv_dir in self.storage_base.iterdir():
                if not conv_dir.is_dir():
                    continue
                
                # Check last modified time of conversation directory
                # (simplified - in production, we'd check individual file timestamps)
                conv_mtime = datetime.fromtimestamp(conv_dir.stat().st_mtime)
                
                if conv_mtime < cutoff_time:
                    # Delete entire conversation directory
                    file_count = sum(1 for _ in conv_dir.rglob("*") if _.is_file())
                    shutil.rmtree(conv_dir)
                    deleted_count += file_count
                    logger.info(f"Cleaned up old conversation attachments: {conv_dir.name} ({file_count} files)")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old attachment files (older than {days} days)")
            
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old attachments: {e}")
            return 0


# Global instance
chat_attachment_service = ChatAttachmentService()
