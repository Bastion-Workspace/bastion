"""
Messaging Attachment Service
Handles file uploads, storage, and cleanup for messaging attachments
"""

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import UploadFile, HTTPException
from fastapi.responses import FileResponse
import asyncpg

from config import settings
from utils.shared_db_pool import get_shared_db_pool

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class MessagingAttachmentService:
    """Service for managing messaging attachments"""
    
    def __init__(self):
        self.db_pool = None
        self.storage_base = Path(settings.MESSAGING_ATTACHMENT_STORAGE_PATH)
        self.max_size = settings.MESSAGING_ATTACHMENT_MAX_SIZE
        self.allowed_types = settings.MESSAGING_ATTACHMENT_ALLOWED_TYPES
    
    async def initialize(self, shared_db_pool=None):
        """Initialize with database pool"""
        if shared_db_pool:
            self.db_pool = shared_db_pool
        else:
            self.db_pool = await get_shared_db_pool()
        
        # Ensure storage directory exists
        self.storage_base.mkdir(parents=True, exist_ok=True)
        logger.info(f"Messaging attachment service initialized. Storage: {self.storage_base}")
    
    async def _ensure_initialized(self):
        """Ensure service is initialized"""
        if not self.db_pool:
            await self.initialize()
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        # Remove path components
        filename = os.path.basename(filename)
        # Remove dangerous characters
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        # Limit length
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:200] + ext
        return filename or "attachment"
    
    def _validate_file(self, file: UploadFile) -> None:
        """Validate file type and size"""
        # Check MIME type
        if file.content_type not in self.allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(self.allowed_types)}"
            )
        
        # Check file size (read first chunk to check)
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)
        
        # Use larger limit for audio files
        max_size = self.max_size
        if file.content_type and file.content_type.startswith("audio/"):
            max_size = settings.AUDIO_ATTACHMENT_MAX_SIZE
        
        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB"
            )
    
    def _get_image_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract image metadata (dimensions, animated status)"""
        metadata = {
            "width": None,
            "height": None,
            "is_animated": False
        }
        
        if not PIL_AVAILABLE:
            return metadata
        
        try:
            with Image.open(file_path) as img:
                metadata["width"], metadata["height"] = img.size
                
                # Check if animated GIF
                if hasattr(img, "is_animated"):
                    metadata["is_animated"] = img.is_animated
                elif img.format == "GIF":
                    # Fallback: try to detect animation by checking frames
                    try:
                        img.seek(1)
                        metadata["is_animated"] = True
                    except EOFError:
                        metadata["is_animated"] = False
        except Exception as e:
            logger.warning(f"Failed to extract image metadata: {e}")
        
        return metadata
    
    async def upload_attachment(
        self,
        room_id: str,
        message_id: str,
        file: UploadFile,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Upload an attachment for a message
        
        Args:
            room_id: Room UUID
            message_id: Message UUID
            file: Uploaded file
            user_id: User ID uploading the file
        
        Returns:
            Dict with attachment details
        """
        await self._ensure_initialized()
        
        # Validate file
        self._validate_file(file)
        
        # Verify user is room participant
        async with self.db_pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
            
            is_participant = await conn.fetchval("""
                SELECT EXISTS(
                    SELECT 1 FROM room_participants
                    WHERE room_id = $1 AND user_id = $2
                )
            """, room_id, user_id)
            
            if not is_participant:
                raise HTTPException(status_code=403, detail="Not a participant in this room")
        
        # Sanitize filename
        sanitized_filename = self._sanitize_filename(file.filename or "attachment")
        
        # Generate unique filename
        timestamp = int(uuid.uuid4().time_low)
        file_ext = Path(sanitized_filename).suffix
        unique_filename = f"{message_id}_{timestamp}{file_ext}"
        
        # Create room directory
        room_dir = self.storage_base / room_id
        room_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = room_dir / unique_filename
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Get file size
            file_size = file_path.stat().st_size
            
            # Extract image metadata
            image_metadata = self._get_image_metadata(file_path)
            
            # Save to database
            async with self.db_pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                attachment_id = await conn.fetchval("""
                    INSERT INTO message_attachments (
                        message_id, room_id, filename, file_path,
                        mime_type, file_size, width, height, is_animated
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING attachment_id
                """, message_id, room_id, sanitized_filename, str(file_path),
                    file.content_type, file_size,
                    image_metadata["width"], image_metadata["height"],
                    image_metadata["is_animated"])
            
            logger.info(f"Uploaded attachment {attachment_id} for message {message_id}")
            
            return {
                "attachment_id": str(attachment_id),
                "message_id": message_id,
                "room_id": room_id,
                "filename": sanitized_filename,
                "file_path": str(file_path),
                "mime_type": file.content_type,
                "file_size": file_size,
                "width": image_metadata["width"],
                "height": image_metadata["height"],
                "is_animated": image_metadata["is_animated"]
            }
        
        except Exception as e:
            # Cleanup file on error
            if file_path.exists():
                file_path.unlink()
            logger.error(f"Failed to upload attachment: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload attachment: {str(e)}")
    
    async def get_attachment(self, attachment_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get attachment metadata"""
        await self._ensure_initialized()
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
            
            row = await conn.fetchrow("""
                SELECT 
                    attachment_id, message_id, room_id, filename,
                    file_path, mime_type, file_size, width, height,
                    is_animated, created_at
                FROM message_attachments
                WHERE attachment_id = $1
            """, attachment_id)
            
            if not row:
                return None
            
            # Verify user is room participant
            is_participant = await conn.fetchval("""
                SELECT EXISTS(
                    SELECT 1 FROM room_participants
                    WHERE room_id = $1 AND user_id = $2
                )
            """, row["room_id"], user_id)
            
            if not is_participant:
                raise HTTPException(status_code=403, detail="Not authorized to access this attachment")
            
            return dict(row)
    
    async def get_message_attachments(self, message_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Get all attachments for a message"""
        await self._ensure_initialized()
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
            
            rows = await conn.fetch("""
                SELECT 
                    attachment_id, message_id, room_id, filename,
                    file_path, mime_type, file_size, width, height,
                    is_animated, created_at
                FROM message_attachments
                WHERE message_id = $1
                ORDER BY created_at ASC
            """, message_id)
            
            # Verify user is room participant for at least one attachment
            if rows:
                room_id = rows[0]["room_id"]
                is_participant = await conn.fetchval("""
                    SELECT EXISTS(
                        SELECT 1 FROM room_participants
                        WHERE room_id = $1 AND user_id = $2
                    )
                """, room_id, user_id)
                
                if not is_participant:
                    raise HTTPException(status_code=403, detail="Not authorized to access these attachments")
            
            return [dict(row) for row in rows]
    
    async def serve_attachment_file(self, attachment_id: str, user_id: str) -> FileResponse:
        """Serve attachment file"""
        attachment = await self.get_attachment(attachment_id, user_id)
        if not attachment:
            raise HTTPException(status_code=404, detail="Attachment not found")
        
        file_path = Path(attachment["file_path"])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Attachment file not found")
        
        return FileResponse(
            path=str(file_path),
            media_type=attachment["mime_type"],
            filename=attachment["filename"]
        )
    
    async def delete_room_attachments(self, room_id: str) -> bool:
        """Delete all attachments for a room (called when room is deleted)"""
        await self._ensure_initialized()
        
        try:
            # Delete room directory
            room_dir = self.storage_base / room_id
            if room_dir.exists() and room_dir.is_dir():
                shutil.rmtree(room_dir)
                logger.info(f"Deleted attachment directory for room {room_id}")
            
            # Database records will be deleted by CASCADE
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete room attachments for {room_id}: {e}")
            return False
    
    async def delete_message_attachments(self, message_id: str) -> bool:
        """Delete all attachments for a message"""
        await self._ensure_initialized()
        
        try:
            # Get attachment file paths before deletion
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT file_path FROM message_attachments
                    WHERE message_id = $1
                """, message_id)
            
            # Delete files
            for row in rows:
                file_path = Path(row["file_path"])
                if file_path.exists():
                    file_path.unlink()
            
            # Database records will be deleted by CASCADE
            logger.info(f"Deleted attachments for message {message_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete message attachments for {message_id}: {e}")
            return False


# Global instance
messaging_attachment_service = MessagingAttachmentService()


