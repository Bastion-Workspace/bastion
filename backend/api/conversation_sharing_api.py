"""
Conversation Sharing API - Multi-User Thread Support
API endpoints for sharing conversations with checkpoint replication
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List
from datetime import datetime

from models.conversation_models import (
    ShareConversationRequest,
    ShareConversationResponse,
    ConversationShareDetail,
    ConversationSharesResponse,
    ConversationParticipantsResponse,
    ConversationParticipant,
    SharedConversationsResponse,
    ConversationSummary,
    UpdateShareRequest,
)
from utils.auth_middleware import get_current_user
from models.api_models import AuthenticatedUserResponse
from services.conversation_sharing_service import get_conversation_sharing_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/conversations/{conversation_id}/share", response_model=ShareConversationResponse)
async def share_conversation(
    conversation_id: str,
    request: ShareConversationRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Share a conversation with another user"""
    try:
        if not request.shared_with_user_id:
            raise HTTPException(status_code=400, detail="shared_with_user_id is required")
        
        # Use share_type if provided, otherwise use permission_level
        share_type = request.share_type or request.permission_level.value
        
        sharing_service = await get_conversation_sharing_service()
        result = await sharing_service.share_conversation(
            conversation_id=conversation_id,
            shared_by_user_id=current_user.user_id,
            shared_with_user_id=request.shared_with_user_id,
            share_type=share_type,
            expires_at=request.expires_at
        )
        
        # Broadcast WebSocket event to recipient
        try:
            from utils.websocket_manager import get_websocket_manager
            websocket_manager = get_websocket_manager()
            if websocket_manager:
                await websocket_manager.send_to_session(
                    message={
                        "type": "conversation_shared",
                        "data": {
                            "conversation_id": conversation_id,
                            "share_id": result["share_id"],
                            "shared_by": current_user.user_id,
                            "share_type": share_type
                        }
                    },
                    session_id=request.shared_with_user_id
                )
        except Exception as ws_error:
            logger.warning(f"Failed to send WebSocket notification: {ws_error}")
        
        return ShareConversationResponse(
            success=result["success"],
            share_id=result["share_id"],
            message=result.get("message", "Conversation shared successfully")
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to share conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/conversations/{conversation_id}/share/{share_id}")
async def unshare_conversation(
    conversation_id: str,
    share_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Revoke a conversation share"""
    try:
        sharing_service = await get_conversation_sharing_service()
        
        # Get share info before deleting to notify recipient
        shares_data = await sharing_service.get_conversation_shares(conversation_id, current_user.user_id)
        share_to_remove = next((s for s in shares_data if s["share_id"] == share_id), None)
        
        success = await sharing_service.unshare_conversation(
            conversation_id=conversation_id,
            share_id=share_id,
            user_id=current_user.user_id
        )
        
        if success:
            # Broadcast WebSocket event to removed user
            if share_to_remove and share_to_remove.get("shared_with_user_id"):
                try:
                    from utils.websocket_manager import get_websocket_manager
                    websocket_manager = get_websocket_manager()
                    if websocket_manager:
                        await websocket_manager.send_to_session(
                            message={
                                "type": "conversation_unshared",
                                "data": {
                                    "conversation_id": conversation_id,
                                    "share_id": share_id
                                }
                            },
                            session_id=share_to_remove["shared_with_user_id"]
                        )
                except Exception as ws_error:
                    logger.warning(f"Failed to send WebSocket notification: {ws_error}")
            
            return {"success": True, "message": "Share revoked successfully"}
        else:
            raise HTTPException(status_code=404, detail="Share not found")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to unshare conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/conversations/{conversation_id}/shares", response_model=ConversationSharesResponse)
async def get_conversation_shares(
    conversation_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get all shares for a conversation"""
    try:
        sharing_service = await get_conversation_sharing_service()
        shares_data = await sharing_service.get_conversation_shares(
            conversation_id=conversation_id,
            user_id=current_user.user_id
        )
        
        shares = [
            ConversationShareDetail(
                share_id=s["share_id"],
                conversation_id=s["conversation_id"],
                shared_by_user_id=s["shared_by_user_id"],
                shared_with_user_id=s["shared_with_user_id"],
                share_type=s["share_type"],
                is_public=s["is_public"],
                expires_at=datetime.fromisoformat(s["expires_at"]) if s["expires_at"] else None,
                created_at=datetime.fromisoformat(s["created_at"]),
                username=s.get("username"),
                email=s.get("email")
            )
            for s in shares_data
        ]
        
        return ConversationSharesResponse(shares=shares)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get conversation shares: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/conversations/shared-with-me", response_model=SharedConversationsResponse)
async def get_shared_conversations(
    skip: int = 0,
    limit: int = 50,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get all conversations shared with the current user"""
    try:
        sharing_service = await get_conversation_sharing_service()
        conversations_data = await sharing_service.get_shared_conversations(
            user_id=current_user.user_id,
            skip=skip,
            limit=limit
        )
        
        conversations = [
            ConversationSummary(
                conversation_id=c["conversation_id"],
                user_id=c["user_id"],
                title=c["title"],
                description=c["description"],
                is_pinned=c["is_pinned"],
                is_archived=c["is_archived"],
                tags=c["tags"],
                metadata_json=c["metadata_json"],
                message_count=c["message_count"],
                last_message_at=datetime.fromisoformat(c["last_message_at"]) if c["last_message_at"] else None,
                manual_order=None,
                order_locked=False,
                created_at=datetime.fromisoformat(c["created_at"]),
                updated_at=datetime.fromisoformat(c["updated_at"])
            )
            for c in conversations_data
        ]
        
        return SharedConversationsResponse(
            conversations=conversations,
            total_count=len(conversations)
        )
        
    except Exception as e:
        logger.error(f"Failed to get shared conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/conversations/{conversation_id}/participants", response_model=ConversationParticipantsResponse)
async def get_conversation_participants(
    conversation_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get all participants (owner + shared users) for a conversation"""
    try:
        sharing_service = await get_conversation_sharing_service()
        participants_data = await sharing_service.get_conversation_participants(
            conversation_id=conversation_id,
            user_id=current_user.user_id
        )
        
        participants = [
            ConversationParticipant(
                user_id=p["user_id"],
                username=p.get("username"),
                email=p.get("email"),
                display_name=p.get("display_name"),
                share_type=p["share_type"],
                is_owner=p["is_owner"]
            )
            for p in participants_data
        ]
        
        return ConversationParticipantsResponse(participants=participants)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get conversation participants: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/conversations/{conversation_id}/share/{share_id}")
async def update_share_permissions(
    conversation_id: str,
    share_id: str,
    request: UpdateShareRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update share permissions (read/comment/edit)"""
    try:
        if not request.share_type:
            raise HTTPException(status_code=400, detail="share_type is required")
        
        if request.share_type not in ["read", "comment", "edit"]:
            raise HTTPException(status_code=400, detail="share_type must be 'read', 'comment', or 'edit'")
        
        sharing_service = await get_conversation_sharing_service()
        success = await sharing_service.update_share_permissions(
            conversation_id=conversation_id,
            share_id=share_id,
            user_id=current_user.user_id,
            new_share_type=request.share_type
        )
        
        if success:
            return {"success": True, "message": "Share permissions updated successfully"}
        else:
            raise HTTPException(status_code=404, detail="Share not found")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update share permissions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

