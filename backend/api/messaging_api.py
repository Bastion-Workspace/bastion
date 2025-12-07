"""
Roosevelt's Messaging API
REST and WebSocket endpoints for user-to-user messaging

BULLY! Let the messages flow like a cavalry charge!
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel, Field

from services.messaging.messaging_service import messaging_service
from services.messaging.messaging_attachment_service import messaging_attachment_service
from utils.auth_middleware import get_current_user
from models.api_models import AuthenticatedUserResponse
from utils.websocket_manager import get_websocket_manager
from config import settings
from fastapi import File, UploadFile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/messaging", tags=["messaging"])


# =====================
# REQUEST/RESPONSE MODELS
# =====================

class CreateRoomRequest(BaseModel):
    participant_ids: List[str] = Field(..., description="List of user IDs to add to room")
    room_name: Optional[str] = Field(None, description="Optional room name")


class SendMessageRequest(BaseModel):
    content: str = Field(..., max_length=10000, description="Message content")
    message_type: str = Field(default="text", description="Message type: text, ai_share, system")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class UpdateRoomNameRequest(BaseModel):
    room_name: str = Field(..., max_length=255, description="New room name")


class AddParticipantRequest(BaseModel):
    user_id: str = Field(..., description="User ID to add to room")
    share_history: bool = Field(default=False, description="Whether new participant can see message history")


class AddReactionRequest(BaseModel):
    emoji: str = Field(..., max_length=10, description="Emoji character")


class UpdatePresenceRequest(BaseModel):
    status: str = Field(..., description="Status: online, offline, away")
    status_message: Optional[str] = Field(None, max_length=255, description="Optional status message")


class UpdateNotificationSettingsRequest(BaseModel):
    settings: Dict[str, Any] = Field(..., description="Notification settings (e.g., {'muted': True})")


# =====================
# ROOM ENDPOINTS
# =====================

@router.post("/rooms")
async def create_room(
    request: CreateRoomRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Create a new chat room"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        # Validate participants exist (basic check)
        if not request.participant_ids:
            raise HTTPException(status_code=400, detail="At least one participant required")
        
        room = await messaging_service.create_room(
            creator_id=current_user.user_id,
            participant_ids=request.participant_ids,
            room_name=request.room_name
        )
        
        # Broadcast new room to all participants
        ws_manager = get_websocket_manager()
        await ws_manager.broadcast_to_room(room['room_id'], {
            "type": "new_room",
            "room": room
        })
        
        logger.info(f"📬 New room {room['room_id']} broadcast to participants")
        
        return room
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to create room: {e}")
        raise HTTPException(status_code=500, detail="Failed to create room")


@router.get("/rooms")
async def get_user_rooms(
    limit: int = Query(20, ge=1, le=100),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get all rooms for current user"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        rooms = await messaging_service.get_user_rooms(
            user_id=current_user.user_id,
            limit=limit,
            include_participants=True
        )
        
        return {"rooms": rooms, "total": len(rooms)}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get user rooms: {e}")
        raise HTTPException(status_code=500, detail="Failed to get rooms")


@router.put("/rooms/{room_id}/name")
async def update_room_name(
    room_id: str,
    request: UpdateRoomNameRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update room name"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        success = await messaging_service.update_room_name(
            room_id=room_id,
            user_id=current_user.user_id,
            new_name=request.room_name
        )
        
        if not success:
            raise HTTPException(status_code=403, detail="Not authorized or room not found")
        
        # Broadcast room update to all participants
        ws_manager = get_websocket_manager()
        await ws_manager.broadcast_to_room(room_id, {
            "type": "room_updated",
            "room_id": room_id,
            "room_name": request.room_name,
            "updated_by": current_user.user_id
        })
        
        logger.info(f"✏️ Room {room_id} renamed to '{request.room_name}' by {current_user.username}")
        
        return {"success": True, "room_id": room_id, "room_name": request.room_name}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update room name: {e}")
        raise HTTPException(status_code=500, detail="Failed to update room name")


@router.put("/rooms/{room_id}/notifications")
async def update_notification_settings(
    room_id: str,
    request: UpdateNotificationSettingsRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update notification settings for a room"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        success = await messaging_service.update_notification_settings(
            room_id=room_id,
            user_id=current_user.user_id,
            settings=request.settings
        )
        
        if not success:
            raise HTTPException(status_code=403, detail="Not authorized or room not found")
        
        logger.info(f"🔕 Updated notification settings for room {room_id} by {current_user.username}")
        
        return {"success": True, "room_id": room_id, "settings": request.settings}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update notification settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to update notification settings")


@router.delete("/rooms/{room_id}")
async def delete_room(
    room_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete a room (removes for all participants)"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        success = await messaging_service.delete_room(
            room_id=room_id,
            user_id=current_user.user_id
        )
        
        if not success:
            raise HTTPException(status_code=403, detail="Not authorized or room not found")
        
        logger.info(f"🗑️ User {current_user.username} deleted room {room_id}")
        return {"success": True, "room_id": room_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete room: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete room")


@router.post("/rooms/{room_id}/participants")
async def add_participant(
    room_id: str,
    request: AddParticipantRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Add a participant to an existing room"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        success = await messaging_service.add_participant(
            room_id=room_id,
            user_id=request.user_id,
            added_by=current_user.user_id,
            share_history=request.share_history
        )
        
        if not success:
            raise HTTPException(status_code=403, detail="Not authorized or room not found")
        
        # Broadcast participant added to all room members
        ws_manager = get_websocket_manager()
        await ws_manager.broadcast_to_room(room_id, {
            "type": "participant_added",
            "room_id": room_id,
            "user_id": request.user_id,
            "added_by": current_user.user_id
        })
        
        logger.info(f"➕ User {current_user.username} added {request.user_id} to room {room_id} (share_history={request.share_history})")
        return {"success": True, "room_id": room_id, "user_id": request.user_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to add participant: {e}")
        raise HTTPException(status_code=500, detail="Failed to add participant")


# =====================
# MESSAGE ENDPOINTS
# =====================

@router.get("/rooms/{room_id}/messages")
async def get_room_messages(
    room_id: str,
    limit: int = Query(50, ge=1, le=100),
    before_message_id: Optional[str] = Query(None),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get messages from a room (paginated)"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        messages = await messaging_service.get_room_messages(
            room_id=room_id,
            user_id=current_user.user_id,
            limit=limit,
            before_message_id=before_message_id
        )
        
        return {
            "messages": messages,
            "total": len(messages),
            "has_more": len(messages) == limit
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get room messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to get messages")


@router.post("/rooms/{room_id}/messages")
async def send_message(
    room_id: str,
    request: SendMessageRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Send a message to a room"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        message = await messaging_service.send_message(
            room_id=room_id,
            sender_id=current_user.user_id,
            content=request.content,
            message_type=request.message_type,
            metadata=request.metadata
        )
        
        if not message:
            raise HTTPException(status_code=403, detail="Not authorized to send messages to this room")
        
        # Broadcast to room via WebSocket
        ws_manager = get_websocket_manager()
        await ws_manager.broadcast_to_room(
            room_id=room_id,
            message={
                "type": "new_message",
                "message": message
            },
            exclude_user_id=current_user.user_id
        )
        
        return message
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to send message: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message")


@router.delete("/messages/{message_id}")
async def delete_message(
    message_id: str,
    delete_for: str = Query("me", regex="^(me|everyone)$"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete a message"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        success = await messaging_service.delete_message(
            message_id=message_id,
            user_id=current_user.user_id,
            delete_for=delete_for
        )
        
        if not success:
            raise HTTPException(status_code=403, detail="Not authorized to delete this message")
        
        return {"success": True, "message_id": message_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete message: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete message")


# =====================
# ATTACHMENT ENDPOINTS
# =====================

@router.post("/rooms/{room_id}/messages/{message_id}/attachments")
async def upload_message_attachment(
    room_id: str,
    message_id: str,
    file: UploadFile = File(...),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Upload an attachment for a message"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        # Initialize attachment service if needed
        await messaging_attachment_service.initialize()
        
        attachment = await messaging_attachment_service.upload_attachment(
            room_id=room_id,
            message_id=message_id,
            file=file,
            user_id=current_user.user_id
        )
        
        return attachment
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload attachment: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload attachment")


@router.get("/attachments/{attachment_id}")
async def get_attachment(
    attachment_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get attachment metadata"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        await messaging_attachment_service.initialize()
        
        attachment = await messaging_attachment_service.get_attachment(
            attachment_id=attachment_id,
            user_id=current_user.user_id
        )
        
        if not attachment:
            raise HTTPException(status_code=404, detail="Attachment not found")
        
        return attachment
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get attachment: {e}")
        raise HTTPException(status_code=500, detail="Failed to get attachment")


@router.get("/attachments/{attachment_id}/file")
async def serve_attachment_file(
    attachment_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Serve attachment file"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        await messaging_attachment_service.initialize()
        
        return await messaging_attachment_service.serve_attachment_file(
            attachment_id=attachment_id,
            user_id=current_user.user_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve attachment: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve attachment")


@router.get("/messages/{message_id}/attachments")
async def get_message_attachments(
    message_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get all attachments for a message"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        await messaging_attachment_service.initialize()
        
        attachments = await messaging_attachment_service.get_message_attachments(
            message_id=message_id,
            user_id=current_user.user_id
        )
        
        return {"attachments": attachments, "total": len(attachments)}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get message attachments: {e}")
        raise HTTPException(status_code=500, detail="Failed to get attachments")


# =====================
# REACTION ENDPOINTS
# =====================

@router.post("/messages/{message_id}/reactions")
async def add_reaction(
    message_id: str,
    request: AddReactionRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Add emoji reaction to a message"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        reaction_id = await messaging_service.add_reaction(
            message_id=message_id,
            user_id=current_user.user_id,
            emoji=request.emoji
        )
        
        if not reaction_id:
            raise HTTPException(status_code=500, detail="Failed to add reaction")
        
        return {"success": True, "reaction_id": reaction_id, "emoji": request.emoji}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to add reaction: {e}")
        raise HTTPException(status_code=500, detail="Failed to add reaction")


@router.delete("/reactions/{reaction_id}")
async def remove_reaction(
    reaction_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Remove an emoji reaction"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        success = await messaging_service.remove_reaction(
            reaction_id=reaction_id,
            user_id=current_user.user_id
        )
        
        if not success:
            raise HTTPException(status_code=403, detail="Not authorized to remove this reaction")
        
        return {"success": True, "reaction_id": reaction_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to remove reaction: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove reaction")


# =====================
# PRESENCE ENDPOINTS
# =====================

@router.put("/presence")
async def update_presence(
    request: UpdatePresenceRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update current user's presence status"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        success = await messaging_service.update_user_presence(
            user_id=current_user.user_id,
            status=request.status,
            status_message=request.status_message
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update presence")
        
        # Broadcast presence update via WebSocket
        ws_manager = get_websocket_manager()
        await ws_manager.broadcast_presence_update(
            user_id=current_user.user_id,
            status=request.status,
            status_message=request.status_message
        )
        
        return {"success": True, "status": request.status}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update presence: {e}")
        raise HTTPException(status_code=500, detail="Failed to update presence")


@router.get("/presence/{user_id}")
async def get_user_presence(
    user_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get presence for a specific user"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        presence = await messaging_service.get_user_presence(user_id=user_id)
        
        if not presence:
            return {"user_id": user_id, "status": "offline", "last_seen_at": None}
        
        return presence
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get user presence: {e}")
        raise HTTPException(status_code=500, detail="Failed to get presence")


@router.get("/rooms/{room_id}/presence")
async def get_room_presence(
    room_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get presence for all participants in a room"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        presence_map = await messaging_service.get_room_participant_presence(room_id=room_id)
        
        return {"room_id": room_id, "presence": presence_map}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get room presence: {e}")
        raise HTTPException(status_code=500, detail="Failed to get room presence")


@router.get("/unread-counts")
async def get_unread_counts(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get unread message counts for all user's rooms"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        counts = await messaging_service.get_unread_counts(user_id=current_user.user_id)
        total_unread = sum(counts.values())
        
        return {
            "unread_by_room": counts,
            "total_unread": total_unread
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get unread counts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get unread counts")


# =====================
# USER LIST ENDPOINT (for creating rooms)
# =====================

@router.get("/users")
async def get_users_for_messaging(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get list of users for starting conversations"""
    try:
        if not settings.MESSAGING_ENABLED:
            raise HTTPException(status_code=503, detail="Messaging is not enabled")
        
        # Get all users from auth service
        from services.auth_service import auth_service
        users_response = await auth_service.get_users(skip=0, limit=1000)
        
        # Filter out current user from the list
        # users_response is a Pydantic model (UsersListResponse), so access attributes directly
        filtered_users = [
            user.dict() if hasattr(user, 'dict') else user
            for user in users_response.users
            if user.user_id != current_user.user_id
        ]
        
        logger.info(f"💬 User {current_user.username} fetched {len(filtered_users)} users for messaging")
        
        return {"users": filtered_users, "total": len(filtered_users)}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get users for messaging: {e}")
        raise HTTPException(status_code=500, detail="Failed to get users")


# =====================
# WEBSOCKET ENDPOINT
# =====================

@router.websocket("/ws/{room_id}")
async def websocket_room_endpoint(websocket: WebSocket, room_id: str):
    """
    WebSocket endpoint for real-time room messaging
    
    Requires token authentication via query parameter
    """
    logger.info(f"💬 Room WebSocket connection attempt for room: {room_id}")
    
    try:
        # Get token from query parameters
        token = websocket.query_params.get("token")
        if not token:
            logger.error(f"❌ Room WebSocket missing token for room: {room_id}")
            await websocket.close(code=4001, reason="Missing token")
            return
        
        logger.info("🔐 Room WebSocket token received")
        
        # Validate token and get user
        try:
            from utils.auth_middleware import decode_jwt_token
            payload = decode_jwt_token(token)
            user_id = payload.get("user_id")
            if not user_id:
                logger.error(f"❌ Room WebSocket invalid token for room: {room_id}")
                await websocket.close(code=4003, reason="Invalid token")
                return
        except Exception as e:
            logger.error(f"❌ Room WebSocket token validation failed for room {room_id}: {e}")
            await websocket.close(code=4003, reason="Invalid token")
            return
        
        logger.info(f"✅ Room WebSocket token validated for room: {room_id}, user: {user_id}")
        
        # Connect to WebSocket manager
        ws_manager = get_websocket_manager()
        await ws_manager.connect_to_room(websocket, room_id, user_id)
        logger.info(f"✅ Room WebSocket connected to manager for room: {room_id}")
        
        # Update user presence to online
        await messaging_service.update_user_presence(user_id, status='online')
        await ws_manager.broadcast_presence_update(user_id, 'online')
        
        try:
            # Keep connection alive and handle messages
            while True:
                # Receive message from client
                data = await websocket.receive_json()
                
                # Handle heartbeat/presence updates
                if data.get("type") == "heartbeat":
                    await messaging_service.update_user_presence(user_id, status='online')
                    await websocket.send_json({"type": "heartbeat_ack"})
                
                # Handle typing indicators
                elif data.get("type") == "typing":
                    await ws_manager.broadcast_to_room(
                        room_id=room_id,
                        message={
                            "type": "typing",
                            "user_id": user_id,
                            "is_typing": data.get("is_typing", True)
                        },
                        exclude_user_id=user_id
                    )
        
        except WebSocketDisconnect:
            logger.info(f"💬 Room WebSocket disconnected for room: {room_id}, user: {user_id}")
        finally:
            # Cleanup
            await ws_manager.disconnect_from_room(websocket, room_id, user_id)
            # Update presence to offline
            await messaging_service.update_user_presence(user_id, status='offline')
            await ws_manager.broadcast_presence_update(user_id, 'offline')
    
    except Exception as e:
        logger.error(f"❌ Room WebSocket error for room {room_id}: {e}")
        try:
            await websocket.close(code=4000, reason="Connection failed")
        except:
            pass


@router.websocket("/ws/user")
async def websocket_user_endpoint(websocket: WebSocket):
    """
    User-level WebSocket for notifications across ALL user rooms
    
    Notifies about:
    - New messages in any room
    - New rooms created
    - Room updates
    - Unread count changes
    """
    logger.info("💬 User WebSocket connection attempt")
    
    try:
        # Get token from query parameters
        token = websocket.query_params.get("token")
        if not token:
            logger.error("❌ User WebSocket missing token")
            await websocket.close(code=4001, reason="Missing token")
            return
        
        logger.info("🔐 User WebSocket token received")
        
        # Validate token and get user
        try:
            from utils.auth_middleware import decode_jwt_token
            payload = decode_jwt_token(token)
            user_id = payload.get("user_id")
            if not user_id:
                logger.error("❌ User WebSocket invalid token")
                await websocket.close(code=4003, reason="Invalid token")
                return
        except Exception as e:
            logger.error(f"❌ User WebSocket token validation failed: {e}")
            await websocket.close(code=4003, reason="Invalid token")
            return
        
        logger.info(f"✅ User WebSocket connected for user: {user_id}")
        
        # Accept the connection
        await websocket.accept()
        
        # Get user's rooms and connect to all of them
        ws_manager = get_websocket_manager()
        user_rooms = await messaging_service.get_user_rooms(user_id=user_id, limit=100)
        
        # Store user connection
        if not hasattr(ws_manager, 'user_connections'):
            ws_manager.user_connections = {}
        ws_manager.user_connections[user_id] = websocket
        
        # Connect to all user rooms for notifications
        for room in user_rooms:
            await ws_manager.connect_to_room(websocket, room['room_id'], user_id)
        
        logger.info(f"✅ User WebSocket connected to {len(user_rooms)} rooms for user: {user_id}")
        
        try:
            # Keep connection alive
            while True:
                data = await websocket.receive_json()
                
                # Handle heartbeat
                if data.get("type") == "heartbeat":
                    await websocket.send_json({"type": "heartbeat_ack"})
        
        except WebSocketDisconnect:
            logger.info(f"💬 User WebSocket disconnected for user: {user_id}")
        finally:
            # Cleanup - disconnect from all rooms
            if hasattr(ws_manager, 'user_connections'):
                ws_manager.user_connections.pop(user_id, None)
            
            for room in user_rooms:
                await ws_manager.disconnect_from_room(websocket, room['room_id'], user_id)
    
    except Exception as e:
        logger.error(f"❌ User WebSocket error: {e}")
        try:
            await websocket.close(code=4000, reason="Connection failed")
        except:
            pass

