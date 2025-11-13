"""
Roosevelt's Messaging Tools
Tools for LangGraph messaging agent

BULLY! Tools for the messaging cavalry!
"""

import logging
from typing import List, Dict, Any
from services.messaging.messaging_service import messaging_service

logger = logging.getLogger(__name__)

# Async wrapper functions for tool registry

async def get_user_rooms_tool(user_id: str, limit: int = 20) -> str:
    """
    Get list of user's chat rooms
    
    Args:
        user_id: User ID to get rooms for
        limit: Maximum number of rooms to return
    
    Returns:
        JSON string with rooms list
    """
    try:
        rooms = await messaging_service.get_user_rooms(user_id, limit, include_participants=True)
        
        # Format for LLM consumption
        room_list = []
        for room in rooms:
            room_info = {
                "room_id": room["room_id"],
                "name": room.get("display_name") or room.get("room_name") or "Unnamed Room",
                "type": room["room_type"],
                "participants": [
                    {
                        "user_id": p["user_id"],
                        "name": p.get("display_name") or p["username"]
                    }
                    for p in room.get("participants", [])
                ],
                "unread_count": room.get("unread_count", 0)
            }
            room_list.append(room_info)
        
        import json
        return json.dumps({"rooms": room_list, "count": len(room_list)})
    
    except Exception as e:
        logger.error(f"❌ Failed to get user rooms: {e}")
        return json.dumps({"error": str(e), "rooms": [], "count": 0})


async def send_room_message_tool(
    user_id: str,
    room_id: str,
    message_content: str,
    message_type: str = "text"
) -> str:
    """
    Send a message to a chat room
    
    Args:
        user_id: User ID sending the message
        room_id: Room UUID to send message to
        message_content: The message text
        message_type: Type of message (text, ai_share, system)
    
    Returns:
        JSON string with result
    """
    try:
        message = await messaging_service.send_message(
            room_id=room_id,
            sender_id=user_id,
            content=message_content,
            message_type=message_type,
            metadata={"source": "messaging_agent"}
        )
        
        if message:
            import json
            return json.dumps({
                "success": True,
                "message_id": message["message_id"],
                "room_id": room_id,
                "content": message_content
            })
        else:
            import json
            return json.dumps({
                "success": False,
                "error": "Failed to send message - not authorized or room not found"
            })
    
    except Exception as e:
        logger.error(f"❌ Failed to send room message: {e}")
        import json
        return json.dumps({"success": False, "error": str(e)})

