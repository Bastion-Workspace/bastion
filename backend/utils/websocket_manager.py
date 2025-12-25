"""
WebSocket connection manager for real-time updates
"""

import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.session_connections: Dict[str, List[WebSocket]] = {}
        self.job_connections: Dict[str, List[WebSocket]] = {}  # Track connections by job_id
        self.conversation_connections: Dict[str, List[WebSocket]] = {}  # Track connections by conversation_id for agent status
        self.user_connections: Dict[str, List[WebSocket]] = {}  # Track connections by user_id for out-of-band updates
        self.room_connections: Dict[str, List[Dict[str, Any]]] = {}  # Track connections by room_id with user context
    
    async def connect(self, websocket: WebSocket, session_id: str = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if session_id:
            if session_id not in self.session_connections:
                self.session_connections[session_id] = []
            self.session_connections[session_id].append(websocket)
        
        logger.info(f"📡 WebSocket connected (session: {session_id})")
    
    async def connect_to_job(self, websocket: WebSocket, job_id: str):
        """Accept a WebSocket connection for job progress tracking"""
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            
            if job_id not in self.job_connections:
                self.job_connections[job_id] = []
            self.job_connections[job_id].append(websocket)
            
            logger.info(f"📡 WebSocket connected to job: {job_id}")
            logger.info(f"📡 Total job connections: {len(self.job_connections)}")
            logger.info(f"📡 Active connections for job {job_id}: {len(self.job_connections[job_id])}")
            logger.info(f"📡 All job IDs with connections: {list(self.job_connections.keys())}")
        except Exception as e:
            logger.error(f"❌ Failed to connect WebSocket to job {job_id}: {e}")
            raise
    
    async def connect_to_conversation(self, websocket: WebSocket, conversation_id: str, user_id: str):
        """
        ROOSEVELT'S AGENT STATUS CHANNEL: Accept WebSocket for real-time agent status updates
        This creates an out-of-band channel for LLM tool execution status
        """
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            
            # Track by conversation_id for targeted updates
            if conversation_id not in self.conversation_connections:
                self.conversation_connections[conversation_id] = []
            self.conversation_connections[conversation_id].append(websocket)
            
            # Also track by user_id for user-level broadcasts
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(websocket)
            
            logger.info(f"🤖 AGENT STATUS CHANNEL: WebSocket connected to conversation {conversation_id} for user {user_id}")
            logger.info(f"📊 Active conversation connections: {len(self.conversation_connections)}")
            logger.info(f"📊 Connections for this conversation: {len(self.conversation_connections[conversation_id])}")
        except Exception as e:
            logger.error(f"❌ Failed to connect WebSocket to conversation {conversation_id}: {e}")
            raise
    
    def disconnect(self, websocket: WebSocket, session_id: str = None):
        """Remove a WebSocket connection"""
        logger.info(f"🔌 Disconnecting WebSocket (session: {session_id})")
        
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"🧹 Removed from active connections")
        
        if session_id and session_id in self.session_connections:
            if websocket in self.session_connections[session_id]:
                self.session_connections[session_id].remove(websocket)
                logger.info(f"🧹 Removed from session connections for {session_id}")
            
            # Clean up empty session lists
            if not self.session_connections[session_id]:
                del self.session_connections[session_id]
                logger.info(f"🧹 Cleaned up empty session {session_id}")
        
        # Also clean up from job connections
        disconnected_jobs = []
        for job_id, connections in self.job_connections.items():
            if websocket in connections:
                connections.remove(websocket)
                disconnected_jobs.append(job_id)
                logger.info(f"🧹 Removed from job connections for {job_id}")
        
        # Clean up empty job connection lists
        empty_jobs = [job_id for job_id, connections in self.job_connections.items() if not connections]
        for job_id in empty_jobs:
            del self.job_connections[job_id]
            logger.info(f"🧹 Cleaned up empty job connections for {job_id}")
        
        # ROOSEVELT'S AGENT STATUS: Clean up from conversation connections
        disconnected_conversations = []
        for conversation_id, connections in self.conversation_connections.items():
            if websocket in connections:
                connections.remove(websocket)
                disconnected_conversations.append(conversation_id)
                logger.info(f"🧹 Removed from conversation connections for {conversation_id}")
        
        # Clean up empty conversation connection lists
        empty_conversations = [conv_id for conv_id, connections in self.conversation_connections.items() if not connections]
        for conv_id in empty_conversations:
            del self.conversation_connections[conv_id]
            logger.info(f"🧹 Cleaned up empty conversation connections for {conv_id}")
        
        # Clean up from user connections
        for user_id, connections in self.user_connections.items():
            if websocket in connections:
                connections.remove(websocket)
                logger.info(f"🧹 Removed from user connections for {user_id}")
        
        # Clean up empty user connection lists
        empty_users = [uid for uid, connections in self.user_connections.items() if not connections]
        for uid in empty_users:
            del self.user_connections[uid]
            logger.info(f"🧹 Cleaned up empty user connections for {uid}")
        
        if disconnected_jobs:
            logger.info(f"📡 WebSocket disconnected from jobs: {disconnected_jobs}")
        elif disconnected_conversations:
            logger.info(f"🤖 WebSocket disconnected from conversations: {disconnected_conversations}")
        else:
            logger.info(f"📡 WebSocket disconnected (session: {session_id})")

    async def send_personal_message(self, message: Any, websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        try:
            if isinstance(message, dict):
                message = json.dumps(message)
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"❌ Failed to send WebSocket message: {e}")
            # Remove broken connection
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def send_to_session(self, message: Any, session_id: str):
        """Send a message to all connections in a session"""
        if session_id in self.session_connections:
            broken_connections = []
            
            for websocket in self.session_connections[session_id]:
                try:
                    if isinstance(message, dict):
                        message_str = json.dumps(message)
                    else:
                        message_str = str(message)
                    await websocket.send_text(message_str)
                except Exception as e:
                    logger.error(f"❌ Failed to send message to session {session_id}: {e}")
                    broken_connections.append(websocket)
            
            # Clean up broken connections
            for websocket in broken_connections:
                self.disconnect(websocket, session_id)

    async def send_to_job(self, message: Any, job_id: str):
        """Send a message to all connections tracking a specific job"""
        logger.info(f"📡 Attempting to send message to job {job_id}")
        logger.info(f"📡 Job connections available: {list(self.job_connections.keys())}")
        logger.info(f"📡 Total active connections: {len(self.active_connections)}")
        
        if job_id in self.job_connections:
            connections = self.job_connections[job_id]
            logger.info(f"📡 Found {len(connections)} connections for job {job_id}")
            broken_connections = []
            
            for i, websocket in enumerate(connections):
                try:
                    if isinstance(message, dict):
                        message_str = json.dumps(message)
                    else:
                        message_str = str(message)
                    await websocket.send_text(message_str)
                    logger.info(f"✅ Successfully sent message to job {job_id} connection {i+1}/{len(connections)}")
                except Exception as e:
                    logger.error(f"❌ Failed to send message to job {job_id} connection {i+1}: {e}")
                    broken_connections.append(websocket)
            
            # Clean up broken connections
            for websocket in broken_connections:
                if websocket in self.job_connections[job_id]:
                    self.job_connections[job_id].remove(websocket)
                    logger.info(f"🧹 Removed broken connection for job {job_id}")
        else:
            logger.warning(f"⚠️ No connections found for job {job_id}")
            logger.info(f"📡 Available job connections: {list(self.job_connections.keys())}")
            logger.info(f"📡 Job ID being looked for: {job_id}")
            logger.info(f"📡 Job ID type: {type(job_id)}")
            for available_job_id in self.job_connections.keys():
                logger.info(f"📡 Available job ID: {available_job_id} (type: {type(available_job_id)})")
                if str(job_id) == str(available_job_id):
                    logger.info(f"📡 String comparison matches for job {job_id}")

    async def broadcast(self, message: Any):
        """Send a message to all active connections"""
        broken_connections = []
        
        for websocket in self.active_connections:
            try:
                if isinstance(message, dict):
                    message_str = json.dumps(message)
                else:
                    message_str = str(message)
                await websocket.send_text(message_str)
            except Exception as e:
                logger.error(f"❌ Failed to broadcast message: {e}")
                broken_connections.append(websocket)
        
        # Clean up broken connections
        for websocket in broken_connections:
            self.disconnect(websocket)

    async def send_folder_update(self, message: Any, user_id: str = None):
        """Send folder/document update to appropriate user sessions"""
        try:
            if isinstance(message, dict):
                message["type"] = "folder_update"
                message["timestamp"] = message.get("timestamp", datetime.now().isoformat())
            
            if user_id:
                # Send to specific user's sessions
                await self.send_to_session(message, user_id)
                logger.info(f"📁 Sent folder update to user {user_id}")
            else:
                # Global folder update - send to all sessions
                await self.broadcast(message)
                logger.info(f"📁 Broadcasted global folder update")
                
        except Exception as e:
            logger.error(f"❌ Failed to send folder update: {e}")

    async def send_document_status_update(self, document_id: str, status: str, folder_id: str = None, user_id: str = None, filename: str = None, proposal_data: Dict[str, Any] = None):
        """Send document status update to appropriate sessions - **BULLY!** Now with filename for toast notifications!"""
        try:
            message = {
                "type": "document_status_update",
                "document_id": document_id,
                "status": status,
                "folder_id": folder_id,
                "filename": filename,  # **ROOSEVELT FIX**: Include filename for UI toast notifications!
                "timestamp": datetime.now().isoformat()
            }
            
            # If status is edit_proposal and proposal_data is provided, send as separate message
            if status == "edit_proposal" and proposal_data:
                proposal_message = {
                    "type": "document_edit_proposal",
                    "document_id": document_id,
                    "proposal_id": proposal_data.get("proposal_id"),
                    "edit_type": proposal_data.get("edit_type"),
                    "operations": proposal_data.get("operations"),
                    "content_edit": proposal_data.get("content_edit"),
                    "agent_name": proposal_data.get("agent_name"),
                    "summary": proposal_data.get("summary"),
                    "timestamp": datetime.now().isoformat()
                }
                
                if user_id:
                    await self.send_to_session(proposal_message, user_id)
                    logger.info(f"📝 Sent document edit proposal to user {user_id}: {document_id} (proposal: {proposal_data.get('proposal_id')})")
                else:
                    await self.broadcast(proposal_message)
                    logger.info(f"📝 Broadcasted document edit proposal: {document_id} (proposal: {proposal_data.get('proposal_id')})")
            
            if user_id:
                await self.send_to_session(message, user_id)
                logger.info(f"📄 Sent document status update to user {user_id}: {document_id} ({filename}) -> {status}")
            else:
                # Global document update
                await self.broadcast(message)
                logger.info(f"📄 Broadcasted global document status update: {document_id} ({filename}) -> {status}")
                
        except Exception as e:
            logger.error(f"❌ Failed to send document status update: {e}")
    
    async def send_agent_status(
        self, 
        conversation_id: str, 
        user_id: str,
        status_type: str,
        message: str,
        agent_type: str = None,
        tool_name: str = None,
        iteration: int = None,
        max_iterations: int = None,
        metadata: Dict[str, Any] = None
    ):
        """
        ROOSEVELT'S AGENT STATUS STREAMING: Send real-time agent tool execution status
        
        This is the OUT-OF-BAND channel for LLM status updates that appear/disappear
        as the agent works through its iterative research process.
        
        Args:
            conversation_id: Target conversation
            user_id: Target user (for security)
            status_type: tool_start, tool_complete, tool_error, iteration_start, synthesis
            message: Human-readable status message
            agent_type: research_agent, chat_agent, etc.
            tool_name: search_local, search_and_crawl, etc.
            iteration: Current iteration number (1-8)
            max_iterations: Maximum iterations (8)
            metadata: Additional context
        """
        try:
            status_message = {
                "type": "agent_status",
                "conversation_id": conversation_id,
                "status_type": status_type,
                "message": message,
                "agent_type": agent_type,
                "tool_name": tool_name,
                "iteration": iteration,
                "max_iterations": max_iterations,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to conversation-specific connections
            sent_to_conversation = False
            if conversation_id in self.conversation_connections:
                connections = self.conversation_connections[conversation_id]
                logger.info(f"🤖 AGENT STATUS: Sending to {len(connections)} conversation connections for {conversation_id}")
                
                broken_connections = []
                for websocket in connections:
                    try:
                        await websocket.send_text(json.dumps(status_message))
                        sent_to_conversation = True
                    except Exception as e:
                        logger.error(f"❌ Failed to send agent status to conversation {conversation_id}: {e}")
                        broken_connections.append(websocket)
                
                # Clean up broken connections
                for websocket in broken_connections:
                    if websocket in self.conversation_connections[conversation_id]:
                        self.conversation_connections[conversation_id].remove(websocket)
            
            # Fallback: Send to user connections if no conversation-specific connection
            if not sent_to_conversation and user_id in self.user_connections:
                connections = self.user_connections[user_id]
                logger.info(f"🤖 AGENT STATUS: Fallback to user connections - sending to {len(connections)} user connections for {user_id}")
                
                broken_connections = []
                for websocket in connections:
                    try:
                        await websocket.send_text(json.dumps(status_message))
                    except Exception as e:
                        logger.error(f"❌ Failed to send agent status to user {user_id}: {e}")
                        broken_connections.append(websocket)
                
                # Clean up broken connections
                for websocket in broken_connections:
                    if websocket in self.user_connections[user_id]:
                        self.user_connections[user_id].remove(websocket)
            
            if sent_to_conversation or (user_id in self.user_connections):
                logger.info(f"✅ AGENT STATUS SENT: {status_type} - {message} (conv: {conversation_id[:8]}...)")
            else:
                logger.debug(f"📡 AGENT STATUS: No active connections for conversation {conversation_id} or user {user_id}")
                
        except Exception as e:
            logger.error(f"❌ Failed to send agent status: {e}")
    
    # =====================
    # ROOSEVELT'S MESSAGING CAVALRY
    # Room-based messaging WebSocket methods
    # =====================
    
    async def connect_to_room(self, websocket: WebSocket, room_id: str, user_id: str):
        """
        Connect a WebSocket to a chat room
        
        Args:
            websocket: WebSocket connection
            room_id: Room UUID
            user_id: User ID of the connected user
        """
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            
            # Track by room_id with user context
            if room_id not in self.room_connections:
                self.room_connections[room_id] = []
            
            self.room_connections[room_id].append({
                "websocket": websocket,
                "user_id": user_id
            })
            
            # Also track by user_id for presence updates - ALWAYS AS A LIST
            if not hasattr(self, 'user_connections'):
                self.user_connections = {}
                
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            
            if websocket not in self.user_connections[user_id]:
                self.user_connections[user_id].append(websocket)
            
            logger.info(f"💬 WebSocket connected to room {room_id} for user {user_id}")
            logger.info(f"📊 Active room connections: {len(self.room_connections)}")
            logger.info(f"📊 Connections for this user: {len(self.user_connections[user_id])}")
        except Exception as e:
            logger.error(f"❌ Failed to connect WebSocket to room {room_id}: {e}")
            raise

    async def disconnect_from_room(self, websocket: WebSocket, room_id: str, user_id: str):
        """
        Disconnect a WebSocket from a chat room
        
        Args:
            websocket: WebSocket connection
            room_id: Room UUID
            user_id: User ID
        """
        if room_id in self.room_connections:
            # Remove from room connections
            self.room_connections[room_id] = [
                conn for conn in self.room_connections[room_id]
                if conn["websocket"] != websocket
            ]
            
            # Clean up empty room lists
            if not self.room_connections[room_id]:
                del self.room_connections[room_id]
                logger.info(f"🧹 Cleaned up empty room {room_id}")
        
        # Also remove from user_connections - handle list correctly
        if hasattr(self, 'user_connections') and user_id in self.user_connections:
            if isinstance(self.user_connections[user_id], list):
                self.user_connections[user_id] = [
                    ws for ws in self.user_connections[user_id]
                    if ws != websocket
                ]
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
                    logger.info(f"🧹 No more active connections for user {user_id}")
            elif self.user_connections[user_id] == websocket:
                del self.user_connections[user_id]
                logger.info(f"🧹 No more active connections for user {user_id}")

        # Also remove from active connections
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        logger.info(f"💬 WebSocket disconnected from room {room_id} for user {user_id}")

    def is_user_connected(self, user_id: str) -> bool:
        """Check if a user has any active WebSocket connections"""
        # 1. Check global user-level connections list
        if hasattr(self, 'user_connections') and user_id in self.user_connections:
            conns = self.user_connections[user_id]
            if isinstance(conns, list) and len(conns) > 0:
                return True
            elif not isinstance(conns, list) and conns:
                return True
        
        # 2. Fallback: Check all room connections as a safety measure
        for connections in self.room_connections.values():
            if any(conn["user_id"] == user_id for conn in connections):
                return True
            
        return False
    
    async def broadcast_to_room(
        self, 
        room_id: str, 
        message: Dict[str, Any],
        exclude_user_id: str = None
    ):
        """
        Broadcast a message to all participants in a room
        
        Args:
            room_id: Room UUID
            message: Message data to broadcast
            exclude_user_id: Optional user ID to exclude (e.g., message sender)
        """
        # 1. Deliver to everyone actively watching this room
        sent_to_room = 0
        if room_id in self.room_connections:
            broken_connections = []
            for conn_info in self.room_connections[room_id]:
                websocket = conn_info["websocket"]
                user_id = conn_info["user_id"]
                
                if exclude_user_id and user_id == exclude_user_id:
                    continue
                
                try:
                    await websocket.send_json(message)
                    sent_to_room += 1
                except Exception as e:
                    logger.error(f"❌ Failed to send message to user {user_id} in room {room_id}: {e}")
                    broken_connections.append(conn_info)
            
            for conn_info in broken_connections:
                if conn_info in self.room_connections[room_id]:
                    self.room_connections[room_id].remove(conn_info)

        # 2. For new messages, also deliver to the global user-level WebSockets of all participants
        # This ensures unread counts update even if the room isn't open
        sent_to_users = 0
        if message.get("type") == "new_message":
            try:
                from services.messaging.messaging_service import messaging_service
                # We need the participant list to know who to notify globally
                # (Ideally we'd have this in the message or state, but fetching from DB is safest for now)
                participants = await messaging_service.get_room_participants(room_id)
                participant_ids = [p['user_id'] for p in participants]
                
                for p_id in participant_ids:
                    if exclude_user_id and p_id == exclude_user_id:
                        continue
                    
                    # Skip if they already received it via the room connection
                    # (Simple check: if they have an active room connection, we skip the global one to avoid duplicates)
                    if room_id in self.room_connections and any(c["user_id"] == p_id for c in self.room_connections[room_id]):
                        continue
                        
                    if hasattr(self, 'user_connections') and p_id in self.user_connections:
                        connections = self.user_connections[p_id]
                        for websocket in (connections if isinstance(connections, list) else [connections]):
                            try:
                                await websocket.send_json(message)
                                sent_to_users += 1
                            except:
                                pass
            except Exception as e:
                logger.error(f"❌ Failed to broadcast new message globally: {e}")

        if sent_to_room > 0 or sent_to_users > 0:
            logger.info(f"✅ Broadcast message type '{message.get('type')}' to {sent_to_room} room and {sent_to_users} user connections")
        else:
            logger.debug(f"📡 No recipients for broadcast in room {room_id}")
    
    async def broadcast_presence_update(self, user_id: str, status: str, status_message: str = None):
        """
        Broadcast user presence update to all relevant rooms and interested participants
        
        Args:
            user_id: User whose presence changed
            status: New status ('online', 'offline', 'away')
            status_message: Optional status message
        """
        presence_message = {
            "type": "presence_update",
            "user_id": user_id,
            "status": status,
            "status_message": status_message,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Find all rooms this user belongs to (from DB to include rooms they just left)
        # We use the messaging_service to get the rooms
        try:
            from services.messaging.messaging_service import messaging_service
            user_rooms = await messaging_service.get_user_rooms(user_id=user_id, limit=100, include_participants=False)
            room_ids = [r['room_id'] for r in user_rooms]
            
            # 1. Broadcast to all active room connections for these rooms
            for room_id in room_ids:
                if room_id in self.room_connections:
                    await self.broadcast_to_room(room_id, presence_message)
            
            # 2. Broadcast to all active user-level connections
            # Since we don't want to spam everyone, we only broadcast to those who share a room
            # but for simplicity and to ensure it works, we can broadcast to all active user connections
            # who are NOT the user themselves.
            if hasattr(self, 'user_connections'):
                for other_user_id, connections in self.user_connections.items():
                    if other_user_id == user_id:
                        continue
                    
                    for websocket in (connections if isinstance(connections, list) else [connections]):
                        try:
                            await websocket.send_json(presence_message)
                        except:
                            pass

            logger.info(f"✅ Broadcast presence update for user {user_id} ({status}) to all active connections")
            
        except Exception as e:
            logger.error(f"❌ Failed to broadcast presence update: {e}")

    async def broadcast_to_users(self, user_ids: List[str], message: Dict[str, Any]):
        """
        Broadcast a message to the user-level WebSockets of specific users
        
        Args:
            user_ids: List of user IDs to notify
            message: Message data to broadcast
        """
        if not hasattr(self, 'user_connections'):
            return
            
        sent_count = 0
        for user_id in user_ids:
            if user_id in self.user_connections:
                connections = self.user_connections[user_id]
                for websocket in (connections if isinstance(connections, list) else [connections]):
                    try:
                        await websocket.send_json(message)
                        sent_count += 1
                    except:
                        pass
        
        if sent_count > 0:
            logger.info(f"✅ Broadcast message type '{message.get('type')}' to {sent_count} user connections")

    def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)

    def get_session_count(self) -> int:
        """Get the number of active sessions"""
        return len(self.session_connections)
    
    def get_job_connection_count(self, job_id: str) -> int:
        """Get the number of connections tracking a specific job"""
        return len(self.job_connections.get(job_id, []))
    
    def get_session_connection_count(self, session_id: str) -> int:
        """Get the number of connections in a specific session"""
        return len(self.session_connections.get(session_id, []))
    
    def get_conversation_connection_count(self, conversation_id: str) -> int:
        """Get the number of connections tracking a specific conversation"""
        return len(self.conversation_connections.get(conversation_id, []))
    
    def get_user_connection_count(self, user_id: str) -> int:
        """Get the number of connections for a specific user"""
        return len(self.user_connections.get(user_id, []))
    
    def get_room_connection_count(self, room_id: str) -> int:
        """Get the number of connections in a specific room"""
        return len(self.room_connections.get(room_id, []))


# Global WebSocket manager instance
_websocket_manager_instance = None


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance (singleton pattern)"""
    global _websocket_manager_instance
    if _websocket_manager_instance is None:
        _websocket_manager_instance = WebSocketManager()
        logger.info("🔌 Created new WebSocket manager instance")
    return _websocket_manager_instance
