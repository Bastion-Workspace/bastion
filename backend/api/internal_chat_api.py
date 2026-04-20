"""
Internal API for service-to-service calls (connections-service -> backend).
Authenticated by X-Internal-Service-Key. Handles external-chat: create/find conversation,
store messages, call orchestrator, return response and images.
Also: agent-initiated conversations and outbound chat_id persistence.
"""

import base64
import json
import logging
import mimetypes
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import grpc
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

from config import settings
from services.conversation_service import ConversationService
from services.grpc_context_gatherer import get_context_gatherer
from services.settings_service import settings_service
from services.user_settings_kv_service import get_user_setting
from services.user_llm_provider_service import user_llm_provider_service
from services.microphone_stt_service import (
    MAX_TRANSCRIBE_AUDIO_BYTES,
    transcribe_audio_bytes_for_user,
)

logger = logging.getLogger(__name__)

# Pattern for document file URLs we can resolve to bytes (for Telegram/Discord)
_DOCUMENT_FILE_URL_PATTERN = re.compile(
    r"^/api/documents/([a-f0-9-]{36})/file$", re.IGNORECASE
)
_API_IMAGES_PREFIX = "/api/images/"
_API_WS_IMAGES_PREFIX = "/api/web-sources/images/"
_STATIC_IMAGES_PREFIX = "/static/images/"


def _model_display_name(model_id: str) -> str:
    """Derive a readable name from a model ID (e.g. anthropic/claude-sonnet-4 -> Anthropic Claude Sonnet 4)."""
    if not model_id:
        return model_id or ""
    return model_id.replace("/", " ").replace("-", " ").strip().title() or model_id

router = APIRouter(prefix="/api/internal", tags=["internal"])


class ExternalChatRequest(BaseModel):
    user_id: str
    conversation_id: str
    query: str
    platform: str  # "telegram" or "discord"
    platform_chat_id: str
    sender_name: str = ""
    images: Optional[List[Dict[str, Any]]] = None  # [{"data": base64}, {"url": "..."}] or [{"data": base64, "mime": "image/jpeg"}]
    start_new_conversation: bool = False  # when True: only ensure conversation exists, no user message, no orchestrator


class ExternalChatModelsRequest(BaseModel):
    """Request for listing enabled models (for /model command)."""
    user_id: str
    conversation_id: Optional[str] = None  # when set, response includes current_model_id for this conversation


class ExternalChatSetModelRequest(BaseModel):
    """Request for setting the model for a conversation (e.g. /model 3)."""
    user_id: str
    conversation_id: str
    model_index: int  # 1-based index into enabled models list


async def verify_internal_service_key(request: Request) -> None:
    key = request.headers.get("X-Internal-Service-Key")
    if not settings.INTERNAL_SERVICE_KEY or key != settings.INTERNAL_SERVICE_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing service key")


_CAPTION_MAX_LEN = 1024  # Telegram photo caption API limit; captions longer than this are truncated


def _build_image_caption(metadata: Dict[str, Any]) -> Optional[str]:
    """
    Build a single-line caption from image metadata for Telegram/Discord.
    Order: title, match_reason, tags, short content snippet (so "Why it matches" survives truncation).
    Cap at 1024 chars (Telegram limit). Returns None when metadata has nothing useful.
    """
    if not metadata:
        return None
    title = (metadata.get("title") or "").strip()
    match_reason = (metadata.get("match_reason") or "").strip()
    content = (metadata.get("content") or "").strip()
    tags_raw = metadata.get("tags")
    tags_list = tags_raw if isinstance(tags_raw, list) else ([tags_raw] if tags_raw else [])
    tags_str = ", ".join(str(t) for t in tags_list if t) if tags_list else ""

    parts = []
    if title:
        parts.append(title)
    if match_reason:
        parts.append(f"Why it matches: {match_reason}")
    if tags_str:
        parts.append(f"Tags: {tags_str}")
    if content:
        snippet = content[:120].rstrip()
        if len(content) > 120:
            snippet += "..."
        parts.append(snippet)

    if not parts:
        return None
    caption = " ".join(parts)
    return caption[:_CAPTION_MAX_LEN] if len(caption) > _CAPTION_MAX_LEN else caption


def _extract_images_from_markdown(text: str) -> List[Dict[str, Any]]:
    """Extract image refs from markdown ![alt](url) or ![alt](data:mime;base64,...)."""
    images: List[Dict[str, Any]] = []
    # Match ![alt](url) or ![alt](data:...)
    for m in re.finditer(r"!\[([^\]]*)\]\((data:[^)]+|[^)]+)\)", text):
        src = m.group(2)
        if src.startswith("data:"):
            parts = src.split(",", 1)
            if len(parts) == 2:
                header = parts[0]  # data:image/png;base64
                b64 = parts[1]
                mime = "image/png"
                if ";" in header:
                    mime = header.split(";")[0].replace("data:", "")
                images.append({"data": b64, "mime": mime})
        else:
            images.append({"url": src})
    return images


async def _get_document_file_bytes(
    user_id: str, doc_id: str
) -> Optional[Tuple[bytes, str]]:
    """
    Resolve document file to bytes for external chat (Telegram/Discord).
    Uses same access rules as document API: global read, team membership, user ownership.
    Returns (bytes, mime) or None if not found or no access.
    """
    try:
        from services.service_container import get_service_container

        container = await get_service_container()
        document_service = container.document_service
    except Exception as e:
        logger.warning("Internal document resolve: service container failed: %s", e)
        return None

    doc_info = await document_service.get_document(doc_id)
    if not doc_info:
        return None

    collection_type = getattr(doc_info, "collection_type", "user")
    doc_user_id = getattr(doc_info, "user_id", None)
    doc_team_id = getattr(doc_info, "team_id", None)

    if collection_type == "global":
        pass
    elif doc_team_id:
        from api.teams_api import team_service

        role = await team_service.check_team_access(doc_team_id, user_id)
        if not role:
            return None
    elif doc_user_id != user_id:
        return None

    filename = getattr(doc_info, "filename", None)
    if not filename:
        return None
    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename in (".", ".."):
        return None

    try:
        from clients.document_service_client import get_document_service_client

        dsc = get_document_service_client()
        await dsc.initialize(required=True)
        chunks: List[bytes] = []
        async for ch in dsc.download_document_stream(
            doc_id,
            user_id,
            role="",
        ):
            if ch.data:
                chunks.append(ch.data)
        data = b"".join(chunks)
    except Exception as e:
        logger.warning("Internal document resolve: DS download failed for %s: %s", doc_id, e)
        return None

    if not data:
        return None

    mime, _ = mimetypes.guess_type(safe_filename)
    if not mime:
        mime = "application/octet-stream"
    return (data, mime)


def _get_image_file_bytes_from_url(url: str) -> Optional[Tuple[bytes, str]]:
    """
    Resolve image URLs under WEB_SOURCES_ROOT/images to bytes (external chat).
    Accepts /api/images/, /api/web-sources/images/, and legacy /static/images/.
    """
    if not url:
        return None
    u = url.strip()
    from urllib.parse import unquote

    if u.startswith(_API_WS_IMAGES_PREFIX):
        prefix = _API_WS_IMAGES_PREFIX
    elif u.startswith(_API_IMAGES_PREFIX):
        prefix = _API_IMAGES_PREFIX
    elif u.startswith(_STATIC_IMAGES_PREFIX):
        prefix = _STATIC_IMAGES_PREFIX
    else:
        return None

    try:
        relative = unquote(u[len(prefix) :].lstrip("/"))
        if not relative:
            return None
        parts = Path(relative).parts
        if not parts or any(p in (".", "..") for p in parts):
            return None
        images_base = Path(settings.WEB_SOURCES_ROOT) / "images"
        image_file_path = (images_base / relative).resolve()
        if not str(image_file_path).startswith(str(images_base.resolve())):
            return None
        if not image_file_path.is_file():
            for subdir in images_base.iterdir() if images_base.exists() else []:
                if subdir.is_dir():
                    candidate = subdir / Path(relative).name
                    if candidate.is_file():
                        image_file_path = candidate
                        break
            else:
                return None
        data = image_file_path.read_bytes()
        mime, _ = mimetypes.guess_type(str(image_file_path))
        if not mime or not mime.startswith("image/"):
            mime = "image/png"
        return (data, mime)
    except Exception as e:
        logger.debug("Internal image URL resolve failed for %s: %s", url[:80], e)
        return None


def _strip_resolved_image_refs(text: str, resolved_urls: List[str]) -> str:
    """Remove markdown image refs whose URLs are in resolved_urls (so Telegram gets clean text)."""
    if not resolved_urls:
        return text
    resolved_set = set(resolved_urls)

    def replace_one(match: re.Match) -> str:
        src = match.group(2)
        return "" if src in resolved_set else match.group(0)

    out = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_one, text)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


@router.post("/transcribe-audio", dependencies=[Depends(verify_internal_service_key)])
async def internal_transcribe_audio(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """
    Multipart STT for connections-service (Telegram/Discord/Slack voice).
    Same provider stack as /api/audio/transcribe for the given user_id.
    """
    uid = (user_id or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id is required")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(audio_bytes) > MAX_TRANSCRIBE_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio exceeds maximum size ({MAX_TRANSCRIBE_AUDIO_BYTES // (1024 * 1024)}MB)",
        )

    try:
        text = await transcribe_audio_bytes_for_user(
            uid,
            audio_bytes,
            file.filename,
            content_type=file.content_type,
            prompt=(prompt.strip() if prompt and prompt.strip() else None),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        logger.error("internal transcribe-audio failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {"success": True, "text": text}


@router.post("/external-chat", dependencies=[Depends(verify_internal_service_key)])
async def external_chat(body: ExternalChatRequest) -> Dict[str, Any]:
    """
    Handle inbound message from Telegram/Discord: ensure conversation exists,
    store user message, call orchestrator, store assistant message, return response and images.
    When start_new_conversation=True, only ensure conversation exists and return a fixed response (no message, no LLM).
    """
    try:
        conversation_service = ConversationService()
        conversation_service.set_current_user(body.user_id)

        if body.start_new_conversation:
            await conversation_service.ensure_conversation_exists(body.conversation_id, body.user_id)
            await conversation_service.update_conversation_metadata(
                body.conversation_id,
                body.user_id,
                {"source_platform": body.platform, "platform_chat_id": body.platform_chat_id},
            )
            return {
                "response": "Started a new chat. How can I help?",
                "images": [],
                "conversation_id": body.conversation_id,
            }

        await conversation_service.add_message(
            conversation_id=body.conversation_id,
            user_id=body.user_id,
            role="user",
            content=body.query,
            metadata={
                "orchestrator_system": True,
                "streaming": False,
                "source_platform": body.platform,
                "platform_chat_id": body.platform_chat_id,
                "sender_name": body.sender_name,
            },
        )
        await conversation_service.update_conversation_metadata(
            body.conversation_id,
            body.user_id,
            {"source_platform": body.platform, "platform_chat_id": body.platform_chat_id},
        )

        lifecycle_info = await conversation_service.lifecycle_manager.get_conversation_lifecycle(
            body.conversation_id, body.user_id
        )
        metadata = (lifecycle_info or {}).get("metadata_json") or {}
        selected_model_id = metadata.get("external_chat_selected_model_id")
        agent_profile_id = metadata.get("agent_profile_id")
        request_context: Dict[str, Any] = {}
        request_context["editor_preference"] = "ignore"
        if agent_profile_id:
            request_context["agent_profile_id"] = str(agent_profile_id)
        else:
            request_context["context_window_size"] = 30
        if selected_model_id:
            request_context["user_chat_model"] = selected_model_id

        context_gatherer = get_context_gatherer()
        grpc_request = await context_gatherer.build_chat_request(
            query=body.query,
            user_id=body.user_id,
            conversation_id=body.conversation_id,
            session_id="default",
            request_context=request_context,
            state=None,
            agent_type=None,
            routing_reason=None,
        )

        orchestrator_host = "llm-orchestrator"
        orchestrator_port = 50051
        options = [
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ]
        accumulated_response = ""
        metadata_received: Dict[str, Any] = {}
        agent_name_used: Optional[str] = None

        async with grpc.aio.insecure_channel(
            f"{orchestrator_host}:{orchestrator_port}", options=options
        ) as channel:
            from protos import orchestrator_pb2_grpc
            stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
            async for chunk in stub.StreamChat(grpc_request):
                if chunk.agent_name and chunk.agent_name not in ("orchestrator", "system"):
                    agent_name_used = chunk.agent_name
                if chunk.type == "content" and chunk.message:
                    accumulated_response += chunk.message
                if chunk.metadata:
                    metadata_received.update(dict(chunk.metadata))

        from utils.history_metadata import (
            filter_history_safe_metadata,
            sanitize_images_for_persistence,
        )
        save_metadata: Dict[str, Any] = {
            "orchestrator_system": True,
            "streaming": False,
            "source_platform": body.platform,
            "platform_chat_id": body.platform_chat_id,
            "sender_name": body.sender_name,
        }
        persist_meta = dict(metadata_received)
        sanitize_images_for_persistence(persist_meta)
        save_metadata.update(filter_history_safe_metadata(persist_meta))
        await conversation_service.add_message(
            conversation_id=body.conversation_id,
            user_id=body.user_id,
            role="assistant",
            content=accumulated_response,
            metadata=save_metadata,
        )

        agent_for_metadata = agent_name_used or metadata_received.get("delegated_agent") or metadata_received.get("agent_name")
        if agent_for_metadata:
            try:
                agent_profile_id_saved = metadata_received.get("agent_profile_id")
                await conversation_service.update_agent_metadata(
                    conversation_id=body.conversation_id,
                    user_id=body.user_id,
                    primary_agent_selected=agent_for_metadata,
                    last_agent=agent_for_metadata,
                    agent_profile_id=agent_profile_id_saved,
                )
                logger.debug("Saved agent metadata for external chat: primary_agent=%s", agent_for_metadata)
            except Exception as agent_save_err:
                logger.warning("Failed to save agent metadata for external chat: %s", agent_save_err)

        title_received = bool(metadata_received.get("title"))
        if not title_received and body.query:
            try:
                conv = await conversation_service.get_conversation(
                    body.conversation_id, body.user_id
                )
                current_title = (conv or {}).get("title") or ""
                if not current_title or current_title == "New Conversation":
                    fallback_title = (body.query[:100] + "...") if len(body.query) > 100 else body.query
                    fallback_title = fallback_title.strip() or "New Conversation"
                    await conversation_service.update_conversation_metadata(
                        body.conversation_id,
                        body.user_id,
                        {"title": fallback_title},
                    )
                    logger.debug("Set fallback title for external chat: %s", fallback_title[:50])
            except Exception as fallback_err:
                logger.warning("Failed to set fallback title: %s", fallback_err)

        images: List[Dict[str, Any]] = []
        raw_images = metadata_received.get("images")
        if raw_images is not None:
            if isinstance(raw_images, list):
                images = raw_images
            elif isinstance(raw_images, str):
                try:
                    images = json.loads(raw_images)
                    if not isinstance(images, list):
                        images = []
                except (json.JSONDecodeError, TypeError):
                    images = []
        if not images:
            images = _extract_images_from_markdown(accumulated_response)

        resolved_urls: List[str] = []
        for img in images:
            url = img.get("url")
            if not url or img.get("data"):
                continue
            url_stripped = url.strip()
            match = _DOCUMENT_FILE_URL_PATTERN.match(url_stripped)
            if match:
                doc_id = match.group(1)
                result = await _get_document_file_bytes(body.user_id, doc_id)
                if result:
                    data_bytes, mime = result
                    img["data"] = base64.b64encode(data_bytes).decode("utf-8")
                    img["mime"] = mime
                    resolved_urls.append(url)
                    del img["url"]
                continue
            if url_stripped.startswith(_API_IMAGES_PREFIX):
                result = _get_image_file_bytes_from_url(url_stripped)
                if result:
                    data_bytes, mime = result
                    img["data"] = base64.b64encode(data_bytes).decode("utf-8")
                    img["mime"] = mime
                    resolved_urls.append(url)
                    del img["url"]

        for img in images:
            meta = img.get("metadata")
            if meta:
                caption = _build_image_caption(meta)
                if caption:
                    img["caption"] = caption

        response_text = accumulated_response
        if resolved_urls:
            response_text = _strip_resolved_image_refs(accumulated_response, resolved_urls)

        return {
            "response": response_text,
            "images": images,
            "conversation_id": body.conversation_id,
        }
    except Exception as e:
        logger.exception("external-chat failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


async def _get_external_chat_enabled_models_for_user(user_id: str) -> List[Dict[str, Any]]:
    """
    Return chat-selectable models for this user for external chat (Telegram/Discord).
    Omits models removed from the provider catalog and the image-generation-only model.
    Each item is {"model_id": str, "display_name": str}.
    """
    from services.model_source_resolver import get_chat_selectable_model_ids_for_user

    model_ids = await get_chat_selectable_model_ids_for_user(user_id)
    return [{"model_id": mid, "display_name": _model_display_name(mid)} for mid in model_ids]


@router.post("/external-chat-models", dependencies=[Depends(verify_internal_service_key)])
async def external_chat_models(body: ExternalChatModelsRequest) -> Dict[str, Any]:
    """
    Return a numbered list of enabled models for the user (for /model command).
    When the user uses their own API keys (use_admin_models=false), returns their
    enabled models; otherwise returns org-enabled models.
    If conversation_id is provided, also returns current_model_id for that conversation.
    Returns: {"models": [{"index": 1, "id": "...", "name": "..."}, ...], "current_model_id": "..." or null}.
    """
    try:
        enabled = await _get_external_chat_enabled_models_for_user(body.user_id)
        models = [
            {"index": i + 1, "id": m["model_id"], "name": m["display_name"]}
            for i, m in enumerate(enabled)
        ]
        out: Dict[str, Any] = {"models": models}
        if body.conversation_id:
            conversation_service = ConversationService()
            conversation_service.set_current_user(body.user_id)
            await conversation_service.ensure_conversation_exists(body.conversation_id, body.user_id)
            lifecycle_info = await conversation_service.lifecycle_manager.get_conversation_lifecycle(
                body.conversation_id, body.user_id
            )
            metadata = (lifecycle_info or {}).get("metadata_json") or {}
            out["current_model_id"] = metadata.get("external_chat_selected_model_id")
        return out
    except Exception as e:
        logger.exception("external-chat-models failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/external-chat-set-model", dependencies=[Depends(verify_internal_service_key)])
async def external_chat_set_model(body: ExternalChatSetModelRequest) -> Dict[str, Any]:
    """
    Set the selected model for a conversation (e.g. /model 3).
    Uses the same user-specific model list as /model (user's models when use_admin_models=false).
    Ensures conversation exists, stores model in conversation metadata, returns confirmation.
    """
    try:
        if body.model_index < 1:
            raise HTTPException(status_code=400, detail="model_index must be at least 1")
        enabled = await _get_external_chat_enabled_models_for_user(body.user_id)
        if not enabled:
            raise HTTPException(status_code=400, detail="No enabled models configured")
        if body.model_index > len(enabled):
            raise HTTPException(
                status_code=400,
                detail=f"model_index must be between 1 and {len(enabled)}",
            )
        model_id = enabled[body.model_index - 1]["model_id"]
        model_name = enabled[body.model_index - 1]["display_name"]

        conversation_service = ConversationService()
        conversation_service.set_current_user(body.user_id)
        await conversation_service.ensure_conversation_exists(body.conversation_id, body.user_id)
        await conversation_service.update_conversation_metadata(
            body.conversation_id,
            body.user_id,
            {"external_chat_selected_model_id": model_id},
        )
        return {
            "response": f"You're now using {model_name}.",
            "model_name": model_name,
            "model_id": model_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("external-chat-set-model failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Agent-Initiated Conversations
# ============================================================================


class AgentConversationRequest(BaseModel):
    """Create or append to an agent-initiated conversation."""
    user_id: str
    message: str
    agent_name: str = ""
    agent_profile_id: str = ""
    title: str = ""
    conversation_id: str = ""  # empty = create new


@router.post("/agent-conversation", dependencies=[Depends(verify_internal_service_key)])
async def create_agent_conversation(body: AgentConversationRequest) -> Dict[str, Any]:
    """
    Create (or append to) an agent-initiated conversation and add an assistant message.
    Always returns {conversation_id, message_id}. The conversation is tagged with
    initiated_by=agent so the frontend can display a badge.
    """
    try:
        conversation_service = ConversationService()
        conversation_service.set_current_user(body.user_id)

        conversation_id = body.conversation_id or str(uuid.uuid4())
        title = body.title or (body.message[:60].strip() + ("..." if len(body.message) > 60 else "")) or "Agent Notification"

        await conversation_service.ensure_conversation_exists(conversation_id, body.user_id)
        await conversation_service.update_conversation_metadata(
            conversation_id,
            body.user_id,
            {
                "title": title,
                "initiated_by": "agent",
                "agent_name": body.agent_name or "Agent",
                "agent_profile_id": body.agent_profile_id or None,
            },
        )

        message_id = str(uuid.uuid4())
        await conversation_service.add_message(
            conversation_id=conversation_id,
            user_id=body.user_id,
            role="assistant",
            content=body.message,
            metadata={
                "orchestrator_system": True,
                "agent_initiated": True,
                "agent_name": body.agent_name or "Agent",
                "message_id": message_id,
            },
        )

        from utils.websocket_manager import get_websocket_manager
        ws_manager = get_websocket_manager()
        if ws_manager:
            await ws_manager.send_to_session(
                {
                    "type": "agent_notification",
                    "conversation_id": conversation_id,
                    "agent_name": body.agent_name or "Agent",
                    "title": title,
                    "preview": body.message[:150] if body.message else "",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                body.user_id,
            )
            await ws_manager.send_to_session(
                {"type": "conversation_created", "conversation_id": conversation_id},
                body.user_id,
            )

        return {"conversation_id": conversation_id, "message_id": message_id}
    except Exception as e:
        logger.exception("agent-conversation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Device proxy (Bastion Local Proxy) - tools-service bridges here for device tool invocations
# ============================================================================


class InvokeDeviceToolRequest(BaseModel):
    """Request to invoke a tool on a user's connected local proxy device."""
    user_id: str
    device_id: Optional[str] = None
    tool: str
    args: Dict[str, Any] = {}
    timeout_seconds: int = 30


class SetDeviceWorkspaceRequest(BaseModel):
    """Request to set the active workspace root on a user's connected local proxy device."""
    user_id: str
    device_id: Optional[str] = None
    workspace_root: str
    timeout_seconds: int = 30


@router.post("/invoke-device-tool", dependencies=[Depends(verify_internal_service_key)])
async def invoke_device_tool_internal(body: InvokeDeviceToolRequest) -> Dict[str, Any]:
    """
    Invoke a tool on a connected Bastion Local Proxy device.
    Used by tools-service gRPC handler; device WebSocket connections live in this process.
    Returns: {success, result_json, error, formatted}.
    """
    try:
        from utils.websocket_manager import get_websocket_manager

        ws_manager = get_websocket_manager()
        result = await ws_manager.invoke_device_tool(
            user_id=body.user_id,
            tool=body.tool,
            args=body.args,
            device_id=body.device_id or None,
            timeout=body.timeout_seconds,
        )
        out = {
            "success": result.get("success", False),
            "result_json": result.get("result_json", "{}"),
            "error": result.get("error", ""),
            "formatted": result.get("formatted", ""),
        }
        if not out["result_json"] and result.get("result") is not None:
            out["result_json"] = json.dumps(result["result"])
        return out
    except Exception as e:
        logger.exception("invoke_device_tool_internal failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set-device-workspace", dependencies=[Depends(verify_internal_service_key)])
async def set_device_workspace_internal(body: SetDeviceWorkspaceRequest) -> Dict[str, Any]:
    """
    Set the device-side active workspace root (relative path base).
    Returns: {success, result_json, error, formatted}.
    """
    try:
        from utils.websocket_manager import get_websocket_manager

        ws_manager = get_websocket_manager()
        result = await ws_manager.set_device_workspace(
            user_id=body.user_id,
            workspace_root=body.workspace_root,
            device_id=body.device_id or None,
            timeout=body.timeout_seconds,
        )
        out = {
            "success": result.get("success", False),
            "result_json": result.get("result_json", "{}"),
            "error": result.get("error", ""),
            "formatted": result.get("formatted", ""),
        }
        return out
    except Exception as e:
        logger.exception("set_device_workspace_internal failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/device-list", dependencies=[Depends(verify_internal_service_key)])
async def get_device_list_internal(user_id: str = Query(...)) -> Dict[str, Any]:
    """
    Return connected Bastion Local Proxy devices for a user.
    Used by tools-service or orchestrator to check capabilities before invoking.
    Returns: {"devices": [{"device_id": "...", "capabilities": ["local_screenshot", ...]}, ...]}.
    """
    try:
        from utils.websocket_manager import get_websocket_manager

        ws_manager = get_websocket_manager()
        devices = ws_manager.get_user_devices(user_id)
        return {"devices": devices}
    except Exception as e:
        logger.exception("get_device_list_internal failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Outbound Chat ID Persistence (for proactive messaging via bots)
# ============================================================================


class UpdateOutboundChatIdRequest(BaseModel):
    """Persist the outbound chat_id for a messaging bot connection."""
    connection_id: str
    outbound_chat_id: str
    sender_name: str = ""


@router.post("/connection-outbound-chat-id", dependencies=[Depends(verify_internal_service_key)])
async def update_outbound_chat_id(body: UpdateOutboundChatIdRequest) -> Dict[str, Any]:
    """Store outbound_chat_id in external_connections.provider_metadata for future proactive messaging."""
    try:
        from services.external_connections_service import external_connections_service

        conn = await external_connections_service.get_connection_by_id(int(body.connection_id))
        if not conn:
            raise HTTPException(status_code=404, detail="Connection not found")

        existing_metadata = external_connections_service._parse_provider_metadata(conn.get("provider_metadata"))
        existing_metadata["outbound_chat_id"] = body.outbound_chat_id
        if body.sender_name:
            existing_metadata["outbound_sender_name"] = body.sender_name

        await external_connections_service.update_provider_metadata(
            int(body.connection_id), existing_metadata
        )
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("update outbound_chat_id failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connection-outbound-chat-id", dependencies=[Depends(verify_internal_service_key)])
async def get_outbound_chat_id(connection_id: str = Query(...)) -> Dict[str, Any]:
    """Retrieve the stored outbound_chat_id for a messaging bot connection."""
    try:
        from services.external_connections_service import external_connections_service

        conn = await external_connections_service.get_connection_by_id(int(connection_id))
        if not conn:
            raise HTTPException(status_code=404, detail="Connection not found")

        metadata = external_connections_service._parse_provider_metadata(conn.get("provider_metadata"))
        return {"outbound_chat_id": metadata.get("outbound_chat_id", "")}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("get outbound_chat_id failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active-chat-bots", dependencies=[Depends(verify_internal_service_key)])
async def get_active_chat_bots() -> Dict[str, Any]:
    """
    Return all active chat_bot connections with decrypted tokens and config.
    Used by connections-service on startup to self-restore bot listeners.
    Returns: {"bots": [{connection_id, user_id, provider, bot_token, display_name, config}, ...]}.
    """
    try:
        from services.database_manager.database_helpers import fetch_all
        from services.external_connections_service import external_connections_service

        rows = await fetch_all("SELECT user_id FROM users")
        user_ids = [r["user_id"] for r in rows] if rows else []
        bots = []
        for uid in user_ids:
            conns = await external_connections_service.get_user_connections(
                uid,
                connection_type="chat_bot",
                active_only=True,
                rls_context={"user_id": uid},
            )
            for conn in conns:
                token = await external_connections_service.get_valid_access_token(
                    conn["id"], rls_context={"user_id": uid}
                )
                if not token:
                    continue
                raw_meta = conn.get("provider_metadata") or {}
                if isinstance(raw_meta, str):
                    try:
                        meta = json.loads(raw_meta) if raw_meta.strip() else {}
                    except Exception:
                        meta = {}
                else:
                    meta = raw_meta if isinstance(raw_meta, dict) else {}
                config = {k: str(v) for k, v in meta.items() if v is not None}
                bots.append({
                    "connection_id": str(conn["id"]),
                    "user_id": uid,
                    "provider": conn.get("provider", ""),
                    "bot_token": token,
                    "display_name": (conn.get("display_name") or conn.get("account_identifier") or ""),
                    "config": config,
                })
        return {"bots": bots}
    except Exception as e:
        logger.exception("get_active_chat_bots failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# User conversations list and validation (for /chats and /loadchat bot commands)
# ============================================================================


@router.get("/user-conversations", dependencies=[Depends(verify_internal_service_key)])
async def list_user_conversations(
    user_id: str = Query(...),
    limit: int = Query(10, ge=1, le=50),
) -> Dict[str, Any]:
    """Return recent conversations for a user (for bot /chats command)."""
    try:
        import asyncpg

        connection_string = (
            f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        )
        conn = await asyncpg.connect(connection_string)
        try:
            await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
            rows = await conn.fetch(
                """
                SELECT
                    conv.conversation_id,
                    conv.title,
                    conv.updated_at,
                    COALESCE(
                        (SELECT COUNT(*) FROM conversation_messages cm
                         WHERE cm.conversation_id = conv.conversation_id
                         AND (cm.is_deleted IS NULL OR cm.is_deleted = FALSE)),
                        conv.message_count,
                        0
                    ) AS message_count
                FROM conversations conv
                WHERE conv.user_id = $1
                ORDER BY conv.updated_at DESC NULLS LAST, conv.created_at DESC
                LIMIT $2
                """,
                user_id,
                limit,
            )
            conversations = []
            for row in rows:
                if row["conversation_id"]:
                    conversations.append({
                        "conversation_id": row["conversation_id"],
                        "title": (row["title"] or "Untitled Conversation").strip(),
                        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                        "message_count": row["message_count"] or 0,
                    })
            return {"conversations": conversations}
        finally:
            await conn.close()
    except Exception as e:
        logger.exception("list_user_conversations failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate-conversation", dependencies=[Depends(verify_internal_service_key)])
async def validate_conversation(
    user_id: str = Query(...),
    conversation_id: str = Query(...),
) -> Dict[str, Any]:
    """Confirm a conversation exists and belongs to the user (for bot /loadchat)."""
    try:
        import asyncpg

        connection_string = (
            f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        )
        conn = await asyncpg.connect(connection_string)
        try:
            row = await conn.fetchrow(
                """
                SELECT title FROM conversations
                WHERE conversation_id = $1 AND user_id = $2
                """,
                conversation_id,
                user_id,
            )
            if not row:
                return {"valid": False, "title": ""}
            return {
                "valid": True,
                "title": (row["title"] or "Untitled Conversation").strip(),
            }
        finally:
            await conn.close()
    except Exception as e:
        logger.exception("validate_conversation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Active conversation per chat (persist /loadchat selection across restarts)
# ============================================================================


class SetActiveConversationRequest(BaseModel):
    """Set the active conversation for a (connection_id, chat_id) for bot /loadchat."""

    connection_id: str
    chat_id: str
    conversation_id: str


@router.post("/connection-active-conversation", dependencies=[Depends(verify_internal_service_key)])
async def set_active_conversation(body: SetActiveConversationRequest) -> Dict[str, Any]:
    """Store active_conversation_id in provider_metadata.active_conversations[chat_id]."""
    try:
        from services.external_connections_service import external_connections_service

        conn = await external_connections_service.get_connection_by_id(int(body.connection_id))
        if not conn:
            raise HTTPException(status_code=404, detail="Connection not found")

        existing_metadata = external_connections_service._parse_provider_metadata(conn.get("provider_metadata"))
        active = existing_metadata.get("active_conversations")
        if not isinstance(active, dict):
            active = {}
        active[str(body.chat_id)] = body.conversation_id
        existing_metadata["active_conversations"] = active

        await external_connections_service.update_provider_metadata(
            int(body.connection_id), existing_metadata
        )
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("set_active_conversation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connection-active-conversation", dependencies=[Depends(verify_internal_service_key)])
async def get_active_conversation(
    connection_id: str = Query(...),
    chat_id: str = Query(...),
) -> Dict[str, Any]:
    """Retrieve the stored active conversation_id for a (connection_id, chat_id)."""
    try:
        from services.external_connections_service import external_connections_service

        conn = await external_connections_service.get_connection_by_id(int(connection_id))
        if not conn:
            raise HTTPException(status_code=404, detail="Connection not found")

        metadata = external_connections_service._parse_provider_metadata(conn.get("provider_metadata"))
        active = metadata.get("active_conversations")
        if not isinstance(active, dict):
            return {"conversation_id": ""}
        cid = active.get(str(chat_id), "")
        return {"conversation_id": cid if cid else ""}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("get_active_conversation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Known chats (recipient dropdown)
# ============================================================================


class AddKnownChatRequest(BaseModel):
    """Add or update a chat in known_chats for a connection (recipient dropdown)."""

    connection_id: str
    chat_id: str
    chat_title: str = ""
    chat_username: str = ""
    chat_type: str = ""


KNOWN_CHATS_MAX = 50


@router.post("/connection-known-chat", dependencies=[Depends(verify_internal_service_key)])
async def add_known_chat(body: AddKnownChatRequest) -> Dict[str, Any]:
    """Add or update a chat in external_connections.provider_metadata.known_chats (for recipient dropdown)."""
    try:
        from services.external_connections_service import external_connections_service

        conn = await external_connections_service.get_connection_by_id(int(body.connection_id))
        if not conn:
            raise HTTPException(status_code=404, detail="Connection not found")

        existing_metadata = external_connections_service._parse_provider_metadata(conn.get("provider_metadata"))
        known_chats: list = existing_metadata.get("known_chats") or []
        if not isinstance(known_chats, list):
            known_chats = []
        entry = {
            "chat_id": body.chat_id,
            "chat_title": (body.chat_title or "").strip(),
            "chat_username": (body.chat_username or "").strip(),
            "chat_type": (body.chat_type or "").strip(),
        }
        known_chats = [c for c in known_chats if isinstance(c, dict) and c.get("chat_id") != body.chat_id]
        known_chats.insert(0, entry)
        known_chats = known_chats[:KNOWN_CHATS_MAX]
        existing_metadata["known_chats"] = known_chats
        await external_connections_service.update_provider_metadata(int(body.connection_id), existing_metadata)
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("add known_chat failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
