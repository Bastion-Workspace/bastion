"""
Internal API for service-to-service calls (connections-service -> backend).
Authenticated by X-Internal-Service-Key. Handles external-chat: create/find conversation,
store messages, call orchestrator, return response and images.
"""

import base64
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import grpc
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from config import settings
from services.conversation_service import ConversationService
from services.grpc_context_gatherer import get_context_gatherer
from services.settings_service import settings_service

logger = logging.getLogger(__name__)

# Pattern for document file URLs we can resolve to bytes (for Telegram/Discord)
_DOCUMENT_FILE_URL_PATTERN = re.compile(
    r"^/api/documents/([a-f0-9-]{36})/file$", re.IGNORECASE
)


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
        folder_service = container.folder_service
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
    folder_id = getattr(doc_info, "folder_id", None)
    if not filename:
        return None
    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename in (".", ".."):
        return None

    try:
        file_path_str = await folder_service.get_document_file_path(
            filename=filename,
            folder_id=folder_id,
            user_id=doc_user_id,
            collection_type=collection_type,
        )
        file_path = Path(file_path_str)
    except Exception as e:
        logger.debug("Internal document resolve: path failed for %s: %s", doc_id, e)
        file_path = None

    if file_path is None or not file_path.exists():
        upload_dir = Path(settings.UPLOAD_DIR)
        for legacy_path in [
            upload_dir / f"{doc_id}_{doc_info.filename}",
            upload_dir / doc_info.filename,
        ]:
            if legacy_path.exists():
                file_path = legacy_path
                break
        else:
            return None

    try:
        uploads_base = Path(settings.UPLOAD_DIR).resolve()
        if not str(file_path.resolve()).startswith(str(uploads_base)):
            return None
    except Exception:
        return None

    try:
        data = file_path.read_bytes()
    except Exception as e:
        logger.warning("Internal document resolve: read failed for %s: %s", doc_id, e)
        return None

    mime, _ = mimetypes.guess_type(str(file_path))
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    return (data, mime)


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
        request_context: Dict[str, Any] = {}
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

        async with grpc.aio.insecure_channel(
            f"{orchestrator_host}:{orchestrator_port}", options=options
        ) as channel:
            from protos import orchestrator_pb2_grpc
            stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
            async for chunk in stub.StreamChat(grpc_request):
                if chunk.type == "content" and chunk.message:
                    accumulated_response += chunk.message
                if chunk.metadata:
                    metadata_received.update(dict(chunk.metadata))

        await conversation_service.add_message(
            conversation_id=body.conversation_id,
            user_id=body.user_id,
            role="assistant",
            content=accumulated_response,
            metadata={"orchestrator_system": True},
        )

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
            match = _DOCUMENT_FILE_URL_PATTERN.match(url.strip())
            if not match:
                continue
            doc_id = match.group(1)
            result = await _get_document_file_bytes(body.user_id, doc_id)
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


@router.post("/external-chat-models", dependencies=[Depends(verify_internal_service_key)])
async def external_chat_models(body: ExternalChatModelsRequest) -> Dict[str, Any]:
    """
    Return a numbered list of enabled models for the user (for /model command).
    Returns: {"models": [{"index": 1, "id": "...", "name": "..."}, ...]}.
    """
    try:
        model_ids = await settings_service.get_enabled_models()
        models = [
            {"index": i + 1, "id": mid, "name": _model_display_name(mid)}
            for i, mid in enumerate(model_ids)
        ]
        return {"models": models}
    except Exception as e:
        logger.exception("external-chat-models failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/external-chat-set-model", dependencies=[Depends(verify_internal_service_key)])
async def external_chat_set_model(body: ExternalChatSetModelRequest) -> Dict[str, Any]:
    """
    Set the selected model for a conversation (e.g. /model 3).
    Ensures conversation exists, stores model in conversation metadata, returns confirmation.
    """
    try:
        if body.model_index < 1:
            raise HTTPException(status_code=400, detail="model_index must be at least 1")
        model_ids = await settings_service.get_enabled_models()
        if not model_ids:
            raise HTTPException(status_code=400, detail="No enabled models configured")
        if body.model_index > len(model_ids):
            raise HTTPException(
                status_code=400,
                detail=f"model_index must be between 1 and {len(model_ids)}",
            )
        model_id = model_ids[body.model_index - 1]
        model_name = _model_display_name(model_id)

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
