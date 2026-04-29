"""
gRPC DocumentMirror: JSON dispatch to ParallelDocumentService for backend facade parity.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from protos import document_service_pb2

from ds_handlers.document_admin_ops import ACTION_OPS

logger = logging.getLogger(__name__)


async def handle_document_mirror_request(
    request: document_service_pb2.JsonToolRequest,
) -> document_service_pb2.JsonToolResponse:
    """
    Payload: {"action": "<name>", ...action-specific fields}
    user_id on the protobuf request is used when payload omits user_id.
    """
    try:
        payload = json.loads(request.payload_json or "{}")
    except json.JSONDecodeError as e:
        return document_service_pb2.JsonToolResponse(success=False, error=f"invalid json: {e}")

    action = (payload.get("action") or "").strip()
    if not action:
        return document_service_pb2.JsonToolResponse(success=False, error="missing action")

    uid = (payload.get("user_id") or request.user_id or "").strip()

    op = ACTION_OPS.get(action)
    if not op:
        return document_service_pb2.JsonToolResponse(
            success=False,
            error=f"unknown action: {action}",
        )

    try:
        result = await op(payload, uid)
        return document_service_pb2.JsonToolResponse(
            success=True,
            result_json=json.dumps(result, default=str),
        )
    except Exception as e:
        logger.exception("DocumentMirror action=%s failed", action)
        return document_service_pb2.JsonToolResponse(success=False, error=str(e))
