"""Document encryption sessions and password operations."""

import json

from protos import document_service_pb2


class EncryptionHandlersMixin:
    async def EncryptDocument(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_encrypt_document(request)

    async def CreateDecryptSession(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_create_decrypt_session(request)

    async def TryDecrypt(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_try_decrypt(request)

    async def EncryptionHeartbeat(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_encryption_heartbeat(request)

    async def EncryptionLock(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_encryption_lock(request)

    async def ChangeEncryptionPassword(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_change_encryption_password(request)

    async def RemoveEncryption(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_remove_encryption(request)

    async def WriteEncryptedContentFromSession(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_write_encrypted_content_from_session(request)

    async def _json_encrypt_document(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import file_encryption_service as enc
            from shims.services.service_container import get_service_container

            c = await get_service_container()
            await enc.encrypt_document(
                c.document_service,
                c.folder_service,
                p["document_id"],
                p["password"],
                p.get("confirm_password") or p["password"],
                p.get("user_id") or request.user_id,
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"ok": True})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_create_decrypt_session(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import file_encryption_service as enc
            from shims.services.service_container import get_service_container

            c = await get_service_container()
            plaintext, session_token = await enc.create_decrypt_session(
                c.document_service,
                c.folder_service,
                p["document_id"],
                p["password"],
                p.get("user_id") or request.user_id,
            )
            return document_service_pb2.JsonToolResponse(
                success=True,
                result_json=json.dumps(
                    {"content": plaintext, "session_token": session_token}
                ),
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_try_decrypt(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import file_encryption_service as enc
            from shims.services.service_container import get_service_container

            c = await get_service_container()
            uid = p.get("user_id") or request.user_id
            r = await enc.try_decrypt_content_with_session(
                c.document_service,
                c.folder_service,
                p["document_id"],
                uid,
                p.get("session_token", ""),
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"content": r})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_encryption_heartbeat(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import file_encryption_service as enc

            uid = p.get("user_id") or request.user_id
            r = await enc.heartbeat_session(
                p["document_id"], uid, p.get("session_token", "")
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"ttl_seconds": r})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_encryption_lock(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import file_encryption_service as enc

            await enc.lock_document(p["document_id"], p.get("user_id") or request.user_id)
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"ok": True})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_change_encryption_password(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import file_encryption_service as enc
            from shims.services.service_container import get_service_container

            c = await get_service_container()
            await enc.change_encryption_password(
                c.document_service,
                c.folder_service,
                p["document_id"],
                p["old_password"],
                p["new_password"],
                p.get("user_id") or request.user_id,
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"ok": True})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_remove_encryption(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import file_encryption_service as enc
            from shims.services.service_container import get_service_container

            c = await get_service_container()
            await enc.remove_encryption(
                c.document_service,
                c.folder_service,
                p["document_id"],
                p["password"],
                p.get("user_id") or request.user_id,
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"ok": True})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_write_encrypted_content_from_session(
        self, request: document_service_pb2.JsonToolRequest
    ):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import file_encryption_service as enc
            from shims.services.service_container import get_service_container

            c = await get_service_container()
            await enc.write_encrypted_content_from_session(
                c.document_service,
                c.folder_service,
                p["document_id"],
                p.get("user_id") or request.user_id,
                p.get("session_token", ""),
                p.get("content", ""),
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"ok": True})
            )
        except PermissionError as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))
