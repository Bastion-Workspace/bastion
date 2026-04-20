"""
Document and folder sharing plus pessimistic edit locks.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from services.database_manager.database_helpers import fetch_all, fetch_one, execute

logger = logging.getLogger(__name__)

_ADMIN_RLS = {"user_id": "", "user_role": "admin"}

LOCK_TTL_SECONDS = 300
HEARTBEAT_GRACE_SECONDS = 180


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DocumentSharingService:
    """Share CRUD, shared-with-me listings, vector search scopes, and document locks."""

    async def _has_outgoing_write_shares(
        self,
        document_id: str,
        folder_id: Optional[str],
        owner_user_id: str,
    ) -> bool:
        """True if the owner has granted active write access on this doc or an ancestor folder."""
        direct = await fetch_one(
            """
            SELECT 1 AS ok
            FROM document_shares
            WHERE document_id = $1
              AND share_type = 'write'
              AND (expires_at IS NULL OR expires_at > NOW())
            LIMIT 1
            """,
            document_id,
            rls_context=_ADMIN_RLS,
        )
        if direct:
            return True
        if not folder_id:
            return False
        folder_hit = await fetch_one(
            """
            WITH RECURSIVE ancestors AS (
                SELECT folder_id, parent_folder_id
                FROM document_folders
                WHERE folder_id = $2::varchar
                UNION ALL
                SELECT p.folder_id, p.parent_folder_id
                FROM document_folders p
                INNER JOIN ancestors a ON p.folder_id = a.parent_folder_id
            )
            SELECT 1 AS ok
            FROM document_shares ds
            INNER JOIN ancestors a ON ds.folder_id = a.folder_id
            WHERE ds.shared_by_user_id = $1::varchar
              AND ds.share_type = 'write'
              AND (ds.expires_at IS NULL OR ds.expires_at > NOW())
            LIMIT 1
            """,
            owner_user_id,
            folder_id,
            rls_context=_ADMIN_RLS,
        )
        return bool(folder_hit)

    def _collab_eligible_from_flags(
        self,
        *,
        can_write: bool,
        team_id: Optional[str],
        is_owner: bool,
        collection_type: str,
        outgoing_write_shares: bool = False,
    ) -> bool:
        if not can_write or collection_type == "global":
            return False
        if team_id:
            return True
        if is_owner:
            return outgoing_write_shares
        return True

    async def check_share_access(
        self,
        document_id: str,
        folder_id: Optional[str],
        user_id: str,
        required_permission: str,
    ) -> bool:
        """True if user has active share with sufficient permission (read or write)."""
        if not user_id:
            return False
        need_write = required_permission in ("write", "delete")
        if required_permission == "delete":
            return False
        row = await fetch_one(
            """
            SELECT document_user_has_share_access($1::varchar, $2::varchar, $3::varchar, $4::boolean) AS ok
            """,
            document_id,
            folder_id,
            user_id,
            need_write,
            rls_context={"user_id": user_id, "user_role": "user"},
        )
        return bool(row and row.get("ok"))

    async def get_sharing_context_for_user(
        self, document_id: str, user_id: str, user_role: str
    ) -> Dict[str, Any]:
        """Owner vs share recipient and effective permissions for API responses."""
        row = await fetch_one(
            """
            SELECT dm.document_id,
                   dm.user_id AS owner_id,
                   dm.folder_id,
                   dm.collection_type,
                   COALESCE(dm.team_id, df.team_id) AS team_id
            FROM document_metadata dm
            LEFT JOIN document_folders df ON df.folder_id = dm.folder_id
            WHERE dm.document_id = $1
            """,
            document_id,
            rls_context={"user_id": user_id, "user_role": user_role},
        )
        if not row:
            return {
                "document_id": document_id,
                "is_owner": False,
                "share_type": None,
                "can_write": False,
                "can_delete": False,
                "collab_eligible": False,
            }
        owner_id = row.get("owner_id")
        folder_id = row.get("folder_id")
        team_id = row.get("team_id")
        if team_id is not None and not isinstance(team_id, str):
            team_id = str(team_id)
        collection_type = row.get("collection_type") or "user"

        if user_role == "admin":
            outgoing = await self._has_outgoing_write_shares(document_id, folder_id, owner_id)
            ce = self._collab_eligible_from_flags(
                can_write=True,
                team_id=team_id,
                is_owner=True,
                collection_type=collection_type,
                outgoing_write_shares=outgoing,
            )
            return {
                "document_id": document_id,
                "is_owner": True,
                "share_type": None,
                "can_write": True,
                "can_delete": True,
                "collab_eligible": ce,
            }

        if collection_type == "global":
            return {
                "document_id": document_id,
                "is_owner": False,
                "share_type": None,
                "can_write": False,
                "can_delete": False,
                "collab_eligible": False,
            }

        if team_id:
            from api.teams_api import team_service

            tm_role = await team_service.check_team_access(team_id, user_id)
            if not tm_role:
                return {
                    "document_id": document_id,
                    "is_owner": False,
                    "share_type": None,
                    "can_write": False,
                    "can_delete": False,
                    "collab_eligible": False,
                }
            is_owner = owner_id == user_id
            return {
                "document_id": document_id,
                "is_owner": is_owner,
                "share_type": None,
                "can_write": True,
                "can_delete": tm_role == "admin",
                "collab_eligible": True,
            }

        is_owner = owner_id == user_id
        if is_owner:
            outgoing = await self._has_outgoing_write_shares(document_id, folder_id, owner_id)
            ce = self._collab_eligible_from_flags(
                can_write=True,
                team_id=None,
                is_owner=True,
                collection_type=collection_type,
                outgoing_write_shares=outgoing,
            )
            return {
                "document_id": document_id,
                "is_owner": True,
                "share_type": None,
                "can_write": True,
                "can_delete": True,
                "collab_eligible": ce,
            }

        read_ok = await self.check_share_access(document_id, folder_id, user_id, "read")
        write_ok = await self.check_share_access(document_id, folder_id, user_id, "write")
        if write_ok:
            share_type = "write"
            can_write = True
        elif read_ok:
            share_type = "read"
            can_write = False
        else:
            share_type = None
            can_write = False

        ce = self._collab_eligible_from_flags(
            can_write=can_write,
            team_id=None,
            is_owner=False,
            collection_type=collection_type,
        )
        return {
            "document_id": document_id,
            "is_owner": False,
            "share_type": share_type,
            "can_write": can_write,
            "can_delete": False,
            "collab_eligible": ce,
        }

    async def get_vector_share_scopes(
        self, shared_with_user_id: str
    ) -> List[Tuple[str, List[str]]]:
        """Pairs of (sharer_user_id, document_ids) for hybrid Qdrant search."""
        if not shared_with_user_id or shared_with_user_id == "system":
            return []

        doc_rows = await fetch_all(
            """
            SELECT shared_by_user_id, document_id FROM document_shares
            WHERE shared_with_user_id = $1
            AND document_id IS NOT NULL
            AND (expires_at IS NULL OR expires_at > NOW())
            """,
            shared_with_user_id,
            rls_context=_ADMIN_RLS,
        )
        folder_rows = await fetch_all(
            """
            SELECT shared_by_user_id, folder_id FROM document_shares
            WHERE shared_with_user_id = $1
            AND folder_id IS NOT NULL
            AND (expires_at IS NULL OR expires_at > NOW())
            """,
            shared_with_user_id,
            rls_context=_ADMIN_RLS,
        )
        by_sharer: Dict[str, Any] = defaultdict(set)
        for r in doc_rows:
            by_sharer[r["shared_by_user_id"]].add(r["document_id"])
        for fs in folder_rows:
            doc_ids = await fetch_all(
                """
                WITH RECURSIVE descendants AS (
                    SELECT folder_id FROM document_folders WHERE folder_id = $1
                    UNION ALL
                    SELECT c.folder_id FROM document_folders c
                    INNER JOIN descendants d ON c.parent_folder_id = d.folder_id
                )
                SELECT dm.document_id FROM document_metadata dm
                WHERE dm.user_id = $2
                AND dm.folder_id IN (SELECT folder_id FROM descendants)
                """,
                fs["folder_id"],
                fs["shared_by_user_id"],
                rls_context=_ADMIN_RLS,
            )
            for d in doc_ids:
                by_sharer[fs["shared_by_user_id"]].add(d["document_id"])
        return [(k, list(v)) for k, v in by_sharer.items() if v]

    async def list_shareable_users(self, exclude_user_id: str) -> List[Dict[str, Any]]:
        return await fetch_all(
            """
            SELECT user_id, username, avatar_url
            FROM users
            WHERE user_id != $1 AND COALESCE(is_active, TRUE)
            ORDER BY LOWER(username)
            """,
            exclude_user_id,
            rls_context=_ADMIN_RLS,
        )

    async def _assert_document_owner(
        self, document_id: str, user_id: str, role: str
    ) -> Dict[str, Any]:
        row = await fetch_one(
            "SELECT document_id, user_id, collection_type, team_id FROM document_metadata WHERE document_id = $1",
            document_id,
            rls_context={"user_id": user_id, "user_role": role},
        )
        if not row:
            raise ValueError("Document not found")
        if role == "admin":
            return row
        if row.get("team_id") or row.get("collection_type") != "user":
            raise PermissionError("Sharing is only supported for personal documents")
        if row.get("user_id") != user_id:
            raise PermissionError("Not the document owner")
        return row

    async def _assert_folder_owner(
        self, folder_id: str, user_id: str, role: str
    ) -> Dict[str, Any]:
        row = await fetch_one(
            """
            SELECT folder_id, user_id, collection_type, team_id
            FROM document_folders
            WHERE folder_id = $1
            """,
            folder_id,
            rls_context={"user_id": user_id, "user_role": role},
        )
        if not row:
            raise ValueError("Folder not found")
        if role == "admin":
            return row
        if row.get("collection_type") != "user" or row.get("team_id"):
            raise PermissionError("Sharing is only supported for personal folders")
        if row.get("user_id") != user_id:
            raise PermissionError("Not the folder owner")
        return row

    async def create_document_share(
        self,
        document_id: str,
        shared_by_user_id: str,
        shared_with_user_id: str,
        share_type: str,
        role: str,
        expires_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        await self._assert_document_owner(document_id, shared_by_user_id, role)
        if shared_with_user_id == shared_by_user_id:
            raise ValueError("Cannot share with yourself")
        sid = str(uuid.uuid4())
        out = await fetch_one(
            """
            INSERT INTO document_shares (
              share_id, document_id, folder_id, shared_by_user_id, shared_with_user_id,
              share_type, expires_at
            ) VALUES ($1, $2, NULL, $3, $4, $5, $6)
            ON CONFLICT (shared_with_user_id, document_id) WHERE document_id IS NOT NULL
            DO UPDATE SET share_type = EXCLUDED.share_type, expires_at = EXCLUDED.expires_at
            RETURNING share_id
            """,
            sid,
            document_id,
            shared_by_user_id,
            shared_with_user_id,
            share_type,
            expires_at,
            rls_context={"user_id": shared_by_user_id, "user_role": role},
        )
        return await self.get_share_by_id(out["share_id"], shared_by_user_id, role)

    async def create_folder_share(
        self,
        folder_id: str,
        shared_by_user_id: str,
        shared_with_user_id: str,
        share_type: str,
        role: str,
        expires_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        await self._assert_folder_owner(folder_id, shared_by_user_id, role)
        if shared_with_user_id == shared_by_user_id:
            raise ValueError("Cannot share with yourself")
        sid = str(uuid.uuid4())
        out = await fetch_one(
            """
            INSERT INTO document_shares (
              share_id, document_id, folder_id, shared_by_user_id, shared_with_user_id,
              share_type, expires_at
            ) VALUES ($1, NULL, $2, $3, $4, $5, $6)
            ON CONFLICT (shared_with_user_id, folder_id) WHERE folder_id IS NOT NULL
            DO UPDATE SET share_type = EXCLUDED.share_type, expires_at = EXCLUDED.expires_at
            RETURNING share_id
            """,
            sid,
            folder_id,
            shared_by_user_id,
            shared_with_user_id,
            share_type,
            expires_at,
            rls_context={"user_id": shared_by_user_id, "user_role": role},
        )
        return await self.get_share_by_id(out["share_id"], shared_by_user_id, role)

    async def get_share_by_id(
        self, share_id: str, acting_user_id: str, role: str
    ) -> Dict[str, Any]:
        row = await fetch_one(
            """
            SELECT ds.share_id, ds.document_id, ds.folder_id, ds.shared_by_user_id,
                   ds.shared_with_user_id, ds.share_type, ds.created_at, ds.expires_at,
                   u.username AS shared_with_username
            FROM document_shares ds
            LEFT JOIN users u ON u.user_id = ds.shared_with_user_id
            WHERE ds.share_id = $1
            """,
            share_id,
            rls_context={"user_id": acting_user_id, "user_role": role},
        )
        if not row:
            raise ValueError("Share not found")
        return dict(row)

    async def list_shares_for_document(
        self, document_id: str, user_id: str, role: str
    ) -> List[Dict[str, Any]]:
        await self._assert_document_owner(document_id, user_id, role)
        rows = await fetch_all(
            """
            SELECT ds.share_id, ds.document_id, ds.folder_id, ds.shared_by_user_id,
                   ds.shared_with_user_id, ds.share_type, ds.created_at, ds.expires_at,
                   u.username AS shared_with_username
            FROM document_shares ds
            LEFT JOIN users u ON u.user_id = ds.shared_with_user_id
            WHERE ds.document_id = $1
            ORDER BY ds.created_at DESC
            """,
            document_id,
            rls_context={"user_id": user_id, "user_role": role},
        )
        return [dict(r) for r in rows]

    async def list_shares_for_folder(
        self, folder_id: str, user_id: str, role: str
    ) -> List[Dict[str, Any]]:
        await self._assert_folder_owner(folder_id, user_id, role)
        rows = await fetch_all(
            """
            SELECT ds.share_id, ds.document_id, ds.folder_id, ds.shared_by_user_id,
                   ds.shared_with_user_id, ds.share_type, ds.created_at, ds.expires_at,
                   u.username AS shared_with_username
            FROM document_shares ds
            LEFT JOIN users u ON u.user_id = ds.shared_with_user_id
            WHERE ds.folder_id = $1
            ORDER BY ds.created_at DESC
            """,
            folder_id,
            rls_context={"user_id": user_id, "user_role": role},
        )
        return [dict(r) for r in rows]

    async def update_share(
        self, share_id: str, share_type: str, acting_user_id: str, role: str
    ) -> Dict[str, Any]:
        row = await fetch_one(
            "SELECT share_id, shared_by_user_id FROM document_shares WHERE share_id = $1",
            share_id,
            rls_context={"user_id": acting_user_id, "user_role": role},
        )
        if not row:
            raise ValueError("Share not found")
        if role != "admin" and row["shared_by_user_id"] != acting_user_id:
            raise PermissionError("Not allowed to update this share")
        await execute(
            "UPDATE document_shares SET share_type = $2 WHERE share_id = $1",
            share_id,
            share_type,
            rls_context={"user_id": acting_user_id, "user_role": role},
        )
        return await self.get_share_by_id(share_id, acting_user_id, role)

    async def revoke_share(self, share_id: str, acting_user_id: str, role: str) -> None:
        row = await fetch_one(
            "SELECT shared_by_user_id FROM document_shares WHERE share_id = $1",
            share_id,
            rls_context={"user_id": acting_user_id, "user_role": role},
        )
        if not row:
            raise ValueError("Share not found")
        if role != "admin" and row["shared_by_user_id"] != acting_user_id:
            raise PermissionError("Not allowed to revoke this share")
        await execute(
            "DELETE FROM document_shares WHERE share_id = $1",
            share_id,
            rls_context={"user_id": acting_user_id, "user_role": role},
        )

    async def get_shared_with_me(self, user_id: str) -> List[Dict[str, Any]]:
        rows = await fetch_all(
            """
            SELECT ds.share_id, ds.share_type, ds.document_id, ds.folder_id,
                   ds.shared_by_user_id, u.username AS sharer_username,
                   dm.title, dm.filename,
                   df.name AS folder_name, df.parent_folder_id AS folder_parent_id
            FROM document_shares ds
            JOIN users u ON u.user_id = ds.shared_by_user_id
            LEFT JOIN document_metadata dm ON dm.document_id = ds.document_id
            LEFT JOIN document_folders df ON df.folder_id = ds.folder_id
            WHERE ds.shared_with_user_id = $1
            AND (ds.expires_at IS NULL OR ds.expires_at > NOW())
            ORDER BY LOWER(u.username), ds.created_at
            """,
            user_id,
            rls_context={"user_id": user_id, "user_role": "user"},
        )
        return [dict(r) for r in rows]

    async def get_lock_row(self, document_id: str, acting_user_id: str, role: str) -> Optional[Dict[str, Any]]:
        row = await fetch_one(
            """
            SELECT dl.document_id, dl.locked_by_user_id, dl.acquired_at, dl.expires_at,
                   dl.heartbeat_at, u.username AS locked_by_username
            FROM document_locks dl
            LEFT JOIN users u ON u.user_id = dl.locked_by_user_id
            WHERE dl.document_id = $1
            """,
            document_id,
            rls_context={"user_id": acting_user_id, "user_role": role},
        )
        return dict(row) if row else None

    async def acquire_lock(self, document_id: str, user_id: str, role: str) -> Dict[str, Any]:
        now = _utcnow()
        stale_before = now - timedelta(seconds=HEARTBEAT_GRACE_SECONDS)
        await execute(
            """
            DELETE FROM document_locks
            WHERE document_id = $1
              AND (expires_at < $2 OR heartbeat_at < $3)
            """,
            document_id,
            now,
            stale_before,
            rls_context={"user_id": user_id, "user_role": role},
        )
        existing = await fetch_one(
            "SELECT locked_by_user_id, expires_at, heartbeat_at FROM document_locks WHERE document_id = $1",
            document_id,
            rls_context={"user_id": user_id, "user_role": role},
        )
        if existing:
            if existing["locked_by_user_id"] == user_id:
                exp = now + timedelta(seconds=LOCK_TTL_SECONDS)
                await execute(
                    """
                    UPDATE document_locks
                    SET expires_at = $2, heartbeat_at = $3
                    WHERE document_id = $1
                    """,
                    document_id,
                    exp,
                    now,
                    rls_context={"user_id": user_id, "user_role": role},
                )
                full = await self.get_lock_row(document_id, user_id, role)
                return {"success": True, "message": "Lock held", "lock": full}
            return {
                "success": False,
                "message": "Document is locked by another user",
                "lock": await self.get_lock_row(document_id, user_id, role),
            }
        exp = now + timedelta(seconds=LOCK_TTL_SECONDS)
        await execute(
            """
            INSERT INTO document_locks (document_id, locked_by_user_id, expires_at, heartbeat_at)
            VALUES ($1, $2, $3, $4)
            """,
            document_id,
            user_id,
            exp,
            now,
            rls_context={"user_id": user_id, "user_role": role},
        )
        full = await self.get_lock_row(document_id, user_id, role)
        return {"success": True, "message": "Lock acquired", "lock": full}

    async def release_lock(self, document_id: str, user_id: str, role: str) -> bool:
        row = await fetch_one(
            "SELECT locked_by_user_id FROM document_locks WHERE document_id = $1",
            document_id,
            rls_context={"user_id": user_id, "user_role": role},
        )
        if not row:
            return True
        if row["locked_by_user_id"] != user_id and role != "admin":
            raise PermissionError("Not the lock holder")
        await execute(
            "DELETE FROM document_locks WHERE document_id = $1",
            document_id,
            rls_context={"user_id": user_id, "user_role": role},
        )
        return True

    async def heartbeat_lock(self, document_id: str, user_id: str, role: str) -> Dict[str, Any]:
        now = _utcnow()
        exp = now + timedelta(seconds=LOCK_TTL_SECONDS)
        row = await fetch_one(
            "SELECT locked_by_user_id FROM document_locks WHERE document_id = $1",
            document_id,
            rls_context={"user_id": user_id, "user_role": role},
        )
        if not row:
            raise ValueError("No active lock")
        if row["locked_by_user_id"] != user_id:
            raise PermissionError("Not the lock holder")
        await execute(
            """
            UPDATE document_locks SET heartbeat_at = $2, expires_at = $3 WHERE document_id = $1
            """,
            document_id,
            now,
            exp,
            rls_context={"user_id": user_id, "user_role": role},
        )
        return await self.get_lock_row(document_id, user_id, role)


document_sharing_service = DocumentSharingService()
