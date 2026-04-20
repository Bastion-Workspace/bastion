"""RLS context dicts for gRPC handlers and background jobs (database_manager set_config keys)."""


def grpc_user_rls(user_id: str) -> dict:
    """Session user for normal agent-factory operations. system → admin bypass."""
    uid = (user_id or "").strip()
    if not uid or uid == "system":
        return {"user_id": "", "user_role": "admin"}
    return {"user_id": uid, "user_role": "user"}


def grpc_admin_rls() -> dict:
    """Cross-tenant reads (e.g. team status board) after service-layer access checks."""
    return {"user_id": "", "user_role": "admin"}
