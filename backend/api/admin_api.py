"""
Admin API endpoints for user management and system administration
"""

import logging
from typing import List, Optional, Any, Dict
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from models.api_models import (
    UserCreateRequest, UserUpdateRequest, PasswordChangeRequest,
    UserResponse, UsersListResponse, AuthenticatedUserResponse
)
from services.auth_service import auth_service
from services.external_connections_service import external_connections_service
from utils.auth_middleware import require_admin, get_current_user
from services.capabilities_service import capabilities_service, FEATURE_KEYS
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["admin"])


# ===== USER MANAGEMENT ENDPOINTS =====

@router.get("/api/admin/users", response_model=UsersListResponse)
async def get_users(
    skip: int = 0, 
    limit: int = 100,
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Get list of users (admin only)"""
    try:
        logger.info(f"🔧 Admin {current_user.username} getting users list")
        return await auth_service.get_users(skip, limit)
    except Exception as e:
        logger.error(f"❌ Get users failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve users")


@router.post("/api/admin/users", response_model=UserResponse)
async def create_user(
    user_request: UserCreateRequest,
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Create a new user (admin only)"""
    try:
        logger.info(f"🔧 Admin {current_user.username} creating user: {user_request.username}")
        
        # Validate password length
        if len(user_request.password) < settings.PASSWORD_MIN_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters long"
            )
        
        result = await auth_service.create_user(user_request)
        if not result:
            raise HTTPException(status_code=400, detail="Username or email already exists")
        
        logger.info(f"✅ Admin {current_user.username} created user: {user_request.username}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ User creation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user")


@router.put("/api/admin/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    update_request: UserUpdateRequest,
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Update user details (admin only)"""
    try:
        logger.info(f"🔧 Admin {current_user.username} updating user: {user_id}")
        
        result = await auth_service.update_user(user_id, update_request)
        if not result:
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.info(f"✅ Admin {current_user.username} updated user: {user_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ User update failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user")


@router.post("/api/admin/users/{user_id}/change-password")
async def change_user_password(
    user_id: str,
    password_request: PasswordChangeRequest,
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Change user password (admin only)"""
    try:
        logger.info(f"🔧 Admin {current_user.username} changing password for user: {user_id}")
        
        # Validate new password length
        if len(password_request.new_password) < settings.PASSWORD_MIN_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters long"
            )
        
        # For admin changing other user's password, current_password is not required
        result = await auth_service.change_password(user_id, password_request)
        
        if not result:
            raise HTTPException(status_code=400, detail="Failed to change password")
        
        logger.info(f"✅ Admin {current_user.username} changed password for user: {user_id}")
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Password change failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to change password")


@router.delete("/api/admin/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Delete a user (admin only)"""
    try:
        logger.info(f"🔧 Admin {current_user.username} deleting user: {user_id}")
        
        # Prevent admin from deleting themselves
        if current_user.user_id == user_id:
            raise HTTPException(status_code=400, detail="Cannot delete your own account")
        
        result = await auth_service.delete_user(user_id)
        if not result:
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.info(f"✅ Admin {current_user.username} deleted user: {user_id}")
        return {"message": "User deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ User deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete user")


# ===== SYSTEM ADMINISTRATION ENDPOINTS =====

@router.post("/api/admin/clear-documents")
async def clear_all_documents(
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Clear all documents from all user folders, vector DB collections, and knowledge graph (admin only)"""
    try:
        logger.info(f"🗑️ Admin {current_user.username} starting complete document clearance")
        
        # Import the function from main.py to avoid code duplication
        from main import clear_all_documents as main_clear_documents
        
        # Call the main function with the current user
        return await main_clear_documents(current_user)
        
    except Exception as e:
        logger.error(f"❌ Admin clearance failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")


@router.post("/api/admin/clear-neo4j")
async def clear_neo4j(
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Clear all data from Neo4j knowledge graph (admin only)"""
    try:
        logger.info(f"🗑️ Admin {current_user.username} clearing Neo4j knowledge graph")
        
        # Import the function from main.py to avoid code duplication
        from main import clear_neo4j as main_clear_neo4j
        
        # Call the main function with the current user
        return await main_clear_neo4j(current_user)
        
    except Exception as e:
        logger.error(f"❌ Failed to clear Neo4j: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear Neo4j: {str(e)}")


@router.post("/api/admin/clear-qdrant")
async def clear_qdrant(
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Clear all data from Qdrant vector database (admin only)"""
    try:
        logger.info(f"🗑️ Admin {current_user.username} clearing Qdrant vector database")
        
        # Import the function from main.py to avoid code duplication
        from main import clear_qdrant as main_clear_qdrant
        
        # Call the main function with the current user
        return await main_clear_qdrant(current_user)
        
    except Exception as e:
        logger.error(f"❌ Failed to clear Qdrant: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear Qdrant: {str(e)}")


        logger.error(f"❌ Failed to review submission: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== SYSTEM EMAIL (admin-only designation) =====


class SystemEmailPutRequest(BaseModel):
    connection_id: Optional[int] = None


class SmtpSettingsPutRequest(BaseModel):
    enabled: bool = False
    host: str = ""
    port: int = 587
    user: str = ""
    password: Optional[str] = None
    use_tls: bool = True
    from_email: str = ""
    from_name: str = ""


class SmtpTestRequest(BaseModel):
    """Optional body for SMTP test: send to this address instead of current user email."""
    to_email: Optional[str] = None


@router.get("/api/admin/smtp-settings")
async def get_smtp_settings(
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    """Get SMTP settings for system outbound email (admin only). Password never returned."""
    rls_context = {"user_id": current_user.user_id, "user_role": current_user.role}
    return await external_connections_service.get_smtp_settings(rls_context=rls_context)


@router.put("/api/admin/smtp-settings")
async def set_smtp_settings(
    body: SmtpSettingsPutRequest,
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    """Save SMTP settings for system outbound email (admin only). Omit password to keep existing."""
    rls_context = {"user_id": current_user.user_id, "user_role": current_user.role}
    await external_connections_service.set_smtp_settings(
        enabled=body.enabled,
        host=body.host,
        port=body.port,
        user=body.user,
        password=body.password,
        use_tls=body.use_tls,
        from_email=body.from_email,
        from_name=body.from_name,
        rls_context=rls_context,
    )
    return {"ok": True}


@router.post("/api/admin/smtp-settings/test")
async def test_smtp_settings(
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
    body: Optional[SmtpTestRequest] = None,
):
    """Send a test email using current SMTP settings (admin only). Uses body.to_email if provided, else current user email."""
    from services.email_service import email_service
    rls_context = {"user_id": current_user.user_id, "user_role": current_user.role}
    config = await external_connections_service.get_smtp_settings_for_sending(rls_context=rls_context)
    if not config:
        raise HTTPException(
            status_code=400,
            detail="SMTP is not configured or disabled. Configure and enable SMTP in System outbound email.",
        )
    to_email = (body and body.to_email and body.to_email.strip()) or (current_user.email or "").strip()
    if not to_email:
        raise HTTPException(
            status_code=400,
            detail="Your account has no email address. Add an email in your profile or enter a test address above.",
        )
    subject = "Bastion System Email Test (SMTP)"
    body = "If you received this, SMTP for system outbound email is configured correctly."
    ok = await email_service.send_email(
        to_email, subject, body, smtp_config=config,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to send test email.")
    return {"ok": True, "message": f"Test email sent to {to_email}"}


@router.get("/api/admin/system-email")
async def get_system_email(
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    """Get current system outbound email connection (admin only)."""
    rls_context = {"user_id": current_user.user_id, "user_role": current_user.role}
    connection_id = await external_connections_service.get_system_email_connection_id(
        rls_context=rls_context
    )
    connection: Optional[Dict[str, Any]] = None
    if connection_id is not None:
        conn = await external_connections_service.get_connection_by_id(
            connection_id, rls_context=rls_context
        )
        if conn:
            connection = {
                "id": conn["id"],
                "account_identifier": conn.get("account_identifier"),
                "display_name": conn.get("display_name"),
                "provider": conn.get("provider"),
            }
    return {"connection_id": connection_id, "connection": connection}


@router.put("/api/admin/system-email")
async def set_system_email(
    body: SystemEmailPutRequest,
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    """Set or clear the system outbound email connection (admin only)."""
    rls_context = {"user_id": current_user.user_id, "user_role": current_user.role}
    connection_id = body.connection_id
    if connection_id is not None:
        conn = await external_connections_service.get_connection_by_id(
            connection_id, rls_context=rls_context
        )
        if not conn or not conn.get("is_active", True):
            raise HTTPException(
                status_code=400,
                detail="Connection not found or inactive",
            )
    await external_connections_service.set_system_email_connection_id(
        connection_id, rls_context=rls_context
    )
    return {"ok": True, "connection_id": connection_id}


@router.post("/api/admin/system-email/test")
async def test_system_email(
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    """Send test email from system account. Tries SMTP (UI-configured) first, then OAuth connection."""
    from services.email_service import email_service
    from datetime import datetime
    rls_context = {"user_id": current_user.user_id, "user_role": current_user.role}
    to_email = current_user.email or ""
    if not to_email:
        raise HTTPException(status_code=400, detail="Your account has no email address.")

    smtp_config = await external_connections_service.get_smtp_settings_for_sending(
        rls_context=rls_context
    )
    if smtp_config:
        subject = "Bastion System Email Test (SMTP)"
        body = "If you received this, system outbound email (SMTP) is configured correctly."
        ok = await email_service.send_email(to_email, subject, body, smtp_config=smtp_config)
        if ok:
            return {"ok": True, "message": f"Test email sent to {to_email}"}
        raise HTTPException(status_code=500, detail="Failed to send test email via SMTP.")

    connection_id = await external_connections_service.get_system_email_connection_id(
        rls_context=rls_context
    )
    if not connection_id:
        raise HTTPException(
            status_code=400,
            detail="No system email configured. Set SMTP (recommended) or designate a Microsoft connection.",
        )
    conn = await external_connections_service.get_connection_by_id(
        connection_id, rls_context=rls_context
    )
    if not conn or not conn.get("is_active", True):
        raise HTTPException(
            status_code=400,
            detail="Designated system connection not found or inactive.",
        )
    try:
        from clients.connections_service_client import get_connections_service_client
        client = await get_connections_service_client()
        token = await external_connections_service.get_valid_access_token(connection_id)
        if not token:
            raise HTTPException(status_code=500, detail="Failed to obtain access token for system connection.")
        subject = "Bastion System Email Test"
        body = f"""This is a test email from the Bastion system.

System email connection: {conn.get('account_identifier')}
Provider: {conn.get('provider')}
Test sent at: {datetime.now().isoformat()}

If you received this, the system email is configured correctly!"""
        provider = conn.get("provider", "microsoft")
        import protos.connections_service_pb2 as cs_pb2
        req = cs_pb2.SendEmailRequest(
            access_token=token,
            provider=provider,
            to_recipients=[current_user.email],
            subject=subject,
            body=body,
        )
        resp = await client.stub.SendEmail(req, timeout=30.0)
        if not resp.success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send test email: {resp.error or 'Unknown error'}",
            )
        return {"ok": True, "message": f"Test email sent to {current_user.email}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Test system email failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to send test email: {str(e)}")


# ===== CAPABILITIES MANAGEMENT =====

@router.get("/api/admin/users/{user_id}/capabilities")
async def get_user_capabilities(
    user_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    try:
        # Users can get their own capabilities, admins can get anyone's
        if current_user.user_id != user_id and current_user.role != "admin":
            raise HTTPException(status_code=403, detail="You can only view your own capabilities")
        
        # Get the target user's role (use current_user if same, otherwise assume "user" for admin queries)
        target_role = current_user.role if current_user.user_id == user_id else "user"
        effective = await capabilities_service.get_effective_capabilities({"user_id": user_id, "role": target_role})
        return {"features": FEATURE_KEYS, "capabilities": effective}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get capabilities for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get capabilities")


@router.post("/api/admin/users/{user_id}/capabilities")
async def set_user_capabilities(
    user_id: str,
    capabilities: dict,
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    try:
        ok = await capabilities_service.set_user_capabilities(user_id, capabilities)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to update capabilities")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to set capabilities for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update capabilities")
