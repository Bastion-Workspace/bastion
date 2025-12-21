"""
Admin API endpoints for user management and system administration
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends

from models.api_models import (
    UserCreateRequest, UserUpdateRequest, PasswordChangeRequest,
    UserResponse, UsersListResponse, AuthenticatedUserResponse
)
from services.auth_service import auth_service
from utils.auth_middleware import require_admin
from services.capabilities_service import capabilities_service, FEATURE_KEYS
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ===== USER MANAGEMENT ENDPOINTS =====

@router.get("/users", response_model=UsersListResponse)
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


@router.post("/users", response_model=UserResponse)
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


@router.put("/users/{user_id}", response_model=UserResponse)
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


@router.post("/users/{user_id}/change-password")
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


@router.delete("/users/{user_id}")
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

@router.post("/clear-documents")
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


@router.post("/clear-neo4j")
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


@router.post("/clear-qdrant")
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


# ===== CAPABILITIES MANAGEMENT =====

@router.get("/users/{user_id}/capabilities")
async def get_user_capabilities(
    user_id: str,
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    try:
        # Admins can view capabilities; admins themselves have all features implicitly
        effective = await capabilities_service.get_effective_capabilities({"user_id": user_id, "role": "user"})
        return {"features": FEATURE_KEYS, "capabilities": effective}
    except Exception as e:
        logger.error(f"❌ Failed to get capabilities for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get capabilities")


@router.post("/users/{user_id}/capabilities")
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
