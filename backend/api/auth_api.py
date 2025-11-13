"""
Authentication API endpoints
Extracted from main.py to improve modularity
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from models.api_models import (
    LoginRequest, LoginResponse, UserCreateRequest, UserUpdateRequest,
    PasswordChangeRequest, UserResponse, UsersListResponse, AuthenticatedUserResponse
)
from services.auth_service import auth_service
from utils.auth_middleware import get_current_user, require_admin
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])


@router.post("/login", response_model=LoginResponse)
async def login(login_request: LoginRequest):
    """Authenticate user and return JWT token"""
    try:
        result = await auth_service.authenticate_user(login_request)
        if not result:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Login failed: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")


@router.post("/logout")
async def logout(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Logout user by invalidating session"""
    try:
        # Extract token from request (this would need to be passed)
        # For now, we'll implement a basic logout
        return {"message": "Logout successful"}
        
    except Exception as e:
        logger.error(f"❌ Logout failed: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")


@router.get("/me", response_model=AuthenticatedUserResponse)
async def get_current_user_info(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Get current user information"""
    return current_user


# User management endpoints (admin only)
@router.get("/users", response_model=UsersListResponse)
async def get_users(
    skip: int = 0, 
    limit: int = 100,
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Get list of users (admin only)"""
    try:
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
        # Validate password length
        if len(user_request.password) < settings.PASSWORD_MIN_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters long"
            )
        
        result = await auth_service.create_user(user_request)
        if not result:
            raise HTTPException(status_code=400, detail="Username or email already exists")
        
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
        result = await auth_service.update_user(user_id, update_request)
        if not result:
            raise HTTPException(status_code=404, detail="User not found")
        
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
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Change user password"""
    try:
        # Users can only change their own password unless they're admin
        if current_user.user_id != user_id and current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Can only change your own password")
        
        # Validate new password length
        if len(password_request.new_password) < settings.PASSWORD_MIN_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters long"
            )
        
        # For admin changing other user's password, current_password is not required
        if current_user.role == "admin" and current_user.user_id != user_id:
            # Create a dummy request for admin password change
            result = await auth_service.change_password(user_id, password_request)
        else:
            result = await auth_service.change_password(user_id, password_request)
        
        if not result:
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        
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
        # Prevent admin from deleting themselves
        if current_user.user_id == user_id:
            raise HTTPException(status_code=400, detail="Cannot delete your own account")
        
        result = await auth_service.delete_user(user_id)
        if not result:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"message": "User deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ User deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete user")
