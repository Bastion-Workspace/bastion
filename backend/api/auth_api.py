"""
Authentication API endpoints
Extracted from main.py to improve modularity
"""

import logging
import os
import re
import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, Request, File, UploadFile
from fastapi.responses import FileResponse
from models.api_models import (
    LoginRequest, LoginResponse, UserCreateRequest, UserUpdateRequest,
    PasswordChangeRequest, UserResponse, UsersListResponse, AuthenticatedUserResponse
)
from services.auth_service import auth_service
from utils.auth_middleware import get_current_user, require_admin
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["authentication"])


@router.post("/api/auth/login", response_model=LoginResponse)
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


@router.post("/api/auth/logout")
async def logout(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Logout user by invalidating session"""
    try:
        # Extract token from request (this would need to be passed)
        # For now, we'll implement a basic logout
        return {"message": "Logout successful"}
        
    except Exception as e:
        logger.error(f"❌ Logout failed: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")


@router.get("/api/auth/me", response_model=AuthenticatedUserResponse)
async def get_current_user_info(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Get current user information"""
    return current_user


@router.post("/api/auth/refresh", response_model=LoginResponse)
async def refresh_token(request: Request):
    """Refresh JWT token"""
    try:
        # Extract token from Authorization header
        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authentication required")
        
        token = authorization.split(" ")[1]
        
        result = await auth_service.refresh_token(token)
        if not result:
            raise HTTPException(status_code=401, detail="Token refresh failed")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Token refresh failed: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")


# User management endpoints (admin only)
@router.get("/api/auth/users", response_model=UsersListResponse)
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


@router.post("/api/auth/users", response_model=UserResponse)
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


@router.put("/api/auth/profile", response_model=UserResponse)
async def update_profile(
    update_request: UserUpdateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Update current user's profile (email, display_name). Same field admins see in User Management."""
    try:
        # Only allow email and display_name for self-update
        request = UserUpdateRequest(
            email=update_request.email,
            display_name=update_request.display_name,
            avatar_url=None,
            role=None,
            is_active=None,
        )
        result = await auth_service.update_user(current_user.user_id, request)
        if not result:
            raise HTTPException(status_code=404, detail="User not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to update profile")


@router.put("/api/auth/users/{user_id}", response_model=UserResponse)
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


@router.post("/api/auth/users/{user_id}/change-password")
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


@router.delete("/api/auth/users/{user_id}")
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


@router.post("/api/auth/users/{user_id}/avatar")
async def upload_user_avatar(
    user_id: str,
    file: UploadFile = File(...),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Upload avatar image for a user"""
    try:
        # Users can only upload their own avatar unless they're admin
        if current_user.user_id != user_id and current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Can only upload your own avatar")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Validate file size (max 5MB)
        content = await file.read()
        file_size = len(content)
        max_size = 5 * 1024 * 1024  # 5MB
        if file_size > max_size:
            raise HTTPException(status_code=400, detail="File size must be less than 5MB")
        
        # Sanitize filename
        sanitized_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
        
        # Create user avatars directory
        uploads_base = Path(os.getenv("UPLOAD_DIR", "/opt/bastion/uploads"))
        avatars_dir = uploads_base / "Users" / user_id / "avatars"
        avatars_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_ext = Path(sanitized_filename).suffix
        unique_filename = f"avatar_{uuid.uuid4()}{file_ext}"
        file_path = avatars_dir / unique_filename
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Update user's avatar_url in database
        avatar_url = f"/api/auth/users/{user_id}/avatar/{unique_filename}"
        update_request = UserUpdateRequest(avatar_url=avatar_url)
        result = await auth_service.update_user(user_id, update_request)
        
        if not result:
            # Cleanup file if update failed
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=500, detail="Failed to update avatar URL")
        
        return {
            "avatar_url": avatar_url,
            "message": "Avatar uploaded successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload user avatar: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload avatar")


@router.get("/api/auth/users/{user_id}/avatar/{filename}")
async def get_user_avatar(
    user_id: str,
    filename: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Serve user avatar image"""
    try:
        # Users can view any user's avatar
        uploads_base = Path(os.getenv("UPLOAD_DIR", "/opt/bastion/uploads"))
        file_path = uploads_base / "Users" / user_id / "avatars" / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Avatar not found")
        
        # Determine media type
        import mimetypes
        media_type, _ = mimetypes.guess_type(str(file_path))
        if not media_type:
            media_type = "image/jpeg"
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=filename
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve user avatar: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve avatar")


# Email verification endpoints
@router.post("/api/auth/verify-email")
async def verify_email(token: str, request: Request):
    """Verify email address using verification token"""
    try:
        result = await auth_service.verify_email_token(token)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message", "Verification failed"))
        
        return {
            "message": result.get("message", "Email verified successfully"),
            "user_id": result.get("user_id"),
            "email": result.get("email")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Email verification failed: {e}")
        raise HTTPException(status_code=500, detail="Email verification failed")


@router.post("/api/auth/resend-verification")
async def resend_verification(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
    request: Request = None
):
    """Resend email verification email"""
    try:
        # Get base URL from request
        base_url = None
        if request:
            base_url = f"{request.url.scheme}://{request.url.netloc}"
        
        result = await auth_service.resend_verification_email(current_user.user_id, base_url)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to resend verification email"))
        
        return {
            "message": result.get("message", "Verification email sent"),
            "email": result.get("email")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Resend verification failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to resend verification email")


@router.get("/api/auth/email-status")
async def get_email_status(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get email verification status and rate limit info"""
    try:
        status = await auth_service.get_email_verification_status(current_user.user_id)
        
        if not status.get("success"):
            raise HTTPException(status_code=400, detail=status.get("message", "Failed to get email status"))
        
        # Get rate limit info
        from services.email_rate_limiter import EmailRateLimiter
        from services.service_container import service_container
        
        if not service_container.is_initialized:
            await service_container.initialize()
        
        rate_limiter = EmailRateLimiter(service_container.db_pool)
        rate_limit_info = await rate_limiter.check_rate_limit(current_user.user_id)
        
        return {
            "email": status.get("email"),
            "email_verified": status.get("email_verified", False),
            "verification_sent_at": status.get("verification_sent_at"),
            "verification_expires_at": status.get("verification_expires_at"),
            "rate_limit_info": rate_limit_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get email status failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get email status")
