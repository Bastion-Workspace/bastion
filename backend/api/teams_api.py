"""
Teams API - REST endpoints for team management, invitations, and posts
"""

import logging
import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from services.team_service import TeamService
from services.team_invitation_service import TeamInvitationService
from services.team_post_service import TeamPostService
from services.messaging.messaging_service import messaging_service
from utils.auth_middleware import get_current_user
from models.api_models import AuthenticatedUserResponse
from config import settings
from models.team_models import (
    CreateTeamRequest, UpdateTeamRequest, AddMemberRequest, UpdateMemberRoleRequest,
    CreatePostRequest, AddReactionRequest, CreateCommentRequest,
    TeamResponse, TeamsListResponse, TeamMembersListResponse, TeamInvitationResponse,
    TeamPostsListResponse, TeamPostResponse, PostCommentsListResponse
)
from models.team_models import TeamRole, PostType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/teams", tags=["teams"])

# Debug: Log all routes when module loads
import sys
print(f"🔍 Teams API routes registered with prefix: /api/teams", file=sys.stderr)

# Global service instances (will be initialized in main.py)
team_service = TeamService()
invitation_service = TeamInvitationService()
post_service = TeamPostService()


# =====================
# TEAM MANAGEMENT ENDPOINTS
# =====================

@router.post("", response_model=TeamResponse)
async def create_team(
    request: CreateTeamRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Create a new team"""
    try:
        team = await team_service.create_team(
            name=request.team_name,
            description=request.description,
            creator_id=current_user.user_id,
            avatar_url=request.avatar_url
        )
        return TeamResponse(**team)
    
    except Exception as e:
        logger.error(f"Failed to create team: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=TeamsListResponse)
async def list_user_teams(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """List all teams user is member of"""
    try:
        teams = await team_service.list_user_teams(current_user.user_id)
        return TeamsListResponse(
            teams=[TeamResponse(**team) for team in teams],
            total=len(teams)
        )
    
    except Exception as e:
        logger.error(f"Failed to list user teams: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{team_id}", response_model=TeamResponse)
async def get_team(
    team_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get team details"""
    try:
        team = await team_service.get_team(team_id, current_user.user_id)
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")
        return TeamResponse(**team)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get team: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{team_id}", response_model=TeamResponse)
async def update_team(
    team_id: str,
    request: UpdateTeamRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update team details (admin only)"""
    try:
        updates = request.dict(exclude_unset=True)
        team = await team_service.update_team(team_id, updates, current_user.user_id)
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")
        return TeamResponse(**team)
    
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update team: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{team_id}")
async def delete_team(
    team_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete team (admin only)"""
    try:
        success = await team_service.delete_team(team_id, current_user.user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Team not found")
        return {"success": True}
    
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete team: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{team_id}/avatar")
async def upload_team_avatar(
    team_id: str,
    file: UploadFile = File(...),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Upload avatar image for a team (admin only)"""
    try:
        # Check team access and admin role
        role = await team_service.check_team_access(team_id, current_user.user_id)
        if role != "admin":
            raise HTTPException(status_code=403, detail="Only team admins can upload team avatars")
        
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
        
        # Create team avatars directory
        uploads_base = Path(settings.UPLOAD_DIR)
        avatars_dir = uploads_base / "Teams" / team_id / "avatars"
        avatars_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_ext = Path(sanitized_filename).suffix
        unique_filename = f"avatar_{uuid.uuid4()}{file_ext}"
        file_path = avatars_dir / unique_filename
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Update team's avatar_url in database
        avatar_url = f"/api/teams/{team_id}/avatar/{unique_filename}"
        updates = {"avatar_url": avatar_url}
        team = await team_service.update_team(team_id, updates, current_user.user_id)
        
        if not team:
            # Cleanup file if update failed
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=500, detail="Failed to update avatar URL")
        
        return {
            "avatar_url": avatar_url,
            "message": "Avatar uploaded successfully"
        }
    
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload team avatar: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload avatar")


@router.get("/{team_id}/avatar/{filename}")
async def get_team_avatar(
    team_id: str,
    filename: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Serve team avatar image"""
    try:
        # Check team access (must be team member to view avatar)
        role = await team_service.check_team_access(team_id, current_user.user_id)
        if not role:
            raise HTTPException(status_code=403, detail="Not a team member")
        
        uploads_base = Path(settings.UPLOAD_DIR)
        file_path = uploads_base / "Teams" / team_id / "avatars" / filename
        
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
    
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve team avatar: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve avatar")


# =====================
# MEMBER MANAGEMENT ENDPOINTS
# =====================

@router.get("/{team_id}/members", response_model=TeamMembersListResponse)
async def get_team_members(
    team_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """List team members with online presence"""
    try:
        members = await team_service.get_team_members(team_id, current_user.user_id)
        from models.team_models import TeamMemberResponse
        return TeamMembersListResponse(
            members=[TeamMemberResponse(**member) for member in members],
            total=len(members)
        )
    
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get team members: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{team_id}/members")
async def add_member(
    team_id: str,
    request: AddMemberRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Add member to team (admin only)"""
    try:
        success = await team_service.add_member(
            team_id=team_id,
            user_id=request.user_id,
            role=request.role,
            added_by=current_user.user_id
        )
        if not success:
            raise HTTPException(status_code=400, detail="Failed to add member")
        return {"success": True}
    
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to add member: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{team_id}/members/{user_id}")
async def remove_member(
    team_id: str,
    user_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Remove member from team (admin only)"""
    try:
        success = await team_service.remove_member(team_id, user_id, current_user.user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Member not found")
        return {"success": True}
    
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove member: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{team_id}/members/{user_id}/role")
async def update_member_role(
    team_id: str,
    user_id: str,
    request: UpdateMemberRoleRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update member role (admin only)"""
    try:
        success = await team_service.update_member_role(
            team_id=team_id,
            user_id=user_id,
            new_role=request.role,
            updated_by=current_user.user_id
        )
        if not success:
            raise HTTPException(status_code=404, detail="Member not found")
        return {"success": True}
    
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update member role: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# INVITATION ENDPOINTS
# =====================

@router.post("/{team_id}/invitations")
async def create_invitation(
    team_id: str,
    invited_user_id: str = Query(..., description="User ID to invite"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Create team invitation (admin only)"""
    try:
        invitation = await invitation_service.create_invitation(
            team_id=team_id,
            invited_user_id=invited_user_id,
            invited_by=current_user.user_id
        )
        return TeamInvitationResponse(**invitation)
    
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create invitation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{team_id}/invitations", response_model=List[TeamInvitationResponse])
async def list_team_invitations(
    team_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """List team invitations (admin only)"""
    try:
        # Check admin permission
        role = await team_service.check_team_access(team_id, current_user.user_id)
        if role != "admin":
            raise HTTPException(status_code=403, detail="Only team admins can view invitations")
        
        # Get invitations from database
        from utils.shared_db_pool import get_shared_db_pool
        db_pool = await get_shared_db_pool()
        async with db_pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.current_user_id', $1, true)", current_user.user_id)
            rows = await conn.fetch("""
                SELECT 
                    ti.*,
                    t.team_name,
                    u_inviter.username as inviter_username,
                    u_inviter.display_name as inviter_display_name
                FROM team_invitations ti
                INNER JOIN teams t ON t.team_id = ti.team_id
                INNER JOIN users u_inviter ON u_inviter.user_id = ti.invited_by
                WHERE ti.team_id = $1
                ORDER BY ti.created_at DESC
            """, team_id)
            
            invitations = [
                {
                    "invitation_id": str(row["invitation_id"]),
                    "team_id": str(row["team_id"]),
                    "team_name": row["team_name"],
                    "invited_user_id": row["invited_user_id"],
                    "invited_by": row["invited_by"],
                    "inviter_name": row["inviter_display_name"] or row["inviter_username"],
                    "status": row["status"],
                    "message_id": str(row["message_id"]) if row["message_id"] else None,
                    "created_at": row["created_at"].isoformat(),
                    "expires_at": row["expires_at"].isoformat(),
                    "responded_at": row["responded_at"].isoformat() if row["responded_at"] else None
                }
                for row in rows
            ]
            
            return [TeamInvitationResponse(**inv) for inv in invitations]
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list team invitations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{team_id}/invitations/{invitation_id}")
async def cancel_invitation(
    team_id: str,
    invitation_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Cancel invitation (admin only)"""
    try:
        success = await invitation_service.cancel_invitation(invitation_id, current_user.user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Invitation not found")
        return {"success": True}
    
    except (PermissionError, ValueError) as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel invitation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/invitations/pending", response_model=List[TeamInvitationResponse])
async def get_pending_invitations(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get user's pending invitations"""
    try:
        invitations = await invitation_service.get_pending_invitations(current_user.user_id)
        return [TeamInvitationResponse(**inv) for inv in invitations]
    
    except Exception as e:
        logger.error(f"Failed to get pending invitations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/invitations/{invitation_id}/accept", response_model=TeamResponse)
async def accept_invitation(
    invitation_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Accept team invitation"""
    try:
        team = await invitation_service.accept_invitation(invitation_id, current_user.user_id)
        return TeamResponse(**team)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to accept invitation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/invitations/{invitation_id}/reject")
async def reject_invitation(
    invitation_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Reject team invitation"""
    try:
        success = await invitation_service.reject_invitation(invitation_id, current_user.user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Invitation not found")
        return {"success": True}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reject invitation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# POST ENDPOINTS
# =====================

# IMPORTANT: More specific routes must come before general ones
# Attachment routes must come before general posts routes

@router.get("/{team_id}/posts/attachments/{filename}")
async def get_post_attachment(
    team_id: str,
    filename: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Serve team post attachment"""
    # Debug: print to stdout (always visible) - MUST appear if route is called
    import sys
    print(f"🔍 GET ATTACHMENT CALLED: team_id={team_id}, filename={filename}, user={current_user.user_id}", file=sys.stderr, flush=True)
    logger.error(f"🔍 GET ATTACHMENT CALLED: team_id={team_id}, filename={filename}, user={current_user.user_id}")
    
    try:
        # Check team access
        role = await team_service.check_team_access(team_id, current_user.user_id)
        if not role:
            print(f"❌ ACCESS DENIED: user {current_user.user_id} not a member of team {team_id}")
            raise HTTPException(status_code=403, detail="Not a team member")
        print(f"✅ ACCESS GRANTED: user {current_user.user_id} has role {role} in team {team_id}")
        
        # Get file path
        from pathlib import Path
        from fastapi.responses import FileResponse
        from urllib.parse import unquote
        import os
        
        # Decode URL-encoded filename
        decoded_filename = unquote(filename)
        
        # SECURITY: Strip any path components to prevent path traversal
        # This prevents attacks like ../../etc/passwd
        safe_filename = os.path.basename(decoded_filename)
        
        if not safe_filename or safe_filename in ('.', '..'):
            logger.warning(f"Invalid filename attempt: {decoded_filename}")
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        uploads_base = Path(settings.UPLOAD_DIR)
        team_posts_dir = uploads_base / "Teams" / team_id / "posts"
        file_path = team_posts_dir / safe_filename
        
        # SECURITY: Verify the resolved path is still within team_posts_dir
        # This is a defense-in-depth measure against path traversal
        try:
            file_path_resolved = file_path.resolve()
            team_posts_dir_resolved = team_posts_dir.resolve()
            
            if not str(file_path_resolved).startswith(str(team_posts_dir_resolved)):
                logger.error(f"Path traversal attempt detected: {filename} -> {file_path_resolved}")
                raise HTTPException(status_code=403, detail="Access denied")
        except Exception as e:
            logger.error(f"Path validation error: {e}")
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Log for debugging (both print and logger)
        print(f"📁 FILE PATH: {file_path}")
        print(f"📁 DIRECTORY EXISTS: {team_posts_dir.exists()}")
        logger.info(f"Serving attachment: team_id={team_id}, filename={filename}, safe_filename={safe_filename}, path={file_path}")
        
        # Check if directory exists
        if not team_posts_dir.exists():
            print(f"❌ DIRECTORY NOT FOUND: {team_posts_dir}")
            logger.warning(f"Team posts directory does not exist: {team_posts_dir}")
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if file exists
        if not file_path.exists():
            # List files in directory for debugging
            existing_files = list(team_posts_dir.glob("*")) if team_posts_dir.exists() else []
            file_list = [f.name for f in existing_files]
            print(f"❌ FILE NOT FOUND: {file_path}")
            print(f"📋 EXISTING FILES: {file_list}")
            logger.warning(
                f"File not found: {file_path}. "
                f"Directory exists: {team_posts_dir.exists()}. "
                f"Files in directory: {file_list}"
            )
            raise HTTPException(status_code=404, detail="File not found")
        
        print(f"✅ FILE FOUND: {file_path}, serving...")
        
        # Determine media type
        import mimetypes
        media_type, _ = mimetypes.guess_type(str(file_path))
        if not media_type:
            media_type = "application/octet-stream"
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=decoded_filename
        )
    
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ EXCEPTION IN GET ATTACHMENT: {e}")
        logger.error(f"Failed to serve post attachment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{team_id}/posts", response_model=TeamPostsListResponse)
async def get_team_posts(
    team_id: str,
    limit: int = Query(20, ge=1, le=100),
    before_post_id: Optional[str] = Query(None),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get team posts (paginated)"""
    try:
        result = await post_service.get_team_posts(
            team_id=team_id,
            user_id=current_user.user_id,
            limit=limit,
            before_post_id=before_post_id
        )
        return TeamPostsListResponse(
            posts=[TeamPostResponse(**post) for post in result["posts"]],
            total=result["total"],
            has_more=result["has_more"]
        )
    
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get team posts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{team_id}/posts/upload")
async def upload_post_attachment(
    team_id: str,
    file: UploadFile = File(...),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Upload attachment for team post"""
    try:
        # Check team access
        role = await team_service.check_team_access(team_id, current_user.user_id)
        if not role:
            raise HTTPException(status_code=403, detail="Not a team member")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Check file size - use larger limit for audio files
        max_size = 10 * 1024 * 1024  # 10MB default
        if file.content_type and file.content_type.startswith("audio/"):
            max_size = settings.AUDIO_ATTACHMENT_MAX_SIZE
        
        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB"
            )
        
        # Sanitize filename
        import re
        sanitized_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
        
        # Create team posts directory
        from pathlib import Path
        uploads_base = Path(settings.UPLOAD_DIR)
        team_posts_dir = uploads_base / "Teams" / team_id / "posts"
        team_posts_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        import uuid
        file_ext = Path(sanitized_filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = team_posts_dir / unique_filename
        
        # Save file
        print(f"💾 UPLOADING FILE: {file_path} (size: {file_size} bytes)")
        try:
            with open(file_path, "wb") as f:
                f.write(content)
                f.flush()
                # Ensure file is synced to disk
                import os
                os.fsync(f.fileno())
            print(f"✅ FILE WRITTEN: {file_path}")
        except Exception as e:
            print(f"❌ FILE WRITE FAILED: {file_path} - {e}")
            logger.error(f"Failed to write file {file_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
        # Verify file was saved
        if not file_path.exists():
            print(f"❌ FILE VERIFICATION FAILED: {file_path} does not exist after write")
            logger.error(f"File was not saved successfully: {file_path}")
            raise HTTPException(status_code=500, detail="Failed to save file")
        
        # Verify file size matches
        actual_size = file_path.stat().st_size
        if actual_size != file_size:
            print(f"⚠️ FILE SIZE MISMATCH: expected {file_size}, got {actual_size}")
            logger.warning(f"File size mismatch: expected {file_size}, got {actual_size}")
        
        print(f"✅ FILE SAVED: {file_path} (size: {file_size} bytes, actual: {actual_size} bytes)")
        logger.info(f"Saved attachment: {file_path} (size: {file_size} bytes, actual: {actual_size} bytes)")
        
        # Get image dimensions if it's an image
        width = None
        height = None
        if file.content_type and file.content_type.startswith("image/"):
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    width, height = img.size
            except Exception as e:
                logger.warning(f"Failed to get image dimensions: {e}")
        
        # Return relative path for API access
        relative_path = f"/api/teams/{team_id}/posts/attachments/{unique_filename}"
        
        response_data = {
            "file_path": relative_path,
            "filename": sanitized_filename,
            "file_size": file_size,
            "mime_type": file.content_type or "application/octet-stream",
            "width": width,
            "height": height
        }
        
        logger.info(
            f"Uploaded attachment for team {team_id}: "
            f"unique_filename={unique_filename}, "
            f"file_path={relative_path}, "
            f"actual_file_path={file_path}, "
            f"file_exists={file_path.exists()}, "
            f"response={response_data}"
        )
        
        return response_data
    
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload post attachment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{team_id}/posts/attachments/debug/list")
async def list_post_attachments_debug(
    team_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Debug endpoint to list all files in team posts directory"""
    try:
        # Check team access
        role = await team_service.check_team_access(team_id, current_user.user_id)
        if not role:
            raise HTTPException(status_code=403, detail="Not a team member")
        
        from pathlib import Path
        
        uploads_base = Path(settings.UPLOAD_DIR)
        team_posts_dir = uploads_base / "Teams" / team_id / "posts"
        
        files = []
        if team_posts_dir.exists():
            for file_path in team_posts_dir.iterdir():
                if file_path.is_file():
                    files.append({
                        "filename": file_path.name,
                        "size": file_path.stat().st_size,
                        "path": str(file_path)
                    })
        
        return {
            "team_id": team_id,
            "upload_dir_env": settings.UPLOAD_DIR,
            "uploads_base": str(uploads_base),
            "directory": str(team_posts_dir),
            "exists": team_posts_dir.exists(),
            "files": files,
            "count": len(files)
        }
    
    except Exception as e:
        logger.error(f"Failed to list attachments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{team_id}/posts", response_model=TeamPostResponse)
async def create_post(
    team_id: str,
    request: CreatePostRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Create team post"""
    try:
        post = await post_service.create_post(
            team_id=team_id,
            author_id=current_user.user_id,
            content=request.content,
            post_type=request.post_type,
            attachments=request.attachments or []
        )
        return TeamPostResponse(**post)
    
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create post: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{team_id}/posts/{post_id}")
async def delete_post(
    team_id: str,
    post_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete post (author or admin only)"""
    try:
        success = await post_service.delete_post(post_id, current_user.user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Post not found")
        return {"success": True}
    
    except (ValueError, PermissionError) as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete post: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# REACTION ENDPOINTS
# =====================

@router.post("/{team_id}/posts/{post_id}/reactions")
async def add_reaction(
    team_id: str,
    post_id: str,
    request: AddReactionRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Add reaction to post"""
    try:
        success = await post_service.add_reaction(
            post_id=post_id,
            user_id=current_user.user_id,
            reaction_type=request.reaction_type
        )
        if not success:
            raise HTTPException(status_code=400, detail="Failed to add reaction")
        return {"success": True}
    
    except (ValueError, PermissionError) as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add reaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{team_id}/posts/{post_id}/reactions/{reaction_type}")
async def remove_reaction(
    team_id: str,
    post_id: str,
    reaction_type: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Remove reaction from post"""
    try:
        success = await post_service.remove_reaction(
            post_id=post_id,
            user_id=current_user.user_id,
            reaction_type=reaction_type
        )
        if not success:
            raise HTTPException(status_code=404, detail="Reaction not found")
        return {"success": True}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove reaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# COMMENT ENDPOINTS
# =====================

@router.post("/{team_id}/posts/{post_id}/comments", response_model=Dict[str, Any])
async def create_comment(
    team_id: str,
    post_id: str,
    request: CreateCommentRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Create comment on post"""
    try:
        from models.team_models import PostCommentResponse
        comment = await post_service.create_comment(
            post_id=post_id,
            author_id=current_user.user_id,
            content=request.content
        )
        return PostCommentResponse(**comment).dict()
    
    except (ValueError, PermissionError) as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{team_id}/posts/{post_id}/comments", response_model=PostCommentsListResponse)
async def get_post_comments(
    team_id: str,
    post_id: str,
    limit: int = Query(50, ge=1, le=100),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get comments for a post"""
    try:
        from models.team_models import PostCommentResponse
        comments = await post_service.get_post_comments(post_id, limit)
        return PostCommentsListResponse(
            comments=[PostCommentResponse(**comment) for comment in comments],
            total=len(comments)
        )
    
    except Exception as e:
        logger.error(f"Failed to get post comments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{team_id}/posts/{post_id}/comments/{comment_id}")
async def delete_comment(
    team_id: str,
    post_id: str,
    comment_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete comment (author or admin only)"""
    try:
        success = await post_service.delete_comment(comment_id, current_user.user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Comment not found")
        return {"success": True}
    
    except (ValueError, PermissionError) as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

