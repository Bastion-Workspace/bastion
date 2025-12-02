"""
Projects API endpoints
Universal project creation for different project types (electronics, fiction, etc.)
"""

import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional

from services.folder_service import FolderService
from services.file_manager.file_manager_service import get_file_manager
from services.file_manager.models.file_placement_models import SourceType, FilePlacementRequest
from utils.auth_middleware import get_current_user, AuthenticatedUserResponse

logger = logging.getLogger(__name__)

# Create folder service instance
_folder_service_instance = None

async def get_folder_service() -> FolderService:
    """Get or create folder service instance"""
    global _folder_service_instance
    if _folder_service_instance is None:
        _folder_service_instance = FolderService()
        await _folder_service_instance.initialize()
    return _folder_service_instance

router = APIRouter(prefix="/api/projects", tags=["projects"])


class CreateProjectRequest(BaseModel):
    parent_folder_id: Optional[str] = Field(None, description="Parent folder ID (None for root)")
    project_name: str = Field(..., description="Project name (becomes folder name and title)")
    project_type: str = Field(..., description="Project type (electronics, fiction, etc.)")


class CreateProjectResponse(BaseModel):
    success: bool
    document_id: str
    folder_id: str
    project_name: str
    project_type: str


@router.post("/create", response_model=CreateProjectResponse)
async def create_project(
    request: CreateProjectRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Create a new project with folder and project_plan.md file
    
    Creates:
    1. A folder with the project name
    2. A project_plan.md file inside with appropriate frontmatter
    """
    try:
        # Validate project type
        allowed_types = ["electronics", "general"]  # Extensible for future types
        if request.project_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid project type. Allowed types: {allowed_types}"
            )
        
        logger.info(f"🔧 Creating {request.project_type} project: {request.project_name}")
        
        # Step 1: Create folder
        folder_service = await get_folder_service()
        folder = await folder_service.create_folder(
            name=request.project_name,
            parent_folder_id=request.parent_folder_id,
            user_id=current_user.user_id,
            collection_type="user",
            current_user_role=current_user.role,
            admin_user_id=current_user.user_id
        )
        
        folder_id = folder.folder_id
        logger.info(f"✅ Created project folder: {folder_id}")
        
        # Step 2: Create project_plan.md file with frontmatter
        from uuid import uuid4
        document_id = str(uuid4())
        
        # Map project_type to frontmatter type
        # "general" projects use "project" in frontmatter to gate general_project_agent
        frontmatter_type = "project" if request.project_type == "general" else request.project_type
        
        # Build frontmatter
        frontmatter = f"""---
type: {frontmatter_type}
title: {request.project_name}
status: planning
files: []
---

"""
        
        # Build body content - different templates for different project types
        creation_date = datetime.utcnow().strftime("%Y-%m-%d")
        
        if request.project_type == "general":
            body_content = f"""# {request.project_name}

## Project Overview

Project created on {creation_date}. Open this file and ask the general project agent to help you plan and manage your project.

## Requirements

Project requirements and goals will be documented here.

## Design

Design decisions and approach will be documented here.

## Tasks

Project tasks and milestones will be tracked here.

## Notes

Project notes and documentation will be added here.
"""
        else:
            # Electronics project template
            body_content = f"""# {request.project_name}

## Project Overview

Project created on {creation_date}. Open this file and ask the {request.project_type} agent to help you design your project.

## Components

Components will be added here as the project develops.

## Design Notes

Design notes and decisions will be added here.
"""
        
        # Combine frontmatter and body
        full_content = frontmatter + body_content
        
        # Create file using FileManager
        file_manager = await get_file_manager()
        
        file_placement_request = FilePlacementRequest(
            content=full_content,
            title=request.project_name,
            filename="project_plan.md",
            source_type=SourceType.MANUAL,
            target_folder_id=folder_id,
            user_id=current_user.user_id,
            collection_type="user",
            process_immediately=False,  # Don't process project plans immediately
            priority=5,  # Normal priority
            source_metadata={},
            author=current_user.username
        )
        
        file_response = await file_manager.place_file(file_placement_request)
        
        logger.info(f"✅ Created project_plan.md: {file_response.document_id}")
        
        return CreateProjectResponse(
            success=True,
            document_id=file_response.document_id,
            folder_id=folder_id,
            project_name=request.project_name,
            project_type=request.project_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to create project: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

