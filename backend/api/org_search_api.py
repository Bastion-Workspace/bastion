"""
Org Search API - Roosevelt's Search Endpoints
Provides search, agenda, and TODO list functionality for org-mode files
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, Query, HTTPException

from utils.auth_middleware import get_current_user, AuthenticatedUserResponse
from services.org_search_service import get_org_search_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Org Tools"])


@router.get("/api/org/search")
async def search_org_files(
    query: str = Query(..., description="Search query string"),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter by"),
    todo_states: Optional[str] = Query(None, description="Comma-separated TODO states to filter by"),
    include_content: bool = Query(True, description="Include content in search, not just headings"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of results"),
    include_archives: bool = Query(False, description="Include _archive.org files in search results"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Search across all org files for the current user
    
    **BULLY!** Full-text search with filtering by tags and TODO states!
    
    **Query Parameters:**
    - **query**: Search text (required)
    - **tags**: Filter by tags, e.g., "work,urgent" (optional)
    - **todo_states**: Filter by states, e.g., "TODO,NEXT" (optional)
    - **include_content**: Search in content or just headings (default: true)
    - **limit**: Max results to return (default: 100)
    - **include_archives**: Include _archive.org files (default: false)
    
    **Returns:**
    - Search results with headings, tags, TODO states, and previews
    - Click results to open the file at the matching heading
    - Archives excluded by default - use include_archives=true to search them
    """
    try:
        # Parse comma-separated filters
        tags_list = [t.strip() for t in tags.split(',')] if tags else None
        states_list = [s.strip().upper() for s in todo_states.split(',')] if todo_states else None
        
        logger.info(f"🔍 ROOSEVELT: User {current_user.username} searching for '{query}' (archives: {include_archives})")
        
        # Get search service
        search_service = await get_org_search_service()
        
        # Execute search
        results = await search_service.search_org_files(
            user_id=current_user.user_id,
            query=query,
            tags=tags_list,
            todo_states=states_list,
            include_content=include_content,
            limit=limit,
            include_archives=include_archives
        )
        
        logger.info(f"✅ Found {results.get('count', 0)} results for '{query}'")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Org search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "results": [],
            "count": 0
        }


@router.get("/api/org/todos")
async def get_all_todos(
    states: Optional[str] = Query(None, description="Comma-separated TODO states to filter by"),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter by"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of results"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Get all TODO items across org files
    
    **BULLY!** View all your TODOs in one place!
    
    **Query Parameters:**
    - **states**: Filter by states, e.g., "TODO,NEXT,WAITING" (optional, default: all non-DONE)
    - **tags**: Filter by tags, e.g., "work,urgent" (optional)
    - **limit**: Max results to return (default: 100)
    
    **Returns:**
    - List of TODO items with metadata
    """
    try:
        # Parse filters
        tags_list = [t.strip() for t in tags.split(',')] if tags else None
        states_list = [s.strip().upper() for s in states.split(',')] if states else None
        
        # Default to non-DONE states if not specified
        if not states_list:
            states_list = ['TODO', 'NEXT', 'STARTED', 'WAITING', 'HOLD']
        
        logger.info(f"✅ ROOSEVELT: User {current_user.username} fetching TODOs")
        
        # Get search service (reuse search functionality)
        search_service = await get_org_search_service()
        
        # Search with empty query to get all matching items
        results = await search_service.search_org_files(
            user_id=current_user.user_id,
            query="",  # Empty query matches all
            tags=tags_list,
            todo_states=states_list,
            include_content=False,  # Just headings for TODO list
            limit=limit
        )
        
        # Filter to only items with TODO states
        if results.get('success'):
            results['results'] = [r for r in results['results'] if r.get('todo_state')]
        
        return results
        
    except Exception as e:
        logger.error(f"❌ TODO fetch failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "count": 0
        }


@router.get("/api/org/contacts")
async def get_all_contacts(
    category: Optional[str] = Query(None, description="Filter by category (tag or parent heading)"),
    limit: int = Query(500, ge=1, le=1000, description="Maximum number of results"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Get all contact entries across org files
    
    **BULLY!** View all your contacts in one place!
    
    Contacts are identified by having properties like EMAIL, PHONE, BIRTHDAY,
    or being in sections labeled "Contacts", "People", etc.
    
    **Query Parameters:**
    - **category**: Filter by tag or parent heading (optional)
    - **limit**: Max results to return (default: 500)
    
    **Returns:**
    - List of contact entries with properties
    """
    try:
        logger.info(f"📞 ROOSEVELT: User {current_user.username} fetching contacts")
        
        # Get search service
        search_service = await get_org_search_service()
        
        # Search with empty query to get all items
        results = await search_service.search_org_files(
            user_id=current_user.user_id,
            query="",  # Empty query matches all
            tags=None,
            todo_states=None,
            include_content=False,
            limit=limit
        )
        
        if not results.get('success'):
            return results
        
        # Filter to items that look like contacts
        # Contacts typically have EMAIL, PHONE, or BIRTHDAY properties
        # Or are under "Contacts", "People", etc. parent headings
        contact_keywords = ['contact', 'people', 'person', 'friend', 'family', 'colleague']
        
        contacts = []
        for item in results['results']:
            properties = item.get('properties', {})
            parent_path = item.get('parent_path', [])
            tags = item.get('tags', [])
            
            # Check if it has contact-like properties
            has_contact_property = any(
                key in properties 
                for key in ['EMAIL', 'PHONE', 'BIRTHDAY', 'COMPANY', 'ADDRESS', 'TITLE']
            )
            
            # Check if it's under a contact-related heading
            has_contact_parent = any(
                keyword in ' '.join(parent_path).lower() 
                for keyword in contact_keywords
            )
            
            # Check if it has contact-related tags
            has_contact_tag = any(
                keyword in tag.lower() 
                for tag in tags 
                for keyword in contact_keywords
            )
            
            if has_contact_property or has_contact_parent or has_contact_tag:
                # Apply category filter if specified
                if category:
                    if category in tags or any(category.lower() in p.lower() for p in parent_path):
                        contacts.append(item)
                else:
                    contacts.append(item)
        
        logger.info(f"✅ ROOSEVELT: Found {len(contacts)} contacts")
        
        return {
            "success": True,
            "results": contacts,
            "count": len(contacts),
            "files_searched": results.get('files_searched', 0)
        }
        
    except Exception as e:
        logger.error(f"❌ Contact fetch failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "count": 0
        }


@router.get("/api/org/agenda")
async def get_agenda(
    days_ahead: int = Query(7, ge=1, le=90, description="Number of days to look ahead"),
    include_scheduled: bool = Query(True, description="Include SCHEDULED items"),
    include_deadlines: bool = Query(True, description="Include DEADLINE items"),
    include_appointments: bool = Query(True, description="Include calendar appointments (active timestamps)"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Get agenda items (scheduled, deadlines, and appointments) for upcoming days
    
    **BULLY!** Your org-mode agenda in one view!
    
    **Query Parameters:**
    - **days_ahead**: Number of days to include (default: 7)
    - **include_scheduled**: Include SCHEDULED items (default: true)
    - **include_deadlines**: Include DEADLINE items (default: true)
    - **include_appointments**: Include calendar appointments with active timestamps (default: true)
    
    **Returns:**
    - Agenda items sorted by date
    - Grouped by day for easy viewing
    - Types: SCHEDULED, DEADLINE, APPOINTMENT
    """
    try:
        logger.info(f"📅 ROOSEVELT: User {current_user.username} fetching agenda")
        
        # Get search service
        search_service = await get_org_search_service()
        
        # Search for all items (empty query)
        all_results = await search_service.search_org_files(
            user_id=current_user.user_id,
            query="",
            tags=None,
            todo_states=None,
            include_content=False,
            limit=1000  # Get all items for agenda
        )
        
        if not all_results.get('success'):
            return all_results
        
        # Filter and organize by date
        from datetime import datetime, timedelta
        
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)
        
        agenda_items = []
        
        for result in all_results['results']:
            # Check for scheduled
            if include_scheduled and result.get('scheduled'):
                try:
                    # Parse org date format: "2025-01-20 Mon" or "2025-01-20 Mon 14:00"
                    scheduled_str = result['scheduled']
                    parts = scheduled_str.split()
                    date_str = parts[0]
                    item_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    
                    # Extract time if present (format: "14:00" or "14:00-15:00")
                    time_str = None
                    time_for_sort = None
                    if len(parts) >= 3:
                        time_str = parts[2]  # Could be "14:00" or "14:00-15:00"
                        # For sorting, use start time
                        if '-' in time_str:
                            time_for_sort = time_str.split('-')[0]
                        else:
                            time_for_sort = time_str
                    
                    if today <= item_date <= end_date:
                        # Create sort datetime (use midnight if no time)
                        if time_for_sort:
                            try:
                                hour, minute = map(int, time_for_sort.split(':'))
                                sort_datetime = datetime.combine(item_date, datetime.min.time().replace(hour=hour, minute=minute))
                            except:
                                sort_datetime = datetime.combine(item_date, datetime.min.time())
                        else:
                            sort_datetime = datetime.combine(item_date, datetime.min.time())
                        
                        agenda_items.append({
                            **result,
                            'agenda_type': 'SCHEDULED',
                            'agenda_date': item_date.isoformat(),
                            'sort_date': item_date,
                            'sort_datetime': sort_datetime,
                            'time': time_str
                        })
                except Exception as e:
                    logger.warning(f"Failed to parse scheduled date: {result.get('scheduled')}: {e}")
            
            # Check for deadline
            if include_deadlines and result.get('deadline'):
                try:
                    # Parse org date format: "2025-01-20 Mon" or "2025-01-20 Mon 18:00"
                    deadline_str = result['deadline']
                    parts = deadline_str.split()
                    date_str = parts[0]
                    item_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    
                    # Extract time if present
                    time_str = None
                    time_for_sort = None
                    if len(parts) >= 3:
                        time_str = parts[2]
                        if '-' in time_str:
                            time_for_sort = time_str.split('-')[0]
                        else:
                            time_for_sort = time_str
                    
                    if today <= item_date <= end_date:
                        # Calculate days until deadline
                        days_until = (item_date - today).days
                        
                        # Create sort datetime (use end of day if no time)
                        if time_for_sort:
                            try:
                                hour, minute = map(int, time_for_sort.split(':'))
                                sort_datetime = datetime.combine(item_date, datetime.min.time().replace(hour=hour, minute=minute))
                            except:
                                sort_datetime = datetime.combine(item_date, datetime.min.time().replace(hour=23, minute=59))
                        else:
                            sort_datetime = datetime.combine(item_date, datetime.min.time().replace(hour=23, minute=59))
                        
                        agenda_items.append({
                            **result,
                            'agenda_type': 'DEADLINE',
                            'agenda_date': item_date.isoformat(),
                            'sort_date': item_date,
                            'sort_datetime': sort_datetime,
                            'days_until': days_until,
                            'is_urgent': days_until <= 3,  # Flag urgent deadlines
                            'time': time_str
                        })
                except Exception as e:
                    logger.warning(f"Failed to parse deadline date: {result.get('deadline')}: {e}")
            
            # Check for active timestamps (calendar appointments)
            # These are timestamps like <2025-10-08 Wed 19:00-19:05>
            active_timestamps = result.get('active_timestamps', [])
            if include_appointments and active_timestamps:
                for timestamp_str in active_timestamps:
                    try:
                        # Parse: "2025-10-08 Wed 19:00-19:05" or "2025-10-23 Thu"
                        date_str = timestamp_str.split()[0]
                        item_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        
                        if today <= item_date <= end_date:
                            # Extract time if present
                            time_str = None
                            time_for_sort = None
                            parts = timestamp_str.split()
                            if len(parts) >= 3:  # Has time component
                                time_str = parts[2]  # Could be "19:00" or "19:00-19:05"
                                if '-' in time_str:
                                    time_for_sort = time_str.split('-')[0]
                                else:
                                    time_for_sort = time_str
                            
                            # Create sort datetime
                            if time_for_sort:
                                try:
                                    hour, minute = map(int, time_for_sort.split(':'))
                                    sort_datetime = datetime.combine(item_date, datetime.min.time().replace(hour=hour, minute=minute))
                                except:
                                    sort_datetime = datetime.combine(item_date, datetime.min.time())
                            else:
                                sort_datetime = datetime.combine(item_date, datetime.min.time())
                            
                            agenda_items.append({
                                **result,
                                'agenda_type': 'APPOINTMENT',
                                'agenda_date': item_date.isoformat(),
                                'sort_date': item_date,
                                'sort_datetime': sort_datetime,
                                'time': time_str,
                                'timestamp': timestamp_str
                            })
                    except Exception as e:
                        logger.warning(f"Failed to parse active timestamp: {timestamp_str}: {e}")
        
        # Sort by datetime (date + time) for proper chronological ordering
        agenda_items.sort(key=lambda x: x.get('sort_datetime', datetime.combine(x['sort_date'], datetime.min.time())))
        
        # Group by date
        grouped = {}
        for item in agenda_items:
            date_key = item['agenda_date']
            if date_key not in grouped:
                grouped[date_key] = []
            grouped[date_key].append(item)
        
        # Sort items within each date group by time
        for date_key in grouped:
            grouped[date_key].sort(key=lambda x: x.get('sort_datetime', datetime.combine(x['sort_date'], datetime.min.time())))
        
        return {
            "success": True,
            "days_ahead": days_ahead,
            "agenda_items": agenda_items,
            "grouped_by_date": grouped,
            "count": len(agenda_items),
            "date_range": {
                "start": today.isoformat(),
                "end": end_date.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Agenda fetch failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "agenda_items": [],
            "count": 0
        }


@router.get("/api/org/backlinks")
async def get_backlinks(
    filename: str = Query(..., description="Filename to find backlinks for"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Find all org files that link to the specified file
    
    **BULLY!** Discover knowledge connections through backlinks!
    
    **Query Parameters:**
    - **filename**: The filename to find backlinks for (e.g., "projects.org")
    
    **Returns:**
    - List of files that contain links to the target file
    - Link count and context for each backlink
    
    **Example:**
    If "ideas.org" contains `[[file:projects.org][My Projects]]`,
    then calling `/api/org/backlinks?filename=projects.org` will return "ideas.org" as a backlink.
    """
    try:
        logger.info(f"🔗 ROOSEVELT: User {current_user.username} finding backlinks for '{filename}'")
        
        # Get search service
        search_service = await get_org_search_service()
        
        # Find backlinks
        backlinks = await search_service.find_backlinks(
            user_id=current_user.user_id,
            target_filename=filename
        )
        
        return {
            "success": True,
            "target_file": filename,
            "backlinks": backlinks,
            "count": len(backlinks)
        }
        
    except Exception as e:
        logger.error(f"❌ Backlink search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "backlinks": [],
            "count": 0
        }


@router.get("/api/org/lookup-document")
async def lookup_document_by_filename(
    filename: str = Query(..., description="Filename to search for"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Look up org file by filename and return document_id from document_metadata
    
    **BULLY!** Find org files in document_metadata for proper opening!
    
    **Query Parameters:**
    - **filename**: The filename to search for (e.g., "inbox.org")
    
    **Returns:**
    - Document metadata including document_id for opening in DocumentViewer
    """
    try:
        logger.info(f"🔍 ROOSEVELT: Looking up org file '{filename}' for {current_user.username}")
        
        from services.database_manager.database_helpers import fetch_all
        from pathlib import Path
        
        # Extract just the filename without path
        target_path = Path(filename)
        target_filename = target_path.name  # "inbox.org" or "path/to/inbox.org" -> "inbox.org"
        
        logger.info(f"🔍 Searching for org file with filename: '{target_filename}'")
        
        # Query document_metadata table for org files matching the filename
        # Search by filename field, not title (title may differ from filename)
        query = """
            SELECT 
                document_id,
                title,
                filename,
                user_id,
                collection_type,
                doc_type,
                processing_status
            FROM document_metadata
            WHERE user_id = $1 
            AND filename ILIKE $2
            AND doc_type = 'org'
            ORDER BY 
                CASE WHEN filename = $3 THEN 0 ELSE 1 END,
                created_at DESC
            LIMIT 10
        """
        
        rows = await fetch_all(query, current_user.user_id, f"%{target_filename}%", target_filename)
        
        if not rows:
            logger.warning(f"⚠️ No org files found matching '{target_filename}'")
            raise HTTPException(
                status_code=404,
                detail=f"No org file found with filename '{target_filename}'"
            )
        
        # Filter to exact filename matches (case-insensitive)
        target_filename_lower = target_filename.lower()
        
        exact_matches = []
        partial_matches = []
        
        for row in rows:
            doc = dict(row)
            doc_filename = doc.get('filename', '')
            doc_filename_lower = doc_filename.lower()
            
            if doc_filename_lower == target_filename_lower:
                exact_matches.append(doc)
            elif target_filename_lower in doc_filename_lower:
                partial_matches.append(doc)
        
        # Prefer exact matches
        results = exact_matches if exact_matches else partial_matches
        
        if not results:
            logger.warning(f"⚠️ No matching org files for '{filename}' after filtering")
            raise HTTPException(
                status_code=404,
                detail=f"No org file found matching '{filename}'"
            )
        
        # Return the best match (first exact match, or first partial match)
        best_match = results[0]
        
        logger.info(f"✅ Found org file: {best_match.get('title')} (ID: {best_match.get('document_id')})")
        
        return {
            "success": True,
            "document": best_match
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Document lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/org/discover-targets")
async def discover_refile_targets(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Discover all potential refile targets for the user
    
    **BULLY!** Find all the places to refile your TODOs!
    
    **Returns:**
    - List of refile targets (files and headings)
    """
    try:
        logger.info(f"🎯 ROOSEVELT: Discovering refile targets for {current_user.username}")
        
        from services.org_refile_service import get_org_refile_service
        
        refile_service = await get_org_refile_service()
        targets = await refile_service.discover_refile_targets(current_user.user_id)
        
        return {
            "success": True,
            "targets": targets,
            "count": len(targets)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to discover refile targets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel

class RefileRequest(BaseModel):
    source_file: str
    source_line: int
    target_file: str
    target_heading_line: Optional[int] = None

@router.post("/api/org/refile")
async def refile_entry(
    request: RefileRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Refile (move) an org entry from one location to another
    
    **ROOSEVELT REFILE OPERATION!** Move that TODO!
    
    **Body Parameters:**
    - **source_file**: Source file relative path (e.g., "OrgMode/inbox.org")
    - **source_line**: Line number of entry to move
    - **target_file**: Target file relative path  
    - **target_heading_line**: Target heading line (optional, None = file root)
    
    **Returns:**
    - Success status and operation details
    """
    try:
        logger.info(f"📦 ROOSEVELT: Refiling from {request.source_file}:{request.source_line} to {request.target_file}")
        
        from services.org_refile_service import get_org_refile_service
        
        refile_service = await get_org_refile_service()
        
        result = await refile_service.refile_entry(
            user_id=current_user.user_id,
            source_file=request.source_file,
            source_line=request.source_line,
            target_file=request.target_file,
            target_heading_line=request.target_heading_line
        )
        
        if result['success']:
            logger.info(f"✅ Refile successful: {result}")
        else:
            logger.error(f"❌ Refile failed: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Refile operation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== ARCHIVE ENDPOINTS =====

class ArchiveRequest(BaseModel):
    source_file: str
    line_number: int
    archive_location: Optional[str] = None

@router.get("/api/org/archive-location")
async def get_archive_location(
    source_file: str = Query(..., description="Source file relative path (e.g., 'OrgMode/tasks.org')"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Get archive location for a source file
    
    **ROOSEVELT ARCHIVE LOCATION!** Check where entries will be archived!
    
    Returns archive location with priority:
    1. File-level #+ARCHIVE: directive (if present)
    2. User's settings default archive location
    3. Default behavior ({filename}_archive.org)
    
    **Query Parameters:**
    - **source_file**: Source file relative path (e.g., "OrgMode/tasks.org")
    
    **Returns:**
    - Archive location information with source and preview
    """
    try:
        from services.org_archive_service import get_org_archive_service
        from services.folder_service import FolderService
        from pathlib import Path
        from config import settings
        
        archive_service = await get_org_archive_service()
        folder_service = FolderService()
        
        # Resolve source file path
        from services.database_manager.database_helpers import fetch_one
        row = await fetch_one("SELECT username FROM users WHERE user_id = $1", current_user.user_id)
        username = row['username'] if row else current_user.user_id
        upload_dir = Path(settings.UPLOAD_DIR)
        user_base_dir = upload_dir / "Users" / username
        source_path = user_base_dir / source_file
        
        if not source_path.exists():
            raise HTTPException(status_code=404, detail=f"Source file not found: {source_file}")
        
        # Check file-level directive
        file_archive = None
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            file_archive = archive_service._parse_file_archive_directive(file_content)
        except Exception as e:
            logger.warning(f"Failed to read file for archive directive: {e}")
        
        # Check settings
        from services.org_settings_service import OrgSettingsService
        settings_service = OrgSettingsService()
        org_settings = await settings_service.get_settings(current_user.user_id)
        settings_archive = None
        if (org_settings.archive_preferences and 
            org_settings.archive_preferences.default_archive_location):
            settings_archive = org_settings.archive_preferences.default_archive_location
        
        # Determine which one is being used
        source_stem = source_path.stem
        archive_location = None
        source_type = None
        
        if file_archive:
            archive_location = file_archive.replace('%s', source_stem)
            source_type = 'file'
        elif settings_archive:
            archive_location = settings_archive.replace('%s', source_stem)
            source_type = 'settings'
        else:
            archive_location = f"{source_stem}_archive.org"
            source_type = 'default'
        
        # Get preview of full path (resolve relative to source file's directory if it's a relative path)
        archive_path = await archive_service._resolve_file_path(
            current_user.user_id,
            archive_location,
            folder_service,
            base_path=source_path
        )
        
        return {
            "success": True,
            "archive_location": archive_location,
            "source_type": source_type,
            "file_archive": file_archive,
            "settings_archive": settings_archive,
            "preview": str(archive_path.relative_to(user_base_dir)) if archive_path.is_relative_to(user_base_dir) else str(archive_path)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get archive location: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/org/archive")
async def archive_entry(
    request: ArchiveRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Archive a single org entry to the archive file
    
    **ROOSEVELT ARCHIVE OPERATION!** Keep your files clean!
    
    **Body Parameters:**
    - **source_file**: Source file relative path (e.g., "OrgMode/tasks.org")
    - **line_number**: Line number of entry to archive (1-based)
    - **archive_location**: Optional custom archive location (overrides file-level and settings)
    
    **Returns:**
    - Success status and operation details
    """
    try:
        logger.info(f"📦 ROOSEVELT: Archiving from {request.source_file}:{request.line_number}")
        
        from services.org_archive_service import get_org_archive_service
        
        archive_service = await get_org_archive_service()
        
        result = await archive_service.archive_entry(
            user_id=current_user.user_id,
            source_file=request.source_file,
            line_number=request.line_number,
            archive_location=request.archive_location
        )
        
        if result['success']:
            logger.info(f"✅ Archive successful: {result}")
        else:
            logger.error(f"❌ Archive failed: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Archive operation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class BulkArchiveRequest(BaseModel):
    source_file: str
    archive_location: Optional[str] = None

@router.post("/api/org/archive-bulk")
async def archive_all_done(
    request: BulkArchiveRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Archive all DONE items from a file
    
    **ROOSEVELT BULK ARCHIVE!** Clean house - archive all completed tasks!
    
    **Body Parameters:**
    - **source_file**: Source file relative path (e.g., "OrgMode/inbox.org")
    - **archive_location**: Optional custom archive location (defaults to {source}_archive.org)
    
    **Returns:**
    - Count of archived items and operation details
    """
    try:
        logger.info(f"📦 ROOSEVELT: Bulk archiving DONE items from {request.source_file}")
        
        from services.org_archive_service import get_org_archive_service
        
        archive_service = await get_org_archive_service()
        
        result = await archive_service.archive_all_done(
            user_id=current_user.user_id,
            source_file=request.source_file,
            archive_location=request.archive_location
        )
        
        if result['success']:
            logger.info(f"✅ Bulk archive successful: {result['archived_count']} items archived")
        else:
            logger.error(f"❌ Bulk archive failed")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Bulk archive operation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== RECURRING TASKS ENDPOINTS =====

class TaskCompletionRequest(BaseModel):
    file_path: str
    line_number: int

@router.post("/api/org/recurring/complete")
async def handle_recurring_task_completion(
    request: TaskCompletionRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Handle completion of a recurring task
    
    **ROOSEVELT RECURRING TASKS!** Build lasting habits!
    
    If task has a repeater (e.g., SCHEDULED: <2025-10-22 Tue +1w>):
    - Resets task to TODO
    - Updates timestamp to next occurrence
    - Logs completion for consistency tracking
    
    **Body Parameters:**
    - **file_path**: File path (e.g., "OrgMode/tasks.org")
    - **line_number**: Line number of task (1-based)
    
    **Returns:**
    - Success status and next occurrence details
    """
    try:
        logger.info(f"🔁 ROOSEVELT: Handling task completion at {request.file_path}:{request.line_number}")
        
        from services.org_recurring_service import get_org_recurring_service
        
        recurring_service = await get_org_recurring_service()
        
        result = await recurring_service.handle_task_completion(
            user_id=current_user.user_id,
            file_path=request.file_path,
            line_number=request.line_number
        )
        
        if result['success'] and result['is_recurring']:
            logger.info(f"✅ Recurring task handled: {result['message']}")
        else:
            logger.info(f"ℹ️ Non-recurring task, no special handling")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Recurring task completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class HabitConsistencyRequest(BaseModel):
    file_path: str
    heading_text: str
    days: int = 30

@router.post("/api/org/recurring/consistency")
async def get_habit_consistency(
    request: HabitConsistencyRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Get habit consistency data for a recurring task
    
    **ROOSEVELT HABIT TRACKING!** Track your consistency!
    
    Analyzes completion history to show:
    - Total completions vs expected
    - Consistency percentage
    - Current streak
    - Last completion date
    
    **Body Parameters:**
    - **file_path**: File path
    - **heading_text**: Task heading to track
    - **days**: Days to analyze (default: 30)
    
    **Returns:**
    - Consistency statistics
    """
    try:
        logger.info(f"📊 ROOSEVELT: Getting consistency for {request.heading_text}")
        
        from services.org_recurring_service import get_org_recurring_service
        
        recurring_service = await get_org_recurring_service()
        
        result = await recurring_service.get_habit_consistency(
            user_id=current_user.user_id,
            file_path=request.file_path,
            heading_text=request.heading_text,
            days=request.days
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Habit consistency check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== TIME TRACKING (CLOCKING) ENDPOINTS =====

class ClockInRequest(BaseModel):
    file_path: str
    line_number: int
    heading: str

@router.post("/api/org/clock/in")
async def clock_in(
    request: ClockInRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Clock in to a task
    
    **ROOSEVELT TIME TRACKING!** Start the productivity meter!
    
    **Body Parameters:**
    - **file_path**: File path (e.g., "OrgMode/tasks.org")
    - **line_number**: Line number of task heading (1-based)
    - **heading**: Task heading text (for display)
    
    **Returns:**
    - Clock start time and details
    """
    try:
        logger.info(f"⏰ ROOSEVELT: Clock in at {request.file_path}:{request.line_number}")
        
        from services.org_clocking_service import get_org_clocking_service
        
        clocking_service = await get_org_clocking_service()
        
        result = await clocking_service.clock_in(
            user_id=current_user.user_id,
            file_path=request.file_path,
            line_number=request.line_number,
            heading=request.heading
        )
        
        if result['success']:
            logger.info(f"✅ Clocked in: {request.heading}")
        else:
            logger.warning(f"⚠️ Clock in failed: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Clock in failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/org/clock/out")
async def clock_out(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Clock out of current task
    
    **ROOSEVELT CLOCK OUT!** Stop the timer and log your progress!
    
    Writes CLOCK entry to LOGBOOK drawer in format:
    ```
    CLOCK: [2025-10-22 Tue 10:00]--[2025-10-22 Tue 11:30] =>  1:30
    ```
    
    **Returns:**
    - Duration worked and CLOCK entry details
    """
    try:
        logger.info(f"⏰ ROOSEVELT: Clock out for user {current_user.user_id}")
        
        from services.org_clocking_service import get_org_clocking_service
        
        clocking_service = await get_org_clocking_service()
        
        result = await clocking_service.clock_out(
            user_id=current_user.user_id
        )
        
        if result['success']:
            logger.info(f"✅ Clocked out: {result['duration_display']}")
        else:
            logger.warning(f"⚠️ Clock out failed: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Clock out failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/org/clock/active")
async def get_active_clock(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Get user's active clock if any
    
    **ROOSEVELT CLOCK CHECK!** See if you're on the clock!
    
    **Returns:**
    - Active clock details with elapsed time, or null if not clocked in
    """
    try:
        from services.org_clocking_service import get_org_clocking_service
        
        clocking_service = await get_org_clocking_service()
        
        active_clock = await clocking_service.get_active_clock(
            user_id=current_user.user_id
        )
        
        return {
            "success": True,
            "active_clock": active_clock
        }
        
    except Exception as e:
        logger.error(f"❌ Get active clock failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TimeReportRequest(BaseModel):
    file_path: Optional[str] = None
    days: int = 30

@router.post("/api/org/clock/report")
async def get_time_report(
    request: TimeReportRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Generate time report from CLOCK entries
    
    **ROOSEVELT TIME REPORT!** See where your time went!
    
    Analyzes LOGBOOK entries to show:
    - Total time tracked
    - Time per task/file
    - Time per day
    - Statistics and trends
    
    **Body Parameters:**
    - **file_path**: Optional specific file (otherwise all files)
    - **days**: Days to include (default: 30)
    
    **Returns:**
    - Time report with statistics
    """
    try:
        logger.info(f"📊 ROOSEVELT: Time report for user {current_user.user_id}")
        
        from services.org_clocking_service import get_org_clocking_service
        
        clocking_service = await get_org_clocking_service()
        
        result = await clocking_service.get_time_report(
            user_id=current_user.user_id,
            file_path=request.file_path,
            days=request.days
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Time report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

