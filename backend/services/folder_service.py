import logging
import asyncio
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime
from pathlib import Path

from models.api_models import (
    DocumentFolder, 
    FolderCreateRequest, 
    FolderUpdateRequest,
    FolderTreeResponse,
    FolderContentsResponse,
    DocumentInfo
)
from repositories.document_repository import DocumentRepository
from config import settings

logger = logging.getLogger(__name__)


class FolderService:
    """Service for managing document folders and hierarchy"""
    
    def __init__(self):
        self.document_repository = DocumentRepository()
        # No more server-side caching - database is the source of truth
        self.uploads_base = Path(settings.UPLOAD_DIR)
        self.global_base = self.uploads_base / "Global"
        self.users_base = self.uploads_base / "Users"
        self.teams_base = self.uploads_base / "Teams"
        
    async def initialize(self):
        """Initialize the folder service"""
        await self.document_repository.initialize()
        
        # Ensure base directory structure exists
        self.global_base.mkdir(parents=True, exist_ok=True)
        self.users_base.mkdir(parents=True, exist_ok=True)
        self.teams_base.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ Folder Service initialized")
        logger.info(f"📂 Global files: {self.global_base}")
        logger.info(f"📂 User files: {self.users_base}")
        logger.info(f"📂 Team files: {self.teams_base}")
    
    def get_user_base_path(self, user_id: str, username: str = None) -> Path:
        """Get the base path for a user's documents"""
        # Use username if provided, otherwise use user_id
        folder_name = username if username else user_id
        return self.users_base / folder_name
    
    def get_team_base_path(self, team_id: str) -> Path:
        """Get the base path for a team's documents"""
        return self.teams_base / team_id / "documents"
    
    async def _get_username(self, user_id: str) -> str:
        """Get username from user_id"""
        try:
            from services.database_manager.database_helpers import fetch_one
            row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
            return row['username'] if row else user_id
        except Exception as e:
            logger.warning(f"⚠️ Could not get username for {user_id}: {e}")
            return user_id
    
    async def get_folder_physical_path(self, folder_id: str) -> Optional[Path]:
        """Get the physical filesystem path for a folder"""
        try:
            folder_data = await self.document_repository.get_folder(folder_id)
            if not folder_data:
                return None
            
            # Build path from root to this folder
            path_components = []
            current_folder = folder_data
            
            while current_folder:
                path_components.insert(0, current_folder['name'])
                parent_id = current_folder.get('parent_folder_id')
                if not parent_id:
                    break
                current_folder = await self.document_repository.get_folder(parent_id)
            
            # Determine base path
            collection_type = folder_data.get('collection_type', 'user')
            user_id = folder_data.get('user_id')
            team_id = folder_data.get('team_id')
            
            # Convert UUID to string if needed (asyncpg returns UUID objects)
            if team_id and not isinstance(team_id, str):
                team_id = str(team_id)
            
            if team_id:
                base_path = self.get_team_base_path(team_id)
            elif collection_type == 'global':
                base_path = self.global_base
            else:
                username = await self._get_username(user_id) if user_id else "unknown"
                base_path = self.get_user_base_path(user_id, username)
            
            # Construct full path
            folder_path = base_path
            for component in path_components:
                folder_path = folder_path / component
            
            return folder_path
            
        except Exception as e:
            logger.error(f"❌ Failed to get physical path for folder {folder_id}: {e}")
            return None
    
    async def _create_physical_directory(self, folder_path: Path) -> bool:
        """Create physical directory on filesystem"""
        try:
            # Only log if directory doesn't exist (to reduce noise)
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"📁 Created physical directory: {folder_path}")
            # Directory already exists - no need to log
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create physical directory {folder_path}: {e}")
            return False
    
    async def get_document_file_path(self, filename: str, folder_id: str = None, user_id: str = None, collection_type: str = "user", team_id: str = None) -> Path:
        """
        Get the physical file path for a document based on folder structure.
        
        Args:
            filename: Name of the file
            folder_id: Folder ID to place file in (optional)
            user_id: User ID for user-specific files
            collection_type: 'user', 'global', or 'team'
            team_id: Team ID for team-specific files
            
        Returns:
            Path object for where the file should be stored
        """
        try:
            # If folder_id provided, get folder's physical path
            if folder_id:
                folder_path = await self.get_folder_physical_path(folder_id)
                if folder_path:
                    # Ensure directory exists
                    await self._create_physical_directory(folder_path)
                    return folder_path / filename
            
            # No folder_id - place at base level
            if team_id:
                base_path = self.get_team_base_path(team_id)
                await self._create_physical_directory(base_path)
            elif collection_type == "global":
                base_path = self.global_base
            else:
                username = await self._get_username(user_id) if user_id else "unknown"
                base_path = self.get_user_base_path(user_id, username)
                # Ensure user base directory exists
                await self._create_physical_directory(base_path)
            
            return base_path / filename
            
        except Exception as e:
            logger.error(f"❌ Failed to get document file path: {e}")
            # Fallback to old flat structure
            return self.uploads_base / filename
    
    async def create_folder(self, name: str, parent_folder_id: str = None, user_id: str = None, collection_type: str = "user", current_user_role: str = "user", admin_user_id: str = None, team_id: str = None) -> DocumentFolder:
        """
        Create a new folder or return existing one if already present.
        
        **ROOSEVELT'S UPSERT CAVALRY!** 🏇
        Now uses database-level UPSERT to handle race conditions properly!
        
        Security Rules:
        - Admins: Can create folders for themselves (user level) and Global
        - Regular Users: Can only create folders for themselves (user level)
        - Regular Users: Cannot create global folders
        """
        try:
            logger.info(f"🔍 Creating folder: name='{name}', parent_folder_id='{parent_folder_id}', user_id='{user_id}', collection_type='{collection_type}'")
            
            # Security validation
            if collection_type == "team":
                # Team folders can be created by any user (will be validated by team membership elsewhere)
                # For now, allow team folder creation
                pass
            elif current_user_role != "admin":
                # Regular users can only create folders for themselves
                if collection_type == "global":
                    raise ValueError("Regular users cannot create global folders")
                if user_id != admin_user_id:  # admin_user_id contains the current user's ID
                    raise ValueError("Regular users can only create folders for themselves")
            else:
                # Admins can create folders for themselves or global
                if collection_type == "user" and user_id != admin_user_id:
                    raise ValueError("Admins can only create folders for themselves or global folders")
            
            folder_id = str(uuid4())
            now = datetime.utcnow()
            
            # Validate parent folder exists and user has access (skip validation for immediate parent creation)
            if parent_folder_id:
                # For immediate parent creation (like in create_default_folders), skip validation
                # as the parent might have been created in the same transaction
                parent_folder = await self.get_folder(parent_folder_id, user_id, current_user_role)
                if not parent_folder:
                    # Log warning but don't fail - this might be a timing issue
                    logger.warning(f"⚠️ Parent folder {parent_folder_id} not found during creation of {name}, but continuing...")
                    # Don't raise error for now - let the database handle foreign key constraints
            
            # Create folder record using UPSERT
            folder_data = {
                "folder_id": folder_id,
                "name": name,
                "parent_folder_id": parent_folder_id,
                "user_id": user_id,
                "collection_type": collection_type,
                "team_id": team_id,
                "created_at": now,
                "updated_at": now
            }
            
            logger.info(f"📁 Attempting to create or get folder: {name}")
            
            # **BULLY!** Use new UPSERT method - handles race conditions at DB level!
            result_folder_data = await self.document_repository.create_or_get_folder(folder_data)
            
            if not result_folder_data:
                logger.error(f"❌ Folder creation/retrieval failed in repository: {name}")
                raise Exception("Folder creation/retrieval failed in repository")
            
            # Check if we got back a different folder_id (means folder already existed)
            actual_folder_id = result_folder_data['folder_id']
            if actual_folder_id != folder_id:
                logger.info(f"📁 Folder already existed: {name} → {actual_folder_id}")
            else:
                logger.info(f"✅ Folder created successfully: {name} → {actual_folder_id}")
            
            # Create physical directory on filesystem
            folder_path = await self.get_folder_physical_path(actual_folder_id)
            if folder_path:
                await self._create_physical_directory(folder_path)
            else:
                logger.warning(f"⚠️ Could not determine physical path for folder {actual_folder_id}, skipping directory creation")
            
            # Return DocumentFolder object with actual data from database
            return DocumentFolder(**result_folder_data)
            
        except Exception as e:
            logger.error(f"❌ Failed to create folder '{name}': {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def get_folder(self, folder_id: str, user_id: str = None, current_user_role: str = "user") -> Optional[DocumentFolder]:
        """Get a specific folder by ID with proper access control.
        
        Access Rules:
        - All users can access global folders (collection_type = 'global')
        - Users can only access their own user folders
        - Admins can access all folders
        """
        try:
            folder_data = await self.document_repository.get_folder(folder_id)
            if not folder_data:
                return None
            
            # Check access permissions
            folder_user_id = folder_data.get("user_id")
            folder_collection_type = folder_data.get("collection_type", "user")
            
            # Global folders are accessible to all users
            if folder_collection_type == "global":
                return DocumentFolder(**folder_data)
            
            # User folders require matching user_id or admin role
            if folder_user_id is not None and folder_user_id != user_id and current_user_role != "admin":
                logger.warning(f"⚠️ Access denied: User {user_id} (role: {current_user_role}) tried to access folder {folder_id} owned by {folder_user_id}")
                return None
            
            return DocumentFolder(**folder_data)
            
        except Exception as e:
            logger.error(f"❌ Failed to get folder {folder_id}: {e}")
            return None
    
    async def update_folder_metadata(self, folder_id: str, category: str = None, tags: List[str] = None, inherit_tags: bool = None) -> bool:
        """
        Update folder metadata (category, tags, inherit_tags)
        
        **ROOSEVELT FOLDER TAGGING PHASE 1**: Store metadata for automatic inheritance on upload!
        """
        try:
            return await self.document_repository.update_folder_metadata(folder_id, category, tags, inherit_tags)
        except Exception as e:
            logger.error(f"❌ Failed to update folder metadata {folder_id}: {e}")
            return False
    
    async def get_folder_metadata(self, folder_id: str) -> Dict[str, Any]:
        """
        Get folder metadata for tag inheritance
        
        **ROOSEVELT FOLDER INHERITANCE**: Get tags to apply to uploaded documents!
        """
        try:
            return await self.document_repository.get_folder_metadata(folder_id)
        except Exception as e:
            logger.error(f"❌ Failed to get folder metadata {folder_id}: {e}")
            return {'category': None, 'tags': [], 'inherit_tags': True}

    async def get_or_create_subfolder(self, parent_folder_id: str, folder_name: str, user_id: str, collection_type: str = "user", current_user_role: str = "user", admin_user_id: str = None) -> DocumentFolder:
        """
        Get or create a subfolder under a parent folder
        
        **SIMPLIFIED**: Just calls create_folder - UPSERT handles race conditions!
        """
        try:
            logger.info(f"📁 Getting or creating subfolder '{folder_name}' under parent {parent_folder_id}")
            
            # **BULLY!** No need to check first - create_folder uses UPSERT!
            return await self.create_folder(
                name=folder_name,
                parent_folder_id=parent_folder_id,
                user_id=user_id,
                collection_type=collection_type,
                current_user_role=current_user_role,
                admin_user_id=admin_user_id
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get or create subfolder '{folder_name}': {e}")
            raise

    async def get_or_create_root_folder(self, folder_name: str, user_id: str, collection_type: str = "user", current_user_role: str = "user", admin_user_id: str = None) -> DocumentFolder:
        """
        Get or create a root folder (no parent)
        
        **SIMPLIFIED**: Just calls create_folder - UPSERT handles race conditions!
        """
        try:
            logger.info(f"📁 Getting or creating root folder '{folder_name}'")
            
            # **BULLY!** No need to check first - create_folder uses UPSERT!
            return await self.create_folder(
                name=folder_name,
                parent_folder_id=None,  # Root folder has no parent
                user_id=user_id,
                collection_type=collection_type,
                current_user_role=current_user_role,
                admin_user_id=admin_user_id
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get or create root folder '{folder_name}': {e}")
            raise
    
    async def get_folder_tree(self, user_id: str = None, collection_type: str = "user") -> List[DocumentFolder]:
        """Get the complete folder tree for a user - always fresh from database"""
        try:
            logger.info(f"📁 Building fresh folder tree for user_id: {user_id}, collection_type: {collection_type}")
            
            # Database is the source of truth - no caching needed
            
            # Get user folders
            user_folders_data = await self.document_repository.get_folders_by_user(user_id, "user")
            logger.info(f"📁 Found {len(user_folders_data)} user folders for user {user_id}")
            for folder in user_folders_data:
                logger.info(f"📁 User folder: {folder.get('name')} (ID: {folder.get('folder_id')}, parent: {folder.get('parent_folder_id')})")
            
            # Get global folders (for admins or if specifically requested)
            global_folders_data = []
            is_admin = await self._is_admin(user_id) if user_id else False
            
            if collection_type == "global" or is_admin:
                global_folders_data = await self.document_repository.get_folders_by_user(None, "global")
                logger.debug(f"📁 Found {len(global_folders_data)} global folders")
            
            # Get team folders for user's teams
            team_folders_data = []
            if user_id:
                try:
                    from services.team_service import TeamService
                    team_service = TeamService()
                    await team_service.initialize()
                    user_teams = await team_service.list_user_teams(user_id)
                    team_ids = [team['team_id'] for team in user_teams]
                    if team_ids:
                        team_folders_data = await self.document_repository.get_folders_by_teams(team_ids)
                        logger.info(f"📁 Found {len(team_folders_data)} team folders for user {user_id}")
                except Exception as e:
                    logger.warning(f"Failed to get team folders for user {user_id}: {e}")
            
            # Combine all folders
            all_folders_data = user_folders_data + global_folders_data + team_folders_data
            
            # Add counts for each folder
            for folder_data in all_folders_data:
                folder_data["document_count"] = await self.document_repository.get_document_count_in_folder(folder_data["folder_id"])
                folder_data["subfolder_count"] = await self.document_repository.get_subfolder_count(folder_data["folder_id"])
            
            folders = [DocumentFolder(**folder_data) for folder_data in all_folders_data]
            
            # Build hierarchical structure
            folder_map = {folder.folder_id: folder for folder in folders}
            root_folders = []
            
            for folder in folders:
                if folder.parent_folder_id:
                    parent = folder_map.get(folder.parent_folder_id)
                    if parent:
                        if not hasattr(parent, 'children'):
                            parent.children = []
                        parent.children.append(folder)
                else:
                    root_folders.append(folder)
            
            # Create virtual root nodes for better organization
            virtual_roots = []
            
            # Create "My Documents" root node with virtual sources for all users
            user_root_folders = [f for f in root_folders if f.collection_type == "user"]
            logger.info(f"🔍 Found {len(user_root_folders)} user root folders for user {user_id}")
            for folder in user_root_folders:
                logger.info(f"🔍 User root folder: {folder.name} (ID: {folder.folder_id})")
            
            # Create virtual sources based on collection type
            virtual_sources = []
            if collection_type == "user":
                # Add virtual sources (only RSS Feeds, no Web Sources) as children for user collection
                virtual_sources = [
                    DocumentFolder(
                        folder_id="rss_feeds_virtual",
                        name="RSS Feeds",
                        parent_folder_id="my_documents_root",
                        user_id=user_id,
                        collection_type="user",
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                        document_count=0,
                        subfolder_count=0,
                        children=[],
                        is_virtual_source=True
                    )
                ]
            
            # Combine user folders with virtual sources
            all_my_documents_children = user_root_folders + virtual_sources
            
            my_documents_root = DocumentFolder(
                folder_id="my_documents_root",
                name="My Documents",
                parent_folder_id=None,
                user_id=user_id,
                collection_type="user",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                document_count=0,
                subfolder_count=len(all_my_documents_children),
                children=all_my_documents_children
            )
            virtual_roots.append(my_documents_root)
            logger.info(f"✅ Created My Documents virtual root with {len(all_my_documents_children)} children (including virtual sources)")
            
            # Create "Global Documents" root node with virtual sources (for admins)
            global_root_folders = [f for f in root_folders if f.collection_type == "global"]
            logger.info(f"🔍 Found {len(global_root_folders)} global root folders")
            
            logger.info(f"🔍 User {user_id} admin status: {is_admin}")
            
            if is_admin:
                # Add virtual sources for Global Documents (only RSS Feeds, no Web Sources)
                global_virtual_sources = [
                    DocumentFolder(
                        folder_id="global_rss_feeds_virtual",
                        name="RSS Feeds",
                        parent_folder_id="global_documents_root",
                        user_id=None,
                        collection_type="global",
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                        document_count=0,
                        subfolder_count=0,
                        children=[],
                        is_virtual_source=True
                    )
                ]
                
                # Combine global folders with virtual sources
                all_global_children = global_root_folders + global_virtual_sources
                
                global_documents_root = DocumentFolder(
                    folder_id="global_documents_root",
                    name="Global Documents",
                    parent_folder_id=None,
                    user_id=None,
                    collection_type="global",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    document_count=0,
                    subfolder_count=len(all_global_children),
                    children=all_global_children
                )
                virtual_roots.append(global_documents_root)
                logger.info(f"✅ Created Global Documents virtual root with {len(all_global_children)} children (including virtual sources)")
            else:
                logger.info(f"⚠️ User {user_id} is not admin - Global Documents not created")
            
            # Add team folders as virtual roots
            team_root_folders = [f for f in root_folders if f.collection_type == "team"]
            logger.info(f"🔍 Found {len(team_root_folders)} team root folders for user {user_id}")
            
            for team_folder in team_root_folders:
                virtual_roots.append(team_folder)
                logger.info(f"✅ Added team folder: {team_folder.name} (ID: {team_folder.folder_id})")
            
            # If no virtual roots were created, return the original root folders
            if not virtual_roots:
                result = root_folders
            else:
                result = virtual_roots
            
            logger.info(f"📁 Built fresh folder tree with {len(result)} root folders")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to get folder tree: {e}")
            return []
    
    async def get_folder_contents(self, folder_id: str, user_id: str = None) -> Optional[FolderContentsResponse]:
        """Get contents of a specific folder"""
        try:
            logger.info(f"📁 Getting contents for folder {folder_id} (user: {user_id})")
            
            # Handle virtual RSS feed folders
            if folder_id in ["rss_feeds_virtual", "global_rss_feeds_virtual"]:
                return await self._get_virtual_rss_folder_contents(folder_id, user_id)
            
            # **ROOSEVELT FIX:** Handle virtual root folders (My Documents, Global Documents)
            # These need to show documents at the root level (folder_id IS NULL)
            if folder_id in ["my_documents_root", "global_documents_root"]:
                return await self._get_virtual_root_contents(folder_id, user_id)
            
            # Virtual web sources folders have been removed - only RSS Feeds virtual folders remain
            
            # Get folder info
            folder = await self.get_folder(folder_id, user_id, "user")  # Default to user role for now
            if not folder:
                logger.warning(f"⚠️ Folder {folder_id} not found or access denied for user {user_id}")
                return None
            
            # Get documents in folder with proper RLS context
            # For global folders, we need to pass user_id=None to get global documents
            # For user folders, we pass the user_id to get user documents
            query_user_id = None if folder.collection_type == "global" else user_id
            
            # Add debug logging for folder query
            logger.info(f"🔍 DEBUG: Folder {folder_id} collection_type: {folder.collection_type}, query_user_id: {query_user_id}")
            
            documents = await self.document_repository.get_documents_by_folder(folder_id, query_user_id)
            logger.info(f"📄 Found {len(documents)} documents in folder {folder_id} (collection_type: {folder.collection_type})")
            
            # **ROOSEVELT DEBUG:** Log what we got back
            if len(documents) == 0:
                logger.warning(f"⚠️ ZERO DOCUMENTS returned for folder {folder_id}!")
                logger.warning(f"⚠️ Query context: folder_id={folder_id}, query_user_id={query_user_id}, collection_type={folder.collection_type}")
            else:
                for doc in documents[:5]:  # Log first 5 docs
                    logger.info(f"📄 Document in response: {doc.document_id} - {doc.filename}")
            
            # Get subfolders
            subfolders_data = await self.document_repository.get_subfolders(folder_id)
            subfolders = [DocumentFolder(**subfolder_data) for subfolder_data in subfolders_data]
            logger.info(f"📁 Found {len(subfolders)} subfolders in folder {folder_id}")
            
            result = FolderContentsResponse(
                folder=folder,
                documents=documents,
                subfolders=subfolders,
                total_documents=len(documents),
                total_subfolders=len(subfolders)
            )
            
            logger.info(f"✅ Folder contents for {folder_id}: {len(documents)} docs, {len(subfolders)} subfolders")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to get folder contents: {e}")
            return None

    async def _get_virtual_root_contents(self, folder_id: str, user_id: str = None) -> Optional[FolderContentsResponse]:
        """Get contents for virtual root folders (My Documents, Global Documents)
        
        **ROOSEVELT FIX:** Virtual roots need to show:
        1. Documents at root level (folder_id IS NULL)
        2. Top-level real folders
        3. Virtual source folders (RSS Feeds)
        """
        try:
            logger.info(f"📁 Getting virtual root contents for {folder_id} (user: {user_id})")
            
            # Determine collection type
            if folder_id == "my_documents_root":
                collection_type = "user"
                folder_name = "My Documents"
                query_user_id = user_id
            else:  # global_documents_root
                collection_type = "global"
                folder_name = "Global Documents"
                query_user_id = None
            
            # Create virtual folder object
            virtual_folder = DocumentFolder(
                folder_id=folder_id,
                name=folder_name,
                parent_folder_id=None,
                user_id=user_id if collection_type == "user" else None,
                collection_type=collection_type,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                document_count=0,
                subfolder_count=0,
                children=[]
            )
            
            # **BULLY!** Get documents at root level (folder_id IS NULL)
            # Need to query with collection_type filter
            logger.info(f"🔍 Querying root-level documents for {collection_type} collection")
            documents = await self.document_repository.get_root_documents_by_collection(collection_type, query_user_id)
            logger.info(f"📄 Found {len(documents)} root-level {collection_type} documents")
            
            # Get top-level real folders (parent_folder_id IS NULL)
            folders_data = await self.document_repository.get_folders_by_user(query_user_id, collection_type)
            # Filter to only root-level folders
            root_folders_data = [f for f in folders_data if f.get('parent_folder_id') is None]
            subfolders = [DocumentFolder(**folder_data) for folder_data in root_folders_data]
            logger.info(f"📁 Found {len(subfolders)} root-level folders")
            
            result = FolderContentsResponse(
                folder=virtual_folder,
                documents=documents,
                subfolders=subfolders,
                total_documents=len(documents),
                total_subfolders=len(subfolders)
            )
            
            logger.info(f"✅ Virtual root contents for {folder_id}: {len(documents)} docs, {len(subfolders)} folders")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to get virtual root contents: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _get_virtual_rss_folder_contents(self, folder_id: str, user_id: str = None) -> Optional[FolderContentsResponse]:
        """Get contents for virtual RSS feed folders"""
        try:
            logger.info(f"📁 Getting virtual RSS folder contents for {folder_id} (user: {user_id})")
            
            # Create virtual folder object
            folder_name = "RSS Feeds"
            parent_folder_id = "my_documents_root" if folder_id == "rss_feeds_virtual" else "global_documents_root"
            collection_type = "user" if folder_id == "rss_feeds_virtual" else "global"
            
            virtual_folder = DocumentFolder(
                folder_id=folder_id,
                name=folder_name,
                parent_folder_id=parent_folder_id,
                user_id=user_id if collection_type == "user" else None,
                collection_type=collection_type,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                document_count=0,
                subfolder_count=0,
                children=[],
                is_virtual_source=True
            )
            
            # For virtual RSS folders, we return empty contents since RSS feeds are handled separately
            # The frontend will populate this with RSS feeds from the RSS API
            result = FolderContentsResponse(
                folder=virtual_folder,
                documents=[],
                subfolders=[],
                total_documents=0,
                total_subfolders=0
            )
            
            logger.info(f"✅ Virtual RSS folder contents for {folder_id}: 0 docs, 0 subfolders")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to get virtual RSS folder contents: {e}")
            return None


    async def update_folder(self, folder_id: str, update_data: FolderUpdateRequest, user_id: str = None, current_user_role: str = "user") -> Optional[DocumentFolder]:
        """Update folder information with proper access control"""
        try:
            # Check if folder exists and user has access
            folder = await self.get_folder(folder_id, user_id, current_user_role)
            if not folder:
                return None
            
            # Get old physical path before any updates
            old_physical_path = await self.get_folder_physical_path(folder_id)
            old_folder_name = folder.name
            
            # Validate parent folder if changing
            if update_data.parent_folder_id and update_data.parent_folder_id != folder.parent_folder_id:
                parent_folder = await self.get_folder(update_data.parent_folder_id, user_id, current_user_role)
                if not parent_folder:
                    raise ValueError("New parent folder not found or access denied")
                
                # Check for circular references
                if await self._would_create_circular_reference(folder_id, update_data.parent_folder_id):
                    raise ValueError("Cannot move folder: would create circular reference")
            
            # Update folder in database
            update_dict = update_data.dict(exclude_unset=True)
            if update_dict:
                update_dict["updated_at"] = datetime.utcnow()
                await self.document_repository.update_folder(folder_id, update_dict)
                
                # Get updated folder
                updated_folder = await self.get_folder(folder_id, user_id, current_user_role)
                
                # If folder name changed, rename physical directory
                if update_data.name and update_data.name != old_folder_name and old_physical_path:
                    new_physical_path = await self.get_folder_physical_path(folder_id)
                    
                    if old_physical_path.exists():
                        try:
                            # Rename the physical directory
                            old_physical_path.rename(new_physical_path)
                            logger.info(f"✅ Renamed physical directory: {old_physical_path} -> {new_physical_path}")
                        except Exception as rename_error:
                            logger.error(f"❌ Failed to rename physical directory: {rename_error}")
                            # Rollback database change
                            await self.document_repository.update_folder(folder_id, {"name": old_folder_name})
                            raise ValueError(f"Failed to rename physical directory: {rename_error}")
                    else:
                        logger.warning(f"⚠️ Physical directory not found, creating new one: {new_physical_path}")
                        await self._create_physical_directory(new_physical_path)
                    
                    # Send WebSocket notification for folder rename
                    try:
                        from services.websocket_manager import get_websocket_manager
                        ws_manager = get_websocket_manager()
                        await ws_manager.send_to_user(user_id, {
                            "type": "folder_update",
                            "action": "renamed",
                            "folder": {
                                "folder_id": updated_folder.folder_id,
                                "name": updated_folder.name,
                                "parent_folder_id": updated_folder.parent_folder_id,
                                "updated_at": updated_folder.updated_at.isoformat() if updated_folder.updated_at else None
                            },
                            "old_name": old_folder_name
                        })
                        logger.info(f"📡 WebSocket notification sent for folder rename: {folder_id}")
                    except Exception as ws_error:
                        logger.warning(f"⚠️ Failed to send WebSocket notification: {ws_error}")
                
                logger.info(f"📁 Folder updated: {folder_id}")
                return updated_folder
            
            return folder
            
        except Exception as e:
            logger.error(f"❌ Failed to update folder {folder_id}: {e}")
            raise
    
    # Cache management removed - database is the source of truth
    
    async def exempt_folder_from_vectorization(self, folder_id: str, user_id: str = None) -> bool:
        """Exempt folder and all descendants from vectorization, delete existing vectors/entities"""
        try:
            logger.info(f"🚫 Exempting folder {folder_id} and all descendants from vectorization")
            
            # Mark folder as exempt (TRUE)
            success = await self.document_repository.update_folder_exemption_status(folder_id, True)
            if not success:
                logger.error(f"Failed to update exemption status for folder {folder_id}")
                return False
            
            # Get all descendant folders and documents
            descendant_folder_ids, descendant_document_ids = await self.document_repository.get_folder_descendants(folder_id)
            
            # Force all descendant folders to exempt (TRUE) to ensure hierarchy aligns with parent
            for desc_folder_id in descendant_folder_ids:
                await self.document_repository.update_folder_exemption_status(desc_folder_id, True)
                logger.info(f"🚫 Set descendant folder {desc_folder_id} to exempt (TRUE)")
            
            # Force all descendant documents to exempt (TRUE) so no existing override prevents shutdown
            for desc_doc_id in descendant_document_ids:
                await self.document_repository.update_document_exemption_status(desc_doc_id, True)
                logger.info(f"🚫 Set descendant document {desc_doc_id} to exempt (TRUE)")
            
            # Get document service for vector/KG deletion
            from services.service_container import get_service_container
            container = await get_service_container()
            document_service = container.document_service
            
            # Delete vectors and entities for all documents
            deleted_count = 0
            for doc_id in descendant_document_ids:
                try:
                    # Get document info for user_id
                    doc_info = await self.document_repository.get_by_id(doc_id)
                    doc_user_id = doc_info.user_id if doc_info else user_id
                    
                    # Delete vectors
                    await document_service.embedding_manager.delete_document_chunks(doc_id, doc_user_id)
                    
                    # Delete KG entities
                    if document_service.kg_service:
                        await document_service.kg_service.delete_document_entities(doc_id)
                    
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete vectors/entities for document {doc_id}: {e}")
            
            logger.info(f"✅ Folder {folder_id} exempted: {len(descendant_folder_ids)} folders, {deleted_count} documents cleaned")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to exempt folder {folder_id}: {e}")
            return False
    
    async def remove_folder_exemption(self, folder_id: str, user_id: str = None) -> bool:
        """Remove exemption from folder (set to inherit from parent), re-process all documents"""
        try:
            logger.info(f"✅ Removing exemption for folder {folder_id} - setting to inherit from parent")
            
            # Set folder to inherit (NULL)
            success = await self.document_repository.update_folder_exemption_status(folder_id, None)
            if not success:
                logger.error(f"Failed to remove exemption status for folder {folder_id}")
                return False
            
            # Get all descendant folders and documents
            descendant_folder_ids, descendant_document_ids = await self.document_repository.get_folder_descendants(folder_id)
            
            # Set descendant folders to inherit (NULL) - they'll inherit from new parent state
            for desc_folder_id in descendant_folder_ids:
                await self.document_repository.update_folder_exemption_status(desc_folder_id, None)
            
            # Set descendant documents to inherit (NULL) - they'll inherit from folder
            for desc_doc_id in descendant_document_ids:
                await self.document_repository.update_document_exemption_status(desc_doc_id, None)
            
            # Get document service for re-processing
            from services.service_container import get_service_container
            container = await get_service_container()
            document_service = container.document_service
            
            # Re-process all documents that are now not exempt
            processed_count = 0
            for doc_id in descendant_document_ids:
                try:
                    # Check if document is now exempt (after inheritance)
                    is_exempt = await self.document_repository.is_document_exempt(doc_id)
                    if not is_exempt:
                        success = await document_service.remove_document_exemption(doc_id, user_id)
                        if success:
                            processed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to re-process document {doc_id}: {e}")
            
            logger.info(f"✅ Folder {folder_id} exemption removed: {len(descendant_folder_ids)} folders, {processed_count} documents re-processed")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to remove exemption for folder {folder_id}: {e}")
            return False
    
    async def override_folder_exemption(self, folder_id: str, user_id: str = None) -> bool:
        """
        Set folder to explicitly NOT exempt (override parent exemption).
        This allows a subfolder to opt out of parent exemption.
        """
        try:
            logger.info(f"✅ Setting folder {folder_id} to explicitly NOT exempt (override parent)")
            
            # Mark folder as not exempt (FALSE - explicit override)
            success = await self.document_repository.update_folder_exemption_status(folder_id, False)
            if not success:
                logger.error(f"Failed to set override status for folder {folder_id}")
                return False
            
            # Get all descendant folders and documents
            descendant_folder_ids, descendant_document_ids = await self.document_repository.get_folder_descendants(folder_id)
            
            # Set descendant folders to inherit (NULL) - they'll inherit from this folder's override
            for desc_folder_id in descendant_folder_ids:
                await self.document_repository.update_folder_exemption_status(desc_folder_id, None)
            
            # Set descendant documents to inherit (NULL) - they'll inherit from folder
            for desc_doc_id in descendant_document_ids:
                await self.document_repository.update_document_exemption_status(desc_doc_id, None)
            
            # Get document service for re-processing
            from services.service_container import get_service_container
            container = await get_service_container()
            document_service = container.document_service
            
            # Re-process all documents that are now not exempt
            processed_count = 0
            for doc_id in descendant_document_ids:
                try:
                    # Check if document is now not exempt (after override)
                    is_exempt = await self.document_repository.is_document_exempt(doc_id)
                    if not is_exempt:
                        success = await document_service.remove_document_exemption(doc_id, user_id)
                        if success:
                            processed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to re-process document {doc_id}: {e}")
            
            logger.info(f"✅ Folder {folder_id} override set: {len(descendant_folder_ids)} folders, {processed_count} documents re-processed")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to set override for folder {folder_id}: {e}")
            return False
    
    async def delete_folder(self, folder_id: str, user_id: str = None, recursive: bool = False, current_user_role: str = "user") -> bool:
        """Delete a folder with proper access control"""
        try:
            # Check if folder exists and user has access
            folder = await self.get_folder(folder_id, user_id, current_user_role)
            if not folder:
                return False
            
            # Prevent deletion of team root folders (must delete team instead)
            if folder.collection_type == "team" and folder.parent_folder_id is None:
                raise ValueError("Cannot delete team root folder. Delete the team instead to remove the folder.")
            
            # No cache to clear - database is the source of truth
            
            # Check if folder has contents
            contents = await self.get_folder_contents(folder_id, user_id)
            if contents and (contents.total_documents > 0 or contents.total_subfolders > 0):
                if not recursive:
                    raise ValueError("Folder is not empty. Use recursive=True to delete with contents.")
                
                # Recursively delete contents
                await self._delete_folder_contents(folder_id, user_id)
            
            # **ROOSEVELT FIX:** Get physical folder path BEFORE deleting from database
            folder_path = await self.get_folder_physical_path(folder_id)
            
            # Delete the folder from database
            await self.document_repository.delete_folder(folder_id)
            logger.info(f"🗑️ Folder deleted from database: {folder_id}")
            
            # **BULLY!** Delete physical directory from filesystem
            if folder_path and folder_path.exists():
                try:
                    import shutil
                    shutil.rmtree(folder_path)
                    logger.info(f"🗑️ Deleted physical directory: {folder_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to delete physical directory {folder_path}: {e}")
                    # Don't fail the operation if directory delete fails - database is source of truth
            else:
                logger.warning(f"⚠️ Physical directory not found or path unavailable: {folder_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete folder {folder_id}: {e}")
            raise
    
    async def move_folder(self, folder_id: str, new_parent_id: str = None, user_id: str = None, current_user_role: str = "user") -> bool:
        """Move a folder to a new parent with proper access control"""
        try:
            update_data = FolderUpdateRequest(parent_folder_id=new_parent_id)
            updated_folder = await self.update_folder(folder_id, update_data, user_id, current_user_role)
            return updated_folder is not None
            
        except Exception as e:
            logger.error(f"❌ Failed to move folder {folder_id}: {e}")
            return False
    
    async def create_default_folders(self, user_id: str) -> List[DocumentFolder]:
        """Create default folder structure for a new user.
        - Root: Org
        - Subfolder: Archive
        """
        try:
            logger.info(f"📁 Creating essential Org folders for user {user_id}")
            # Create root 'Org' folder for the user
            org_folder = await self.create_folder(
                name="Org",
                parent_folder_id=None,
                user_id=user_id,
                collection_type="user",
                current_user_role="user",
                admin_user_id=user_id
            )
            # Create 'Archive' subfolder under Org
            archive_folder = await self.create_folder(
                name="Archive",
                parent_folder_id=org_folder.folder_id,
                user_id=user_id,
                collection_type="user",
                current_user_role="user",
                admin_user_id=user_id
            )
            return [org_folder, archive_folder]
            
        except Exception as e:
            logger.error(f"❌ Failed to create default folders for user {user_id}: {e}")
            return []
    
    async def create_global_folder_structure(self) -> List[DocumentFolder]:
        """Create global folder structure for shared content"""
        try:
            # No longer create RSS Feeds and Web Sources as database folders
            # These are now handled as virtual sources in get_folder_tree
            logger.info("📁 No global essential folders needed - using virtual sources")
            return []
            
        except Exception as e:
            logger.error(f"❌ Failed to create global folder structure: {e}")
            return []
    
    async def ensure_default_folders_exist(self, user_id: str) -> List[DocumentFolder]:
        """Ensure default folders exist for a user, create if they don't"""
        try:
            # Check if user already has folders
            existing_folders = await self.document_repository.get_folders_by_user(user_id, "user")
            
            if existing_folders:
                # Clean up any existing RSS Feeds and Web Sources database folders
                # These should now be virtual sources only
                folders_to_remove = []
                for folder in existing_folders:
                    if folder.get('name') in ['RSS Feeds', 'Web Sources']:
                        folders_to_remove.append(folder)
                
                if folders_to_remove:
                    logger.info(f"📁 Cleaning up old RSS/Web Sources database folders for user {user_id}")
                    for folder in folders_to_remove:
                        await self.delete_folder(folder['folder_id'], user_id, recursive=True)
                
                # Check if we have the old nested structure (My Documents -> subfolders)
                has_old_structure = any(folder.get('name') == 'My Documents' for folder in existing_folders)
                
                # Check if we have non-essential folders that should be removed
                has_notes_folder = any(folder.get('name') == 'Notes' for folder in existing_folders)
                
                if has_old_structure or has_notes_folder:
                    logger.info(f"📁 Detected old user folder structure for {user_id}, cleaning up...")
                    # Delete all existing user folders to recreate with new structure
                    for folder in existing_folders:
                        await self.delete_folder(folder['folder_id'], user_id, recursive=True)
                    # Create new structure
                    logger.info(f"📁 Creating new essential folders for user {user_id}")
                    return await self.create_default_folders(user_id)
                else:
                    logger.debug(f"📁 User {user_id} already has {len(existing_folders)} folders")
                    return [DocumentFolder(**folder) for folder in existing_folders]
            
            # Create default folders if none exist
            logger.info(f"📁 Creating default folders for user {user_id}")
            return await self.create_default_folders(user_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to ensure default folders for user {user_id}: {e}")
            return []
    
    async def ensure_global_folders_exist(self) -> List[DocumentFolder]:
        """Ensure global folder structure exists, create if it doesn't"""
        try:
            # Check if global folders already exist
            existing_global_folders = await self.document_repository.get_folders_by_user(None, "global")
            
            if existing_global_folders:
                # Clean up any existing RSS Feeds and Web Sources database folders
                # These should now be virtual sources only
                folders_to_remove = []
                for folder in existing_global_folders:
                    if folder.get('name') in ['RSS Feeds', 'Web Sources']:
                        folders_to_remove.append(folder)
                
                if folders_to_remove:
                    logger.info("📁 Cleaning up old RSS/Web Sources global database folders")
                    for folder in folders_to_remove:
                        await self.delete_folder(folder['folder_id'], None, recursive=True)
                
                # Check if we have the old nested structure (Global Documents -> subfolders)
                has_old_structure = any(folder.get('name') == 'Global Documents' for folder in existing_global_folders)
                
                # Check if we have non-essential folders that should be removed
                has_notes_folder = any(folder.get('name') == 'Notes' for folder in existing_global_folders)
                
                if has_old_structure or has_notes_folder:
                    logger.info("📁 Detected old global folder structure, cleaning up...")
                    # Delete all existing global folders to recreate with new structure
                    for folder in existing_global_folders:
                        await self.delete_folder(folder['folder_id'], None, recursive=True)
                    # Create new structure
                    logger.info("📁 Creating new global essential folder structure")
                    return await self.create_global_folder_structure()
                else:
                    logger.debug(f"📁 Global folder structure already exists: {len(existing_global_folders)} folders")
                    return [DocumentFolder(**folder) for folder in existing_global_folders]
            
            # Create global folder structure if none exist
            logger.info("📁 Creating global folder structure")
            return await self.create_global_folder_structure()
            
        except Exception as e:
            logger.error(f"❌ Failed to ensure global folders exist: {e}")
            return []
    
    async def _would_create_circular_reference(self, folder_id: str, new_parent_id: str) -> bool:
        """Check if moving a folder would create a circular reference"""
        try:
            current_parent = new_parent_id
            visited = {folder_id}
            
            while current_parent:
                if current_parent in visited:
                    return True
                visited.add(current_parent)
                
                parent_folder = await self.document_repository.get_folder(current_parent)
                if not parent_folder:
                    break
                current_parent = parent_folder["parent_folder_id"]
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking circular reference: {e}")
            return True

    async def _is_admin(self, user_id: str) -> bool:
        """Check if user is an admin"""
        try:
            from services.database_manager.database_helpers import fetch_one
            
            # Get user from database to check their role
            row = await fetch_one("""
                SELECT role FROM users WHERE user_id = $1
            """, user_id)
            
            if row:
                return row['role'] == 'admin'
            else:
                logger.warning(f"⚠️ User {user_id} not found in database")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to check admin status for user {user_id}: {e}")
            return False
    
    def _is_essential_automated_folder(self, folder_name: str) -> bool:
        """Check if a folder is an essential automated folder that should not accept manual uploads"""
        essential_folders = {
            "Web Sources",
            "RSS Feeds"
        }
        return folder_name in essential_folders
    
    async def _is_essential_automated_folder_by_id(self, folder_id: str) -> bool:
        """Check if a folder is an essential automated folder by its ID"""
        try:
            folder_data = await self.document_repository.get_folder(folder_id)
            if not folder_data:
                return False
            
            return self._is_essential_automated_folder(folder_data.get('name', ''))
        except Exception as e:
            logger.error(f"❌ Failed to check if folder {folder_id} is essential: {e}")
            return False
    
    async def validate_document_folder_assignment(self, folder_id: str, user_id: str = None) -> bool:
        """Validate if a document can be assigned to a specific folder"""
        try:
            # Check if folder exists and user has access
            folder = await self.get_folder(folder_id, user_id)
            if not folder:
                logger.warning(f"⚠️ Folder {folder_id} not found or access denied for user {user_id}")
                return False
            
            # Check if it's an essential automated folder
            if self._is_essential_automated_folder(folder.name):
                logger.warning(f"⚠️ Cannot assign documents to essential automated folder: {folder.name}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to validate folder assignment: {e}")
            return False
    
    async def assign_document_to_essential_folder(self, document_id: str, folder_id: str, automated_process: str = "system") -> bool:
        """Assign a document to an essential folder (for automated processes only)"""
        try:
            # Check if folder exists
            folder = await self.get_folder(folder_id)
            if not folder:
                logger.error(f"❌ Essential folder {folder_id} not found")
                return False
            
            # Verify it's actually an essential folder
            if not self._is_essential_automated_folder(folder.name):
                logger.warning(f"⚠️ Attempted to use essential folder assignment for non-essential folder: {folder.name}")
                return False
            
            # Update document folder
            success = await self.document_repository.update_document_folder(document_id, folder_id)
            if success:
                logger.info(f"🤖 Automated process '{automated_process}' assigned document {document_id} to essential folder: {folder.name}")
                return True
            else:
                logger.error(f"❌ Failed to assign document {document_id} to essential folder {folder.name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to assign document to essential folder: {e}")
            return False
    
    async def get_essential_folder_id(self, folder_name: str, user_id: str = None, collection_type: str = "user") -> Optional[str]:
        """Get the folder ID for an essential folder by name"""
        try:
            if not self._is_essential_automated_folder(folder_name):
                logger.warning(f"⚠️ Requested folder '{folder_name}' is not an essential automated folder")
                return None
            
            # Get folders for the specified user/collection type
            folders_data = await self.document_repository.get_folders_by_user(user_id, collection_type)
            
            # Find the essential folder by name
            for folder_data in folders_data:
                if folder_data.get('name') == folder_name:
                    return folder_data.get('folder_id')
            
            logger.warning(f"⚠️ Essential folder '{folder_name}' not found for user_id={user_id}, collection_type={collection_type}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get essential folder ID: {e}")
            return None
    
    async def create_or_get_folder(self, folder_name: str, parent_folder_id: str = None, user_id: str = None, collection_type: str = "user", current_user_role: str = "user", admin_user_id: str = None) -> str:
        """Create a folder or get existing folder ID"""
        try:
            logger.info(f"📁 Creating or getting folder: '{folder_name}' (parent: {parent_folder_id}, user: {user_id}, collection: {collection_type})")
            
            if parent_folder_id:
                # Create or get subfolder
                folder = await self.get_or_create_subfolder(parent_folder_id, folder_name, user_id, collection_type, current_user_role, admin_user_id)
            else:
                # Create or get root folder
                folder = await self.get_or_create_root_folder(folder_name, user_id, collection_type, current_user_role, admin_user_id)
            
            logger.info(f"✅ Folder '{folder_name}' ready: {folder.folder_id}")
            return folder.folder_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create or get folder '{folder_name}': {e}")
            raise
    
    async def _delete_folder_contents(self, folder_id: str, user_id: str = None):
        """Recursively delete all contents of a folder"""
        try:
            contents = await self.get_folder_contents(folder_id, user_id)
            if not contents:
                return
            
            # Delete subfolders recursively
            for subfolder in contents.subfolders:
                await self.delete_folder(subfolder.folder_id, user_id, recursive=True)
            
            # Move documents to root (or delete them)
            for document in contents.documents:
                await self.document_repository.update_document_folder(document.document_id, None)
            
        except Exception as e:
            logger.error(f"❌ Error deleting folder contents: {e}")
            raise 