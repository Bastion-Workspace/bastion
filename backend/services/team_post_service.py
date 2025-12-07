"""
Team Post Service - Handles team posts, comments, and reactions
"""

import logging
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncpg

from utils.shared_db_pool import get_shared_db_pool
from services.team_service import TeamService
from services.embedding_service_wrapper import get_embedding_service
from utils.document_processor import get_document_processor
# PostType enum imported for type hints only

logger = logging.getLogger(__name__)


class TeamPostService:
    """
    Service for managing team posts, comments, and reactions
    
    Handles:
    - Post creation and retrieval
    - Comments on posts
    - Reactions to posts
    - File attachments
    """
    
    def __init__(self):
        self.db_pool = None
        self.team_service = None
    
    async def initialize(self, shared_db_pool=None, team_service=None):
        """Initialize with database pool and team service"""
        if shared_db_pool:
            self.db_pool = shared_db_pool
        else:
            self.db_pool = await get_shared_db_pool()
        
        self.team_service = team_service
        logger.info("Team post service initialized")
    
    async def _ensure_initialized(self):
        """Ensure service is initialized"""
        if not self.db_pool:
            await self.initialize()
    
    async def create_post(
        self,
        team_id: str,
        author_id: str,
        content: str,
        post_type: str,  # "text", "image", or "file"
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create a team post
        
        Args:
            team_id: Team ID
            author_id: User ID creating post
            content: Post content
            post_type: Post type
            attachments: List of attachment metadata
        
        Returns:
            Post dict
        """
        await self._ensure_initialized()
        
        # Check permissions
        if self.team_service:
            permissions = await self.team_service.get_team_permissions(team_id, author_id)
            if not permissions.get("can_post"):
                raise PermissionError("Not allowed to post in this team")
        
        post_id = str(uuid.uuid4())
        attachments = attachments or []
        
        # Convert attachments to JSON string for JSONB column
        attachments_json = json.dumps(attachments)
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, true)", author_id)
                
                # Create post
                await conn.execute("""
                    INSERT INTO team_posts (
                        post_id, team_id, author_id, content, post_type, attachments
                    )
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                """, post_id, team_id, author_id, content, post_type, attachments_json)
                
                # Update team updated_at
                await conn.execute("""
                    UPDATE teams
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE team_id = $1
                """, team_id)
                
                logger.info(f"Created post {post_id} in team {team_id} by user {author_id}")
                
                # Get created post with author info
                post_dict = await self._get_post_with_details(conn, post_id, author_id)
                
                # Vectorize post text content with author metadata for search
                # This allows Research Agent to know who said what when searching team posts
                # Skip vectorization for very short posts (< 50 chars) as they provide little semantic value
                content_stripped = content.strip() if content else ""
                if content_stripped and len(content_stripped) >= 50:
                    try:
                        await self._vectorize_post_content(
                            post_id=post_id,
                            content=content,
                            team_id=team_id,
                            author_id=author_id,
                            author_name=post_dict.get("author_name", "Unknown")
                        )
                    except Exception as e:
                        # Don't fail post creation if vectorization fails
                        logger.warning(f"Failed to vectorize post {post_id} content: {e}")
                elif content_stripped:
                    logger.debug(f"Skipping vectorization for short post {post_id} ({len(content_stripped)} chars)")
                
                return post_dict
        
        except PermissionError:
            raise
        except Exception as e:
            logger.error(f"Failed to create post: {e}")
            raise
    
    async def get_team_posts(
        self,
        team_id: str,
        user_id: str,
        limit: int = 20,
        before_post_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get team posts (paginated)
        
        Args:
            team_id: Team ID
            user_id: User ID requesting
            limit: Number of posts to return
            before_post_id: Get posts before this post ID (for pagination)
        
        Returns:
            Dict with posts list and has_more flag
        """
        await self._ensure_initialized()
        
        # Check access
        if self.team_service:
            role = await self.team_service.check_team_access(team_id, user_id)
            if not role:
                raise PermissionError("Not a team member")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, true)", user_id)
                
                # Build query with pagination
                query = """
                    SELECT 
                        tp.*,
                        u.username as author_username,
                        u.display_name as author_display_name,
                        u.avatar_url as author_avatar
                    FROM team_posts tp
                    INNER JOIN users u ON u.user_id = tp.author_id
                    WHERE tp.team_id = $1 AND tp.deleted_at IS NULL
                """
                params = [team_id]
                param_num = 2
                
                if before_post_id:
                    query += f" AND tp.created_at < (SELECT created_at FROM team_posts WHERE post_id = ${param_num})"
                    params.append(before_post_id)
                    param_num += 1
                
                query += f" ORDER BY tp.created_at DESC LIMIT ${param_num}"
                params.append(limit + 1)  # Get one extra to check if more exists
                
                rows = await conn.fetch(query, *params)
                
                has_more = len(rows) > limit
                posts = rows[:limit]
                
                # Get reactions and comments for each post
                post_dicts = []
                for row in posts:
                    post_dict = await self._row_to_post_dict(conn, row, user_id)
                    post_dicts.append(post_dict)
                
                return {
                    "posts": post_dicts,
                    "has_more": has_more,
                    "total": len(post_dicts)
                }
        
        except PermissionError:
            raise
        except Exception as e:
            logger.error(f"Failed to get team posts: {e}")
            raise
    
    async def delete_post(self, post_id: str, user_id: str) -> bool:
        """
        Delete post (author or admin only)
        
        Args:
            post_id: Post ID
            user_id: User ID deleting
        
        Returns:
            True if deleted
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, true)", user_id)
                
                # Get post
                post_row = await conn.fetchrow("""
                    SELECT tp.*, tm.role
                    FROM team_posts tp
                    LEFT JOIN team_members tm ON tm.team_id = tp.team_id AND tm.user_id = $2
                    WHERE tp.post_id = $1
                """, post_id, user_id)
                
                if not post_row:
                    raise ValueError("Post not found")
                
                # Check permission (author or admin)
                if post_row["author_id"] != user_id and post_row["role"] != "admin":
                    raise PermissionError("Only post author or team admin can delete posts")
                
                # Get team_id and attachments before soft delete for cleanup
                team_id = str(post_row["team_id"])
                attachments = post_row.get("attachments")
                
                # Parse attachments if needed
                if isinstance(attachments, str):
                    try:
                        attachments = json.loads(attachments) if attachments else []
                    except (json.JSONDecodeError, TypeError):
                        attachments = []
                elif attachments is None:
                    attachments = []
                elif not isinstance(attachments, list):
                    attachments = []
                
                # Soft delete
                result = await conn.execute("""
                    UPDATE team_posts
                    SET deleted_at = CURRENT_TIMESTAMP
                    WHERE post_id = $1
                """, post_id)
                
                if result == "UPDATE 1":
                    logger.info(f"Deleted post {post_id} by user {user_id}")
                    
                    # Remove vectorized content from team collection
                    try:
                        from services.vector_store_service import VectorStoreService
                        vector_store = VectorStoreService()
                        team_collection_name = vector_store._get_team_collection_name(team_id)
                        
                        await vector_store.delete_points_by_filter(
                            document_id=post_id,
                            collection_name=team_collection_name
                        )
                        logger.info(f"Removed vectorized content for post {post_id} from team collection")
                    except Exception as e:
                        # Don't fail deletion if vector cleanup fails
                        logger.warning(f"Failed to remove vectorized content for post {post_id}: {e}")
                    
                    # Delete attachment files from disk
                    if attachments:
                        await self._cleanup_attachment_files(team_id, attachments)
                    
                    return True
                else:
                    return False
        
        except (ValueError, PermissionError):
            raise
        except Exception as e:
            logger.error(f"Failed to delete post: {e}")
            raise
    
    async def add_reaction(self, post_id: str, user_id: str, reaction_type: str) -> bool:
        """
        Add reaction to post
        
        Args:
            post_id: Post ID
            user_id: User ID reacting
            reaction_type: Emoji reaction
        
        Returns:
            True if added
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, true)", user_id)
                
                # Check post exists and user has access
                post_row = await conn.fetchrow("""
                    SELECT tp.team_id
                    FROM team_posts tp
                    INNER JOIN team_members tm ON tm.team_id = tp.team_id AND tm.user_id = $2
                    WHERE tp.post_id = $1 AND tp.deleted_at IS NULL
                """, post_id, user_id)
                
                if not post_row:
                    raise ValueError("Post not found or no access")
                
                # Check permissions
                if self.team_service:
                    permissions = await self.team_service.get_team_permissions(
                        str(post_row["team_id"]), user_id
                    )
                    if not permissions.get("can_react"):
                        raise PermissionError("Not allowed to react in this team")
                
                # Add reaction (upsert)
                await conn.execute("""
                    INSERT INTO post_reactions (post_id, user_id, reaction_type)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (post_id, user_id, reaction_type) DO NOTHING
                """, post_id, user_id, reaction_type)
                
                logger.info(f"Added reaction {reaction_type} to post {post_id} by user {user_id}")
                return True
        
        except (ValueError, PermissionError):
            raise
        except Exception as e:
            logger.error(f"Failed to add reaction: {e}")
            raise
    
    async def remove_reaction(self, post_id: str, user_id: str, reaction_type: str) -> bool:
        """
        Remove reaction from post
        
        Args:
            post_id: Post ID
            user_id: User ID removing reaction
            reaction_type: Emoji reaction to remove
        
        Returns:
            True if removed
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, true)", user_id)
                
                # Remove reaction
                result = await conn.execute("""
                    DELETE FROM post_reactions
                    WHERE post_id = $1 AND user_id = $2 AND reaction_type = $3
                """, post_id, user_id, reaction_type)
                
                if result == "DELETE 1":
                    logger.info(f"Removed reaction {reaction_type} from post {post_id} by user {user_id}")
                    return True
                else:
                    return False
        
        except Exception as e:
            logger.error(f"Failed to remove reaction: {e}")
            raise
    
    async def get_post_reactions(self, post_id: str) -> List[Dict[str, Any]]:
        """
        Get reactions for a post grouped by type
        
        Args:
            post_id: Post ID
        
        Returns:
            List of reaction dicts with counts
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get reactions grouped by type
                rows = await conn.fetch("""
                    SELECT 
                        reaction_type,
                        COUNT(*) as count,
                        ARRAY_AGG(user_id) as user_ids
                    FROM post_reactions
                    WHERE post_id = $1
                    GROUP BY reaction_type
                    ORDER BY count DESC
                """, post_id)
                
                return [
                    {
                        "reaction_type": row["reaction_type"],
                        "count": row["count"],
                        "users": row["user_ids"]
                    }
                    for row in rows
                ]
        
        except Exception as e:
            logger.error(f"Failed to get post reactions: {e}")
            raise
    
    async def create_comment(self, post_id: str, author_id: str, content: str) -> Dict[str, Any]:
        """
        Create comment on post
        
        Args:
            post_id: Post ID
            author_id: User ID creating comment
            content: Comment content
        
        Returns:
            Comment dict
        """
        await self._ensure_initialized()
        
        comment_id = str(uuid.uuid4())
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, true)", author_id)
                
                # Check post exists and user has access
                post_row = await conn.fetchrow("""
                    SELECT tp.team_id
                    FROM team_posts tp
                    INNER JOIN team_members tm ON tm.team_id = tp.team_id AND tm.user_id = $2
                    WHERE tp.post_id = $1 AND tp.deleted_at IS NULL
                """, post_id, author_id)
                
                if not post_row:
                    raise ValueError("Post not found or no access")
                
                # Check permissions
                if self.team_service:
                    permissions = await self.team_service.get_team_permissions(
                        str(post_row["team_id"]), author_id
                    )
                    if not permissions.get("can_comment"):
                        raise PermissionError("Not allowed to comment in this team")
                
                # Create comment
                await conn.execute("""
                    INSERT INTO post_comments (comment_id, post_id, author_id, content)
                    VALUES ($1, $2, $3, $4)
                """, comment_id, post_id, author_id, content)
                
                logger.info(f"Created comment {comment_id} on post {post_id} by user {author_id}")
                
                # Get created comment with author info
                row = await conn.fetchrow("""
                    SELECT 
                        pc.*,
                        u.username as author_username,
                        u.display_name as author_display_name,
                        u.avatar_url as author_avatar
                    FROM post_comments pc
                    INNER JOIN users u ON u.user_id = pc.author_id
                    WHERE pc.comment_id = $1
                """, comment_id)
                
                return {
                    "comment_id": str(row["comment_id"]),
                    "post_id": str(row["post_id"]),
                    "author_id": row["author_id"],
                    "author_name": row["author_display_name"] or row["author_username"],
                    "author_avatar": row["author_avatar"],
                    "content": row["content"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat()
                }
        
        except (ValueError, PermissionError):
            raise
        except Exception as e:
            logger.error(f"Failed to create comment: {e}")
            raise
    
    async def get_post_comments(self, post_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get comments for a post
        
        Args:
            post_id: Post ID
            limit: Maximum comments to return
        
        Returns:
            List of comment dicts
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        pc.*,
                        u.username as author_username,
                        u.display_name as author_display_name,
                        u.avatar_url as author_avatar
                    FROM post_comments pc
                    INNER JOIN users u ON u.user_id = pc.author_id
                    WHERE pc.post_id = $1 AND pc.deleted_at IS NULL
                    ORDER BY pc.created_at ASC
                    LIMIT $2
                """, post_id, limit)
                
                return [
                    {
                        "comment_id": str(row["comment_id"]),
                        "post_id": str(row["post_id"]),
                        "author_id": row["author_id"],
                        "author_name": row["author_display_name"] or row["author_username"],
                        "author_avatar": row["author_avatar"],
                        "content": row["content"],
                        "created_at": row["created_at"].isoformat(),
                        "updated_at": row["updated_at"].isoformat()
                    }
                    for row in rows
                ]
        
        except Exception as e:
            logger.error(f"Failed to get post comments: {e}")
            raise
    
    async def delete_comment(self, comment_id: str, user_id: str) -> bool:
        """
        Delete comment (author or admin only)
        
        Args:
            comment_id: Comment ID
            user_id: User ID deleting
        
        Returns:
            True if deleted
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, true)", user_id)
                
                # Get comment with post team
                comment_row = await conn.fetchrow("""
                    SELECT pc.*, tp.team_id, tm.role
                    FROM post_comments pc
                    INNER JOIN team_posts tp ON tp.post_id = pc.post_id
                    LEFT JOIN team_members tm ON tm.team_id = tp.team_id AND tm.user_id = $2
                    WHERE pc.comment_id = $1
                """, comment_id, user_id)
                
                if not comment_row:
                    raise ValueError("Comment not found")
                
                # Check permission (author or admin)
                if comment_row["author_id"] != user_id and comment_row["role"] != "admin":
                    raise PermissionError("Only comment author or team admin can delete comments")
                
                # Soft delete
                result = await conn.execute("""
                    UPDATE post_comments
                    SET deleted_at = CURRENT_TIMESTAMP
                    WHERE comment_id = $1
                """, comment_id)
                
                if result == "UPDATE 1":
                    logger.info(f"Deleted comment {comment_id} by user {user_id}")
                    return True
                else:
                    return False
        
        except (ValueError, PermissionError):
            raise
        except Exception as e:
            logger.error(f"Failed to delete comment: {e}")
            raise
    
    async def _get_post_with_details(self, conn, post_id: str, user_id: str) -> Dict[str, Any]:
        """Get post with all details"""
        row = await conn.fetchrow("""
            SELECT 
                tp.*,
                u.username as author_username,
                u.display_name as author_display_name,
                u.avatar_url as author_avatar
            FROM team_posts tp
            INNER JOIN users u ON u.user_id = tp.author_id
            WHERE tp.post_id = $1
        """, post_id)
        
        if not row:
            raise ValueError("Post not found")
        
        return await self._row_to_post_dict(conn, row, user_id)
    
    async def _row_to_post_dict(self, conn, row, user_id: str) -> Dict[str, Any]:
        """Convert post row to dict with reactions and comments"""
        post_id = str(row["post_id"])
        
        # Get reactions
        reactions = await self.get_post_reactions(post_id)
        
        # Get comment count
        comment_count = await conn.fetchval("""
            SELECT COUNT(*) FROM post_comments
            WHERE post_id = $1 AND deleted_at IS NULL
        """, post_id)
        
        # Parse attachments JSONB (asyncpg returns as dict/list, but ensure it's a list)
        attachments = row["attachments"]
        if isinstance(attachments, str):
            try:
                attachments = json.loads(attachments) if attachments else []
            except (json.JSONDecodeError, TypeError):
                attachments = []
        elif attachments is None:
            attachments = []
        elif not isinstance(attachments, list):
            attachments = []
        
        return {
            "post_id": post_id,
            "team_id": str(row["team_id"]),
            "author_id": row["author_id"],
            "author_name": row["author_display_name"] or row["author_username"],
            "author_avatar": row["author_avatar"],
            "content": row["content"],
            "post_type": row["post_type"],
            "attachments": attachments,
            "reactions": reactions,
            "comment_count": comment_count,
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat()
        }
    
    async def _vectorize_post_content(
        self,
        post_id: str,
        content: str,
        team_id: str,
        author_id: str,
        author_name: str
    ):
        """
        Vectorize team post text content with author metadata
        
        This allows the Research Agent to know who said what when searching team posts.
        The content is stored in the team's vector collection with author metadata.
        """
        try:
            # Get document processor and embedding service
            document_processor = await get_document_processor()
            embedding_service = await get_embedding_service()
            
            # Process content into chunks
            metadata = {
                "post_id": post_id,
                "team_id": team_id,
                "author_id": author_id,
                "source": "team_post"
            }
            
            chunks = await document_processor.process_text_content(
                content=content,
                document_id=post_id,
                metadata=metadata
            )
            
            if not chunks:
                logger.warning(f"No chunks generated for post {post_id}")
                return
            
            # Vectorize with author metadata
            # This ensures Research Agent can see who authored the content
            await embedding_service.embed_and_store_chunks(
                chunks=chunks,
                team_id=team_id,
                document_title=f"Team Post by {author_name}",
                document_author=author_name,
                document_filename=f"post_{post_id}.txt",
                document_category="team_post",
                document_tags=["team_post", f"team_{team_id}", f"author_{author_id}"]
            )
            
            logger.info(
                f"Vectorized post {post_id} content: {len(chunks)} chunks, "
                f"author={author_name}, team={team_id}"
            )
            
        except Exception as e:
            logger.error(f"Failed to vectorize post {post_id} content: {e}", exc_info=True)
            raise
    
    async def _cleanup_attachment_files(
        self,
        team_id: str,
        attachments: List[Dict[str, Any]]
    ):
        """
        Delete attachment files from disk
        
        Args:
            team_id: Team ID
            attachments: List of attachment metadata dicts
        """
        from pathlib import Path
        from config import settings
        
        try:
            uploads_base = Path(settings.UPLOAD_DIR)
            team_posts_dir = uploads_base / "Teams" / team_id / "posts"
            
            deleted_count = 0
            failed_count = 0
            
            for att in attachments:
                # Get filename from file_path
                file_path_str = att.get("file_path", "")
                
                if file_path_str:
                    # Extract filename from path like /api/teams/{team_id}/posts/attachments/{filename}
                    # The actual file is at: uploads/Teams/{team_id}/posts/{filename}
                    filename = file_path_str.split("/")[-1]
                    
                    if filename:
                        file_path = team_posts_dir / filename
                        
                        try:
                            if file_path.exists():
                                file_path.unlink()
                                deleted_count += 1
                                logger.info(f"Deleted attachment file: {file_path}")
                            else:
                                logger.warning(f"Attachment file not found: {file_path}")
                                failed_count += 1
                        except Exception as e:
                            logger.error(f"Failed to delete attachment file {file_path}: {e}")
                            failed_count += 1
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} attachment files for team {team_id}")
            if failed_count > 0:
                logger.warning(f"Failed to delete {failed_count} attachment files for team {team_id}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup attachment files for team {team_id}: {e}")
            # Don't raise - attachment cleanup shouldn't fail post deletion

