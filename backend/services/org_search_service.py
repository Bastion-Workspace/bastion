"""
Org Search Service - Roosevelt's Search Cavalry
Parses and searches org-mode files with full-text and metadata support
"""

import logging
import re
from collections import Counter
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from datetime import datetime
import orgparse

from config import settings

logger = logging.getLogger(__name__)


class OrgSearchService:
    """Service for searching and parsing org-mode files"""
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)

    @staticmethod
    def _heading_matches_tags(
        heading_tags: Optional[List[str]],
        filter_tags: Optional[List[str]],
        tags_match: str,
    ) -> bool:
        """Return True if heading tags satisfy filter (any or all), case-insensitive."""
        if not filter_tags:
            return True
        ft = [t.strip().lower() for t in filter_tags if t and str(t).strip()]
        if not ft:
            return True
        ht = {str(t).lower() for t in (heading_tags or []) if t}
        if tags_match == "all":
            return all(f in ht for f in ft)
        return any(f in ht for f in ft)

    async def list_org_tags(
        self,
        user_id: str,
        include_archives: bool = False,
    ) -> Dict[str, Any]:
        """Aggregate org headline tags across the user's .org files (count = headings per tag)."""
        try:
            org_files = await self._find_user_org_files(user_id, include_archives=include_archives)
            counter: Counter = Counter()
            for file_path in org_files:
                try:
                    content = file_path.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning("list_org_tags: skip %s: %s", file_path, e)
                    continue
                headings = self._parse_org_headings(content)
                for heading in headings:
                    for tag in heading.get("tags") or []:
                        if tag:
                            counter[str(tag)] += 1
            tags_sorted = sorted(
                [{"tag": t, "count": c} for t, c in counter.items()],
                key=lambda x: (-x["count"], x["tag"].lower()),
            )
            return {
                "success": True,
                "tags": tags_sorted,
                "files_scanned": len(org_files),
                "total_distinct_tags": len(counter),
            }
        except Exception as e:
            logger.error("list_org_tags failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "tags": [],
                "files_scanned": 0,
                "total_distinct_tags": 0,
            }
    
    async def search_org_files(
        self,
        user_id: str,
        query: str,
        tags: Optional[List[str]] = None,
        todo_states: Optional[List[str]] = None,
        include_content: bool = True,
        limit: int = 100,
        include_archives: bool = False,
        tags_match: str = "any",
    ) -> Dict[str, Any]:
        """
        Search across all org files for a user
        
        Args:
            user_id: User ID to search files for
            query: Search query string
            tags: Filter by tags (e.g., ["work", "urgent"])
            todo_states: Filter by TODO states (e.g., ["TODO", "NEXT"])
            include_content: Include content in results or just headings
            limit: Maximum number of results
            include_archives: Include _archive.org files (default: False)
            tags_match: When filtering by tags, "any" (OR) or "all" (AND)
        
        Returns:
            Dict with search results and metadata
        """
        try:
            # Get all org files for user (exclude archives by default)
            org_files = await self._find_user_org_files(user_id, include_archives=include_archives)
            
            # Build filename -> document_id map from database
            document_id_map = await self._get_document_id_map(user_id)
            
            if not org_files:
                return {
                    "success": True,
                    "query": query,
                    "results": [],
                    "count": 0,
                    "files_searched": 0
                }
            
            logger.debug("Searching %d org files for '%s'", len(org_files), query)
            
            # Search each file and add document_ids
            all_results = []
            for file_path in org_files:
                file_results = await self._search_org_file(
                    file_path=file_path,
                    query=query,
                    tags=tags,
                    todo_states=todo_states,
                    include_content=include_content,
                    tags_match=tags_match,
                )
                
                # Add document_id to each result
                filename_stem = file_path.stem  # "inbox.org" -> "inbox"
                document_id = document_id_map.get(filename_stem)
                for result in file_results:
                    result['document_id'] = document_id
                
                all_results.extend(file_results)
            
            # Sort by relevance (simple: heading matches first, then content matches)
            all_results.sort(key=lambda r: (
                -int(r.get('heading_match', False)),
                -r.get('match_count', 0)
            ))
            
            # Limit results
            limited_results = all_results[:limit]
            
            return {
                "success": True,
                "query": query,
                "results": limited_results,
                "count": len(limited_results),
                "total_matches": len(all_results),
                "files_searched": len(org_files),
                "filters": {
                    "tags": tags,
                    "todo_states": todo_states,
                    "tags_match": tags_match,
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Org search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": [],
                "count": 0
            }
    
    async def _get_document_id_map(self, user_id: str) -> Dict[str, str]:
        """
        Get mapping of filename -> document_id from database
        
        Fetches document IDs once instead of repeated lookups.
        """
        try:
            from services.database_manager.database_helpers import fetch_all
            
            query = """
                SELECT title, document_id
                FROM document_metadata
                WHERE user_id = $1 AND doc_type = 'org'
            """
            
            rows = await fetch_all(query, user_id)
            
            # Build map: title -> document_id
            doc_map = {row['title']: row['document_id'] for row in rows}
            
            logger.debug("Built document_id map for %d org files", len(doc_map))
            return doc_map
            
        except Exception as e:
            logger.error(f"❌ Failed to build document_id map: {e}")
            return {}
    
    async def _find_user_org_files(self, user_id: str, include_archives: bool = False) -> List[Path]:
        """
        Find all .org files for a user by recursively searching their entire directory tree.
        
        Searches the user directory tree for org files.
        
        Args:
            user_id: User ID
            include_archives: If False, exclude files ending with _archive.org (default: False)
        """
        try:
            # Look in user-specific directories
            from services.database_manager.database_helpers import fetch_one
            
            # Get username for folder structure
            row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
            username = row['username'] if row else user_id
            
            org_files = []
            
            # Primary strategy: Recursively search ALL of user's directory
            user_base_dir = self.upload_dir / "Users" / username
            
            if user_base_dir.exists():
                logger.debug("Recursively searching all of %s", user_base_dir)
                
                # Find all .org files under user's directory
                all_org_files = list(user_base_dir.rglob("*.org"))
                
                # Filter out empty files and apply any exclusions
                for org_file in all_org_files:
                    # Skip files inside .versions directories (historical snapshots)
                    if "/.versions/" in str(org_file) or "\\.versions\\" in str(org_file):
                        logger.debug(f"⏭️  Skipping .versions file: {org_file}")
                        continue
                    
                    # Skip empty files
                    if org_file.stat().st_size == 0:
                        logger.debug(f"⏭️  Skipping empty file: {org_file}")
                        continue
                    
                    # Skip common system/backup files
                    if org_file.name.startswith('.') or org_file.name.endswith('~'):
                        logger.debug(f"⏭️  Skipping system/backup file: {org_file}")
                        continue
                    
                    # Skip archive files unless explicitly requested
                    if not include_archives and org_file.name.endswith('_archive.org'):
                        logger.debug(f"⏭️  Skipping archive file: {org_file.name}")
                        continue
                    
                    org_files.append(org_file)
                
                logger.debug("Found %d org files in user directory", len(org_files))
            else:
                logger.warning(f"⚠️  User directory does not exist: {user_base_dir}")
            
            # Also check legacy locations (root-level org files for backward compatibility)
            legacy_locations = [
                self.upload_dir / "inbox.org",
                self.upload_dir / "Tasks.org",
                self.upload_dir / "calendar.org"
            ]
            
            for legacy_org in legacy_locations:
                if legacy_org.exists() and legacy_org not in org_files:
                    # Only include if it has content (non-empty)
                    if legacy_org.stat().st_size > 0:
                        org_files.append(legacy_org)
                        logger.debug("Including legacy org file: %s", legacy_org.name)
            
            # Sort for consistent ordering
            org_files.sort(key=lambda f: str(f))
            
            logger.debug("Total %d org files found for user %s", len(org_files), username)
            
            return org_files
            
        except Exception as e:
            logger.error(f"❌ Failed to find org files: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return []
    
    async def _search_org_file(
        self,
        file_path: Path,
        query: str,
        tags: Optional[List[str]],
        todo_states: Optional[List[str]],
        include_content: bool,
        tags_match: str = "any",
    ) -> List[Dict[str, Any]]:
        """Search a single org file and return matching headings"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Parse org file structure
            headings = self._parse_org_headings(content)
            logger.debug("Parsed %d headings from %s", len(headings), file_path.name)
            
            # Filter and search
            results = []
            query_lower = query.lower()
            
            for heading in headings:
                # Apply filters
                if tags and not self._heading_matches_tags(heading.get("tags"), tags, tags_match):
                    logger.debug("Filtered out '%s' - tag mismatch", heading.get("heading", ""))
                    continue
                
                if todo_states and heading.get('todo_state') not in todo_states:
                    logger.debug(f"🔍 Filtered out '{heading['heading']}' - TODO state '{heading.get('todo_state')}' not in {todo_states}")
                    continue
                
                # Search in heading and content
                # If query is empty, match everything (for filtering without search)
                if query_lower:
                    heading_match = query_lower in heading['heading'].lower()
                    content_match = query_lower in heading.get('content', '').lower() if include_content else False
                    
                    if not (heading_match or content_match):
                        continue
                else:
                    # Empty query - include all filtered items
                    heading_match = False
                    content_match = False
                
                # At this point, item passes all filters and query checks
                logger.debug(f"✅ Including '{heading['heading']}' with TODO state '{heading.get('todo_state')}'")
                
                match_count = heading['heading'].lower().count(query_lower) if query_lower else 0
                if include_content and query_lower:
                    match_count += heading.get('content', '').lower().count(query_lower)
                
                # Extract preview snippet
                preview = self._extract_preview(
                    heading['heading'], 
                    heading.get('content', ''),
                    query,
                    heading_match
                )
                
                result_item = {
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "heading": heading['heading'],
                    "level": heading['level'],
                    "line_number": heading['line_number'],
                    "todo_state": heading.get('todo_state'),
                    "tags": heading.get('tags', []),
                    "properties": heading.get('properties', {}),
                    "preview": preview,
                    "heading_match": heading_match,
                    "content_match": content_match,
                    "match_count": match_count,
                    "scheduled": heading.get('scheduled'),
                    "deadline": heading.get('deadline'),
                    "active_timestamps": heading.get('active_timestamps', []),
                    "parent_path": heading.get('parent_path', []),
                    "parent_levels": heading.get('parent_levels', []),
                    "priority": heading.get('priority'),
                    "effort": heading.get('effort'),
                    "category": heading.get('category'),
                }
                if include_content:
                    result_item["body"] = heading.get('content', '') or ''
                results.append(result_item)
            
            logger.debug("Returning %d results from %s", len(results), file_path.name)
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to search file {file_path}: {e}")
            return []
    
    def _parse_org_headings(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse org file into structured headings using orgparse library
        
        Uses orgparse library for reliable parsing.
        """
        headings = []
        
        try:
            # Parse with orgparse
            root = orgparse.loads(content)
            file_category = None
            kw = getattr(root, 'keywords', None)
            if isinstance(kw, dict):
                file_category = kw.get('CATEGORY')
            elif hasattr(root, 'get_keyword') and callable(getattr(root, 'get_keyword')):
                file_category = root.get_keyword('CATEGORY')

            # Recursively traverse the tree to get ALL nodes with parent hierarchy tracking
            def traverse(node, depth=0, parent_path=None, parent_levels=None, file_category=file_category):
                """Recursively traverse orgparse tree with parent hierarchy tracking"""
                if parent_path is None:
                    parent_path = []
                if parent_levels is None:
                    parent_levels = []
                
                logger.debug("Visiting node: level=%s, heading='%s'", node.level, (node.heading[:30] if node.heading else "ROOT"))
                
                # Process this node if it's not the root (root has no heading)
                if node.heading:
                    # Extract heading text (without TODO state or tags)
                    heading_text = node.heading
                    
                    # Get TODO state
                    todo_state = node.todo if node.todo else None
                    
                    # Get tags (orgparse returns tags as a set)
                    tags = list(node.tags) if node.tags else []
                    
                    # Get properties
                    properties = dict(node.properties) if node.properties else {}
                    
                    # Priority: [#A], [#B], [#C] on the headline
                    priority = getattr(node, 'priority', None)
                    if not priority:
                        m = re.search(r'\[#([ABC])\]', heading_text or '')
                        priority = m.group(1) if m else None
                    
                    # Effort: :EFFORT: property in the PROPERTIES drawer
                    effort = properties.get('EFFORT') or properties.get('Effort') or None
                    
                    # Category: per-heading :CATEGORY: property, else file-level #+CATEGORY:
                    category = properties.get('CATEGORY') or properties.get('Category') or file_category or None
                    
                    # Get scheduled and deadline from SCHEDULED: and DEADLINE: keywords
                    scheduled = None
                    deadline = None
                    
                    if node.scheduled:
                        scheduled = str(node.scheduled)
                    
                    if node.deadline:
                        deadline = str(node.deadline)
                    
                    closed = None
                    if getattr(node, 'closed', None):
                        closed = str(node.closed)
                    
                    # Get body content
                    content_text = node.body if node.body else ''
                    
                    # Extract active timestamps from body (for calendar appointments)
                    # Active timestamps: <2025-10-08 Wed 19:00-19:05> or <2025-10-23 Thu>
                    active_timestamps = self._extract_active_timestamps(content_text)
                    
                    # Get line number (orgparse provides this as 'linenumber' or '_lines')
                    line_number = 0
                    if hasattr(node, 'linenumber'):
                        line_number = node.linenumber
                    elif hasattr(node, '_lines') and node._lines:
                        line_number = node._lines[0] if node._lines else 0
                    
                    heading_dict = {
                        'level': node.level,
                        'heading': heading_text,
                        'todo_state': todo_state,
                        'tags': tags,
                        'line_number': line_number,
                        'content': content_text,
                        'properties': properties,
                        'priority': priority,
                        'effort': effort,
                        'category': category,
                        'scheduled': scheduled,
                        'deadline': deadline,
                        'closed': closed,  # When item was marked DONE (for "completed in last week" etc.)
                        'active_timestamps': active_timestamps,  # Calendar appointments
                        'parent_path': list(parent_path),  # Copy to avoid mutation
                        'parent_levels': list(parent_levels)  # Copy to avoid mutation
                    }
                    
                    headings.append(heading_dict)
                    logger.debug(f"✅ Parsed heading: level={node.level}, todo={todo_state}, heading='{heading_text[:50]}', parents={parent_path}")
                
                # Recursively process children with updated parent path
                if hasattr(node, 'children'):
                    # Build new parent path for children
                    if node.heading:
                        new_parent_path = parent_path + [node.heading]
                        new_parent_levels = parent_levels + [node.level]
                    else:
                        new_parent_path = parent_path
                        new_parent_levels = parent_levels
                    
                    for child in node.children:
                        traverse(child, depth + 1, new_parent_path, new_parent_levels, file_category)
            
            traverse(root)
            
            return headings
            
        except Exception as e:
            logger.error(f"❌ Failed to parse org file with orgparse: {e}")
            logger.error(f"   Content preview: {content[:200]}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return []
    
    def _extract_active_timestamps(self, content: str) -> List[str]:
        """
        Extract active timestamps from org content.
        
        Active timestamps look like:
        - <2025-10-08 Wed 19:00-19:05>  (with time range)
        - <2025-10-23 Thu>  (date only)
        - <2025-10-15 Wed>  (with newline after)
        """
        if not content:
            return []
        
        # Regex pattern for active timestamps
        # Matches: <YYYY-MM-DD Day> or <YYYY-MM-DD Day HH:MM-HH:MM>
        timestamp_pattern = r'<(\d{4}-\d{2}-\d{2}\s+\w{3}(?:\s+\d{1,2}:\d{2}(?:-\d{1,2}:\d{2})?)?)>'
        
        matches = re.findall(timestamp_pattern, content)
        
        if matches:
            logger.debug(f"📅 Found {len(matches)} active timestamps: {matches}")
        
        return matches
    
    def _extract_preview(
        self,
        heading: str,
        content: str,
        query: str,
        heading_match: bool,
        max_length: int = 200
    ) -> str:
        """Extract a preview snippet around the search query"""
        query_lower = query.lower()
        
        if heading_match:
            # If query is in heading, just return the heading
            return heading
        
        # Find query in content
        content_lower = content.lower()
        idx = content_lower.find(query_lower)
        
        if idx == -1:
            # No match, return first bit of content
            return content[:max_length].strip() + ('...' if len(content) > max_length else '')
        
        # Extract context around match
        start = max(0, idx - 50)
        end = min(len(content), idx + len(query) + 150)
        
        snippet = content[start:end].strip()
        if start > 0:
            snippet = '...' + snippet
        if end < len(content):
            snippet = snippet + '...'
        
        return snippet
    
    async def find_backlinks(self, user_id: str, target_filename: str) -> List[Dict[str, Any]]:
        """
        Find all org files that link to the target file.
        
        Finds files that link to the target filename.
        
        Args:
            user_id: User ID to search files for
            target_filename: The filename to find backlinks to (e.g., "projects.org")
        
        Returns:
            List of files that contain links to the target file
        """
        try:
            # Get all org files for user
            org_files = await self._find_user_org_files(user_id)
            
            if not org_files:
                return []
            
            logger.debug("Searching for backlinks to '%s' across %d files", target_filename, len(org_files))
            
            # Extract just the filename without path for matching
            target_name = Path(target_filename).name
            
            # Patterns to match org-mode links to this file
            # [[file:target.org][Description]]
            # [[file:path/to/target.org][Description]]
            # [[file:target.org::*Heading][Description]]
            # [[./target.org][Description]]
            # [[../target.org][Description]]
            # [[../../reference/target.org][Description]]
            link_patterns = [
                rf'\[\[file:[^\]]*{re.escape(target_name)}[^\]]*\](?:\[[^\]]*\])?\]',  # file: links
                rf'\[\[\.+[/\\][^\]]*{re.escape(target_name)}[^\]]*\](?:\[[^\]]*\])?\]',  # relative links (./ ../ ../../)
            ]
            
            backlinks = []
            
            for file_path in org_files:
                # Skip the target file itself
                if file_path.name == target_name:
                    continue
                
                try:
                    content = file_path.read_text(encoding='utf-8')
                    
                    # Check if any link pattern matches
                    found_links = []
                    for pattern in link_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            found_links.extend(matches)
                    
                    if found_links:
                        # Count how many links to the target
                        link_count = len(found_links)
                        
                        # Try to extract context around the first link
                        first_link = found_links[0]
                        link_index = content.find(first_link)
                        
                        # Get surrounding context (50 chars before and after)
                        context_start = max(0, link_index - 50)
                        context_end = min(len(content), link_index + len(first_link) + 50)
                        context = content[context_start:context_end].strip()
                        
                        if context_start > 0:
                            context = '...' + context
                        if context_end < len(content):
                            context = context + '...'
                        
                        backlinks.append({
                            'filename': file_path.name,
                            'file_path': str(file_path.relative_to(self.upload_dir)),
                            'link_count': link_count,
                            'context': context,
                            'links': found_links
                        })
                        
                        logger.debug("Found %d link(s) in %s", link_count, file_path.name)
                
                except Exception as e:
                    logger.warning(f"⚠️ Failed to check file {file_path}: {e}")
                    continue
            
            logger.debug("Found %d file(s) with backlinks to '%s'", len(backlinks), target_filename)
            return backlinks
            
        except Exception as e:
            logger.error(f"❌ Backlink search failed: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return []


# Singleton instance
_org_search_service = None


async def get_org_search_service() -> OrgSearchService:
    """Get singleton org search service instance"""
    global _org_search_service
    if _org_search_service is None:
        _org_search_service = OrgSearchService()
    return _org_search_service

