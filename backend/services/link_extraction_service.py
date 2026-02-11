"""
Link Extraction Service - parse and store document links for file relation graph.

Extracts org-mode and markdown file links from document content, resolves relative
paths to document_ids, and upserts into document_links with RLS-safe ownership.
Also extracts proprietary frontmatter file references (e.g. outline, style,
components) and stores them in document_links with link_type 'frontmatter_ref'.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from services.database_manager.database_helpers import fetch_one, execute, fetch_all
from repositories.document_repository import DocumentRepository
from services.folder_service import FolderService

logger = logging.getLogger(__name__)

# Frontmatter keys that reference other files (path or document_id). List keys (e.g. components, characters) hold multiple refs.
_FRONTMATTER_FILE_REF_KEYS = frozenset({
    "outline", "style", "rules", "pr_outline", "pr_req_document_id", "pr_style_document_id",
    "company_knowledge_id", "components", "characters",
})

# Org-mode: [[file:path][desc]], [[./path]], [[../path]], [[id:uuid]]
_ORG_LINK_RE = re.compile(
    r'\[\[([^\]]+)\](?:\[([^\]]*)\])?\]',
    re.IGNORECASE
)
# Markdown: [text](url) - capture url (filter http/https in code)
_MD_LINK_RE = re.compile(
    r'\[([^\]]*)\]\(([^)]+)\)'
)


def _parse_org_links(content: str) -> List[Tuple[str, str, str, int]]:
    """Parse org-mode links. Returns list of (target, description, link_type, line_number)."""
    results = []
    for i, line in enumerate(content.splitlines(), start=1):
        for m in _ORG_LINK_RE.finditer(line):
            target = (m.group(1) or '').strip()
            desc = (m.group(2) or '').strip() or target
            if not target:
                continue
            if target.startswith('file:'):
                path = target[5:].strip()
                if path:
                    results.append((path, desc, 'org_file', i))
            elif target.startswith('id:'):
                results.append((target, desc, 'org_id', i))
            elif target.startswith(('http://', 'https://')):
                continue
            elif target.startswith(('#') or target.startswith('*')):
                continue
            elif re.match(r'^\.?\.?/[^\s]*\.(org|md|txt)$', target, re.I) or re.match(r'^[^\s]+\.(org|md|txt)$', target, re.I):
                results.append((target, desc, 'org_file', i))
            else:
                if '.' in target and target.split('.')[-1].lower() in ('org', 'md', 'txt'):
                    results.append((target, desc, 'org_file', i))
    return results


def _parse_markdown_links(content: str) -> List[Tuple[str, str, int]]:
    """Parse markdown links. Returns list of (url, text, line_number). Skip external URLs."""
    results = []
    for i, line in enumerate(content.splitlines(), start=1):
        for m in _MD_LINK_RE.finditer(line):
            url = (m.group(2) or '').strip()
            text = (m.group(1) or '').strip()
            if not url or url.startswith('http://') or url.startswith('https://') or url.startswith('#'):
                continue
            if url.startswith('mailto:'):
                continue
            results.append((url, text, i))
    return results


def _parse_frontmatter_block(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse frontmatter from full document content. Returns (frontmatter_dict, body). Supports list values via YAML when available."""
    if not content:
        return {}, content
    trimmed = content[1:] if content.startswith("\ufeff") else content
    m = re.match(r"^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n?", trimmed)
    if not m:
        return {}, content
    yaml_block = m.group(1)
    body = trimmed[m.end():]
    try:
        import yaml
        data = yaml.safe_load(yaml_block)
        if not isinstance(data, dict):
            data = {}
        return data, body
    except Exception:
        pass
    data: Dict[str, Any] = {}
    for line in re.split(r"\r?\n", yaml_block):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                inner = v[1:-1].strip()
                data[k.strip()] = [s.strip().strip('"\'') for s in inner.split(",")] if inner else []
            else:
                data[k.strip()] = v
    return data, body


def _looks_like_document_id(value: str) -> bool:
    """True if value is likely a document_id (uuid or doc_xxx)."""
    if not value or len(value) > 255:
        return False
    s = value.strip()
    if re.match(r"^doc_[a-zA-Z0-9_-]+$", s):
        return True
    if re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", s, re.I):
        return True
    return False


def _looks_like_file_path(value: str) -> bool:
    """True if value looks like a relative file path to .md/.org/.txt."""
    if not value or not isinstance(value, str):
        return False
    s = value.strip()
    if s.startswith("./") or s.startswith("../") or "/" in s:
        return True
    if s.endswith((".md", ".org", ".txt")) or re.match(r"^[^\s]+\.(md|org|txt)$", s, re.I):
        return True
    return False


def _extract_frontmatter_refs(content: str) -> List[Tuple[str, str]]:
    """
    Extract file references from YAML frontmatter. Returns list of (raw_path_or_id, key).
    Uses known file-reference keys and heuristics for path/document_id values.
    """
    frontmatter, _ = _parse_frontmatter_block(content)
    if not frontmatter:
        return []
    refs: List[Tuple[str, str]] = []
    for key, value in frontmatter.items():
        if value is None:
            continue
        key_lower = key.lower()
        if key_lower not in _FRONTMATTER_FILE_REF_KEYS and not key_lower.endswith("_document_id") and not key_lower.endswith("_id"):
            if not _looks_like_file_path(str(value)) and not _looks_like_document_id(str(value)):
                continue
        if isinstance(value, list):
            for item in value:
                s = str(item).strip() if item is not None else ""
                if s and (_looks_like_file_path(s) or _looks_like_document_id(s)):
                    refs.append((s, key))
        else:
            s = str(value).strip()
            if s and (_looks_like_file_path(s) or _looks_like_document_id(s)):
                refs.append((s, key))
    return refs


def _normalize_path_segments(raw_path: str) -> Tuple[List[str], str]:
    """Normalize relative path to list of folder segments and filename. Returns (segment_list, filename)."""
    raw = raw_path.replace('\\', '/').strip()
    parts = [p for p in raw.split('/') if p and p != '.']
    if not parts:
        return [], ''
    # Resolve ..
    resolved = []
    for p in parts:
        if p == '..':
            if resolved:
                resolved.pop()
            continue
        resolved.append(p)
    if not resolved:
        return [], ''
    filename = resolved[-1]
    folder_segments = resolved[:-1]
    return folder_segments, filename


class LinkExtractionService:
    """Extract document links from content and store in document_links with RLS."""

    def __init__(self):
        self.document_repository = DocumentRepository()
        self.folder_service = FolderService()

    async def initialize(self):
        await self.document_repository.initialize()
        await self.folder_service.initialize()

    async def _resolve_target_folder(
        self,
        source_folder_id: Optional[str],
        path_segments: List[str],
        user_id: Optional[str],
        collection_type: str,
        team_id: Optional[str],
        user_role: str,
    ) -> Optional[str]:
        """Walk folder hierarchy from source_folder_id. Returns resolved folder_id or None (root)."""
        current_id = source_folder_id
        for name in path_segments:
            if name == '..':
                if not current_id:
                    return None
                folder = await self.document_repository.get_folder(
                    current_id, user_id=user_id or '', user_role=user_role
                )
                if not folder:
                    return None
                current_id = folder.get('parent_folder_id')
                continue
            if not current_id:
                return None
            subfolders = await self.document_repository.get_subfolders(
                current_id, user_id=user_id, user_role=user_role
            )
            found = None
            for sf in subfolders:
                if (sf.get('name') or '').strip() == name:
                    found = sf.get('folder_id')
                    break
            if not found:
                return None
            current_id = found
        return current_id

    async def _resolve_link_target(
        self,
        raw_path: str,
        source_doc: Dict[str, Any],
        rls_context: Dict[str, str],
    ) -> Optional[str]:
        """Resolve raw link path to target document_id. Returns None if unresolved."""
        user_id = source_doc.get('user_id')
        collection_type = (source_doc.get('collection_type') or 'user')
        team_id = source_doc.get('team_id')
        if team_id and not isinstance(team_id, str):
            team_id = str(team_id)
        source_folder_id = source_doc.get('folder_id')
        user_role = rls_context.get('user_role', 'user')

        segments, filename = _normalize_path_segments(raw_path)
        if not filename:
            return None

        target_folder_id = await self._resolve_target_folder(
            source_folder_id,
            segments,
            user_id,
            collection_type,
            team_id,
            user_role,
        )
        if target_folder_id is None and segments:
            logger.debug("Link target unresolved: path=%r (folder walk failed for segments=%s)", raw_path, segments)
            return None

        doc = await self.document_repository.find_by_filename_and_context(
            filename,
            user_id,
            collection_type,
            target_folder_id,
            case_insensitive=True,
        )
        if doc:
            return doc.document_id
        doc = await self.document_repository.find_by_filename_in_user_collection(
            filename, user_id, collection_type
        )
        if doc:
            logger.info(
                "Link resolved via fallback: path=%r -> document_id=%s (folder-scoped lookup had no match)",
                raw_path, doc.document_id,
            )
            return doc.document_id
        logger.info(
            "Link target unresolved: path=%r filename=%r source_folder_id=%r (no document in DB)",
            raw_path, filename, source_folder_id,
        )
        return None

    async def extract_and_store_links(
        self,
        document_id: str,
        content: str,
        rls_context: Dict[str, str],
    ) -> int:
        """
        Extract all file links from content, resolve targets, and upsert into document_links.
        Inherits user_id, collection_type, team_id from source document. Returns count of links stored.
        """
        try:
            doc_row = await self.document_repository.get_document_by_id(
                document_id, user_id=rls_context.get('user_id')
            )
            if not doc_row:
                logger.warning("Source document not found for link extraction: %s", document_id)
                return 0

            user_id = doc_row.get('user_id')
            collection_type = (doc_row.get('collection_type') or 'user')
            team_id = doc_row.get('team_id')
            if team_id and not isinstance(team_id, str):
                team_id = str(team_id)

            links_to_insert: List[Dict[str, Any]] = []

            # Org-mode links
            for raw_path, desc, link_type, line_no in _parse_org_links(content):
                if link_type == 'org_id':
                    id_val = raw_path.replace('id:', '').strip()
                    target_id = None
                    try:
                        check = await self.document_repository.get_document_by_id(
                            id_val, user_id=user_id
                        )
                        if check:
                            target_id = check.get('document_id')
                    except Exception:
                        pass
                else:
                    target_id = await self._resolve_link_target(
                        raw_path, doc_row, rls_context
                    )
                links_to_insert.append({
                    'source_document_id': document_id,
                    'target_document_id': target_id,
                    'target_raw_path': raw_path,
                    'link_type': link_type,
                    'description': desc or None,
                    'line_number': line_no,
                    'user_id': user_id,
                    'collection_type': collection_type,
                    'team_id': team_id,
                })

            # Markdown file links (treat as relative path)
            for url, text, line_no in _parse_markdown_links(content):
                target_id = await self._resolve_link_target(url, doc_row, rls_context)
                links_to_insert.append({
                    'source_document_id': document_id,
                    'target_document_id': target_id,
                    'target_raw_path': url,
                    'link_type': 'markdown_file',
                    'description': text or None,
                    'line_number': line_no,
                    'user_id': user_id,
                    'collection_type': collection_type,
                    'team_id': team_id,
                })

            # Frontmatter file references (proprietary keys: outline, style, components, etc.)
            for raw_ref, key in _extract_frontmatter_refs(content):
                if _looks_like_document_id(raw_ref):
                    target_id = None
                    try:
                        check = await self.document_repository.get_document_by_id(
                            raw_ref, user_id=user_id
                        )
                        if check:
                            target_id = check.get('document_id')
                    except Exception:
                        pass
                else:
                    target_id = await self._resolve_link_target(
                        raw_ref, doc_row, rls_context
                    )
                # Unique per (source, key+path) via compound target_raw_path; line_number=0 for frontmatter
                target_raw_path = f"frontmatter:{key}:{raw_ref}"
                links_to_insert.append({
                    'source_document_id': document_id,
                    'target_document_id': target_id,
                    'target_raw_path': target_raw_path,
                    'link_type': 'frontmatter_ref',
                    'description': key,
                    'line_number': 0,
                    'user_id': user_id,
                    'collection_type': collection_type,
                    'team_id': team_id,
                })

            # Delete existing links for this document (upsert pattern)
            await execute(
                """DELETE FROM document_links WHERE source_document_id = $1""",
                document_id,
                rls_context=rls_context,
            )

            if not links_to_insert:
                return 0

            count = 0
            resolved = 0
            for link in links_to_insert:
                try:
                    await execute(
                        """
                        INSERT INTO document_links (
                            source_document_id, target_document_id, target_raw_path,
                            link_type, description, line_number, user_id, collection_type, team_id
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (source_document_id, target_raw_path, line_number) DO UPDATE SET
                            target_document_id = EXCLUDED.target_document_id,
                            updated_at = NOW()
                        """,
                        link['source_document_id'],
                        link['target_document_id'],
                        link['target_raw_path'],
                        link['link_type'],
                        link['description'],
                        link['line_number'],
                        link['user_id'],
                        link['collection_type'],
                        link['team_id'],
                        rls_context=rls_context,
                    )
                    count += 1
                    if link.get('target_document_id'):
                        resolved += 1
                except Exception as e:
                    logger.warning("Failed to insert link %s -> %s: %s", document_id, link['target_raw_path'], e)

            logger.info("Stored %d links for document %s (%d resolved to target document)", count, document_id, resolved)
            return count
        except Exception as e:
            logger.error("Link extraction failed for %s: %s", document_id, e)
            return 0


_link_extraction_service: Optional[LinkExtractionService] = None


async def get_link_extraction_service() -> LinkExtractionService:
    global _link_extraction_service
    if _link_extraction_service is None:
        _link_extraction_service = LinkExtractionService()
        await _link_extraction_service.initialize()
    return _link_extraction_service
