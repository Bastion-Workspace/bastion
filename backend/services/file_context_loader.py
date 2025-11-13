"""
Reusable file context loader to resolve relative frontmatter references.

Capabilities:
- Resolve paths like "../outlines/outline.md" relative to a base file directory
- Enforce project root boundary (settings.UPLOAD_DIR)
- Load and parse outline frontmatter, then cascade its references: rules, style, characters

Notes:
- Frontmatter parsing is minimal (key: value). For characters, we support either a single
  "characters" value (comma-separated) or any keys starting with "character".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import settings
from utils.frontmatter_utils import parse_frontmatter


logger = logging.getLogger(__name__)


@dataclass
class LoadedDoc:
    path: str
    content: str
    frontmatter: Dict[str, str]


@dataclass
class LoadedContext:
    outline: Optional[LoadedDoc]
    rules: Optional[LoadedDoc]
    style: Optional[LoadedDoc]
    characters: List[LoadedDoc]


class FileContextLoader:
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = Path(project_root or settings.UPLOAD_DIR).resolve()

    def _resolve_path(self, base_dir: Path, ref: str) -> Optional[Path]:
        try:
            ref = str(ref).strip()
            if not ref:
                return None
            
            logger.info(f"🔍 PATH RESOLUTION: ref='{ref}', base_dir='{base_dir}'")
            
            # Ensure .md extension fallback
            if not Path(ref).suffix:
                ref_with_md = ref + ".md"
            else:
                ref_with_md = ref

            candidate = (base_dir / ref_with_md).resolve() if not Path(ref_with_md).is_absolute() else Path(ref_with_md).resolve()
            logger.info(f"🔍 PATH RESOLUTION: Candidate path: {candidate}")
            # Enforce boundary: candidate must be under project_root
            try:
                candidate.relative_to(self.project_root)
            except ValueError:
                logger.warning(f"⚠️ FileContextLoader: Path outside project root blocked: {candidate}")
                return None
            if not candidate.exists() or not candidate.is_file():
                # Attempt to resolve by upload naming scheme: <doc_id>_<basename>
                try:
                    from glob import glob
                    basename = Path(ref_with_md).name
                    pattern = str(self.project_root / f"*_{basename}")
                    matches = glob(pattern)
                    if matches:
                        candidate = Path(matches[0]).resolve()
                    else:
                        logger.warning(f"⚠️ FileContextLoader: File not found: {candidate}")
                        return None
                except Exception:
                    logger.warning(f"⚠️ FileContextLoader: File not found: {candidate}")
                    return None
            return candidate
        except Exception as e:
            logger.error(f"❌ Path resolution error: {e}")
            return None

    def _read_doc(self, path: Path) -> Optional[LoadedDoc]:
        try:
            text = path.read_text(encoding="utf-8")
            fm, body = parse_frontmatter(text)
            # Preserve original full text (including body) for context; we don't strip frontmatter here
            return LoadedDoc(path=str(path), content=text, frontmatter=fm)
        except Exception as e:
            logger.error(f"❌ Failed to read {path}: {e}")
            return None

    def _extract_character_paths(self, outline_fm: Dict[str, str]) -> List[str]:
        char_paths: List[str] = []
        # 1) characters: value1, value2
        chars_val = outline_fm.get("characters")
        if chars_val:
            # split on comma and whitespace
            parts = [p.strip() for p in str(chars_val).split(',') if p.strip()]
            char_paths.extend(parts)
        # 2) Any key starting with character
        for k, v in outline_fm.items():
            if k.lower().startswith("character") and v:
                char_paths.append(str(v).strip())
        # Deduplicate while preserving order
        seen = set()
        deduped: List[str] = []
        for p in char_paths:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        return deduped

    def load_referenced_context(
        self,
        manuscript_filename: str,
        manuscript_frontmatter: Dict[str, str],
    ) -> LoadedContext:
        # Determine base directory from manuscript filename
        try:
            manuscript_path = Path(manuscript_filename)
            base_dir = manuscript_path.parent if manuscript_path.parent != Path('.') else Path(self.project_root)
            logger.info(f"📂 CONTEXT LOADER: Initial base_dir from filename '{manuscript_filename}': {base_dir}")
        except Exception as e:
            base_dir = Path(self.project_root)
            logger.warning(f"⚠️ CONTEXT LOADER: Failed to parse filename, using project root: {e}")
        
        # If canonical path is present in frontmatter (from active_editor), prefer its parent
        canonical = manuscript_frontmatter.get("__canonical_path__")
        logger.info(f"📂 CONTEXT LOADER: canonical_path from frontmatter: {canonical}")
        
        if canonical:
            try:
                cp = Path(str(canonical)).resolve()
                logger.info(f"📂 CONTEXT LOADER: Resolved canonical path: {cp}")
                logger.info(f"📂 CONTEXT LOADER: Project root: {self.project_root}")
                logger.info(f"📂 CONTEXT LOADER: Path starts with root? {str(cp).startswith(str(self.project_root))}")
                
                if str(cp).startswith(str(self.project_root)):
                    base_dir = cp.parent
                    logger.info(f"✅ CONTEXT LOADER: Using canonical base_dir: {base_dir}")
                else:
                    logger.warning(f"⚠️ CONTEXT LOADER: Canonical path outside project root, using fallback base_dir")
            except Exception as e:
                logger.error(f"❌ CONTEXT LOADER: Failed to resolve canonical path: {e}")
                pass
        else:
            logger.warning(f"⚠️ CONTEXT LOADER: No canonical_path provided, relative paths may not resolve correctly!")

        # Load outline if present
        outline_doc: Optional[LoadedDoc] = None
        rules_doc: Optional[LoadedDoc] = None
        style_doc: Optional[LoadedDoc] = None
        characters_docs: List[LoadedDoc] = []

        outline_ref = manuscript_frontmatter.get("outline")
        if outline_ref:
            logger.info(f"📄 CONTEXT LOADER: Resolving outline ref '{outline_ref}' from base_dir: {base_dir}")
            outline_path = self._resolve_path(base_dir, outline_ref)
            if outline_path:
                logger.info(f"✅ CONTEXT LOADER: Resolved outline to: {outline_path}")
                outline_doc = self._read_doc(outline_path)
            else:
                logger.warning(f"⚠️ CONTEXT LOADER: Failed to resolve outline: {outline_ref}")

        # Cascade from outline frontmatter
        if outline_doc and outline_doc.frontmatter:
            o_fm = outline_doc.frontmatter
            outline_dir = Path(outline_doc.path).parent
            # rules
            if o_fm.get("rules"):
                p = self._resolve_path(outline_dir, o_fm["rules"]) 
                if p:
                    rules_doc = self._read_doc(p)
            # style
            if o_fm.get("style"):
                p = self._resolve_path(outline_dir, o_fm["style"]) 
                if p:
                    style_doc = self._read_doc(p)
            # characters list/keys
            for ref in self._extract_character_paths(o_fm):
                p = self._resolve_path(outline_dir, ref)
                if p:
                    doc = self._read_doc(p)
                    if doc:
                        characters_docs.append(doc)

        # If no style was provided via outline cascade, allow manuscript frontmatter to reference style directly
        if style_doc is None and manuscript_frontmatter.get("style"):
            style_ref = manuscript_frontmatter.get("style")
            logger.info(f"📄 CONTEXT LOADER: Resolving style ref '{style_ref}' from base_dir: {base_dir}")
            p = self._resolve_path(base_dir, style_ref)
            if p:
                logger.info(f"✅ CONTEXT LOADER: Resolved style to: {p}")
                style_doc = self._read_doc(p)
            else:
                logger.warning(f"⚠️ CONTEXT LOADER: Failed to resolve style: {style_ref}")

        return LoadedContext(
            outline=outline_doc,
            rules=rules_doc,
            style=style_doc,
            characters=characters_docs,
        )




