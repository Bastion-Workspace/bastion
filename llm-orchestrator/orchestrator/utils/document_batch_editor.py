"""
Document Batch Editor - Universal batched editing system for LangGraph agents

This module provides a DocumentEditBatch class that allows agents to collect
multiple document edits and apply them atomically in a single write operation.

This solves the problem of sequential read-modify-write cycles that lead to:
- Race conditions and stale content reads
- Inefficient full-content replacements (multiple writes for multiple edits)
- Risk of partial updates if operations fail mid-stream
- Duplicate frontmatter blocks from sequential frontmatter/section updates
"""

import re
import logging
import yaml
from typing import Dict, Any, List, Optional, Literal, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from orchestrator.tools.document_tools import get_document_content_tool
from orchestrator.tools.document_editing_tools import update_document_content_tool
from orchestrator.utils.frontmatter_utils import update_frontmatter_field

logger = logging.getLogger(__name__)


@dataclass
class EditOperation:
    """Base class for document edit operations"""
    operation_type: Literal["frontmatter", "section_replace", "section_append", "section_delete"]
    target: str  # Section name or "frontmatter"
    content: Any  # String content or dict for frontmatter
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentEditBatch:
    """
    Collects multiple document edits and applies them atomically.
    
    Usage:
        batch = DocumentEditBatch(document_id, user_id, "agent_name")
        await batch.initialize()  # Reads current content once
        
        # Queue edits
        batch.add_frontmatter_update({"title": "New Title"}, {"files": ["./file.md"]})
        batch.add_section_replace("Requirements", "New requirements...")
        batch.add_section_append("Tasks", "- New task")
        batch.add_section_delete("Obsolete Section")
        
        # Apply all edits atomically
        result = await batch.apply()
    """
    
    def __init__(self, document_id: str, user_id: str, agent_name: str = "unknown"):
        """
        Initialize a document edit batch.
        
        Args:
            document_id: Document ID to edit
            user_id: User ID for permissions
            agent_name: Name of agent making edits (for logging)
        """
        self.document_id = document_id
        self.user_id = user_id
        self.agent_name = agent_name
        self.operations: List[EditOperation] = []
        self.initial_content: Optional[str] = None
        self.current_content: Optional[str] = None
    
    async def initialize(self) -> bool:
        """
        Read document content once.
        
        Returns:
            True if content was loaded successfully, False otherwise
        """
        try:
            self.initial_content = await get_document_content_tool(self.document_id, self.user_id)
            if self.initial_content.startswith("Error"):
                logger.error(f"Failed to read document {self.document_id}: {self.initial_content}")
                return False
            
            self.current_content = self.initial_content
            logger.debug(f"Initialized batch editor for document {self.document_id} ({len(self.initial_content)} chars)")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize batch editor: {e}")
            return False
    
    def add_frontmatter_update(
        self,
        field_updates: Dict[str, Any],
        list_updates: Optional[Dict[str, List[str]]] = None
    ):
        """
        Queue a frontmatter update operation.
        
        Args:
            field_updates: Dict of scalar field updates (e.g., {"title": "New Title"})
            list_updates: Dict of list field updates (e.g., {"files": ["./file1.md"]})
        """
        operation = EditOperation(
            operation_type="frontmatter",
            target="frontmatter",
            content={"fields": field_updates, "lists": list_updates or {}}
        )
        self.operations.append(operation)
        logger.debug(f"Queued frontmatter update: {list(field_updates.keys())}")
    
    def add_section_replace(self, section: str, content: str):
        """
        Queue a section replacement operation.
        
        Args:
            section: Section name to replace
            content: New content for the section
        """
        operation = EditOperation(
            operation_type="section_replace",
            target=section,
            content=content
        )
        self.operations.append(operation)
        logger.debug(f"Queued section replace: {section}")
    
    def add_section_append(self, section: str, content: str):
        """
        Queue a section append operation.
        
        Args:
            section: Section name to append to
            content: Content to append
        """
        operation = EditOperation(
            operation_type="section_append",
            target=section,
            content=content
        )
        self.operations.append(operation)
        logger.debug(f"Queued section append: {section}")
    
    def add_section_delete(self, section: str):
        """
        Queue a section deletion operation.
        
        Args:
            section: Section name to delete
        """
        operation = EditOperation(
            operation_type="section_delete",
            target=section,
            content=None
        )
        self.operations.append(operation)
        logger.debug(f"Queued section delete: {section}")
    
    def _find_section(self, content: str, section_name: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        """
        Find a section in markdown content.
        
        Args:
            content: Document content to search
            section_name: Name of section to find
            
        Returns:
            Tuple of (start_index, end_index, used_header) or (None, None, None) if not found
        """
        section_headers = [
            f"## {section_name}",
            f"### {section_name}",
            f"# {section_name}"
        ]
        
        for header in section_headers:
            # Match section header and everything until next header or end of document
            pattern = rf"{re.escape(header)}.*?(?=\n##|\n###|\n#|$)"
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                return match.start(), match.end(), header
        
        return None, None, None
    
    def _apply_operation(self, op: EditOperation, content: str) -> str:
        """
        Apply a single operation to content.
        
        Args:
            op: Edit operation to apply
            content: Current document content
            
        Returns:
            Updated content after applying operation
        """
        if op.operation_type == "frontmatter":
            # Use frontmatter utility
            field_updates = op.content.get("fields", {})
            list_updates = op.content.get("lists", {})
            
            # update_frontmatter_field is async, but we're in a sync context
            # We'll need to handle this differently - call it in apply() instead
            # For now, return content unchanged (will be handled in apply())
            return content
        
        elif op.operation_type == "section_replace":
            section_start, section_end, used_header = self._find_section(content, op.target)
            
            if section_start is None:
                # Section not found - append at end
                logger.warning(f"Section '{op.target}' not found, appending instead")
                new_section = f"\n\n## {op.target}\n\n{op.content}\n"
                return content + new_section
            
            # Replace section
            updated_section = f"{used_header}\n\n{op.content}\n"
            return content[:section_start] + updated_section + content[section_end:]
        
        elif op.operation_type == "section_append":
            section_start, section_end, used_header = self._find_section(content, op.target)
            
            if section_start is None:
                # Section not found - create new section
                logger.warning(f"Section '{op.target}' not found, creating new section")
                new_section = f"\n\n## {op.target}\n\n{op.content}\n"
                return content + new_section
            
            # Append to existing section
            existing_section = content[section_start:section_end]
            updated_section = existing_section.rstrip() + f"\n\n{op.content}\n"
            return content[:section_start] + updated_section + content[section_end:]
        
        elif op.operation_type == "section_delete":
            section_start, section_end, _ = self._find_section(content, op.target)
            
            if section_start is None:
                logger.warning(f"Section '{op.target}' not found for deletion")
                return content
            
            # Delete section
            return content[:section_start] + content[section_end:]
        
        else:
            logger.warning(f"Unknown operation type: {op.operation_type}")
            return content
    
    async def apply(self) -> Dict[str, Any]:
        """
        Apply all queued operations in order, write once.
        
        Returns:
            Dict with success status, metrics, and error details
        """
        if not self.current_content:
            return {
                "success": False,
                "error": "Batch not initialized - call initialize() first",
                "document_id": self.document_id
            }
        
        if not self.operations:
            logger.warning("No operations queued for batch editor")
            return {
                "success": True,
                "operations_attempted": 0,
                "operations_succeeded": 0,
                "operations_failed": 0,
                "message": "No operations to apply",
                "document_id": self.document_id
            }
        
        working_content = self.current_content
        successful_ops = []
        failed_ops = []
        
        # Process operations in order
        for i, op in enumerate(self.operations):
            try:
                if op.operation_type == "frontmatter":
                    # Frontmatter updates need async utility
                    field_updates = op.content.get("fields", {})
                    list_updates = op.content.get("lists", {})
                    
                    updated_content, success = await update_frontmatter_field(
                        content=working_content,
                        field_updates=field_updates,
                        list_updates=list_updates
                    )
                    
                    if success:
                        working_content = updated_content
                        successful_ops.append(op)
                        logger.debug(f"Applied frontmatter update: {list(field_updates.keys())}")
                    else:
                        raise Exception("Frontmatter update returned success=False")
                else:
                    # Section operations are synchronous
                    working_content = self._apply_operation(op, working_content)
                    successful_ops.append(op)
                    logger.debug(f"Applied {op.operation_type} on {op.target}")
            
            except Exception as e:
                logger.error(f"Operation {i+1} failed: {op.operation_type} on {op.target}: {e}")
                failed_ops.append({
                    "operation": op.operation_type,
                    "target": op.target,
                    "error": str(e)
                })
                # Continue with other operations (best effort)
        
        # Write the accumulated changes if anything changed
        if working_content != self.initial_content:
            try:
                await update_document_content_tool(
                    document_id=self.document_id,
                    content=working_content,
                    user_id=self.user_id,
                    append=False
                )
                logger.info(
                    f"✅ Batch editor applied {len(successful_ops)}/{len(self.operations)} operations "
                    f"to document {self.document_id} ({len(self.initial_content)} → {len(working_content)} chars)"
                )
            except Exception as e:
                logger.error(f"Failed to write batch edits: {e}")
                return {
                    "success": False,
                    "error": f"Failed to write changes: {str(e)}",
                    "operations_attempted": len(self.operations),
                    "operations_succeeded": len(successful_ops),
                    "operations_failed": len(failed_ops),
                    "failed_operations": failed_ops,
                    "document_id": self.document_id
                }
        else:
            logger.info(f"No changes detected after applying {len(self.operations)} operations")
        
        return {
            "success": len(failed_ops) == 0,
            "operations_attempted": len(self.operations),
            "operations_succeeded": len(successful_ops),
            "operations_failed": len(failed_ops),
            "failed_operations": failed_ops,
            "initial_length": len(self.initial_content),
            "final_length": len(working_content),
            "document_id": self.document_id,
            "agent_name": self.agent_name
        }

