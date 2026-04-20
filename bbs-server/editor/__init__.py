"""BBS telnet document editor (read raw text, save via PUT /api/documents/.../content)."""

from .run import is_editable_filename, run_document_editor

__all__ = ["is_editable_filename", "run_document_editor"]
