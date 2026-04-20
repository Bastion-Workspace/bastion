"""
FileManager service package.

Centralized file management and real-time updates for agents and tools.
"""

from .file_manager_service import FileManagerService, get_file_manager

__all__ = [
    "FileManagerService",
    "get_file_manager",
]
