"""
FileManager Service Package
Centralized file management and real-time updates for all agents and tools
"""

from .file_manager_service import FileManagerService, get_file_manager
from .agent_helpers import (
    place_chat_response,
    place_code_file,
    place_web_content,
    place_manual_file
)

__all__ = [
    "FileManagerService", 
    "get_file_manager",
    "place_chat_response",
    "place_code_file", 
    "place_web_content",
    "place_manual_file"
]
