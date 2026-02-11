"""
Org-Mode Settings Models
Pydantic models for org-mode configuration and preferences
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class TodoStateSequence(BaseModel):
    """
    A TODO state sequence with active and done states
    
    Example:
        name: "Work Tasks"
        active_states: ["TODO", "NEXT", "WAITING"]
        done_states: ["DONE", "CANCELED"]
    """
    name: str = Field(..., description="Name of this sequence (e.g., 'Work', 'Personal')")
    active_states: List[str] = Field(default=["TODO"], description="Active/incomplete states")
    done_states: List[str] = Field(default=["DONE"], description="Completed states")
    is_default: bool = Field(default=False, description="Whether this is the default sequence")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Work Tasks",
                "active_states": ["TODO", "NEXT", "WAITING"],
                "done_states": ["DONE", "CANCELED"],
                "is_default": True
            }
        }


class TagDefinition(BaseModel):
    """
    A tag definition with display preferences
    """
    name: str = Field(..., description="Tag name (e.g., 'work', 'urgent')")
    category: Optional[str] = Field(None, description="Tag category (e.g., 'context', 'priority')")
    color: Optional[str] = Field(None, description="Display color (hex code)")
    icon: Optional[str] = Field(None, description="Icon emoji or unicode")
    description: Optional[str] = Field(None, description="Tag description")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "urgent",
                "category": "priority",
                "color": "#ff0000",
                "icon": "🔥",
                "description": "High priority items"
            }
        }


class AgendaPreferences(BaseModel):
    """
    Preferences for agenda view
    """
    default_view: str = Field(default="week", description="Default view: 'day', 'week', 'month'")
    default_days_ahead: int = Field(default=7, description="Default number of days to show")
    deadline_warning_days: int = Field(default=3, description="Days before deadline to show warning")
    week_start_day: int = Field(default=1, description="Week start day (0=Sunday, 1=Monday)")
    show_scheduled: bool = Field(default=True, description="Show scheduled items")
    show_deadlines: bool = Field(default=True, description="Show deadline items")
    group_by_date: bool = Field(default=True, description="Group items by date")
    
    class Config:
        json_schema_extra = {
            "example": {
                "default_view": "week",
                "default_days_ahead": 7,
                "deadline_warning_days": 3,
                "week_start_day": 1,
                "show_scheduled": True,
                "show_deadlines": True,
                "group_by_date": True
            }
        }


class DisplayPreferences(BaseModel):
    """
    Display preferences for org-mode rendering
    """
    todo_state_colors: Dict[str, str] = Field(
        default={
            "TODO": "#ff0000",
            "NEXT": "#ff8800",
            "WAITING": "#ffaa00",
            "DONE": "#00aa00",
            "CANCELED": "#888888"
        },
        description="Color mapping for TODO states"
    )
    default_collapsed: bool = Field(default=False, description="Start with headings collapsed")
    show_properties: bool = Field(default=True, description="Show property drawers by default")
    show_tags_inline: bool = Field(default=True, description="Show tags inline with headings")
    highlight_current_line: bool = Field(default=True, description="Highlight current line in editor")
    indent_subheadings: bool = Field(default=True, description="Indent subheadings visually")
    indent_content_to_heading: bool = Field(default=False, description="Visually indent body content to match heading level (display-only)")

    class Config:
        json_schema_extra = {
            "example": {
                "todo_state_colors": {
                    "TODO": "#ff0000",
                    "DONE": "#00aa00"
                },
                "default_collapsed": False,
                "show_properties": True,
                "show_tags_inline": True,
                "highlight_current_line": True,
                "indent_subheadings": True,
                "indent_content_to_heading": False
            }
        }


class JournalOrganizationMode(str, Enum):
    """Journal file organization strategies"""
    MONOLITHIC = "monolithic"  # Single journal.org
    YEARLY = "yearly"          # YYYY - Journal.org files
    MONTHLY = "monthly"        # YYYY-MM - Journal.org files
    DAILY = "daily"            # YYYY-MM-DD - Journal.org files


class JournalPreferences(BaseModel):
    """Journal configuration"""
    enabled: bool = Field(default=False, description="Enable journal functionality")
    journal_location: Optional[str] = Field(
        default=None,
        description="Folder path relative to user directory (e.g., 'Journal' or 'Notes/Journal')"
    )
    organization_mode: JournalOrganizationMode = Field(
        default=JournalOrganizationMode.MONOLITHIC,
        description="How to organize journal files"
    )
    include_timestamps: bool = Field(default=True, description="Include timestamps in entries")
    default_tags: List[str] = Field(default=["journal"], description="Default tags for journal entries")
    prompt_for_title: bool = Field(default=True, description="Prompt user for entry title")
    
    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "journal_location": "Journal",
                "organization_mode": "yearly",
                "include_timestamps": True,
                "default_tags": ["journal", "personal"],
                "prompt_for_title": True
            }
        }


class ArchivePreferences(BaseModel):
    """
    Preferences for archiving org-mode entries
    
    Supports format specifiers:
    - %s = source filename without extension
    - %s_archive = source filename with _archive suffix
    - Custom paths like 'OrgMode/archive.org' = single central archive
    """
    default_archive_location: Optional[str] = Field(
        default=None,
        description="Default archive file path. Use %s for source filename. Examples: '%s_archive' (same dir), 'OrgMode/archives/%s_archive' (central folder), 'OrgMode/archive.org' (single file)"
    )
    archive_to_heading: Optional[str] = Field(
        default=None,
        description="Optional heading within archive file (e.g., '* Archived Tasks'). Only used if archive_location points to a file."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "default_archive_location": "%s_archive",
                "archive_to_heading": None
            }
        }


class OrgModeSettings(BaseModel):
    """
    Complete org-mode settings for a user
    """
    user_id: str = Field(..., description="User ID these settings belong to")
    inbox_file: Optional[str] = Field(
        default=None,
        description="Path to inbox.org file (relative to user's directory). If None, will auto-discover."
    )
    refile_max_level: int = Field(
        default=2,
        description="Maximum heading level to show as refile target (1 = only *, 2 = * and **, etc.)"
    )
    todo_sequences: List[TodoStateSequence] = Field(
        default=[
            TodoStateSequence(
                name="Default",
                active_states=["TODO", "NEXT"],
                done_states=["DONE", "CANCELED"],
                is_default=True
            )
        ],
        description="TODO state sequences"
    )
    tags: List[TagDefinition] = Field(default=[], description="Predefined tags")
    agenda_preferences: AgendaPreferences = Field(
        default_factory=AgendaPreferences,
        description="Agenda view preferences"
    )
    display_preferences: DisplayPreferences = Field(
        default_factory=DisplayPreferences,
        description="Display preferences"
    )
    journal_preferences: JournalPreferences = Field(
        default_factory=JournalPreferences,
        description="Journal preferences"
    )
    archive_preferences: ArchivePreferences = Field(
        default_factory=ArchivePreferences,
        description="Archive preferences"
    )
    created_at: Optional[datetime] = Field(None, description="When settings were created")
    updated_at: Optional[datetime] = Field(None, description="When settings were last updated")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "todo_sequences": [
                    {
                        "name": "Work",
                        "active_states": ["TODO", "NEXT", "WAITING"],
                        "done_states": ["DONE", "CANCELED"],
                        "is_default": True
                    }
                ],
                "tags": [
                    {
                        "name": "urgent",
                        "category": "priority",
                        "color": "#ff0000",
                        "icon": "🔥"
                    }
                ],
                "agenda_preferences": {
                    "default_view": "week",
                    "deadline_warning_days": 3
                },
                "display_preferences": {
                    "default_collapsed": False,
                    "show_properties": True
                }
            }
        }


class OrgModeSettingsUpdate(BaseModel):
    """
    Request model for updating org-mode settings
    """
    inbox_file: Optional[str] = Field(None, description="Path to inbox.org file (relative to user's directory)")
    refile_max_level: Optional[int] = Field(None, description="Maximum heading level to show as refile target")
    todo_sequences: Optional[List[TodoStateSequence]] = Field(None, description="TODO state sequences")
    tags: Optional[List[TagDefinition]] = Field(None, description="Predefined tags")
    agenda_preferences: Optional[AgendaPreferences] = Field(None, description="Agenda preferences")
    display_preferences: Optional[DisplayPreferences] = Field(None, description="Display preferences")
    journal_preferences: Optional[JournalPreferences] = Field(None, description="Journal preferences")
    archive_preferences: Optional[ArchivePreferences] = Field(None, description="Archive preferences")
    
    class Config:
        json_schema_extra = {
            "example": {
                "todo_sequences": [
                    {
                        "name": "Work",
                        "active_states": ["TODO", "NEXT"],
                        "done_states": ["DONE"],
                        "is_default": True
                    }
                ]
            }
        }


class OrgModeSettingsResponse(BaseModel):
    """
    Response model for org-mode settings API
    """
    success: bool = Field(..., description="Whether the operation was successful")
    settings: Optional[OrgModeSettings] = Field(None, description="The settings data")
    message: Optional[str] = Field(None, description="Success or error message")

