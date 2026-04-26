"""Pydantic models for Zettelkasten / PKM settings."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DailyNoteFormat(str, Enum):
    """Filename pattern for markdown daily notes."""

    ISO = "YYYY-MM-DD"
    ISO_DAY = "YYYY-MM-DD-dddd"
    COMPACT = "YYYYMMDD"


class ZettelkastenSettings(BaseModel):
    user_id: str = Field(..., description="Owner user id")
    enabled: bool = Field(default=False, description="Master ZK features toggle")
    daily_note_location: Optional[str] = Field(
        default=None,
        description="Folder path relative to user library (e.g. Notes/Daily)",
    )
    daily_note_format: DailyNoteFormat = Field(
        default=DailyNoteFormat.ISO,
        description="Daily note filename pattern",
    )
    daily_note_template: str = Field(
        default="",
        description="Markdown body template for new daily notes (optional)",
    )
    note_id_prefix: bool = Field(
        default=False,
        description="If true, prefix new capture filenames with timestamp",
    )
    backlinks_enabled: bool = Field(default=True, description="Show backlinks panel when supported")
    wikilink_autocomplete: bool = Field(default=True, description="Autocomplete after [[ in markdown editor")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ZettelkastenSettingsUpdate(BaseModel):
    enabled: Optional[bool] = None
    daily_note_location: Optional[str] = None
    daily_note_format: Optional[DailyNoteFormat] = None
    daily_note_template: Optional[str] = None
    note_id_prefix: Optional[bool] = None
    backlinks_enabled: Optional[bool] = None
    wikilink_autocomplete: Optional[bool] = None


class ZettelkastenSettingsResponse(BaseModel):
    success: bool
    settings: Optional[ZettelkastenSettings] = None
    message: Optional[str] = None
