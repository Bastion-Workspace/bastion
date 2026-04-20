"""Pydantic models for OPDS / ebook reader / KoSync settings APIs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OpdsCatalogEntry(BaseModel):
    id: str = Field(..., description="Stable client-generated id for this catalog")
    title: str = Field(..., description="Display name")
    root_url: str = Field(..., description="OPDS catalog root URL")
    http_basic_b64: Optional[str] = Field(
        default=None,
        description="Optional Base64(user:pass) for HTTP Basic on catalog host (write-only on PUT when set)",
    )
    verify_ssl: bool = Field(default=True)


class OpdsCatalogEntryResponse(BaseModel):
    """Catalog row returned by GET /api/ebooks/settings (no secret material)."""

    id: str
    title: str
    root_url: str
    verify_ssl: bool = Field(default=True)
    http_basic_configured: bool = Field(
        default=False,
        description="True when HTTP Basic credentials are stored; credentials are not returned",
    )


class EbookKosyncStored(BaseModel):
    base_url: str = Field(default="", description="HTTPS base URL of kosync server")
    username: str = Field(default="")
    userkey: str = Field(default="", description="MD5 hex of password, same as KOReader stores")
    verify_ssl: bool = Field(default=True)


class EbooksSettingsResponse(BaseModel):
    catalogs: List[OpdsCatalogEntryResponse] = Field(default_factory=list)
    reader_prefs: Dict[str, Any] = Field(default_factory=dict)
    recently_opened: List[Dict[str, Any]] = Field(default_factory=list)
    kosync: Dict[str, Any] = Field(
        default_factory=dict,
        description="base_url, username, verify_ssl, configured (bool); never includes userkey",
    )


class EbooksSettingsUpdate(BaseModel):
    catalogs: Optional[List[OpdsCatalogEntry]] = None
    reader_prefs: Optional[Dict[str, Any]] = None
    recently_opened: Optional[List[Dict[str, Any]]] = None


class OpdsFetchRequest(BaseModel):
    catalog_id: str
    url: str = Field(..., description="Absolute acquisition or navigation URL under catalog origin")
    want: str = Field("atom", description="'atom' parses as JSON; 'binary' returns base64 (epub)")


class KosyncRegisterRequest(BaseModel):
    username: str
    password: str = Field(..., description="Plain password; server stores MD5 hex only")
    base_url: Optional[str] = Field(None, description="If set, saved to KoSync settings before register")
    verify_ssl: bool = Field(default=True)


class KosyncAuthTestRequest(BaseModel):
    base_url: str
    username: str
    password: str = Field(..., description="Plain password for one-off test")
    verify_ssl: bool = Field(default=True)


class KosyncProgressPut(BaseModel):
    document: str = Field(..., description="32-char partial MD5 digest")
    progress: str = Field(..., description="EPUB CFI or XPath-like locator string")
    percentage: float = Field(..., ge=0.0, le=1.0)
    device: str = Field(default="BastionWeb")
    device_id: str = Field(default="")
