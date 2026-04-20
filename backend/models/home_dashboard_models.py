"""
Pydantic models for user home dashboards (Phase 1 layout + Phase 2 multi-page + Phase 3 grid + widgets).

Phase 1 key `home_dashboard_v1` is migrated into Phase 2 envelope `home_dashboards_v2` on first read.
"""

from __future__ import annotations

import uuid
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Discriminator, Field, model_validator

HOME_DASHBOARD_SETTING_KEY = "home_dashboard_v1"
HOME_DASHBOARDS_SETTING_KEY_V2 = "home_dashboards_v2"
HOME_DASHBOARD_SCHEMA_VERSION = 1
USER_DASHBOARDS_ENVELOPE_VERSION = 2
MAX_USER_DASHBOARDS = 20
MAX_DASHBOARD_NAME_LEN = 100
MAX_HOME_DASHBOARD_WIDGETS = 20
MAX_MARKDOWN_BODY_CHARS = 50_000
MAX_NAV_LINK_ITEMS = 30
MAX_FOLDER_SHORTCUT_ITEMS = 20

SCRATCH_PAD_COUNT = 4
MAX_SCRATCH_PAD_LABEL_LEN = 30
MAX_SCRATCH_PAD_BODY_CHARS = 10_000
SCRATCH_PAD_SETTING_KEY = "scratchpad_pads"


class WidgetGrid(BaseModel):
    """12-column grid cell for react-grid-layout (Phase 3)."""

    x: int = Field(0, ge=0)
    y: int = Field(0, ge=0)
    w: int = Field(6, ge=1, le=12)
    h: int = Field(4, ge=1, le=20)


class NavLinkItem(BaseModel):
    label: str = Field(..., min_length=1, max_length=200)
    path: Optional[str] = Field(None, max_length=500, description="In-app route, e.g. /documents")
    href: Optional[str] = Field(None, max_length=2000, description="External URL")

    @model_validator(mode="after")
    def exactly_one_target(self) -> NavLinkItem:
        p = (self.path or "").strip()
        h = (self.href or "").strip()
        if bool(p) == bool(h):
            raise ValueError("Each link must have exactly one of path or href")
        if p:
            if not p.startswith("/") or p.startswith("//"):
                raise ValueError("path must be an app-relative path starting with /")
        if h and not h.startswith(("http://", "https://")):
            raise ValueError("href must start with http:// or https://")
        return self.model_copy(update={"path": p or None, "href": h or None})


class NavLinksConfig(BaseModel):
    items: List[NavLinkItem] = Field(default_factory=list, max_length=MAX_NAV_LINK_ITEMS)


class MarkdownCardConfig(BaseModel):
    title: Optional[str] = Field(None, max_length=200)
    body: str = Field(..., min_length=0, max_length=MAX_MARKDOWN_BODY_CHARS)


class RssHeadlinesConfig(BaseModel):
    feed_id: Optional[str] = Field(
        None,
        max_length=64,
        description="If set, headlines from this feed only; if null, widget aggregates recent feeds client-side",
    )
    limit: int = Field(default=8, ge=1, le=50)


class OrgAgendaConfig(BaseModel):
    days_ahead: int = Field(default=7, ge=1, le=30)
    include_scheduled: bool = True
    include_deadlines: bool = True
    include_appointments: bool = True


class FolderShortcutItem(BaseModel):
    folder_id: str = Field(..., min_length=1, max_length=255)
    label: Optional[str] = Field(None, max_length=200)


class FolderShortcutsConfig(BaseModel):
    items: List[FolderShortcutItem] = Field(
        default_factory=list,
        max_length=MAX_FOLDER_SHORTCUT_ITEMS,
    )


class PinnedDocumentsConfig(BaseModel):
    limit: int = Field(default=10, ge=1, le=50)
    show_preview: bool = False


class FolderImageSlideshowConfig(BaseModel):
    """Random/shuffled image viewer for documents in one folder."""

    folder_id: Optional[str] = Field(
        None,
        max_length=255,
        description="Folder whose image documents are shown",
    )
    scan_limit: int = Field(
        default=500,
        ge=50,
        le=5000,
        description="Max image files to collect when scanning folder pages",
    )
    include_subfolders: bool = Field(
        default=False,
        description="If true, include images from nested subfolders (breadth-first by folder)",
    )


class NavLinksWidget(BaseModel):
    type: Literal["nav_links"] = "nav_links"
    id: str = Field(..., min_length=1, max_length=64)
    config: NavLinksConfig
    grid: Optional[WidgetGrid] = None


class MarkdownCardWidget(BaseModel):
    type: Literal["markdown_card"] = "markdown_card"
    id: str = Field(..., min_length=1, max_length=64)
    config: MarkdownCardConfig
    grid: Optional[WidgetGrid] = None


class RssHeadlinesWidget(BaseModel):
    type: Literal["rss_headlines"] = "rss_headlines"
    id: str = Field(..., min_length=1, max_length=64)
    config: RssHeadlinesConfig
    grid: Optional[WidgetGrid] = None


class OrgAgendaWidget(BaseModel):
    type: Literal["org_agenda"] = "org_agenda"
    id: str = Field(..., min_length=1, max_length=64)
    config: OrgAgendaConfig
    grid: Optional[WidgetGrid] = None


class FolderShortcutsWidget(BaseModel):
    type: Literal["folder_shortcuts"] = "folder_shortcuts"
    id: str = Field(..., min_length=1, max_length=64)
    config: FolderShortcutsConfig
    grid: Optional[WidgetGrid] = None


class PinnedDocumentsWidget(BaseModel):
    type: Literal["pinned_documents"] = "pinned_documents"
    id: str = Field(..., min_length=1, max_length=64)
    config: PinnedDocumentsConfig
    grid: Optional[WidgetGrid] = None


class FolderImageSlideshowWidget(BaseModel):
    type: Literal["folder_images"] = "folder_images"
    id: str = Field(..., min_length=1, max_length=64)
    config: FolderImageSlideshowConfig
    grid: Optional[WidgetGrid] = None


class ArtifactEmbedConfig(BaseModel):
    """Reference to a row in `saved_artifacts` (user-scoped)."""

    artifact_id: Optional[str] = Field(
        None,
        max_length=64,
        description="UUID of saved artifact; null until user selects one in editor",
    )


class ArtifactEmbedWidget(BaseModel):
    type: Literal["artifact_embed"] = "artifact_embed"
    id: str = Field(..., min_length=1, max_length=64)
    config: ArtifactEmbedConfig
    grid: Optional[WidgetGrid] = None


class ScratchPadPad(BaseModel):
    label: str = Field(..., min_length=0, max_length=MAX_SCRATCH_PAD_LABEL_LEN)
    body: str = Field(..., min_length=0, max_length=MAX_SCRATCH_PAD_BODY_CHARS)


class ScratchPadData(BaseModel):
    """Persisted in user_settings under SCRATCH_PAD_SETTING_KEY; shared across dashboards."""

    pads: List[ScratchPadPad] = Field(..., min_length=SCRATCH_PAD_COUNT, max_length=SCRATCH_PAD_COUNT)
    active_index: int = Field(0, ge=0, le=SCRATCH_PAD_COUNT - 1)


class ScratchPadConfig(BaseModel):
    """Widget placement only; pad content lives in user_settings."""

    show_labels: bool = Field(default=True, description="Show pad labels on tabs")


class ScratchPadWidget(BaseModel):
    type: Literal["scratchpad"] = "scratchpad"
    id: str = Field(..., min_length=1, max_length=64)
    config: ScratchPadConfig = Field(default_factory=ScratchPadConfig)
    grid: Optional[WidgetGrid] = None


def default_scratchpad_data() -> ScratchPadData:
    return ScratchPadData(
        pads=[
            ScratchPadPad(label=f"Pad {i + 1}", body="")
            for i in range(SCRATCH_PAD_COUNT)
        ],
        active_index=0,
    )


HomeDashboardWidget = Annotated[
    Union[
        NavLinksWidget,
        MarkdownCardWidget,
        RssHeadlinesWidget,
        OrgAgendaWidget,
        FolderShortcutsWidget,
        PinnedDocumentsWidget,
        FolderImageSlideshowWidget,
        ArtifactEmbedWidget,
        ScratchPadWidget,
    ],
    Discriminator("type"),
]


class HomeDashboardLayout(BaseModel):
    schema_version: Literal[1] = 1
    layout_mode: Literal["stack", "grid"] = "stack"
    widgets: List[HomeDashboardWidget] = Field(default_factory=list)

    @model_validator(mode="after")
    def cap_widgets(self) -> HomeDashboardLayout:
        if len(self.widgets) > MAX_HOME_DASHBOARD_WIDGETS:
            raise ValueError(f"At most {MAX_HOME_DASHBOARD_WIDGETS} widgets allowed")
        return self


def default_home_dashboard_layout() -> HomeDashboardLayout:
    """Starter layout when the user has not saved one yet."""
    return HomeDashboardLayout(
        schema_version=1,
        layout_mode="stack",
        widgets=[
            MarkdownCardWidget(
                id=str(uuid.uuid4()),
                type="markdown_card",
                config=MarkdownCardConfig(
                    title="Welcome",
                    body="Use **Edit layout** to add navigation links, notes, or RSS headlines.",
                ),
            ),
            NavLinksWidget(
                id=str(uuid.uuid4()),
                type="nav_links",
                config=NavLinksConfig(
                    items=[
                        NavLinkItem(label="Documents", path="/documents"),
                        NavLinkItem(label="Chat", path="/chat"),
                    ]
                ),
            ),
        ],
    )


class UserDashboardEntry(BaseModel):
    """One named dashboard page with its widget layout."""

    id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=MAX_DASHBOARD_NAME_LEN)
    is_default: bool = False
    layout: HomeDashboardLayout


class UserDashboardsEnvelope(BaseModel):
    """All dashboard pages for a user (Phase 2)."""

    schema_version: Literal[2] = 2
    dashboards: List[UserDashboardEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dashboards(self) -> UserDashboardsEnvelope:
        if len(self.dashboards) > MAX_USER_DASHBOARDS:
            raise ValueError(f"At most {MAX_USER_DASHBOARDS} dashboards allowed")
        if not self.dashboards:
            raise ValueError("At least one dashboard is required")
        ids = [d.id for d in self.dashboards]
        if len(ids) != len(set(ids)):
            raise ValueError("Dashboard ids must be unique")
        defaults = [d for d in self.dashboards if d.is_default]
        if len(defaults) != 1:
            raise ValueError("Exactly one dashboard must be marked is_default")
        return self


class UserDashboardSummary(BaseModel):
    id: str
    name: str
    is_default: bool


class UserDashboardsListResponse(BaseModel):
    dashboards: List[UserDashboardSummary]


class UserDashboardCreateRequest(BaseModel):
    name: Optional[str] = Field(
        None,
        max_length=MAX_DASHBOARD_NAME_LEN,
        description="Display name; defaults to 'New dashboard'",
    )
    duplicate_from_id: Optional[str] = Field(
        None,
        description="If set, copy layout from this dashboard id",
    )


class UserDashboardPatchRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=MAX_DASHBOARD_NAME_LEN)
    is_default: Optional[bool] = None


def envelope_from_legacy_v1_layout(layout: HomeDashboardLayout) -> UserDashboardsEnvelope:
    """Wrap a Phase-1-only layout as a single default dashboard."""
    return UserDashboardsEnvelope(
        schema_version=2,
        dashboards=[
            UserDashboardEntry(
                id=str(uuid.uuid4()),
                name="Home",
                is_default=True,
                layout=layout,
            )
        ],
    )


def default_user_dashboards_envelope() -> UserDashboardsEnvelope:
    """First-time user: one default dashboard with starter widgets."""
    return envelope_from_legacy_v1_layout(default_home_dashboard_layout())
