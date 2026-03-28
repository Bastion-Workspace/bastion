"""
Structured models for browser automation governance.
Used by agents to declare action plans and file disposition before executing Playwright automation.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


FinalActionType = Literal["download", "click", "extract", "screenshot"]


class BrowserFinalAction(BaseModel):
    """Final action after running steps: download, click, extract, or screenshot."""

    final_action_type: FinalActionType = Field(
        ...,
        description="download: click to trigger file, save to folder_path; click: click only; extract: get text; screenshot: capture image, optional save to folder_path",
    )
    final_selector: str = Field(
        "",
        description="For download: element to click to trigger download; click: element to click; extract: selector (empty=body); screenshot: optional element (empty=full page)",
    )
    folder_path: str = Field(
        "",
        description="Required for download; optional for screenshot (where to save PNG); unused for click/extract",
    )


class BrowserStep(BaseModel):
    """A single step in a browser automation sequence."""

    action: Literal["navigate", "click", "fill", "wait", "download", "extract"] = Field(
        ..., description="Action to perform"
    )
    selector: Optional[str] = Field(None, description="CSS selector or aria-label")
    value: Optional[str] = Field(None, description="Value for fill actions")
    wait_for: Optional[str] = Field(None, description="CSS selector to wait for after action")
    verification_text: Optional[str] = Field(
        None, description="Text near element to confirm correct target"
    )
    reasoning: str = Field("", description="Why this step is needed")
    url: Optional[str] = Field(None, description="URL for navigate action")


class FileDisposition(BaseModel):
    """Where a downloaded file is stored and how it is processed."""

    folder_path: str = Field(..., description="e.g. 'Research/Papers' — resolved to folder_id")
    title: Optional[str] = Field(None, description="Override the filename's title")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    category: Optional[str] = Field(None, description="Document category")
    ingest: bool = Field(True, description="Whether to run through the document pipeline")
    sidecar: Optional[Dict[str, str]] = Field(
        None, description="Extra metadata passed to file watcher"
    )


class BrowserActionPlan(BaseModel):
    """Structured plan for browser automation; emitted by LLM before execution."""

    goal: str = Field(..., description="Natural language description of what this achieves")
    target_url: str = Field(..., description="Initial or primary URL")
    requires_auth: bool = Field(False, description="Whether login/credentials are needed")
    connection_id: Optional[str] = Field(
        None, description="Links to external_connections for credentials"
    )
    steps: List[BrowserStep] = Field(default_factory=list, description="Ordered action steps")
    download_step_index: int = Field(
        -1, description="Index of the step that produces the file (-1 if last step)"
    )
    requires_approval: bool = Field(
        False, description="True for form submissions, purchases, logins"
    )
    file_disposition: Optional[FileDisposition] = Field(
        None, description="Where the file lands and how it is ingested"
    )
    final_action_type: Optional[FinalActionType] = Field(
        None, description="Final action after steps: download, click, extract, screenshot"
    )
    final_selector: Optional[str] = Field(
        None, description="Selector for the final action (e.g. download button, submit button)"
    )
