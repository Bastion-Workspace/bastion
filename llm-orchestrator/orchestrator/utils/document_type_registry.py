"""
Document Type Registry - Single source of truth for document types, frontmatter schemas,
reference configs, cascade configs, and body templates.

Supports two reference models:
- Hub-and-spoke: project, electronics, outline (hubs); fiction, project_child, electronics_child (hub children)
- Shared library: character, rules, style (standalone, optional explicit refs)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Default body template placeholders: {{title}}, {{creation_date}}, {{query}}
_DEFAULT_PROJECT_BODY = """# {{title}}

## Project Overview

Project created on {{creation_date}}. Open this file and ask the general project agent to help you plan and manage your project.

## Requirements

Project requirements and goals will be documented here.

## Design

Design decisions and approach will be documented here.

## Tasks

Project tasks and milestones will be tracked here.

## Notes

Project notes and documentation will be added here.
"""

_DEFAULT_ELECTRONICS_BODY = """# {{title}}

## Project Overview

Project created on {{creation_date}}. Open this file and ask the electronics agent to help you design your project.

## Components

Components will be added here as the project develops.

## Design Notes

Design notes and decisions will be added here.
"""

_DEFAULT_OUTLINE_BODY = """# {{title}}

## Outline

Outline content and structure will be added here.
"""

_DEFAULT_FICTION_BODY = """# {{title}}

## Chapters

Manuscript content will be added here.
"""

_DEFAULT_CHARACTER_BODY = """# {{title}}

## Character Profile

Character details will be added here.
"""

_DEFAULT_RULES_BODY = """# {{title}}

## Universe / Story Rules

Rules and world-building constraints will be documented here.
"""

_DEFAULT_STYLE_BODY = """# {{title}}

## Style Guide

Writing style, voice, and formatting guidelines will be documented here.
"""

_DEFAULT_REFERENCE_BODY = """# {{title}}

Reference content, notes, or data will be documented here.
"""

_DEFAULT_HUB_CHILD_BODY = """# {{title}}

Content will be added here.
"""


@dataclass
class DocumentTypeSpec:
    """Spec for a document type: frontmatter schema, reference config, cascade, body template."""

    type_key: str
    model: str  # "hub", "hub_child", "shared_library"
    default_frontmatter: Dict[str, Any] = field(default_factory=dict)
    reference_categories: Dict[str, List[str]] = field(default_factory=dict)
    cascade_config: Optional[Dict[str, Dict[str, List[str]]]] = None
    hub_key: Optional[str] = None  # frontmatter key pointing to parent hub (hub_child only)
    parent_frontmatter_list: Optional[str] = None  # which list in hub to add this child to
    body_template: str = ""

    def get_reference_config(self) -> Dict[str, List[str]]:
        """Return reference_categories for load_referenced_files."""
        return self.reference_categories


# ---------------------------------------------------------------------------
# Hub types (they ARE the hub; they reference children)
# ---------------------------------------------------------------------------

PROJECT_SPEC = DocumentTypeSpec(
    type_key="project",
    model="hub",
    default_frontmatter={"type": "project", "title": "", "status": "planning", "files": []},
    reference_categories={
        "specifications": ["specifications", "spec", "specs", "specification"],
        "design": ["design", "design_docs", "architecture"],
        "tasks": ["tasks", "task", "todo", "checklist"],
        "notes": ["notes", "note", "documentation", "docs"],
        "other": ["references", "reference", "files", "related", "documents"],
    },
    cascade_config=None,
    hub_key=None,
    parent_frontmatter_list=None,
    body_template=_DEFAULT_PROJECT_BODY,
)

ELECTRONICS_SPEC = DocumentTypeSpec(
    type_key="electronics",
    model="hub",
    default_frontmatter={"type": "electronics", "title": "", "description": "", "files": []},
    reference_categories={
        "components": ["components", "component", "component_docs"],
        "protocols": ["protocols", "protocol", "protocol_docs"],
        "schematics": ["schematics", "schematic", "schematic_docs"],
        "specifications": ["specifications", "spec", "specs", "specification"],
        "firmware": ["firmware", "code", "software"],
        "bom": ["bom", "bill_of_materials", "parts_list"],
        "other": ["references", "reference", "docs", "documents", "related", "files"],
    },
    cascade_config=None,
    hub_key=None,
    parent_frontmatter_list=None,
    body_template=_DEFAULT_ELECTRONICS_BODY,
)

OUTLINE_SPEC = DocumentTypeSpec(
    type_key="outline",
    model="hub",
    default_frontmatter={"type": "outline", "title": ""},
    reference_categories={
        "style": ["style"],
        "rules": ["rules"],
        "characters": ["characters", "character_*"],
        "series": ["series"],
        "other": ["references", "reference", "related"],
    },
    cascade_config={
        "outline": {
            "rules": ["rules"],
            "style": ["style"],
            "characters": ["characters", "character_*"],
            "series": ["series"],
        }
    },
    hub_key=None,
    parent_frontmatter_list=None,
    body_template=_DEFAULT_OUTLINE_BODY,
)

# ---------------------------------------------------------------------------
# Hub-child types (they reference a single hub; hub lists them)
# ---------------------------------------------------------------------------

FICTION_SPEC = DocumentTypeSpec(
    type_key="fiction",
    model="hub_child",
    default_frontmatter={"type": "fiction", "title": ""},
    reference_categories={},  # cascade from outline
    cascade_config=None,
    hub_key="outline",
    parent_frontmatter_list="files",  # outline may list manuscript; often outline is referenced by fiction, not vice versa
    body_template=_DEFAULT_FICTION_BODY,
)

PROJECT_CHILD_SPEC = DocumentTypeSpec(
    type_key="project_child",
    model="hub_child",
    default_frontmatter={"type": "project", "title": "", "summary": ""},
    reference_categories={},
    cascade_config=None,
    hub_key="project_plan",
    parent_frontmatter_list="files",
    body_template=_DEFAULT_HUB_CHILD_BODY,
)

ELECTRONICS_CHILD_SPEC = DocumentTypeSpec(
    type_key="electronics_child",
    model="hub_child",
    default_frontmatter={"type": "electronics", "title": "", "description": ""},
    reference_categories={},
    cascade_config=None,
    hub_key="project_plan",
    parent_frontmatter_list="files",
    body_template=_DEFAULT_HUB_CHILD_BODY,
)

# ---------------------------------------------------------------------------
# Shared library types (optional explicit refs; no hub)
# ---------------------------------------------------------------------------

CHARACTER_SPEC = DocumentTypeSpec(
    type_key="character",
    model="shared_library",
    default_frontmatter={"type": "character", "title": ""},
    reference_categories={
        "rules": ["rules"],
        "style": ["style"],
        "other": ["references", "reference", "related"],
    },
    cascade_config=None,
    hub_key=None,
    parent_frontmatter_list=None,
    body_template=_DEFAULT_CHARACTER_BODY,
)

RULES_SPEC = DocumentTypeSpec(
    type_key="rules",
    model="shared_library",
    default_frontmatter={"type": "rules", "title": ""},
    reference_categories={
        "style": ["style"],
        "other": ["references", "reference", "related"],
    },
    cascade_config=None,
    hub_key=None,
    parent_frontmatter_list=None,
    body_template=_DEFAULT_RULES_BODY,
)

STYLE_SPEC = DocumentTypeSpec(
    type_key="style",
    model="shared_library",
    default_frontmatter={"type": "style", "title": ""},
    reference_categories={},
    cascade_config=None,
    hub_key=None,
    parent_frontmatter_list=None,
    body_template=_DEFAULT_STYLE_BODY,
)

REFERENCE_SPEC = DocumentTypeSpec(
    type_key="reference",
    model="shared_library",
    default_frontmatter={"type": "reference", "title": ""},
    reference_categories={
        "other": ["references", "reference", "related", "sources"],
    },
    cascade_config=None,
    hub_key=None,
    parent_frontmatter_list=None,
    body_template=_DEFAULT_REFERENCE_BODY,
)

# ---------------------------------------------------------------------------
# Registry map and lookup functions
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, DocumentTypeSpec] = {
    "project": PROJECT_SPEC,
    "electronics": ELECTRONICS_SPEC,
    "outline": OUTLINE_SPEC,
    "fiction": FICTION_SPEC,
    "project_child": PROJECT_CHILD_SPEC,
    "electronics_child": ELECTRONICS_CHILD_SPEC,
    "character": CHARACTER_SPEC,
    "rules": RULES_SPEC,
    "style": STYLE_SPEC,
    "reference": REFERENCE_SPEC,
}

# Backend uses "general" to mean project
_REGISTRY["general"] = PROJECT_SPEC


def get_type_spec(type_key: str) -> Optional[DocumentTypeSpec]:
    """Look up a document type spec by key."""
    return _REGISTRY.get((type_key or "").strip().lower())


def get_reference_config(type_key: str) -> Dict[str, List[str]]:
    """Shortcut for agents: return reference_categories for load_referenced_files."""
    spec = get_type_spec(type_key)
    if spec:
        return spec.reference_categories
    return {}


def list_types() -> List[str]:
    """Return all registered type keys (excluding aliases like general)."""
    return [k for k in _REGISTRY.keys() if k != "general"]


def is_hub_type(type_key: str) -> bool:
    """True if this type is a hub (outline, project, electronics)."""
    spec = get_type_spec(type_key)
    return spec is not None and spec.model == "hub"


def get_hub_child_spec_for_frontmatter(frontmatter: Dict[str, Any]) -> Optional[DocumentTypeSpec]:
    """
    If this document is a hub child (has a back-reference to a hub), return its spec.
    Used by reference_file_loader to decide whether to cascade up to the hub.
    """
    for spec in _REGISTRY.values():
        if spec.model == "hub_child" and spec.hub_key and frontmatter.get(spec.hub_key):
            return spec
    return None
