"""
Document type templates for project plan creation.

Mirrors the frontmatter and body templates from the orchestrator's document_type_registry
for project and electronics hub types. Used by projects_api when creating projects
so backend stays self-contained (no dependency on llm-orchestrator).
"""

from datetime import datetime
from typing import Optional


def get_project_plan_content(project_type: str, project_name: str) -> str:
    """
    Return full document content (frontmatter + body) for a new project plan.

    project_type: "general" or "electronics" (general uses frontmatter type "project")
    project_name: Title and heading text.
    """
    creation_date = datetime.utcnow().strftime("%Y-%m-%d")
    frontmatter_type = "project" if project_type == "general" else project_type

    frontmatter = f"""---
type: {frontmatter_type}
title: {project_name}
status: planning
files: []
---

"""

    if project_type == "general":
        body = f"""# {project_name}

## Project Overview

Project created on {creation_date}. Open this file and ask the general project agent to help you plan and manage your project.

## Requirements

Project requirements and goals will be documented here.

## Design

Design decisions and approach will be documented here.

## Tasks

Project tasks and milestones will be tracked here.

## Notes

Project notes and documentation will be added here.
"""
    else:
        body = f"""# {project_name}

## Project Overview

Project created on {creation_date}. Open this file and ask the {project_type} agent to help you design your project.

## Components

Components will be added here as the project develops.

## Design Notes

Design notes and decisions will be added here.
"""

    return frontmatter + body


def get_allowed_project_types() -> list:
    """Return list of project types allowed for API creation."""
    return ["electronics", "general"]
