"""
Team Tools - Read team posts and create team posts/comments for Agent Factory.

Provides read_team_posts_tool (schedule-driven: fetch unread posts) and post_to_team_tool
(create top-level post or comment). Used by team-aware agents and output_router team_post destination.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.line_context import line_id_from_metadata

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_uuid(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and _UUID_RE.match(s.strip()))


# ── I/O Models: read_team_posts ─────────────────────────────────────────────


class ReadTeamPostsInputs(BaseModel):
    """Inputs for read_team_posts. team_id optional when running in team context (from pipeline metadata)."""
    team_id: Optional[str] = Field(default=None, description="Team UUID; omit to use current team from context")


class ReadTeamPostsParams(BaseModel):
    """Optional configuration for read_team_posts."""
    since_last_read: bool = Field(
        default=True,
        description="If true, return only posts created after the user's last_read_at",
    )
    limit: int = Field(default=20, ge=1, le=100, description="Max number of posts to return")
    mark_as_read: bool = Field(
        default=True,
        description="If true, update last_read_at after fetching (so next run gets only newer posts)",
    )


class ReadTeamPostsOutputs(BaseModel):
    """Outputs for read_team_posts."""
    posts: List[Dict[str, Any]] = Field(description="List of post dicts with post_id, author_id, author_name, content, post_type, created_at")
    count: int = Field(description="Number of posts returned")
    team_name: str = Field(description="Display name of the team")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


# ── I/O Models: post_to_team ────────────────────────────────────────────────


class PostToTeamInputs(BaseModel):
    """Inputs for post_to_team. team_id optional when running in team context (from pipeline metadata)."""
    team_id: Optional[str] = Field(default=None, description="Team UUID; omit to use current team from context")
    content: str = Field(description="Post or comment content (wire from e.g. {step_1.formatted})")


class PostToTeamParams(BaseModel):
    """Optional configuration for post_to_team."""
    post_type: str = Field(default="text", description="Post type: text, image, file")
    reply_to_post_id: str = Field(
        default="",
        description="If set, creates a comment on this post instead of a top-level post (e.g. {metadata.post_id} when triggered by team post)",
    )


class PostToTeamOutputs(BaseModel):
    """Outputs for post_to_team."""
    post_id: str = Field(description="ID of the created post or comment")
    success: bool = Field(description="Whether the create succeeded")
    formatted: str = Field(description="Human-readable summary")


# ── Tool functions ─────────────────────────────────────────────────────────


async def read_team_posts_tool(
    team_id: Optional[str] = None,
    user_id: str = "system",
    since_last_read: bool = True,
    limit: int = 20,
    mark_as_read: bool = True,
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Read team posts (optionally since last read). team_id from pipeline metadata when running in team context.
    Returns structured dict with posts, count, team_name, formatted.
    """
    resolved_team = team_id or line_id_from_metadata(_pipeline_metadata or {})
    if not resolved_team:
        return {
            "posts": [],
            "count": 0,
            "team_name": "",
            "formatted": "line_id is required (or run in an agent line context so it is provided automatically).",
        }
    if not _is_uuid(resolved_team):
        return {
            "posts": [],
            "count": 0,
            "team_name": "",
            "formatted": "line_id must be the agent line UUID, not the line name.",
        }
    client = await get_backend_tool_client()
    result = await client.read_team_posts(
        team_id=resolved_team,
        user_id=user_id,
        since_last_read=since_last_read,
        limit=limit,
        mark_as_read=mark_as_read,
    )
    posts = result.get("posts", [])
    count = result.get("count", 0)
    team_name = result.get("team_name", "")
    success = result.get("success", False)
    error = result.get("error", "")
    if not success:
        formatted = f"Failed to read team posts: {error}" if error else "Failed to read team posts."
        return {
            "posts": [],
            "count": 0,
            "team_name": team_name,
            "formatted": formatted,
        }
    lines = [f"Team: {team_name}. {count} post(s) since last read:"]
    for i, p in enumerate(posts, 1):
        author = p.get("author_name", p.get("author_id", "?"))
        content = (p.get("content", "") or "").strip()[:200]
        if len((p.get("content") or "")) > 200:
            content += "..."
        lines.append(f"{i}. [{author}]: {content}")
    formatted = "\n".join(lines)
    return {
        "posts": posts,
        "count": count,
        "team_name": team_name,
        "formatted": formatted,
    }


async def post_to_team_tool(
    content: str,
    team_id: Optional[str] = None,
    user_id: str = "system",
    post_type: str = "text",
    reply_to_post_id: Optional[str] = None,
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a team post or comment. team_id from pipeline metadata when running in team context.
    If reply_to_post_id is set, creates a comment on that post. Returns dict with post_id, success, formatted.
    """
    resolved_team = team_id or line_id_from_metadata(_pipeline_metadata or {})
    if not resolved_team:
        return {
            "post_id": "",
            "success": False,
            "formatted": "line_id is required (or run in an agent line context so it is provided automatically).",
        }
    if not _is_uuid(resolved_team):
        return {
            "post_id": "",
            "success": False,
            "formatted": "line_id must be the agent line UUID, not the line name.",
        }
    client = await get_backend_tool_client()
    result = await client.create_team_post(
        team_id=resolved_team,
        user_id=user_id,
        content=content,
        post_type=post_type or "text",
        reply_to_post_id=reply_to_post_id or "",
    )
    post_id = result.get("post_id", "")
    success = result.get("success", False)
    error = result.get("error", "")
    if success:
        kind = "comment" if (reply_to_post_id and reply_to_post_id.strip()) else "post"
        formatted = f"Created team {kind} (id: {post_id})."
    else:
        formatted = f"Failed to create team post: {error}" if error else "Failed to create team post."
    return {
        "post_id": post_id,
        "success": success,
        "formatted": formatted,
    }


# ── Registry ─────────────────────────────────────────────────────────────────


register_action(
    name="read_team_posts",
    category="teams",
    description="Read posts from a team feed (optionally since last read). Use in scheduled playbooks to process new team activity.",
    inputs_model=ReadTeamPostsInputs,
    params_model=ReadTeamPostsParams,
    outputs_model=ReadTeamPostsOutputs,
    tool_function=read_team_posts_tool,
)

register_action(
    name="post_to_team",
    category="teams",
    description="Create a team post or comment. Set reply_to_post_id to comment on a specific post (e.g. when triggered by new post).",
    inputs_model=PostToTeamInputs,
    params_model=PostToTeamParams,
    outputs_model=PostToTeamOutputs,
    tool_function=post_to_team_tool,
)
