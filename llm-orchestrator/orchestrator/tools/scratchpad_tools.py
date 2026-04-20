"""
Scratchpad Tools - Read and write the user's dashboard scratch pad via backend gRPC.
Zone 2: implementation in backend grpc_tool_service; this module is the orchestrator wrapper.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)

SCRATCH_PAD_COUNT = 4


# ---------------------------------------------------------------------------
# read_scratchpad
# ---------------------------------------------------------------------------


class ReadScratchpadInputs(BaseModel):
    """Optional inputs; user_id is injected by the engine."""

    pad_index: int = Field(
        default=-1,
        description="Which pad to read: 0–3 for a specific pad, -1 (default) to read all four pads.",
    )


class ScratchpadPadOutput(BaseModel):
    index: int = Field(description="Pad index (0–3)")
    label: str = Field(description="User-assigned tab label, e.g. 'Pad 1'")
    body: str = Field(description="Pad content")


class ReadScratchpadOutputs(BaseModel):
    pads: List[ScratchpadPadOutput] = Field(description="Returned pads")
    active_index: int = Field(description="Index of the currently active pad on the dashboard")
    formatted: str = Field(description="Human-readable pad content for LLM/chat display")


async def read_scratchpad_tool(
    pad_index: int = -1,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Read the user's dashboard scratch pad.
    Pass pad_index=-1 to return all four pads, or pad_index 0-3 to read a single pad.
    Always call this tool when the user asks what is in their scratch pad or notes.
    Do not fabricate pad content; always call this tool to get real data.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.read_scratchpad(user_id=user_id, pad_index=pad_index)
        if not result.get("success", False):
            err = result.get("error", "Unknown error")
            return {
                "pads": [],
                "active_index": 0,
                "formatted": f"Could not read scratch pad: {err}",
            }
        pads = result.get("pads") or []
        active_index = result.get("active_index", 0)

        lines = []
        for p in pads:
            label = p.get("label") or f"Pad {p.get('index', 0) + 1}"
            body = p.get("body") or ""
            lines.append(f"### {label}")
            lines.append(body if body.strip() else "(empty)")
            lines.append("")

        formatted = "\n".join(lines).strip() if lines else "Scratch pad is empty."

        pad_outputs = [
            {"index": p.get("index", 0), "label": p.get("label", ""), "body": p.get("body", "")}
            for p in pads
        ]

        return {
            "pads": pad_outputs,
            "active_index": active_index,
            "formatted": formatted,
        }
    except Exception as e:
        logger.error("read_scratchpad_tool error: %s", e)
        return {
            "pads": [],
            "active_index": 0,
            "formatted": f"Error reading scratch pad: {str(e)}",
        }


register_action(
    name="read_scratchpad",
    category="user",
    description=(
        "Read the contents of the user's dashboard scratch pad. "
        "Returns up to four named pads. Pass pad_index=-1 for all pads (default), "
        "or 0-3 for a specific pad. "
        "Always call this when the user asks what is in their scratch pad, notes, or dashboard pads."
    ),
    short_description="Read the user's dashboard scratch pad",
    inputs_model=ReadScratchpadInputs,
    params_model=None,
    outputs_model=ReadScratchpadOutputs,
    tool_function=read_scratchpad_tool,
)


# ---------------------------------------------------------------------------
# write_scratchpad_pad
# ---------------------------------------------------------------------------


class WriteScratchpadPadInputs(BaseModel):
    pad_index: int = Field(
        description="Which pad to write to (0–3). The user should specify this explicitly.",
    )
    body: str = Field(
        description="New content for the pad. This REPLACES the existing body entirely.",
    )
    label: Optional[str] = Field(
        default=None,
        description="Optional new tab label for the pad. Omit to keep the existing label.",
    )


class WriteScratchpadPadOutputs(BaseModel):
    success: bool = Field(description="Whether the pad was updated successfully")
    formatted: str = Field(description="Confirmation message for LLM/chat display")


async def write_scratchpad_pad_tool(
    pad_index: int,
    body: str,
    label: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Overwrite the body of one of the user's dashboard scratch pad tabs (pad_index 0-3).
    This REPLACES the existing content of the specified pad on the user's visible home dashboard.
    Only call this when the user has explicitly asked you to update their scratch pad.
    Always confirm which pad to use (0-3) and the intended content before calling.
    Optionally rename the pad tab by passing a new label.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.write_scratchpad_pad(
            pad_index=pad_index,
            body=body,
            label=label or "",
            user_id=user_id,
        )
        if not result.get("success", False):
            err = result.get("error", "Unknown error")
            return {
                "success": False,
                "formatted": f"Could not update scratch pad: {err}",
            }
        pad_num = pad_index + 1
        label_note = f" (renamed to '{label}')" if label else ""
        return {
            "success": True,
            "formatted": f"Scratch pad {pad_num}{label_note} updated successfully.",
        }
    except Exception as e:
        logger.error("write_scratchpad_pad_tool error: %s", e)
        return {
            "success": False,
            "formatted": f"Error updating scratch pad: {str(e)}",
        }


register_action(
    name="write_scratchpad_pad",
    category="user",
    description=(
        "Overwrite the body of one of the user's four dashboard scratch pads (pad_index 0-3). "
        "This REPLACES the existing visible content on the user's home dashboard. "
        "Only call when the user explicitly asks you to write to their scratch pad. "
        "Confirm the pad index and content with the user before calling."
    ),
    short_description="Write to the user's dashboard scratch pad",
    inputs_model=WriteScratchpadPadInputs,
    params_model=None,
    outputs_model=WriteScratchpadPadOutputs,
    tool_function=write_scratchpad_pad_tool,
)
