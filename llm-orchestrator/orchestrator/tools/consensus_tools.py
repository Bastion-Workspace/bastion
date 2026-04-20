"""
Optional consensus helpers: store proposals and votes in team workspace (JSON ballot).
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.line_context import line_id_from_metadata

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
BALLOT_KEY = "_consensus_ballot"


def _is_uuid(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and _UUID_RE.match(s.strip()))


def _load_ballot(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            return {"proposals": [], "votes": []}
        if "proposals" not in data:
            data["proposals"] = []
        if "votes" not in data:
            data["votes"] = []
        return data
    except json.JSONDecodeError:
        return {"proposals": [], "votes": []}


class ProposeActionInputs(BaseModel):
    action_type: str = Field(description="e.g. create_task, send_message")
    title: str = Field(default="", description="Short title or summary")
    description: str = Field(default="", description="Details")
    target_agent_id: str = Field(default="", description="Assignee or message target UUID or @handle")


class ProposeActionOutputs(BaseModel):
    formatted: str
    proposal_id: str = ""
    success: bool = False


async def propose_action_tool(
    action_type: str,
    title: str = "",
    description: str = "",
    target_agent_id: str = "",
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Record a structured proposal on the team workspace ballot for consensus workflows.
    Returns proposal_id for vote_on_proposal.
    """
    metadata = _pipeline_metadata or {}
    line_id = line_id_from_metadata(metadata)
    agent_profile_id = metadata.get("agent_profile_id")
    if not line_id or not _is_uuid(line_id):
        return {
            "formatted": "No valid line_id in context; consensus tools require an agent line.",
            "proposal_id": "",
            "success": False,
        }
    pid = str(uuid.uuid4())
    try:
        client = await get_backend_tool_client()
        r = await client.read_workspace(team_id=line_id, user_id=user_id, key=BALLOT_KEY)
        entry_body = ""
        if r.get("success") and r.get("single") and isinstance(r.get("entry"), dict):
            entry_body = str(r["entry"].get("value") or "")
        ballot = _load_ballot(entry_body)
        proposal = {
            "proposal_id": pid,
            "action_type": (action_type or "").strip(),
            "title": (title or "").strip()[:500],
            "description": (description or "").strip()[:8000],
            "target_agent_id": (target_agent_id or "").strip(),
            "proposed_by": agent_profile_id or "",
        }
        ballot["proposals"].append(proposal)
        wr = await client.set_workspace_entry(
            team_id=line_id,
            key=BALLOT_KEY,
            value=json.dumps(ballot),
            user_id=user_id,
            updated_by_agent_id=agent_profile_id,
        )
        if not wr.get("success"):
            return {
                "formatted": wr.get("error") or "Failed to save proposal.",
                "proposal_id": "",
                "success": False,
            }
        return {
            "formatted": f"Proposal recorded (proposal_id={pid}).",
            "proposal_id": pid,
            "success": True,
        }
    except Exception as e:
        logger.warning("propose_action_tool failed: %s", e)
        return {"formatted": str(e), "proposal_id": "", "success": False}


class VoteOnProposalInputs(BaseModel):
    proposal_id: str = Field(description="proposal_id from propose_action")
    vote: str = Field(description="approve, reject, or abstain")


class VoteOnProposalOutputs(BaseModel):
    formatted: str
    success: bool = False


async def vote_on_proposal_tool(
    proposal_id: str,
    vote: str,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Cast a vote on a proposal in the workspace ballot."""
    metadata = _pipeline_metadata or {}
    line_id = line_id_from_metadata(metadata)
    agent_profile_id = metadata.get("agent_profile_id")
    if not line_id or not _is_uuid(line_id):
        return {"formatted": "No valid line_id in context.", "success": False}
    v = (vote or "").strip().lower()
    if v not in ("approve", "reject", "abstain"):
        v = "abstain"
    try:
        client = await get_backend_tool_client()
        r = await client.read_workspace(team_id=line_id, user_id=user_id, key=BALLOT_KEY)
        entry_body = ""
        if r.get("success") and r.get("single") and isinstance(r.get("entry"), dict):
            entry_body = str(r["entry"].get("value") or "")
        ballot = _load_ballot(entry_body)
        ballot["votes"].append(
            {
                "proposal_id": (proposal_id or "").strip(),
                "vote": v,
                "voter_agent_id": agent_profile_id or "",
            }
        )
        wr = await client.set_workspace_entry(
            team_id=line_id,
            key=BALLOT_KEY,
            value=json.dumps(ballot),
            user_id=user_id,
            updated_by_agent_id=agent_profile_id,
        )
        if not wr.get("success"):
            return {"formatted": wr.get("error") or "Failed to save vote.", "success": False}
        return {"formatted": f"Vote recorded ({v}) for proposal {proposal_id}.", "success": True}
    except Exception as e:
        logger.warning("vote_on_proposal_tool failed: %s", e)
        return {"formatted": str(e), "success": False}


class TallyProposalsInputs(BaseModel):
    """No required inputs; line_id from pipeline metadata."""


class TallyProposalsOutputs(BaseModel):
    formatted: str
    tallies: List[Dict[str, Any]] = Field(default_factory=list)
    success: bool = False


async def tally_proposals_tool(
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Summarize proposals and vote counts from the workspace ballot."""
    metadata = _pipeline_metadata or {}
    line_id = line_id_from_metadata(metadata)
    if not line_id or not _is_uuid(line_id):
        return {"formatted": "No valid line_id in context.", "tallies": [], "success": False}
    try:
        client = await get_backend_tool_client()
        r = await client.read_workspace(team_id=line_id, user_id=user_id, key=BALLOT_KEY)
        entry_body = ""
        if r.get("success") and r.get("single") and isinstance(r.get("entry"), dict):
            entry_body = str(r["entry"].get("value") or "")
        ballot = _load_ballot(entry_body)
        proposals = ballot.get("proposals") or []
        votes = ballot.get("votes") or []
        counts: Dict[str, Dict[str, int]] = {}
        for v in votes:
            pid = str(v.get("proposal_id") or "")
            if not pid:
                continue
            if pid not in counts:
                counts[pid] = {"approve": 0, "reject": 0, "abstain": 0}
            vv = (v.get("vote") or "abstain").lower()
            if vv in counts[pid]:
                counts[pid][vv] += 1
        tallies = []
        lines = ["Consensus ballot tally:"]
        for p in proposals:
            pid = p.get("proposal_id")
            c = counts.get(str(pid), {"approve": 0, "reject": 0, "abstain": 0})
            tallies.append({"proposal": p, "votes": c})
            lines.append(
                f"- {pid}: {p.get('action_type')} / {p.get('title')[:40]} — "
                f"approve={c['approve']} reject={c['reject']} abstain={c['abstain']}"
            )
        formatted = "\n".join(lines) if proposals else "No proposals on ballot."
        return {"formatted": formatted, "tallies": tallies, "success": True}
    except Exception as e:
        logger.warning("tally_proposals_tool failed: %s", e)
        return {"formatted": str(e), "tallies": [], "success": False}


register_action(
    name="propose_action",
    category="agent_communication",
    description="Record a structured proposal on the team consensus ballot (workspace)",
    inputs_model=ProposeActionInputs,
    params_model=None,
    outputs_model=ProposeActionOutputs,
    tool_function=propose_action_tool,
)
register_action(
    name="vote_on_proposal",
    category="agent_communication",
    description="Vote approve/reject/abstain on a consensus proposal_id",
    inputs_model=VoteOnProposalInputs,
    params_model=None,
    outputs_model=VoteOnProposalOutputs,
    tool_function=vote_on_proposal_tool,
)
register_action(
    name="tally_proposals",
    category="agent_communication",
    description="Summarize consensus ballot proposals and votes",
    inputs_model=TallyProposalsInputs,
    params_model=None,
    outputs_model=TallyProposalsOutputs,
    tool_function=tally_proposals_tool,
)
