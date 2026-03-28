"""
Contact Tools - O365 and org-mode contact operations via backend gRPC.
Unified contacts (O365 + org-mode) when include_org=True, similar to calendar unification.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.tool_type_models import ContactRef

logger = logging.getLogger(__name__)


def _parse_contacts_from_result(result: Any) -> tuple[List[Dict[str, Any]], str]:
    """Parse backend result (formatted string) into list of contact dicts and display text."""
    if isinstance(result, str):
        text = result
        contacts = []
        for line in result.strip().split("\n"):
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            id_match = re.search(r"contact_id:\s*(\S+)", line)
            if id_match:
                contact_id = id_match.group(1)
                name = ""
                if "|" in line:
                    parts = line.split("|", 2)
                    if len(parts) >= 2:
                        name = parts[1].strip()
                contacts.append({
                    "contact_id": contact_id,
                    "display_name": name or "(No name)",
                    "given_name": "",
                    "surname": "",
                    "email_addresses": [],
                    "phone_numbers": [],
                    "company_name": "",
                    "job_title": "",
                    "birthday": None,
                    "source": "microsoft",
                })
        return contacts, text
    contacts = result.get("contacts") or result.get("items") or []
    text = result.get("formatted") or result.get("content") or str(result)
    out = []
    for c in contacts:
        emails = [e.get("address", "") for e in (c.get("email_addresses") or []) if isinstance(e, dict) and e.get("address")]
        if not emails and c.get("email_addresses"):
            emails = [a for a in c["email_addresses"] if isinstance(a, str) and a]
        phones = [p.get("number", "") for p in (c.get("phone_numbers") or []) if isinstance(p, dict) and p.get("number")]
        if not phones and c.get("phone_numbers"):
            phones = [n for n in c["phone_numbers"] if isinstance(n, str) and n]
        out.append({
            "contact_id": c.get("id", c.get("contact_id", "")),
            "display_name": c.get("display_name", ""),
            "given_name": c.get("given_name", ""),
            "surname": c.get("surname", ""),
            "email_addresses": emails,
            "phone_numbers": phones,
            "company_name": c.get("company_name", ""),
            "job_title": c.get("job_title", ""),
            "birthday": c.get("birthday"),
            "source": c.get("source", "microsoft"),
        })
    return out, text


class GetContactsInputs(BaseModel):
    """Inputs for get_contacts_tool."""
    folder_id: str = Field(default="", description="Contact folder ID (empty = default)")
    top: int = Field(default=100, description="Max contacts to return")
    sources: str = Field(
        default="all",
        description="Sources: all (O365+org), microsoft, org, caldav",
    )


class GetContactByIdInputs(BaseModel):
    """Inputs for get_contact_by_id_tool."""
    contact_id: str = Field(description="O365 contact ID")


class CreateContactInputs(BaseModel):
    """Inputs for create_contact_tool."""
    display_name: str = Field(default="", description="Display name")
    given_name: str = Field(default="", description="Given name")
    surname: str = Field(default="", description="Surname")
    company_name: str = Field(default="", description="Company name")
    job_title: str = Field(default="", description="Job title")
    birthday: str = Field(default="", description="Birthday ISO 8601")
    notes: str = Field(default="", description="Notes")


class UpdateContactInputs(BaseModel):
    """Inputs for update_contact_tool."""
    contact_id: str = Field(description="Contact ID to update")
    display_name: Optional[str] = Field(default=None, description="Display name")
    given_name: Optional[str] = Field(default=None, description="Given name")
    surname: Optional[str] = Field(default=None, description="Surname")
    company_name: Optional[str] = Field(default=None, description="Company name")
    job_title: Optional[str] = Field(default=None, description="Job title")
    birthday: Optional[str] = Field(default=None, description="Birthday ISO 8601")
    notes: Optional[str] = Field(default=None, description="Notes")


class DeleteContactInputs(BaseModel):
    """Inputs for delete_contact_tool."""
    contact_id: str = Field(description="Contact ID to delete")


class GetContactsOutputs(BaseModel):
    """Outputs for get_contacts_tool."""
    contacts: List[ContactRef] = Field(description="List of contacts")
    count: int = Field(description="Number of contacts")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class GetContactByIdOutputs(BaseModel):
    """Outputs for get_contact_by_id_tool."""
    contact: Optional[ContactRef] = Field(default=None, description="Contact if found")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class CreateContactOutputs(BaseModel):
    """Outputs for create_contact_tool."""
    success: bool = Field(description="Whether creation succeeded")
    contact_id: Optional[str] = Field(default=None, description="New contact ID if created")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class UpdateContactOutputs(BaseModel):
    """Outputs for update_contact_tool."""
    success: bool = Field(description="Whether update succeeded")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class DeleteContactOutputs(BaseModel):
    """Outputs for delete_contact_tool."""
    success: bool = Field(description="Whether delete succeeded")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class SearchContactsInputs(BaseModel):
    """Inputs for search_contacts_tool."""
    query: str = Field(description="Search query (name, email, company)")
    sources: str = Field(default="all", description="Sources: all, microsoft, org, caldav")
    top: int = Field(default=20, description="Max results to return")


class SearchContactsOutputs(BaseModel):
    """Outputs for search_contacts_tool."""
    contacts: List[ContactRef] = Field(description="Matching contacts")
    count: int = Field(description="Number of results")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def get_contacts_tool(
    user_id: str = "system",
    connection_id: Optional[int] = None,
    folder_id: str = "",
    top: int = 100,
    sources: str = "all",
) -> Dict[str, Any]:
    """Get contacts. sources: all (O365+org), microsoft, org, caldav. Returns dict with contacts, count, formatted."""
    try:
        logger.info("get_contacts (sources=%s)", sources)
        client = await get_backend_tool_client()
        result = await client.get_contacts(
            user_id=user_id,
            connection_id=connection_id,
            folder_id=folder_id,
            top=top,
            sources=(sources or "all").strip().lower() or "all",
        )
        # Unified response is JSON string with contacts + formatted
        if isinstance(result, str) and result.strip().startswith("{"):
            try:
                data = json.loads(result)
                if isinstance(data, dict) and "contacts" in data:
                    result = data
            except json.JSONDecodeError:
                pass
        contacts_list, text = _parse_contacts_from_result(result)
        refs = [
            ContactRef(
                contact_id=c.get("contact_id", ""),
                display_name=c.get("display_name", ""),
                given_name=c.get("given_name", ""),
                surname=c.get("surname", ""),
                email_addresses=c.get("email_addresses", []),
                phone_numbers=c.get("phone_numbers", []),
                company_name=c.get("company_name", ""),
                job_title=c.get("job_title", ""),
                birthday=c.get("birthday"),
                source=c.get("source", "microsoft"),
            )
            for c in contacts_list
        ]
        return {
            "contacts": [r.model_dump() for r in refs],
            "count": len(refs),
            "formatted": text,
        }
    except Exception as e:
        logger.error("get_contacts_tool error: %s", e)
        err = str(e)
        return {"contacts": [], "count": 0, "formatted": f"Error: {err}"}


async def get_contact_by_id_tool(
    user_id: str = "system",
    contact_id: str = "",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Get a single O365 contact by contact_id. Returns dict with contact, formatted."""
    try:
        if not contact_id:
            msg = "Error: contact_id is required."
            return {"contact": None, "formatted": msg}
        logger.info("get_contact_by_id: %s", contact_id[:50] if contact_id else "")
        client = await get_backend_tool_client()
        result = await client.get_contact_by_id(
            user_id=user_id,
            contact_id=contact_id,
            connection_id=connection_id,
        )
        text = result if isinstance(result, str) else str(result)
        if not result or "Error" in text or "not found" in text.lower():
            return {"contact": None, "formatted": text}
        contact = None
        if isinstance(result, dict) and result.get("contact"):
            c = result["contact"]
            emails = [e.get("address", "") for e in (c.get("email_addresses") or []) if e.get("address")]
            phones = [p.get("number", "") for p in (c.get("phone_numbers") or []) if p.get("number")]
            contact = ContactRef(
                contact_id=c.get("id", contact_id),
                display_name=c.get("display_name", ""),
                given_name=c.get("given_name", ""),
                surname=c.get("surname", ""),
                email_addresses=emails,
                phone_numbers=phones,
                company_name=c.get("company_name", ""),
                job_title=c.get("job_title", ""),
                birthday=c.get("birthday"),
                source="microsoft",
            )
        return {"contact": contact.model_dump() if contact else None, "formatted": text}
    except Exception as e:
        logger.error("get_contact_by_id_tool error: %s", e)
        err = str(e)
        return {"contact": None, "formatted": f"Error: {err}"}


async def create_contact_tool(
    user_id: str = "system",
    display_name: str = "",
    given_name: str = "",
    surname: str = "",
    connection_id: Optional[int] = None,
    folder_id: str = "",
    email_addresses: Optional[List[Dict[str, str]]] = None,
    phone_numbers: Optional[List[Dict[str, str]]] = None,
    company_name: str = "",
    job_title: str = "",
    birthday: str = "",
    notes: str = "",
) -> Dict[str, Any]:
    """Create an O365 contact. Returns dict with success, contact_id, formatted."""
    try:
        logger.info("create_contact: %s", (display_name or given_name or surname or "unnamed")[:50])
        client = await get_backend_tool_client()
        result = await client.create_contact(
            user_id=user_id,
            display_name=display_name,
            given_name=given_name,
            surname=surname,
            connection_id=connection_id,
            folder_id=folder_id,
            email_addresses=email_addresses,
            phone_numbers=phone_numbers,
            company_name=company_name,
            job_title=job_title,
            birthday=birthday,
            notes=notes,
        )
        text = result if isinstance(result, str) else str(result)
        success = "successfully" in text and "Error" not in text
        contact_id = None
        if success and "contact_id:" in text:
            m = re.search(r"contact_id:\s*([A-Za-z0-9_-]+)", text)
            if m:
                contact_id = m.group(1)
        return {"success": success, "contact_id": contact_id, "formatted": text}
    except Exception as e:
        logger.error("create_contact_tool error: %s", e)
        err = str(e)
        return {"success": False, "contact_id": None, "formatted": f"Error: {err}"}


async def update_contact_tool(
    user_id: str = "system",
    contact_id: str = "",
    connection_id: Optional[int] = None,
    display_name: Optional[str] = None,
    given_name: Optional[str] = None,
    surname: Optional[str] = None,
    email_addresses: Optional[List[Dict[str, str]]] = None,
    phone_numbers: Optional[List[Dict[str, str]]] = None,
    company_name: Optional[str] = None,
    job_title: Optional[str] = None,
    birthday: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Update an O365 contact. Returns dict with success, formatted."""
    try:
        if not contact_id:
            msg = "Error: contact_id is required."
            return {"success": False, "formatted": msg}
        logger.info("update_contact: %s", contact_id[:50])
        client = await get_backend_tool_client()
        result = await client.update_contact(
            user_id=user_id,
            contact_id=contact_id,
            connection_id=connection_id,
            display_name=display_name,
            given_name=given_name,
            surname=surname,
            email_addresses=email_addresses,
            phone_numbers=phone_numbers,
            company_name=company_name,
            job_title=job_title,
            birthday=birthday,
            notes=notes,
        )
        text = result if isinstance(result, str) else str(result)
        success = "successfully" in text and "Error" not in text
        return {"success": success, "formatted": text}
    except Exception as e:
        logger.error("update_contact_tool error: %s", e)
        err = str(e)
        return {"success": False, "formatted": f"Error: {err}"}


async def delete_contact_tool(
    user_id: str = "system",
    contact_id: str = "",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Delete an O365 contact. Returns dict with success, formatted."""
    try:
        if not contact_id:
            msg = "Error: contact_id is required."
            return {"success": False, "formatted": msg}
        logger.info("delete_contact: %s", contact_id[:50])
        client = await get_backend_tool_client()
        result = await client.delete_contact(
            user_id=user_id,
            contact_id=contact_id,
            connection_id=connection_id,
        )
        text = result if isinstance(result, str) else str(result)
        success = "successfully" in text and "Error" not in text
        return {"success": success, "formatted": text}
    except Exception as e:
        logger.error("delete_contact_tool error: %s", e)
        err = str(e)
        return {"success": False, "formatted": f"Error: {err}"}


async def search_contacts_tool(
    user_id: str = "system",
    query: str = "",
    sources: str = "all",
    top: int = 20,
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Search contacts by query (name, email, company). Returns dict with contacts, count, formatted."""
    try:
        if not (query or "").strip():
            return {"contacts": [], "count": 0, "formatted": "Provide a search query."}
        logger.info("search_contacts: query=%s", query[:80])
        client = await get_backend_tool_client()
        result = await client.search_contacts(
            user_id=user_id,
            query=query.strip(),
            sources=(sources or "all").strip().lower() or "all",
            top=top,
            connection_id=connection_id,
        )
        if isinstance(result, str) and result.strip().startswith("{"):
            try:
                data = json.loads(result)
                if isinstance(data, dict) and "contacts" in data:
                    result = data
            except json.JSONDecodeError:
                pass
        contacts_list, text = _parse_contacts_from_result(result)
        refs = [
            ContactRef(
                contact_id=c.get("contact_id", ""),
                display_name=c.get("display_name", ""),
                given_name=c.get("given_name", ""),
                surname=c.get("surname", ""),
                email_addresses=c.get("email_addresses", []),
                phone_numbers=c.get("phone_numbers", []),
                company_name=c.get("company_name", ""),
                job_title=c.get("job_title", ""),
                birthday=c.get("birthday"),
                source=c.get("source", "microsoft"),
            )
            for c in contacts_list
        ]
        return {
            "contacts": [r.model_dump() for r in refs],
            "count": len(refs),
            "formatted": text,
        }
    except Exception as e:
        logger.error("search_contacts_tool error: %s", e)
        return {"contacts": [], "count": 0, "formatted": f"Error: {e}"}


register_action(
    name="list_contacts",
    category="contacts",
    description="Get contacts for the user from O365 and optionally org-mode (unified, like calendars).",
    inputs_model=GetContactsInputs,
    params_model=None,
    outputs_model=GetContactsOutputs,
    tool_function=get_contacts_tool,
)
register_action(
    name="get_contact_by_id",
    category="contacts",
    description="Get a single O365 contact by ID.",
    inputs_model=GetContactByIdInputs,
    params_model=None,
    outputs_model=GetContactByIdOutputs,
    tool_function=get_contact_by_id_tool,
)
register_action(
    name="create_contact",
    category="contacts",
    description="Create an O365 contact.",
    inputs_model=CreateContactInputs,
    params_model=None,
    outputs_model=CreateContactOutputs,
    tool_function=create_contact_tool,
)
register_action(
    name="update_contact",
    category="contacts",
    description="Update an O365 contact.",
    inputs_model=UpdateContactInputs,
    params_model=None,
    outputs_model=UpdateContactOutputs,
    tool_function=update_contact_tool,
)
register_action(
    name="delete_contact",
    category="contacts",
    description="Delete an O365 contact.",
    inputs_model=DeleteContactInputs,
    params_model=None,
    outputs_model=DeleteContactOutputs,
    tool_function=delete_contact_tool,
)
register_action(
    name="search_contacts",
    category="contacts",
    description="Search contacts by name, email, or company.",
    inputs_model=SearchContactsInputs,
    params_model=None,
    outputs_model=SearchContactsOutputs,
    tool_function=search_contacts_tool,
)
