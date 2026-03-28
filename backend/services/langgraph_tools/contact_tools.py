"""
Contact Tools - LangGraph tools for O365 and org-mode contact operations.
Unified contacts (O365 + org-mode) when include_org is requested, similar to calendar unification.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from clients.connections_service_client import get_connections_service_client

logger = logging.getLogger(__name__)

CONTACT_KEYWORDS = ["contact", "people", "person", "friend", "family", "colleague"]
CONTACT_PROPERTY_KEYS = ["EMAIL", "PHONE", "BIRTHDAY", "COMPANY", "ADDRESS", "TITLE"]


def _format_contacts(
    contacts: List[Dict[str, Any]],
    max_items: int = 100,
    include_source: bool = False,
) -> str:
    if not contacts:
        return "No contacts found."
    lines = []
    for i, c in enumerate(contacts[:max_items], 1):
        cid = c.get("contact_id") or c.get("id", "")
        name = c.get("display_name") or f"{c.get('given_name', '')} {c.get('surname', '')}".strip() or "(No name)"
        emails = c.get("email_addresses") or []
        email_str = ", ".join(e.get("address", "") for e in emails if isinstance(e, dict) and e.get("address")) or ", ".join(e for e in emails if isinstance(e, str) and e)
        line = f"{i}. contact_id: {cid} | {name}"
        if email_str:
            line += f" | {email_str}"
        if include_source and c.get("source"):
            line += f" [source: {c['source']}]"
        lines.append(line)
    if len(contacts) > max_items:
        lines.append(f"... and {len(contacts) - max_items} more.")
    return "\n".join(lines)


def _normalize_o365_contact(c: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize O365 contact to unified shape with source=microsoft."""
    cid = c.get("id", "")
    return {
        "id": cid,
        "contact_id": cid,
        "display_name": c.get("display_name", ""),
        "given_name": c.get("given_name", ""),
        "surname": c.get("surname", ""),
        "email_addresses": c.get("email_addresses") or [],
        "phone_numbers": c.get("phone_numbers") or [],
        "company_name": c.get("company_name", ""),
        "job_title": c.get("job_title", ""),
        "birthday": c.get("birthday"),
        "source": "microsoft",
    }


def _normalize_org_contact(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize org-mode search result to unified contact shape with source=org."""
    props = item.get("properties") or {}
    doc_id = item.get("document_id", "")
    line_no = item.get("line_number") or ""
    heading = item.get("heading") or item.get("title", "")
    contact_id = f"{doc_id}#{line_no}#{heading}" if doc_id or line_no or heading else ""
    email = (props.get("EMAIL") or props.get("email") or "").strip()
    emails = [{"address": email, "name": ""}] if email else []
    phone = (props.get("PHONE") or props.get("phone") or props.get("MOBILE") or "").strip()
    phones = [{"number": phone, "type": ""}] if phone else []
    return {
        "id": contact_id,
        "contact_id": contact_id,
        "display_name": heading or "(No name)",
        "given_name": "",
        "surname": "",
        "email_addresses": emails,
        "phone_numbers": phones,
        "company_name": (props.get("COMPANY") or props.get("company") or "").strip(),
        "job_title": (props.get("TITLE") or props.get("title") or "").strip(),
        "birthday": (props.get("BIRTHDAY") or props.get("birthday") or "").strip() or None,
        "source": "org",
    }


def _format_contact_detail(c: Dict[str, Any]) -> str:
    if not c:
        return "Contact not found."
    name = c.get("display_name") or f"{c.get('given_name', '')} {c.get('surname', '')}".strip() or "(No name)"
    lines = [
        f"contact_id: {c.get('id', '')}",
        f"Display name: {name}",
    ]
    if c.get("given_name") or c.get("surname"):
        lines.append(f"Given name: {c.get('given_name', '')} | Surname: {c.get('surname', '')}")
    emails = c.get("email_addresses") or []
    if emails:
        lines.append("Emails: " + ", ".join(f"{e.get('address', '')} ({e.get('name', '')})" for e in emails if e.get("address")))
    phones = c.get("phone_numbers") or []
    if phones:
        lines.append("Phones: " + ", ".join(f"{p.get('number', '')} ({p.get('type', '')})" for p in phones if p.get("number")))
    if c.get("company_name"):
        lines.append(f"Company: {c.get('company_name')}")
    if c.get("job_title"):
        lines.append(f"Job title: {c.get('job_title')}")
    if c.get("birthday"):
        lines.append(f"Birthday: {c.get('birthday')}")
    if (c.get("notes") or "").strip():
        notes = (c.get("notes") or "")[:500]
        if len((c.get("notes") or "")) > 500:
            notes += "..."
        lines.append(f"Notes: {notes}")
    return "\n".join(lines)


async def _get_org_contacts_for_tool(user_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    """Fetch org-mode contact entries (same logic as contacts API) and return normalized contact dicts."""
    try:
        from services.org_search_service import get_org_search_service

        search_service = await get_org_search_service()
        results = await search_service.search_org_files(
            user_id=user_id,
            query="",
            tags=None,
            todo_states=None,
            include_content=False,
            limit=limit,
        )
        if not results.get("success"):
            return []
        contact_list = []
        for item in results.get("results", []):
            properties = item.get("properties", {})
            parent_path = item.get("parent_path", [])
            tags = item.get("tags", [])
            has_contact_property = any(k in properties for k in CONTACT_PROPERTY_KEYS)
            has_contact_parent = any(
                kw in " ".join(parent_path).lower() for kw in CONTACT_KEYWORDS
            )
            has_contact_tag = any(
                kw in tag.lower() for tag in tags for kw in CONTACT_KEYWORDS
            )
            if has_contact_property or has_contact_parent or has_contact_tag:
                contact_list.append(_normalize_org_contact(item))
        return contact_list
    except Exception as e:
        logger.exception("get_org_contacts_for_tool failed: %s", e)
        return []


async def get_contacts_unified(
    user_id: str,
    connection_id: Optional[int] = None,
    folder_id: str = "",
    top: int = 100,
) -> str:
    """
    Get O365 + org-mode contacts merged (unified like calendars).
    Returns JSON string: { "contacts": [...], "formatted": "..." } for orchestrator parsing.
    """
    merged = []
    try:
        client = await get_connections_service_client()
        o365_result = await client.get_contacts(
            user_id=user_id,
            connection_id=connection_id,
            folder_id=folder_id,
            top=top,
        )
        if not o365_result.get("error") and o365_result.get("contacts"):
            for c in o365_result["contacts"]:
                merged.append(_normalize_o365_contact(c))
        org_list = await _get_org_contacts_for_tool(user_id, limit=top)
        merged.extend(org_list)
        formatted = _format_contacts(merged, max_items=top, include_source=True)
        return json.dumps({"contacts": merged, "formatted": formatted})
    except Exception as e:
        logger.exception("get_contacts_unified failed: %s", e)
        return json.dumps(
            {"contacts": [], "formatted": f"Error fetching unified contacts: {e}"}
        )


async def get_contacts(
    user_id: str,
    connection_id: Optional[int] = None,
    folder_id: str = "",
    top: int = 100,
) -> str:
    """Get the user's O365 contacts. Returns a formatted list for the LLM."""
    try:
        client = await get_connections_service_client()
        result = await client.get_contacts(
            user_id=user_id,
            connection_id=connection_id,
            folder_id=folder_id,
            top=top,
        )
        if result.get("error") and not result.get("contacts"):
            return f"Error: {result.get('error', 'Failed to get contacts')}. Ensure an O365 connection with Contacts is configured in Settings."
        contacts = result.get("contacts", [])
        return _format_contacts(contacts, max_items=top)
    except Exception as e:
        logger.exception("get_contacts failed: %s", e)
        return f"Error fetching contacts: {e}"


async def search_contacts(
    user_id: str,
    query: str,
    sources: str = "all",
    top: int = 20,
    connection_id: Optional[int] = None,
) -> str:
    """Search contacts by query (substring match on name, email, company). Returns JSON string with contacts and formatted."""
    if not (query or "").strip():
        return json.dumps({"contacts": [], "formatted": "Provide a search query."})
    merged: List[Dict[str, Any]] = []
    if sources in ("all", "microsoft"):
        client = await get_connections_service_client()
        o365_result = await client.get_contacts(
            user_id=user_id,
            connection_id=connection_id,
            folder_id="",
            top=500,
        )
        if not o365_result.get("error") and o365_result.get("contacts"):
            for c in o365_result["contacts"]:
                merged.append(_normalize_o365_contact(c))
    if sources in ("all", "org"):
        org_list = await _get_org_contacts_for_tool(user_id, limit=500)
        merged.extend(org_list)
    q = (query or "").strip().lower()
    filtered = []
    for c in merged:
        name = " ".join(
            [
                (c.get("display_name") or "").lower(),
                (c.get("given_name") or "").lower(),
                (c.get("surname") or "").lower(),
                (c.get("company_name") or "").lower(),
                (c.get("job_title") or "").lower(),
            ]
        )
        emails = " ".join(
            e.get("address", "")
            for e in (c.get("email_addresses") or [])
            if isinstance(e, dict) and e.get("address")
        ).lower()
        if q in name or q in emails:
            filtered.append(c)
    filtered = filtered[:top]
    formatted = _format_contacts(filtered, max_items=top, include_source=True)
    return json.dumps({"contacts": filtered, "formatted": formatted})


async def get_contact_by_id(
    user_id: str,
    contact_id: str,
    connection_id: Optional[int] = None,
) -> str:
    """Get a single O365 contact by contact_id. Returns full contact details for the LLM."""
    try:
        client = await get_connections_service_client()
        result = await client.get_contact_by_id(
            user_id=user_id,
            contact_id=contact_id,
            connection_id=connection_id,
        )
        if result.get("error") and not result.get("contact"):
            return f"Error: {result.get('error', 'Contact not found')}."
        return _format_contact_detail(result.get("contact"))
    except Exception as e:
        logger.exception("get_contact_by_id failed: %s", e)
        return f"Error fetching contact: {e}"


async def create_contact(
    user_id: str,
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
) -> str:
    """Create an O365 contact. Returns success message with contact_id or error."""
    try:
        client = await get_connections_service_client()
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
        if result.get("success"):
            cid = result.get("contact_id", "")
            return f"Contact created successfully. contact_id: {cid}"
        return f"Failed to create contact: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.exception("create_contact failed: %s", e)
        return f"Error creating contact: {e}"


async def update_contact(
    user_id: str,
    contact_id: str,
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
) -> str:
    """Update an O365 contact. Only provided fields are updated. Returns success or error message."""
    try:
        client = await get_connections_service_client()
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
        if result.get("success"):
            return "Contact updated successfully."
        return f"Failed to update contact: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.exception("update_contact failed: %s", e)
        return f"Error updating contact: {e}"


async def delete_contact(
    user_id: str,
    contact_id: str,
    connection_id: Optional[int] = None,
) -> str:
    """Delete an O365 contact. Returns success or error message."""
    try:
        client = await get_connections_service_client()
        result = await client.delete_contact(
            user_id=user_id,
            contact_id=contact_id,
            connection_id=connection_id,
        )
        if result.get("success"):
            return "Contact deleted successfully."
        return f"Failed to delete contact: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.exception("delete_contact failed: %s", e)
        return f"Error deleting contact: {e}"
