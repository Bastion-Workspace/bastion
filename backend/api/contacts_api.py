"""
Contacts API - REST endpoints for Contacts view (O365 + org-mode).
Exposes list connections, O365 contacts (CRUD), and org contacts for the frontend.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Body

from utils.auth_middleware import get_current_user, AuthenticatedUserResponse
from services.external_connections_service import external_connections_service
from services.org_search_service import get_org_search_service
from clients.connections_service_client import get_connections_service_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Contacts"])


@router.get("/api/contacts/connections")
async def list_contacts_connections(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """
    List current user's contacts-capable connections (Microsoft O365; same as calendar).
    Returns connection id, provider, and display name for the Contacts source selector.
    """
    try:
        out = []
        microsoft = await external_connections_service.get_user_connections(
            current_user.user_id,
            provider="microsoft",
            connection_type="email",
            active_only=True,
        )
        for c in microsoft:
            out.append({
                "id": c["id"],
                "provider": c.get("provider", "microsoft"),
                "display_name": c.get("display_name") or c.get("account_identifier") or "Microsoft",
            })
        return {"connections": out, "error": None}
    except Exception as e:
        logger.exception("list_contacts_connections failed: %s", e)
        return {"connections": [], "error": str(e)}


@router.get("/api/contacts/o365")
async def get_o365_contacts(
    connection_id: Optional[int] = Query(None, description="Connection ID (default: first Microsoft)"),
    folder_id: Optional[str] = Query("", description="Contact folder ID (empty = default)"),
    top: int = Query(100, ge=1, le=1000, description="Max contacts"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """
    Get O365 contacts for the given connection (or first Microsoft connection).
    """
    try:
        client = await get_connections_service_client()
        result = await client.get_contacts(
            user_id=current_user.user_id,
            connection_id=connection_id,
            folder_id=folder_id or "",
            top=top,
            rls_context={"user_id": current_user.user_id},
        )
        if result.get("error") and not result.get("contacts"):
            return {
                "contacts": [],
                "total_count": 0,
                "error": result.get("error", "No contacts connection or token"),
            }
        return {
            "contacts": result.get("contacts", []),
            "total_count": result.get("total_count", 0),
            "error": result.get("error"),
        }
    except Exception as e:
        logger.exception("get_o365_contacts failed: %s", e)
        return {"contacts": [], "total_count": 0, "error": str(e)}


@router.post("/api/contacts/o365")
async def create_o365_contact(
    body: Dict[str, Any] = Body(..., description="Contact fields: display_name, given_name, surname, email_addresses, phone_numbers, company_name, job_title, birthday, notes"),
    connection_id: Optional[int] = Query(None, description="Connection ID (default: first Microsoft)"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Create an O365 contact."""
    try:
        client = await get_connections_service_client()
        result = await client.create_contact(
            user_id=current_user.user_id,
            connection_id=connection_id,
            display_name=body.get("display_name", ""),
            given_name=body.get("given_name", ""),
            surname=body.get("surname", ""),
            folder_id=body.get("folder_id", ""),
            email_addresses=body.get("email_addresses"),
            phone_numbers=body.get("phone_numbers"),
            company_name=body.get("company_name", ""),
            job_title=body.get("job_title", ""),
            birthday=body.get("birthday", ""),
            notes=body.get("notes", ""),
            rls_context={"user_id": current_user.user_id},
        )
        if not result.get("success"):
            return {"success": False, "contact_id": None, "error": result.get("error", "Failed to create contact")}
        return {"success": True, "contact_id": result.get("contact_id"), "error": result.get("error")}
    except Exception as e:
        logger.exception("create_o365_contact failed: %s", e)
        return {"success": False, "contact_id": None, "error": str(e)}


@router.patch("/api/contacts/o365/{contact_id}")
async def update_o365_contact(
    contact_id: str,
    body: Dict[str, Any] = Body(..., description="Fields to update (partial)"),
    connection_id: Optional[int] = Query(None),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Update an O365 contact."""
    try:
        client = await get_connections_service_client()
        result = await client.update_contact(
            user_id=current_user.user_id,
            contact_id=contact_id,
            connection_id=connection_id,
            display_name=body.get("display_name"),
            given_name=body.get("given_name"),
            surname=body.get("surname"),
            email_addresses=body.get("email_addresses"),
            phone_numbers=body.get("phone_numbers"),
            company_name=body.get("company_name"),
            job_title=body.get("job_title"),
            birthday=body.get("birthday"),
            notes=body.get("notes"),
            rls_context={"user_id": current_user.user_id},
        )
        return {"success": result.get("success", False), "error": result.get("error")}
    except Exception as e:
        logger.exception("update_o365_contact failed: %s", e)
        return {"success": False, "error": str(e)}


@router.delete("/api/contacts/o365/{contact_id}")
async def delete_o365_contact(
    contact_id: str,
    connection_id: Optional[int] = Query(None),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Delete an O365 contact."""
    try:
        client = await get_connections_service_client()
        result = await client.delete_contact(
            user_id=current_user.user_id,
            contact_id=contact_id,
            connection_id=connection_id,
            rls_context={"user_id": current_user.user_id},
        )
        return {"success": result.get("success", False), "error": result.get("error")}
    except Exception as e:
        logger.exception("delete_o365_contact failed: %s", e)
        return {"success": False, "error": str(e)}


@router.get("/api/contacts/org")
async def get_org_contacts(
    category: Optional[str] = Query(None, description="Filter by category (tag or parent heading)"),
    limit: int = Query(500, ge=1, le=1000, description="Maximum number of results"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """
    Get org-mode contact entries (proxy to same logic as /api/org/contacts).
    Contacts are identified by EMAIL/PHONE/BIRTHDAY properties or contact-related headings/tags.
    """
    try:
        search_service = await get_org_search_service()
        results = await search_service.search_org_files(
            user_id=current_user.user_id,
            query="",
            tags=None,
            todo_states=None,
            include_content=False,
            limit=limit,
        )
        if not results.get("success"):
            return {
                "success": False,
                "results": [],
                "count": 0,
                "error": results.get("error"),
            }
        contact_keywords = ["contact", "people", "person", "friend", "family", "colleague"]
        contacts = []
        for item in results.get("results", []):
            properties = item.get("properties", {})
            parent_path = item.get("parent_path", [])
            tags = item.get("tags", [])
            has_contact_property = any(
                key in properties
                for key in ["EMAIL", "PHONE", "BIRTHDAY", "COMPANY", "ADDRESS", "TITLE"]
            )
            has_contact_parent = any(
                kw in " ".join(parent_path).lower() for kw in contact_keywords
            )
            has_contact_tag = any(
                kw in tag.lower() for tag in tags for kw in contact_keywords
            )
            if has_contact_property or has_contact_parent or has_contact_tag:
                if category:
                    if category in tags or any(category.lower() in p.lower() for p in parent_path):
                        contacts.append(item)
                else:
                    contacts.append(item)
        return {
            "success": True,
            "results": contacts,
            "count": len(contacts),
            "files_searched": results.get("files_searched", 0),
        }
    except Exception as e:
        logger.exception("get_org_contacts failed: %s", e)
        return {"success": False, "results": [], "count": 0, "error": str(e)}
