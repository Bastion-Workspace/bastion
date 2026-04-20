"""gRPC handlers for Org-mode and Todo operations."""

import logging

import grpc
from protos import tool_service_pb2

logger = logging.getLogger(__name__)


class OrgTodoHandlersMixin:
    """Mixin providing Org-mode and Todo gRPC handlers.

    Mixed into ToolServiceImplementation in grpc_tool_service.py.
    """

    # ===== Org-mode Operations =====

    async def SearchOrgFiles(
        self,
        request: tool_service_pb2.OrgSearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.OrgSearchResponse:
        """Search org-mode files"""
        try:
            logger.info(f"SearchOrgFiles: query={request.query}")
            
            # Placeholder implementation
            response = tool_service_pb2.OrgSearchResponse()
            return response
            
        except Exception as e:
            logger.error(f"SearchOrgFiles error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Org search failed: {str(e)}")

    async def GetOrgInboxItems(
        self,
        request: tool_service_pb2.OrgInboxRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.OrgInboxResponse:
        """Get org-mode inbox items"""
        try:
            logger.info(f"GetOrgInboxItems: user={request.user_id}")
            
            # Placeholder implementation
            response = tool_service_pb2.OrgInboxResponse()
            return response
            
        except Exception as e:
            logger.error(f"GetOrgInboxItems error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Get inbox items failed: {str(e)}")

    # ===== Org Inbox Management Operations =====

    async def ListOrgInboxItems(
        self,
        request: tool_service_pb2.ListOrgInboxItemsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ListOrgInboxItemsResponse:
        """List all org inbox items for user"""
        try:
            logger.info(f"ListOrgInboxItems: user={request.user_id}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_list_items, org_inbox_path
            
            # Get inbox path
            path = await org_inbox_path(request.user_id)
            
            # List items
            listing = await org_inbox_list_items(request.user_id)
            
            # Convert to proto response
            response = tool_service_pb2.ListOrgInboxItemsResponse(
                success=True,
                path=path
            )
            
            for item in listing.get("items", []):
                item_details = tool_service_pb2.OrgInboxItemDetails(
                    line_index=item.get("line_index", 0),
                    text=item.get("text", ""),
                    item_type=item.get("item_type", "plain"),
                    todo_state=item.get("todo_state", ""),
                    tags=item.get("tags", []),
                    is_done=item.get("is_done", False)
                )
                response.items.append(item_details)
            
            logger.info(f"ListOrgInboxItems: Found {len(response.items)} items")
            return response
            
        except Exception as e:
            logger.error(f"❌ ListOrgInboxItems error: {e}")
            return tool_service_pb2.ListOrgInboxItemsResponse(
                success=False,
                error=str(e)
            )

    async def AddOrgInboxItem(
        self,
        request: tool_service_pb2.AddOrgInboxItemRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.AddOrgInboxItemResponse:
        """Add new item to org inbox"""
        try:
            logger.info(f"AddOrgInboxItem: user={request.user_id}, kind={request.kind}, text={request.text[:50]}...")
            
            from services.langgraph_tools.org_inbox_tools import (
                org_inbox_add_item,
                org_inbox_append_text,
                org_inbox_list_items,
                org_inbox_set_schedule_and_repeater,
                org_inbox_apply_tags
            )
            from services.org_todo_service import _strip_trailing_org_tags_from_title

            # If tags will be applied, strip any trailing org-style tags from text to avoid duplicating
            text = request.text or ""
            if request.tags:
                text = _strip_trailing_org_tags_from_title(text)
            
            # Handle different kinds of entries
            if request.kind == "contact":
                # Build contact entry with PROPERTIES drawer
                headline = f"* {text}"
                org_entry = f"{headline}\n"
                
                if request.contact_properties:
                    org_entry += ":PROPERTIES:\n"
                    for key, value in request.contact_properties.items():
                        if value:
                            org_entry += f":{key}: {value}\n"
                    org_entry += ":END:\n"
                
                result = await org_inbox_append_text(org_entry, request.user_id)
                line_index = None  # Will determine after listing
                
            elif request.kind == "note":
                # Headline without TODO (plain note)
                headline = f"* {text}"
                org_entry = f"{headline}\n"
                result = await org_inbox_append_text(org_entry, request.user_id)
                listing = await org_inbox_list_items(request.user_id)
                items = listing.get("items", [])
                line_index = items[-1].get("line_index") if items else None
                
            elif request.schedule or request.kind == "event":
                # Build a proper org-mode entry with schedule
                org_type = "TODO" if request.kind == "todo" else ""
                headline = f"* {org_type} {text}".strip()
                org_entry = f"{headline}\n"
                result = await org_inbox_append_text(org_entry, request.user_id)
                
                # Get the line index of the newly added item
                listing = await org_inbox_list_items(request.user_id)
                items = listing.get("items", [])
                line_index = items[-1].get("line_index") if items else None
                
                # Set schedule if provided
                if line_index is not None and request.schedule:
                    await org_inbox_set_schedule_and_repeater(
                        line_index=line_index,
                        scheduled=request.schedule,
                        repeater=request.repeater if request.repeater else None,
                        user_id=request.user_id
                    )
            else:
                # Regular todo or checkbox
                kind = "todo" if request.kind != "checkbox" else "checkbox"
                result = await org_inbox_add_item(text=text, kind=kind, user_id=request.user_id)
                line_index = result.get("line_index")
            
            # Apply tags if provided
            if line_index is not None and request.tags:
                await org_inbox_apply_tags(line_index=line_index, tags=list(request.tags), user_id=request.user_id)
            elif line_index is None and request.tags:
                # Best effort: get last item's index
                listing = await org_inbox_list_items(request.user_id)
                items = listing.get("items", [])
                if items:
                    line_index = items[-1].get("line_index")
                    if line_index is not None:
                        await org_inbox_apply_tags(line_index=line_index, tags=list(request.tags), user_id=request.user_id)
            
            logger.info(f"✅ AddOrgInboxItem: Added item successfully")
            return tool_service_pb2.AddOrgInboxItemResponse(
                success=True,
                line_index=line_index if line_index is not None else 0,
                message=f"Added '{text}' to inbox.org"
            )
            
        except Exception as e:
            logger.error(f"❌ AddOrgInboxItem error: {e}")
            return tool_service_pb2.AddOrgInboxItemResponse(
                success=False,
                error=str(e)
            )

    async def CaptureJournalEntry(
        self,
        request: tool_service_pb2.CaptureJournalEntryRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.CaptureJournalEntryResponse:
        """Append a journal entry; respects user journal preferences and date hierarchy."""
        try:
            from services.org_journal_service import get_org_journal_service
            from models.org_capture_models import OrgCaptureRequest

            content = request.content or ""
            if request.HasField("title") and request.title:
                content = f"{request.title}\n{content}"
            capture_req = OrgCaptureRequest(
                content=content,
                template_type="journal",
                tags=list(request.tags) if request.tags else None,
                entry_date=request.entry_date if request.HasField("entry_date") and request.entry_date else None,
            )
            svc = await get_org_journal_service()
            response = await svc.capture_journal_entry(request.user_id, capture_req)
            return tool_service_pb2.CaptureJournalEntryResponse(
                success=response.success,
                message=response.message,
                entry_preview=response.entry_preview or "",
                file_path=response.file_path or "",
                document_id="",
            )
        except Exception as e:
            logger.error("CaptureJournalEntry error: %s", e)
            return tool_service_pb2.CaptureJournalEntryResponse(
                success=False,
                message="",
                error=str(e),
            )

    async def GetJournalEntry(
        self,
        request: tool_service_pb2.GetJournalEntryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetJournalEntryResponse:
        """Read one date's journal entry (section-aware)."""
        try:
            from services.org_journal_service import get_org_journal_service
            svc = await get_org_journal_service()
            date_str = request.date or "today"
            result = await svc.get_journal_entry(request.user_id, date_str)
            return tool_service_pb2.GetJournalEntryResponse(
                success=result.get("success", False),
                content=result.get("content", ""),
                date=result.get("date", ""),
                heading=result.get("heading", ""),
                document_id=result.get("document_id") or "",
                file_path=result.get("file_path") or "",
                has_content=result.get("has_content", False),
                error=result.get("error") or "",
            )
        except Exception as e:
            logger.error("GetJournalEntry error: %s", e)
            return tool_service_pb2.GetJournalEntryResponse(
                success=False,
                content="",
                date="",
                heading="",
                has_content=False,
                error=str(e),
            )

    async def GetJournalEntries(
        self,
        request: tool_service_pb2.GetJournalEntriesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetJournalEntriesResponse:
        """Get full content of journal entries in a date range (review/lookback)."""
        try:
            from services.org_journal_service import get_org_journal_service
            svc = await get_org_journal_service()
            max_entries = 100
            if request.HasField("max_entries") and request.max_entries > 0:
                max_entries = request.max_entries
            result = await svc.get_journal_entries(
                request.user_id,
                start_date=request.start_date if request.HasField("start_date") and request.start_date else None,
                end_date=request.end_date if request.HasField("end_date") and request.end_date else None,
                max_entries=max_entries,
            )
            if not result.get("success"):
                return tool_service_pb2.GetJournalEntriesResponse(
                    success=False,
                    total=0,
                    error=result.get("error") or "",
                )
            entries = [
                tool_service_pb2.JournalEntryWithContent(
                    date=e["date"],
                    content=e.get("content", ""),
                    heading=e.get("heading", ""),
                    has_content=e.get("has_content", False),
                )
                for e in result.get("entries", [])
            ]
            return tool_service_pb2.GetJournalEntriesResponse(
                success=True,
                entries=entries,
                total=result.get("total", 0),
            )
        except Exception as e:
            logger.error("GetJournalEntries error: %s", e)
            return tool_service_pb2.GetJournalEntriesResponse(
                success=False,
                total=0,
                error=str(e),
            )

    async def UpdateJournalEntry(
        self,
        request: tool_service_pb2.UpdateJournalEntryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateJournalEntryResponse:
        """Replace or append to a single date's journal section only."""
        try:
            from services.org_journal_service import get_org_journal_service
            svc = await get_org_journal_service()
            result = await svc.update_journal_entry(
                request.user_id,
                request.date or "",
                request.content or "",
                request.mode or "replace",
            )
            return tool_service_pb2.UpdateJournalEntryResponse(
                success=result.get("success", False),
                date=result.get("date", ""),
                error=result.get("error") or "",
            )
        except Exception as e:
            logger.error("UpdateJournalEntry error: %s", e)
            return tool_service_pb2.UpdateJournalEntryResponse(
                success=False,
                date=request.date or "",
                error=str(e),
            )

    async def ListJournalEntries(
        self,
        request: tool_service_pb2.ListJournalEntriesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListJournalEntriesResponse:
        """List journal entries in a date range with metadata."""
        try:
            from services.org_journal_service import get_org_journal_service
            svc = await get_org_journal_service()
            result = await svc.list_journal_entries(
                request.user_id,
                start_date=request.start_date if request.HasField("start_date") and request.start_date else None,
                end_date=request.end_date if request.HasField("end_date") and request.end_date else None,
            )
            if not result.get("success"):
                return tool_service_pb2.ListJournalEntriesResponse(
                    success=False,
                    total=0,
                    error=result.get("error") or "",
                )
            entries = [
                tool_service_pb2.JournalEntryMeta(
                    date=e["date"],
                    word_count=e.get("word_count", 0),
                    has_content=e.get("has_content", False),
                )
                for e in result.get("entries", [])
            ]
            return tool_service_pb2.ListJournalEntriesResponse(
                success=True,
                entries=entries,
                total=result.get("total", 0),
            )
        except Exception as e:
            logger.error("ListJournalEntries error: %s", e)
            return tool_service_pb2.ListJournalEntriesResponse(
                success=False,
                total=0,
                error=str(e),
            )

    async def SearchJournal(
        self,
        request: tool_service_pb2.SearchJournalRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SearchJournalResponse:
        """Search within journal entry content in a date range."""
        try:
            from services.org_journal_service import get_org_journal_service
            svc = await get_org_journal_service()
            result = await svc.search_journal_entries(
                request.user_id,
                request.query or "",
                start_date=request.start_date if request.HasField("start_date") and request.start_date else None,
                end_date=request.end_date if request.HasField("end_date") and request.end_date else None,
            )
            if not result.get("success"):
                return tool_service_pb2.SearchJournalResponse(
                    success=False,
                    count=0,
                    error=result.get("error") or "",
                )
            results = [
                tool_service_pb2.JournalSearchResult(
                    date=r["date"],
                    excerpt=r.get("excerpt", ""),
                )
                for r in result.get("results", [])
            ]
            return tool_service_pb2.SearchJournalResponse(
                success=True,
                results=results,
                count=result.get("count", 0),
            )
        except Exception as e:
            logger.error("SearchJournal error: %s", e)
            return tool_service_pb2.SearchJournalResponse(
                success=False,
                count=0,
                error=str(e),
            )

    async def ToggleOrgInboxItem(
        self,
        request: tool_service_pb2.ToggleOrgInboxItemRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ToggleOrgInboxItemResponse:
        """Toggle DONE status of org inbox item"""
        try:
            logger.info(f"ToggleOrgInboxItem: user={request.user_id}, line={request.line_index}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_toggle_done
            
            result = await org_inbox_toggle_done(line_index=request.line_index, user_id=request.user_id)
            
            if result.get("error"):
                return tool_service_pb2.ToggleOrgInboxItemResponse(
                    success=False,
                    error=result.get("error")
                )
            
            return tool_service_pb2.ToggleOrgInboxItemResponse(
                success=True,
                updated_index=result.get("updated_index", request.line_index),
                new_line=result.get("new_line", "")
            )
            
        except Exception as e:
            logger.error(f"❌ ToggleOrgInboxItem error: {e}")
            return tool_service_pb2.ToggleOrgInboxItemResponse(
                success=False,
                error=str(e)
            )

    async def UpdateOrgInboxItem(
        self,
        request: tool_service_pb2.UpdateOrgInboxItemRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.UpdateOrgInboxItemResponse:
        """Update org inbox item text"""
        try:
            logger.info(f"UpdateOrgInboxItem: user={request.user_id}, line={request.line_index}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_update_line
            
            result = await org_inbox_update_line(
                line_index=request.line_index,
                new_text=request.new_text,
                user_id=request.user_id
            )
            
            if result.get("error"):
                return tool_service_pb2.UpdateOrgInboxItemResponse(
                    success=False,
                    error=result.get("error")
                )
            
            return tool_service_pb2.UpdateOrgInboxItemResponse(
                success=True,
                updated_index=result.get("updated_index", request.line_index),
                new_line=result.get("new_line", "")
            )
            
        except Exception as e:
            logger.error(f"❌ UpdateOrgInboxItem error: {e}")
            return tool_service_pb2.UpdateOrgInboxItemResponse(
                success=False,
                error=str(e)
            )

    async def SetOrgInboxSchedule(
        self,
        request: tool_service_pb2.SetOrgInboxScheduleRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.SetOrgInboxScheduleResponse:
        """Set schedule and repeater for org inbox item"""
        try:
            logger.info(f"SetOrgInboxSchedule: user={request.user_id}, line={request.line_index}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_set_schedule_and_repeater
            
            result = await org_inbox_set_schedule_and_repeater(
                line_index=request.line_index,
                scheduled=request.scheduled,
                repeater=request.repeater if request.repeater else None,
                user_id=request.user_id
            )
            
            if result.get("error"):
                return tool_service_pb2.SetOrgInboxScheduleResponse(
                    success=False,
                    error=result.get("error")
                )
            
            return tool_service_pb2.SetOrgInboxScheduleResponse(
                success=True,
                updated_index=result.get("updated_index", request.line_index),
                scheduled_line=result.get("scheduled_line", "")
            )
            
        except Exception as e:
            logger.error(f"❌ SetOrgInboxSchedule error: {e}")
            return tool_service_pb2.SetOrgInboxScheduleResponse(
                success=False,
                error=str(e)
            )

    async def ApplyOrgInboxTags(
        self,
        request: tool_service_pb2.ApplyOrgInboxTagsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ApplyOrgInboxTagsResponse:
        """Apply tags to org inbox item"""
        try:
            logger.info(f"ApplyOrgInboxTags: user={request.user_id}, line={request.line_index}, tags={list(request.tags)}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_apply_tags
            
            result = await org_inbox_apply_tags(
                line_index=request.line_index,
                tags=list(request.tags),
                user_id=request.user_id
            )
            
            if result.get("error"):
                return tool_service_pb2.ApplyOrgInboxTagsResponse(
                    success=False,
                    error=result.get("error")
                )
            
            return tool_service_pb2.ApplyOrgInboxTagsResponse(
                success=True,
                applied_tags=list(request.tags)
            )
            
        except Exception as e:
            logger.error(f"❌ ApplyOrgInboxTags error: {e}")
            return tool_service_pb2.ApplyOrgInboxTagsResponse(
                success=False,
                error=str(e)
            )

    async def ArchiveOrgInboxDone(
        self,
        request: tool_service_pb2.ArchiveOrgInboxDoneRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ArchiveOrgInboxDoneResponse:
        """Archive all DONE items from org inbox"""
        try:
            logger.info(f"ArchiveOrgInboxDone: user={request.user_id}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_archive_done
            
            result = await org_inbox_archive_done(request.user_id)
            
            if result.get("error"):
                return tool_service_pb2.ArchiveOrgInboxDoneResponse(
                    success=False,
                    error=result.get("error")
                )
            
            archived_count = result.get("archived_count", 0)
            
            return tool_service_pb2.ArchiveOrgInboxDoneResponse(
                success=True,
                archived_count=archived_count,
                message=f"Archived {archived_count} DONE items"
            )
            
        except Exception as e:
            logger.error(f"❌ ArchiveOrgInboxDone error: {e}")
            return tool_service_pb2.ArchiveOrgInboxDoneResponse(
                success=False,
                error=str(e)
            )

    async def AppendOrgInboxText(
        self,
        request: tool_service_pb2.AppendOrgInboxTextRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.AppendOrgInboxTextResponse:
        """Append raw org-mode text to inbox"""
        try:
            logger.info(f"AppendOrgInboxText: user={request.user_id}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_append_text
            
            result = await org_inbox_append_text(request.text, request.user_id)
            
            if result.get("error"):
                return tool_service_pb2.AppendOrgInboxTextResponse(
                    success=False,
                    error=result.get("error")
                )
            
            return tool_service_pb2.AppendOrgInboxTextResponse(
                success=True,
                message="Text appended to inbox.org"
            )
            
        except Exception as e:
            logger.error(f"❌ AppendOrgInboxText error: {e}")
            return tool_service_pb2.AppendOrgInboxTextResponse(
                success=False,
                error=str(e)
            )

    async def GetOrgInboxPath(
        self,
        request: tool_service_pb2.GetOrgInboxPathRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.GetOrgInboxPathResponse:
        """Get path to user's inbox.org file"""
        try:
            logger.info(f"GetOrgInboxPath: user={request.user_id}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_path
            
            path = await org_inbox_path(request.user_id)
            
            return tool_service_pb2.GetOrgInboxPathResponse(
                success=True,
                path=path
            )
            
        except Exception as e:
            logger.error(f"❌ GetOrgInboxPath error: {e}")
            return tool_service_pb2.GetOrgInboxPathResponse(
                success=False,
                error=str(e)
            )

    # ===== Universal Todo Operations =====

    async def ListTodos(
        self,
        request: tool_service_pb2.ListTodosRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ListTodosResponse:
        """List todos; scope is all, inbox, or file path."""
        logger.info("ListTodos: user=%s scope=%s query=%s", request.user_id, request.scope or "all", (request.query or "")[:80])
        try:
            from services.org_todo_service import get_org_todo_service
            service = await get_org_todo_service()
            # Proto default limit=0: no cap — return all matching todos for tools/agents.
            result = await service.list_todos(
                user_id=request.user_id,
                scope=request.scope or "all",
                states=list(request.states) if request.states else None,
                tags=list(request.tags) if request.tags else None,
                query=request.query or "",
                limit=int(request.limit),
                include_archives=request.include_archives or False,
                include_body=getattr(request, "include_body", False) or False,
                closed_since_days=request.closed_since_days if getattr(request, "closed_since_days", 0) > 0 else None,
            )
            if not result.get("success"):
                return tool_service_pb2.ListTodosResponse(success=False, error=result.get("error", ""))
            response = tool_service_pb2.ListTodosResponse(success=True, count=result.get("count", 0), files_searched=result.get("files_searched", 0))
            for r in result.get("results", []):
                response.results.append(tool_service_pb2.TodoResult(
                    filename=r.get("filename", ""),
                    file_path=r.get("file_path", ""),
                    heading=r.get("heading", ""),
                    level=r.get("level", 0),
                    line_number=r.get("line_number", 0),
                    todo_state=r.get("todo_state", ""),
                    tags=r.get("tags", []),
                    scheduled=r.get("scheduled", "") or "",
                    deadline=r.get("deadline", "") or "",
                    document_id=r.get("document_id", "") or "",
                    preview=r.get("preview", "") or "",
                    body=r.get("body", "") or "",
                    closed=r.get("closed", "") or "",
                ))
            return response
        except Exception as e:
            logger.exception("ListTodos error")
            return tool_service_pb2.ListTodosResponse(success=False, error=str(e))

    async def CreateTodo(
        self,
        request: tool_service_pb2.CreateTodoRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.CreateTodoResponse:
        logger.info("CreateTodo: user=%s text=%s", request.user_id, (request.text or "")[:50])
        try:
            from services.org_todo_service import get_org_todo_service
            service = await get_org_todo_service()
            has_hl = getattr(request, "HasField", lambda _: False)("heading_level")
            has_ins = getattr(request, "HasField", lambda _: False)("insert_after_line_number")
            heading_level = getattr(request, "heading_level", None) if has_hl else None
            insert_after = getattr(request, "insert_after_line_number", None) if has_ins else None
            result = await service.create_todo(
                user_id=request.user_id,
                text=request.text,
                file_path=request.file_path if request.file_path else None,
                state=request.state or "TODO",
                tags=list(request.tags) if request.tags else None,
                scheduled=request.scheduled if request.scheduled else None,
                deadline=request.deadline if request.deadline else None,
                priority=request.priority if request.priority else None,
                body=(getattr(request, "body", "") or "").strip() or None,
                heading_level=heading_level,
                insert_after_line_number=insert_after,
            )
            if not result.get("success"):
                return tool_service_pb2.CreateTodoResponse(success=False, error=result.get("error", ""))
            return tool_service_pb2.CreateTodoResponse(
                success=True,
                file_path=result.get("file_path", ""),
                line_number=result.get("line_number", 0),
                heading=result.get("heading", ""),
            )
        except Exception as e:
            logger.exception("CreateTodo error")
            return tool_service_pb2.CreateTodoResponse(success=False, error=str(e))

    async def UpdateTodo(
        self,
        request: tool_service_pb2.UpdateTodoRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.UpdateTodoResponse:
        logger.info("UpdateTodo: user=%s file_path=%s line_number=%s new_state=%s", request.user_id, request.file_path, request.line_number, request.new_state or "")
        try:
            from services.org_todo_service import get_org_todo_service
            service = await get_org_todo_service()
            result = await service.update_todo(
                user_id=request.user_id,
                file_path=request.file_path,
                line_number=request.line_number,
                heading_text=request.heading_text if request.heading_text else None,
                new_state=request.new_state if request.new_state else None,
                new_text=request.new_text if request.new_text else None,
                add_tags=list(request.add_tags) if request.add_tags else None,
                remove_tags=list(request.remove_tags) if request.remove_tags else None,
                scheduled=request.scheduled if request.scheduled else None,
                deadline=request.deadline if request.deadline else None,
                priority=request.priority if request.priority else None,
                new_body=(getattr(request, "new_body", "") or "").strip() or None,
            )
            if not result.get("success"):
                return tool_service_pb2.UpdateTodoResponse(success=False, error=result.get("error", ""))
            return tool_service_pb2.UpdateTodoResponse(
                success=True,
                file_path=result.get("file_path", ""),
                line_number=result.get("line_number", 0),
                new_line=result.get("new_line", ""),
            )
        except Exception as e:
            logger.exception("UpdateTodo error")
            return tool_service_pb2.UpdateTodoResponse(success=False, error=str(e))

    async def ToggleTodo(
        self,
        request: tool_service_pb2.ToggleTodoRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ToggleTodoResponse:
        logger.info("ToggleTodo: user=%s file_path=%s line_number=%s", request.user_id, request.file_path, request.line_number)
        try:
            from services.org_todo_service import get_org_todo_service
            service = await get_org_todo_service()
            result = await service.toggle_todo(
                user_id=request.user_id,
                file_path=request.file_path,
                line_number=request.line_number,
                heading_text=request.heading_text if request.heading_text else None,
            )
            if not result.get("success"):
                logger.warning("ToggleTodo failed: %s", result.get("error", ""))
                return tool_service_pb2.ToggleTodoResponse(success=False, error=result.get("error", ""))
            return tool_service_pb2.ToggleTodoResponse(
                success=True,
                file_path=result.get("file_path", ""),
                line_number=result.get("line_number", 0),
                new_line=result.get("new_line", ""),
            )
        except Exception as e:
            logger.exception("ToggleTodo error")
            return tool_service_pb2.ToggleTodoResponse(success=False, error=str(e))

    async def DeleteTodo(
        self,
        request: tool_service_pb2.DeleteTodoRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DeleteTodoResponse:
        logger.info("DeleteTodo: user=%s file_path=%s line_number=%s", request.user_id, request.file_path, request.line_number)
        try:
            from services.org_todo_service import get_org_todo_service
            service = await get_org_todo_service()
            result = await service.delete_todo(
                user_id=request.user_id,
                file_path=request.file_path,
                line_number=request.line_number,
                heading_text=request.heading_text if request.heading_text else None,
            )
            if not result.get("success"):
                return tool_service_pb2.DeleteTodoResponse(success=False, error=result.get("error", ""))
            return tool_service_pb2.DeleteTodoResponse(
                success=True,
                file_path=result.get("file_path", ""),
                deleted_line_count=result.get("deleted_line_count", 0),
            )
        except Exception as e:
            logger.exception("DeleteTodo error")
            return tool_service_pb2.DeleteTodoResponse(success=False, error=str(e))

    async def ArchiveDoneTodos(
        self,
        request: tool_service_pb2.ArchiveDoneTodosRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ArchiveDoneTodosResponse:
        logger.info(
            "ArchiveDoneTodos: user=%s file_path=%s preview_only=%s line_number=%s",
            request.user_id, request.file_path or "inbox", getattr(request, "preview_only", False),
            getattr(request, "line_number", None),
        )
        try:
            from services.org_todo_service import get_org_todo_service
            service = await get_org_todo_service()
            line_number = None
            if hasattr(request, "line_number") and request.HasField("line_number"):
                line_number = request.line_number
            result = await service.archive_done(
                user_id=request.user_id,
                file_path=request.file_path if request.file_path else None,
                preview_only=getattr(request, "preview_only", False),
                line_number=line_number,
            )
            if result.get("error"):
                return tool_service_pb2.ArchiveDoneTodosResponse(success=False, error=result.get("error", ""))
            resp = tool_service_pb2.ArchiveDoneTodosResponse(
                success=True,
                path=result.get("path", ""),
                archived_to=result.get("archived_to", ""),
                archived_count=result.get("archived_count", 0),
            )
            if hasattr(resp, "directive_found"):
                resp.directive_found = result.get("directive_found", False)
            if hasattr(resp, "directive_value"):
                resp.directive_value = result.get("directive_value", "")
            return resp
        except Exception as e:
            logger.exception("ArchiveDoneTodos error")
            return tool_service_pb2.ArchiveDoneTodosResponse(success=False, error=str(e))

    async def RefileTodo(
        self,
        request: tool_service_pb2.RefileTodoRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.RefileTodoResponse:
        """Move a todo entry (and its subtree) from one org file to another."""
        logger.info("RefileTodo: user=%s source=%s:%s target=%s", request.user_id, request.source_file, request.source_line, request.target_file)
        try:
            from services.org_refile_service import get_org_refile_service
            service = await get_org_refile_service()
            target_heading_line = None
            if request.HasField("target_heading_line"):
                target_heading_line = request.target_heading_line + 1
            result = await service.refile_entry(
                user_id=request.user_id,
                source_file=request.source_file,
                source_line=request.source_line + 1,
                target_file=request.target_file,
                target_heading_line=target_heading_line,
            )
            if not result.get("success"):
                return tool_service_pb2.RefileTodoResponse(
                    success=False,
                    source_file=result.get("source_file", request.source_file),
                    target_file=result.get("target_file", request.target_file),
                    lines_moved=0,
                    error=result.get("error", "Unknown error"),
                )
            return tool_service_pb2.RefileTodoResponse(
                success=True,
                source_file=result.get("source_file", request.source_file),
                target_file=result.get("target_file", request.target_file),
                lines_moved=result.get("lines_moved", 0),
            )
        except Exception as e:
            logger.exception("RefileTodo error")
            return tool_service_pb2.RefileTodoResponse(
                success=False,
                source_file=request.source_file,
                target_file=request.target_file,
                lines_moved=0,
                error=str(e),
            )

    async def DiscoverRefileTargets(
        self,
        request: tool_service_pb2.DiscoverRefileTargetsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DiscoverRefileTargetsResponse:
        """List all org files and headings available as refile destinations."""
        logger.info("DiscoverRefileTargets: user=%s", request.user_id)
        try:
            from services.org_refile_service import get_org_refile_service
            service = await get_org_refile_service()
            targets = await service.discover_refile_targets(request.user_id)
            response = tool_service_pb2.DiscoverRefileTargetsResponse(success=True)
            for t in targets:
                heading_line = t.get("heading_line", 0)
                if heading_line > 0:
                    heading_line -= 1
                response.targets.append(tool_service_pb2.RefileTarget(
                    file=t.get("file", ""),
                    filename=t.get("filename", ""),
                    heading_path=t.get("heading_path", []),
                    heading_line=heading_line,
                    display_name=t.get("display_name", ""),
                    level=t.get("level", 0),
                ))
            return response
        except Exception as e:
            logger.exception("DiscoverRefileTargets error")
            return tool_service_pb2.DiscoverRefileTargetsResponse(success=False, error=str(e))
