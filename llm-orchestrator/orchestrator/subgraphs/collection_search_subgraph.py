"""
Collection Search Subgraph - Fast path for collection-specific queries.

Executes hybrid search (vector concept + SQL filters) via search_images_tool,
then synthesizes a short description from metadata (including tags) that relates
results back to the user's query. Uses the user's selected chat model for synthesis.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from orchestrator.agents.base_agent import BaseAgent
from orchestrator.models.query_classification_models import QueryPlan
from orchestrator.tools.image_search_tools import search_images_tool
from orchestrator.models.agent_response_contract import AgentResponse, TaskStatus

logger = logging.getLogger(__name__)


async def execute_collection_search(
    plan: QueryPlan,
    state: Dict[str, Any],
    original_query: str,
) -> Dict[str, Any]:
    """
    Run hybrid collection search: semantic concept + structured filters (series, type, author, date).
    Uses search_images_tool which performs vector search and SQL type/series filtering in the backend.
    """
    user_id = state.get("user_id", "system")
    query_for_vector = (plan.concept or original_query).strip()
    if not query_for_vector:
        query_for_vector = original_query

    image_type = plan.content_type or None
    series = plan.series or None
    author = plan.author or None
    date_range = plan.date_range or None
    raw_limit = plan.requested_count if getattr(plan, "requested_count", None) is not None else 10
    limit = max(1, min(50, int(raw_limit))) if isinstance(raw_limit, (int, float)) else 10

    shared_memory = state.get("shared_memory", {})
    exclude_ids = shared_memory.get("shown_document_ids", [])

    logger.info(
        f"Collection search: concept={query_for_vector}, series={series}, type={image_type}, author={author}, date={date_range}"
    )

    try:
        result = await search_images_tool(
            query=query_for_vector,
            image_type=image_type,
            date=date_range,
            author=author,
            series=series,
            limit=limit,
            user_id=user_id,
            is_random=False,
            exclude_document_ids=exclude_ids if exclude_ids else None,
        )
    except Exception as e:
        logger.error(f"Collection search failed: {e}")
        return AgentResponse(
            response=f"I couldn't search your collection: {str(e)}",
            task_status=TaskStatus.ERROR,
            agent_type="research_agent",
            timestamp=datetime.utcnow().isoformat() + "Z",
            error=str(e),
        ).model_dump(exclude_none=True)

    images_markdown = result.get("images_markdown", "")
    metadata_list = result.get("metadata", [])
    structured_images = result.get("images")
    expanded_to_all_types = False

    # Treat "No images found" message as no results (backend returns that string in images_markdown).
    no_actual_results = (
        (not metadata_list and not structured_images)
        or ("No images found" in (images_markdown or ""))
    )

    # If no results and we filtered by type, retry without type filter so we don't overlook
    # images that weren't tagged (e.g. user forgot to set type: photo).
    if no_actual_results and image_type:
        try:
            logger.info(f"Collection search: no results for type '{image_type}', retrying without type filter")
            fallback = await search_images_tool(
                    query=query_for_vector,
                    image_type=None,
                    date=date_range,
                    author=author,
                    series=series,
                    limit=limit,
                    user_id=user_id,
                    is_random=False,
                    exclude_document_ids=exclude_ids if exclude_ids else None,
                )
            fm = fallback.get("images_markdown", "")
            flist = fallback.get("metadata", [])
            fstruct = fallback.get("images")
            if (fm and "No images found" not in fm) or fstruct:
                result = fallback
                images_markdown = fm
                metadata_list = flist
                structured_images = fstruct
                expanded_to_all_types = True
        except Exception as e:
            logger.warning(f"Fallback search without type filter failed: {e}")

    # After fallback, again treat "No images found" as no results for final response
    no_actual_results = (
        (not metadata_list and not structured_images)
        or ("No images found" in (images_markdown or ""))
    )
    if no_actual_results:
        no_match_msg = result.get("images_markdown", "")
        if no_match_msg and "No images found" in no_match_msg:
            response_text = no_match_msg
        else:
            filters = []
            if series:
                filters.append(f"series '{series}'")
            if image_type:
                filters.append(f"type '{image_type}'")
            if author:
                filters.append(f"author '{author}'")
            if date_range:
                filters.append(f"date '{date_range}'")
            filter_str = " and ".join(filters) if filters else "your filters"
            response_text = f"No items in your collection match '{query_for_vector}' with {filter_str}."
        return AgentResponse(
            response=response_text,
            task_status=TaskStatus.COMPLETE,
            agent_type="research_agent",
            timestamp=datetime.utcnow().isoformat() + "Z",
            images=structured_images if structured_images else None,
        ).model_dump(exclude_none=True)

    # Add match_reason to each structured image (why it matches the query)
    query_terms = [t.lower() for t in original_query.split() if len(t) > 2]
    for i, img in enumerate(structured_images or []):
        img_meta = (img.get("metadata") or {}).copy()
        meta = metadata_list[i] if i < len(metadata_list) else {}
        content_lower = (meta.get("content") or "").lower()
        tags_list = meta.get("tags") or []
        tags_text = " ".join(tags_list).lower() if isinstance(tags_list, list) else ""
        reasons = []
        for t in query_terms:
            if t in content_lower or t in tags_text:
                reasons.append(f"'{t}' in description")
        if reasons:
            img_meta["match_reason"] = "; ".join(reasons)
        img_meta["content"] = meta.get("content", "")  # Full description for modal
        img["metadata"] = img_meta

    # Build rich response with metadata descriptions
    count = len(metadata_list) or len(structured_images or [])
    if expanded_to_all_types:
        intro = f"No images of type '{image_type}' matched; here are {count} matching item(s) from your collection (all image types)."
    else:
        intro = f"Here are {count} matching item(s) from your collection."

    # Synthesize a short description from metadata that relates results back to the query (uses user's selected chat model)
    synthesis_text = ""
    try:
        items_context = []
        for i, meta in enumerate(metadata_list, 1):
            title = meta.get("title", "Untitled")
            date = meta.get("date", "")
            series = meta.get("series", "")
            author = meta.get("author", "")
            content = (meta.get("content") or "")[:400]
            tags = meta.get("tags") or []
            tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
            items_context.append(
                f"{i}. {title}" + (f" ({date})" if date else "")
                + (f" — {series}" if series else "")
                + (f" by {author}" if author else "")
                + (f" | Tags: {tags_str}" if tags_str else "")
                + (f"\n   Description: {content}" if content else "")
            )
        context_block = "\n".join(items_context)
        base_agent = BaseAgent("collection_synthesis")
        llm = base_agent._get_llm(temperature=0.3, state=state)
        continuation_hint = ""
        if exclude_ids:
            n = len(exclude_ids)
            continuation_hint = f"The user has already seen {n} result(s) from this search. These are ADDITIONAL results. "

        synthesis_prompt = f"""{continuation_hint}The user asked: "{original_query}"

They have these matching items from their collection:

{context_block}

Write a short synthesis (2–4 sentences) that:
- Relates these results back to the user's query.
- Mentions relevant tags or themes that connect to what they asked for.
- Sounds natural and concise, not a raw list.
Do not repeat the full list; the details are shown below."""
        response = await llm.ainvoke([
            SystemMessage(content="You synthesize collection search results into a brief, natural summary that connects the findings to the user's question and highlights relevant tags or themes."),
            HumanMessage(content=synthesis_prompt),
        ])
        synthesis_text = (response.content if hasattr(response, "content") else str(response)).strip()
        if synthesis_text:
            logger.info(f"Collection search synthesis: {len(synthesis_text)} chars")
    except Exception as e:
        logger.warning(f"Collection search synthesis failed (using plain list): {e}")

    # Add descriptions from metadata (full description, no truncation)
    descriptions = []
    for i, meta in enumerate(metadata_list, 1):
        parts = []
        title = meta.get("title", "Untitled")
        date = meta.get("date", "")
        if date and date not in title:
            parts.append(f"**{title}** ({date})")
        else:
            parts.append(f"**{title}**")
        series = meta.get("series", "")
        author = meta.get("author", "")
        if series and author:
            parts.append(f"from *{series}* by {author}")
        elif series:
            parts.append(f"from *{series}*")
        elif author:
            parts.append(f"by {author}")
        content = meta.get("content", "")
        if content:
            parts.append(f"\n  {content}")
        tags = meta.get("tags", [])
        if tags and isinstance(tags, list):
            parts.append(f"\n  *Tags: {', '.join(tags)}*")
        descriptions.append(f"{i}. {' '.join(parts)}")

    # Combine intro, optional synthesis, details
    # NOTE: Frontend uses EITHER structured images OR markdown, not both
    # - If structured_images exist: frontend renders from metadata (no markdown needed); omit descriptions to avoid duplication
    # - If NO structured_images: frontend extracts from markdown in response text; include descriptions
    response_parts = [intro]
    if synthesis_text:
        response_parts.append(synthesis_text)
    if descriptions and not structured_images:
        response_parts.append("\n\n".join(descriptions))
    
    # Only include markdown when we DON'T have structured images
    # (frontend shows images once from structured data when available)
    if not structured_images and images_markdown:
        response_parts.append(images_markdown)

    response_text = "\n\n".join(response_parts)

    newly_shown = [
        img.get("metadata", {}).get("document_id")
        for img in (structured_images or [])
        if img.get("metadata", {}).get("document_id")
    ]
    out = AgentResponse(
        response=response_text,
        task_status=TaskStatus.COMPLETE,
        agent_type="research_agent",
        timestamp=datetime.utcnow().isoformat() + "Z",
        images=structured_images if structured_images else None,
    ).model_dump(exclude_none=True)
    out["_shown_document_ids"] = newly_shown
    return out