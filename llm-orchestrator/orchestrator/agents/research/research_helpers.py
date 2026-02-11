"""
Standalone helper functions for the research agent (no agent instance required).
"""

from typing import Any, Dict, List, Optional


def is_local_search_intent(query: str) -> bool:
    """
    Detect if the user is asking about their own content (local docs/photos) rather than general knowledge.
    Such queries must run local search and must not short-circuit on quick answer from general knowledge.
    """
    if not query or not query.strip():
        return False
    q = query.lower().strip()
    local_phrases = [
        "do we have",
        "do i have",
        "any photos we have",
        "any documents we have",
        "our photos",
        "our documents",
        "our collection",
        "in our library",
        "in our",
        "my photos",
        "my documents",
        "my collection",
        "have we got",
        "have i got",
        "do we have any",
        "do i have any",
        "anything we have",
        "what we have",
        "what i have",
    ]
    return any(phrase in q for phrase in local_phrases)


def detect_research_tier(query: str, metadata: Dict[str, Any]) -> str:
    """
    Lightweight tier detection without LLM calls.

    Returns:
        "fast" - Simple existence queries
        "standard" - Content queries needing synthesis
        "web" - Queries requiring latest/current information
    """
    if not query or not query.strip():
        return "standard"
    query_lower = query.lower().strip()

    fast_indicators = [
        "do we have",
        "do i have",
        "show me",
        "display",
        "any photos",
        "any documents",
        "our photos",
        "our documents",
        "have we got",
        "have i got",
    ]
    if any(phrase in query_lower for phrase in fast_indicators):
        return "fast"

    web_indicators = ["latest", "current", "recent", "now", "today", "2026"]
    if any(word in query_lower for word in web_indicators):
        return "web"

    return "standard"


def doc_display_name(doc: Dict[str, Any]) -> str:
    """Return display name for a document (title or filename). Handles flat search results."""
    doc_inner = doc.get("document") or {}
    return (
        doc.get("title")
        or doc.get("filename")
        or doc_inner.get("title")
        or doc_inner.get("filename")
        or doc.get("metadata", {}).get("title")
        or doc.get("metadata", {}).get("image_filename")
        or "Untitled"
    )


def doc_filename(doc: Dict[str, Any]) -> str:
    """Return filename for a document. Handles flat search results."""
    doc_inner = doc.get("document") or {}
    return (
        doc.get("filename")
        or doc_inner.get("filename")
        or doc.get("metadata", {}).get("image_filename")
        or doc.get("metadata", {}).get("filename")
        or ""
    )


def is_image_document(doc: Dict[str, Any]) -> bool:
    """True if document is an image file by extension."""
    filename = doc_filename(doc)
    if not filename:
        return False
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    return ext in ("webp", "png", "jpg", "jpeg", "gif", "bmp", "svg", "heic")


def build_fast_path_response(
    documents: List[Dict[str, Any]],
    structured_images: Optional[List[Dict[str, Any]]],
    image_results: Optional[str],
) -> tuple:
    """
    Build rich response text and structured_images for fast path.

    Returns (response_text, structured_images, images_markdown).
    """
    if not documents and not image_results:
        return "No matching items found in your collection.", structured_images or None, ""

    images_for_markdown = list(structured_images) if structured_images else []

    if not images_for_markdown and documents:
        image_docs = [d for d in documents if is_image_document(d)]
        for doc in image_docs:
            doc_id = doc.get("document_id") or (doc.get("document") or {}).get("document_id")
            if not doc_id:
                continue
            display_name = doc_display_name(doc)
            filename = doc_filename(doc)
            url = f"/api/documents/{doc_id}/file"
            images_for_markdown.append({
                "url": url,
                "alt_text": display_name or filename or "Image",
                "type": "search_result",
                "metadata": {
                    "document_id": doc_id,
                    "filename": filename,
                    "title": display_name,
                },
            })
        if images_for_markdown:
            structured_images = images_for_markdown

    lines = []
    count = len(documents) if documents else len(images_for_markdown)
    lines.append(f"Found {count} relevant item(s) in your collection:")
    if images_for_markdown:
        for i, img in enumerate(images_for_markdown, 1):
            meta = img.get("metadata", {}) if isinstance(img, dict) else {}
            title = meta.get("title") or meta.get("filename") or img.get("alt_text") or f"Image {i}"
            lines.append(f"- **{title}**")
    for doc in (documents or [])[:10]:
        name = doc_display_name(doc)
        if is_image_document(doc) and images_for_markdown:
            continue
        preview = ""
        content = doc.get("content_preview") or doc.get("full_content") or ""
        if content and isinstance(content, str) and len(content) > 20:
            preview = content[:200].strip() + "..." if len(content) > 200 else content.strip()
        if preview:
            lines.append(f"- **{name}**: {preview}")
        else:
            lines.append(f"- **{name}**")
    response_text = "\n".join(lines)

    images_markdown = ""
    if images_for_markdown:
        images_markdown = "\n".join(
            f"![]({img.get('url', '')})" for img in images_for_markdown if img.get("url")
        )

    return response_text, structured_images, images_markdown
