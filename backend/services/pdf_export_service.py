"""
PDF Export Service

Builds PDF documents from Markdown, chat messages, or conversations using WeasyPrint.
Uses proper Markdown parsing (not regex) for robust HTML generation.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import markdown
from weasyprint import HTML, CSS

from utils.frontmatter_utils import parse_frontmatter
from models.api_models import (
    PdfBookExportOptions,
    PdfBookSectionOverride,
    PdfExportKind,
    PdfExportLayout,
    PdfExportRequest,
    PdfHeadingOutlineItem,
)
from services.pdf_export_css import build_article_print_css, build_book_print_css
from services.pdf_org_to_markdown import prepare_body_for_pdf_markdown


logger = logging.getLogger(__name__)

# Extensions supported by Python-Markdown without extra deps (no Pygments required).
# Avoids codehilite/toc/sane_lists which can fail if optional deps or entry points differ by version.
_MARKDOWN_EXTENSIONS = [
    "fenced_code",
    "tables",
    "nl2br",
]

# PDF pipeline: no nl2br so block elements (headings, paragraphs) parse correctly for WeasyPrint.
_PDF_MARKDOWN_EXTENSIONS = [
    "fenced_code",
    "tables",
]

# Lines made only of these chars are unbreakable as text and overflow the PDF; map to --- → <hr>.
_RE_ASCII_HR = re.compile(r"^[-*_]{3,}$")
_RE_UNICODE_DECORATIVE_HR = re.compile(
    r"^[\u2500-\u257F\u2010-\u2015\uFE31\uFE58\uFF0D]{3,}$"
)

# Paragraphs that are only an ATX line (ASCII or U+FF03 hashes) after markdown.
_RE_HTML_P_ATX_ONLY = re.compile(
    r"<p>\s*((?:#|\uFF03){1,6})\s+([^<]+?)\s*</p>",
    re.IGNORECASE,
)
_RE_ATX_TITLE_TRAILING_HASHES = re.compile(r"\s+#+\s*$")


def _is_markdown_table_separator_line(line: str) -> bool:
    """True if line is a GFM/PHP-MExtra style column alignment row (|---|--:| etc.)."""
    stripped = line.strip()
    if "|" not in stripped or "-" not in stripped:
        return False
    inner = stripped.replace("|", "")
    return bool(re.match(r"^[\s\-:]+$", inner))


def _looks_like_markdown_table_header_line(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped or _is_markdown_table_separator_line(stripped):
        return False
    return stripped.count("|") >= 2


class PdfExportService:
    @staticmethod
    def _is_decorative_horizontal_rule_line(trimmed: str) -> bool:
        if len(trimmed) < 3:
            return False
        if _RE_ASCII_HR.match(trimmed):
            return True
        return bool(_RE_UNICODE_DECORATIVE_HR.match(trimmed))

    def _normalize_decorative_separator_lines(self, text: str) -> str:
        """
        Replace standalone decorative Unicode/ASCII rule lines with Markdown '---' so they become
        <hr> and respect page width (long ═ runs are one unbreakable text line otherwise).
        Skips lines inside fenced code blocks.
        """
        if not text:
            return text
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n")
        in_fence = False
        out: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                out.append(line)
                continue
            if not in_fence and stripped and self._is_decorative_horizontal_rule_line(stripped):
                out.append("---")
            else:
                out.append(line)
        return "\n".join(out)

    def _merge_broken_pipe_table_rows(self, text: str) -> str:
        """
        Join table rows that were split across lines (common in LLM/editor output).
        Markdown pipe tables require one row per line; otherwise the parser emits
        literal pipes in <p> tags.
        """
        if not text:
            return text
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n")
        out: list[str] = []
        i = 0
        in_fence = False
        max_row_merges = 40

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                out.append(line)
                i += 1
                continue
            if in_fence:
                out.append(line)
                i += 1
                continue

            if (
                i + 1 < len(lines)
                and _looks_like_markdown_table_header_line(lines[i])
                and _is_markdown_table_separator_line(lines[i + 1])
            ):
                out.append(lines[i])
                out.append(lines[i + 1])
                i += 2
                broke_on_non_row = False
                while i < len(lines):
                    row_line = lines[i]
                    if row_line.strip() == "":
                        out.append(row_line)
                        i += 1
                        break
                    if "|" not in row_line:
                        broke_on_non_row = True
                        break
                    merges = 0
                    while (
                        merges < max_row_merges
                        and i + 1 < len(lines)
                        and not row_line.rstrip().endswith("|")
                    ):
                        nxt = lines[i + 1]
                        if nxt.strip() == "":
                            break
                        row_line = row_line.rstrip() + " " + nxt.lstrip()
                        i += 1
                        merges += 1
                    out.append(row_line)
                    i += 1
                if broke_on_non_row:
                    out.append(lines[i])
                    i += 1
                continue

            out.append(line)
            i += 1

        return "\n".join(out)

    def _strip_frontmatter(self, content: str) -> str:
        """Strip YAML frontmatter from markdown content if present."""
        _, body = parse_frontmatter(content)
        return body
    
    def _repair_unparsed_atx_paragraphs(self, html_body: str) -> str:
        """
        Promote <p># Title</p> (or fullwidth ＃) to <h1>-<h6> when markdown left them as paragraphs.
        """

        if not html_body:
            return html_body

        def repl(m: re.Match) -> str:
            raw_hashes = m.group(1).replace("\uff03", "#")
            if not re.fullmatch(r"#+", raw_hashes):
                return m.group(0)
            level = len(raw_hashes)
            if not 1 <= level <= 6:
                return m.group(0)
            inner = m.group(2).strip()
            inner = _RE_ATX_TITLE_TRAILING_HASHES.sub("", inner).strip()
            if not inner:
                return m.group(0)
            esc = html.escape(html.unescape(inner))
            return f"<h{level}>{esc}</h{level}>"

        return _RE_HTML_P_ATX_ONLY.sub(repl, html_body)

    def _markdown_to_html(self, markdown_text: str, source_format: str = "markdown") -> str:
        """Convert markdown to HTML using proper markdown library."""
        if not markdown_text:
            return ""
        prepared = prepare_body_for_pdf_markdown(markdown_text, source_format)
        merged_tables = self._merge_broken_pipe_table_rows(prepared)
        normalized = self._normalize_decorative_separator_lines(merged_tables)
        html_body = markdown.markdown(
            normalized,
            extensions=_PDF_MARKDOWN_EXTENSIONS,
        )
        return self._repair_unparsed_atx_paragraphs(html_body)

    _HEADING_RE = re.compile(r"<h([1-6])(\s[^>]*)?>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)

    @staticmethod
    def _strip_html_inner_text(s: str) -> str:
        t = re.sub(r"<[^>]+>", "", s or "")
        return html.unescape(re.sub(r"\s+", " ", t).strip())

    def _inject_heading_ids(self, html_body: str) -> Tuple[str, List[Tuple[int, str, str]]]:
        """Add stable id attributes to h1-h6 for TOC anchors. Returns (html, list of (level, plain_text, id))."""
        headings: List[Tuple[int, str, str]] = []
        n = [0]

        def repl(m) -> str:
            level_s = m.group(1)
            attrs = m.group(2) or ""
            inner = m.group(3)
            level = int(level_s)
            id_match = re.search(r'\bid\s*=\s*["\']([^"\']+)["\']', attrs, re.I)
            if id_match:
                aid = id_match.group(1)
            else:
                n[0] += 1
                aid = f"toc-{n[0]}"
                gap = "" if attrs.strip() else " "
                attrs = f"{attrs}{gap}id=\"{aid}\""
            text = self._strip_html_inner_text(inner)
            headings.append((level, text, aid))
            return f"<h{level_s}{attrs}>{inner}</h{level_s}>"

        out = self._HEADING_RE.sub(repl, html_body)
        return out, headings

    def _build_toc_html(self, headings: List[Tuple[int, str, str]], toc_depth: int) -> str:
        """Build TOC block with leader dots; page numbers via CSS target-counter on links."""
        items: List[str] = []
        depth = max(1, min(6, int(toc_depth)))
        for level, text, aid in headings:
            if level > depth or not text:
                continue
            esc_title = html.escape(text)
            safe_aid = html.escape(aid).replace('"', "&quot;")
            items.append(
                f'<li class="toc-h{level}">'
                f'<a href="#{safe_aid}">'
                f'<span class="toc-title">{esc_title}</span>'
                f'<span class="toc-leader"></span>'
                f"</a></li>"
            )
        if not items:
            return ""
        return (
            '<div class="toc-wrap">'
            "<h2>Table of Contents</h2>"
            '<ul class="toc-list">'
            + "".join(items)
            + "</ul></div>"
        )

    @staticmethod
    def _inject_watermark_before_body_close(
        html_doc: str, watermark_text: Optional[str], all_pages: bool
    ) -> str:
        raw = (watermark_text or "").strip()
        if not raw:
            return html_doc
        cls = "watermark watermark-all-pages" if all_pages else "watermark watermark-first-page"
        div = f'<div class="{cls}">{html.escape(raw)}</div>'
        if "</body>" not in html_doc:
            return html_doc + div
        return html_doc.replace("</body>", f"{div}</body>", 1)

    @staticmethod
    def _heading_id_from_match(m: re.Match) -> str:
        attrs = m.group(2) or ""
        id_match = re.search(r'\bid\s*=\s*["\']([^"\']+)["\']', attrs, re.I)
        if id_match:
            return id_match.group(1)
        return ""

    def _split_html_book_segments(
        self, html_body: str, break_levels: List[int]
    ) -> List[Tuple[Optional[str], str]]:
        """Split body HTML at headings whose level is in break_levels."""
        levels = {int(x) for x in break_levels if 1 <= int(x) <= 6}
        if not levels:
            return [(None, html_body)]

        matches = list(self._HEADING_RE.finditer(html_body))
        bmatches = [m for m in matches if int(m.group(1)) in levels]
        if not bmatches:
            return [(None, html_body)]

        segments: List[Tuple[Optional[str], str]] = []
        segments.append((None, html_body[: bmatches[0].start()]))
        for i, bm in enumerate(bmatches):
            start = bm.start()
            end = bmatches[i + 1].start() if i + 1 < len(bmatches) else len(html_body)
            hid = self._heading_id_from_match(bm)
            segments.append((hid or None, html_body[start:end]))
        return segments

    @staticmethod
    def _override_map(
        overrides: Optional[List[PdfBookSectionOverride]],
    ) -> Dict[str, PdfBookSectionOverride]:
        out: Dict[str, PdfBookSectionOverride] = {}
        for o in overrides or []:
            if o.heading_id:
                out[o.heading_id] = o
        return out

    @staticmethod
    def _segment_page_style_key(
        heading_id: Optional[str],
        ov_map: Dict[str, PdfBookSectionOverride],
        global_page_numbers: bool,
    ) -> Tuple[bool, bool]:
        nums = global_page_numbers
        plain_first = False
        if heading_id and heading_id in ov_map:
            o = ov_map[heading_id]
            if o.page_numbers is not None:
                nums = o.page_numbers
            plain_first = bool(o.plain_first_page)
        return (nums, plain_first)

    def _wrap_book_segments(
        self,
        segments: List[Tuple[Optional[str], str]],
        ov_map: Dict[str, PdfBookSectionOverride],
        global_page_numbers: bool,
    ) -> Tuple[str, Dict[str, Tuple[bool, bool]]]:
        key_to_name: Dict[Tuple[bool, bool], str] = {}
        template_for_page_name: Dict[str, Tuple[bool, bool]] = {}

        def page_name_for_key(key: Tuple[bool, bool]) -> str:
            if key not in key_to_name:
                name = f"bk{len(key_to_name)}"
                key_to_name[key] = name
                template_for_page_name[name] = key
            return key_to_name[key]

        parts: List[str] = []
        first_segment = True
        for hid, chunk in segments:
            if not (chunk or "").strip():
                continue
            key = self._segment_page_style_key(hid, ov_map, global_page_numbers)
            pname = page_name_for_key(key)
            cls = "pdf-book-segment"
            if first_segment:
                cls += " pdf-book-segment-first"
                first_segment = False
            parts.append(f'<div class="{cls}" style="page: {pname};">{chunk}</div>')
        inner = f'<div class="main-body book">{"".join(parts)}</div>'
        return inner, template_for_page_name

    def get_heading_outline(
        self, content: str, source_format: str = "markdown"
    ) -> List[PdfHeadingOutlineItem]:
        """Heading list with same ids as PDF export (for book section UI)."""
        body_content = self._strip_frontmatter(content or "")
        html_body = self._markdown_to_html(body_content, source_format)
        _, headings = self._inject_heading_ids(html_body)
        return [
            PdfHeadingOutlineItem(id=aid, level=level, text=text)
            for level, text, aid in headings
            if text
        ]

    def _build_markdown_document_html(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        include_toc: bool = False,
        toc_depth: int = 3,
        page_break_before_headings: Optional[List[int]] = None,
        watermark_text: Optional[str] = None,
        watermark_on_all_pages: bool = True,
        pdf_layout: PdfExportLayout = PdfExportLayout.article,
        book_options: Optional[PdfBookExportOptions] = None,
        book_section_overrides: Optional[List[PdfBookSectionOverride]] = None,
        global_page_numbers: bool = True,
        pdf_source_format: str = "markdown",
    ) -> Tuple[str, Optional[Dict[str, Tuple[bool, bool]]]]:
        """
        Build HTML for markdown document export.
        Returns (full_html, book_template_map_or_none). book_template_map maps @page name -> (show_nums, plain_first).
        """
        body_content = self._strip_frontmatter(content)
        html_body = self._markdown_to_html(body_content, pdf_source_format)
        html_body, headings = self._inject_heading_ids(html_body)

        toc_block = ""
        if include_toc and headings:
            toc_block = self._build_toc_html(headings, toc_depth)

        book_tpl_map: Optional[Dict[str, Tuple[bool, bool]]] = None
        pb_list = list(page_break_before_headings or [])

        if pdf_layout == PdfExportLayout.book:
            ov_map = self._override_map(book_section_overrides)
            segments = self._split_html_book_segments(html_body, pb_list)
            main_inner, book_tpl_map = self._wrap_book_segments(
                segments, ov_map, global_page_numbers
            )
            if "pdf-book-segment" not in main_inner:
                main_inner, book_tpl_map = self._wrap_book_segments(
                    [(None, html_body)], ov_map, global_page_numbers
                )
            inner = f"{toc_block}{main_inner}" if toc_block else main_inner
        else:
            inner = (
                f"{toc_block}<div class=\"main-body\">{html_body}</div>"
                if toc_block
                else f'<div class="main-body">{html_body}</div>'
            )

        title = "Document"
        if metadata and metadata.get("title"):
            title = metadata.get("title")
        else:
            h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", html_body, re.IGNORECASE | re.DOTALL)
            if h1_match:
                title = self._strip_html_inner_text(h1_match.group(1))

        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{html.escape(str(title))}</title>
</head>
<body>
{inner}
</body>
</html>"""

        out = self._inject_watermark_before_body_close(
            html_doc, watermark_text, watermark_on_all_pages
        )
        return out, book_tpl_map
    
    def _build_chat_message_html(self, content: str, timestamp: Optional[str] = None, role: Optional[str] = None) -> str:
        """Build HTML for single chat message export."""
        # Convert markdown to HTML
        html_content = self._markdown_to_html(content)
        
        # Format timestamp
        timestamp_str = ""
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                timestamp_str = timestamp
        
        role_label = role or "Message"
        
        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Chat Message</title>
</head>
<body>
    <div class="chat-header">
        <div class="chat-title">Chat Message</div>
        {f'<div class="chat-meta">{html.escape(timestamp_str)}</div>' if timestamp_str else ''}
    </div>
    <div class="message-card">
        <div class="message-header {role_label.lower() if role_label else ''}">
            {html.escape(role_label)}
            {f'<span class="message-timestamp">{html.escape(timestamp_str)}</span>' if timestamp_str else ''}
        </div>
        <div class="message-content">
            {html_content}
        </div>
    </div>
</body>
</html>"""
        
        return html_doc
    
    def _build_conversation_html(self, title: str, created_at: Optional[str], messages: list) -> str:
        """Build HTML for conversation export."""
        # Format creation timestamp
        created_str = ""
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                created_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                created_str = created_at
        
        # Build message cards
        message_html = []
        for msg in messages:
            role = msg.get("role", "unknown")
            timestamp = msg.get("timestamp", "")
            content = msg.get("content", "")
            
            # Format message timestamp
            msg_timestamp_str = ""
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    msg_timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    msg_timestamp_str = timestamp
            
            # Convert message content markdown to HTML
            msg_html = self._markdown_to_html(content)
            
            role_label = "User" if role == "user" else "Assistant" if role == "assistant" else role.capitalize()
            
            message_html.append(f"""
    <div class="message-card">
        <div class="message-header {role}">
            {html.escape(role_label)}
            {f'<span class="message-timestamp">{html.escape(msg_timestamp_str)}</span>' if msg_timestamp_str else ''}
        </div>
        <div class="message-content">
            {msg_html}
        </div>
    </div>""")
        
        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{html.escape(title)}</title>
</head>
<body>
    <div class="chat-header">
        <div class="chat-title">Conversation: {html.escape(title)}</div>
        {f'<div class="chat-meta">Created: {html.escape(created_str)}</div>' if created_str else ''}
    </div>
    {''.join(message_html)}
</body>
</html>"""
        
        return html_doc
    
    async def export_to_pdf(self, request: PdfExportRequest, user_id: Optional[str] = None) -> bytes:
        """Export content to PDF based on request kind."""
        try:
            include_toc = bool(request.include_toc)
            toc_depth = max(1, min(6, int(request.toc_depth)))
            page_numbers = request.page_numbers is not False
            wm = request.watermark_text
            wm_all = request.watermark_on_all_pages is not False
            pb_levels: List[int] = list(request.page_break_before_headings or [])
            book_tpl_map: Optional[Dict[str, Tuple[bool, bool]]] = None
            font_preset = (getattr(request, "pdf_font_preset", None) or "liberation").lower()
            face_style = (getattr(request, "pdf_typeface_style", None) or "mixed").lower()
            if face_style not in ("serif", "sans", "mixed"):
                face_style = "mixed"
            pdf_src = (getattr(request, "pdf_source_format", None) or "markdown").lower()
            if pdf_src not in ("markdown", "org"):
                pdf_src = "markdown"

            if request.kind == PdfExportKind.markdown_document:
                if not request.content:
                    raise ValueError("content is required for markdown_document export")
                layout = request.pdf_layout
                if isinstance(layout, str):
                    layout = PdfExportLayout(layout)
                html_content, book_tpl_map = self._build_markdown_document_html(
                    request.content,
                    request.metadata,
                    include_toc=include_toc,
                    toc_depth=toc_depth,
                    page_break_before_headings=pb_levels,
                    watermark_text=wm,
                    watermark_on_all_pages=wm_all,
                    pdf_layout=layout,
                    book_options=request.book_options,
                    book_section_overrides=request.book_section_overrides,
                    global_page_numbers=page_numbers,
                    pdf_source_format=pdf_src,
                )
                include_toc_style = include_toc
                pb_for_css = pb_levels
            elif request.kind == PdfExportKind.chat_message:
                if not request.message_content:
                    raise ValueError("message_content is required for chat_message export")
                html_content = self._build_chat_message_html(
                    request.message_content,
                    request.message_timestamp,
                    request.message_role,
                )
                html_content = self._inject_watermark_before_body_close(html_content, wm, wm_all)
                include_toc_style = False
                pb_for_css = []
            elif request.kind == PdfExportKind.conversation:
                if not request.conversation_title or not request.messages:
                    raise ValueError("conversation_title and messages are required for conversation export")
                html_content = self._build_conversation_html(
                    request.conversation_title,
                    request.conversation_created_at,
                    request.messages,
                )
                html_content = self._inject_watermark_before_body_close(html_content, wm, wm_all)
                include_toc_style = False
                pb_for_css = []
            else:
                raise ValueError(f"Unknown export kind: {request.kind}")

            orient = (request.page_orientation or "portrait").lower()
            if orient not in ("portrait", "landscape"):
                orient = "portrait"

            layout = getattr(request, "pdf_layout", PdfExportLayout.article)
            if isinstance(layout, str):
                layout = PdfExportLayout(layout)

            use_book_css = (
                request.kind == PdfExportKind.markdown_document
                and layout == PdfExportLayout.book
                and book_tpl_map is not None
            )
            if use_book_css:
                bo = request.book_options or PdfBookExportOptions()
                css_content = build_book_print_css(
                    request.page_size,
                    orient,
                    bo,
                    global_page_numbers=page_numbers,
                    watermark_text=wm,
                    include_toc_style=include_toc_style,
                    template_for_page_name=book_tpl_map,
                    suppress_first_doc_page=bo.suppress_page_number_on_first_page,
                    font_preset=font_preset,
                    typeface_style=face_style,
                )
            else:
                css_content = build_article_print_css(
                    request.page_size,
                    orient,
                    page_numbers=page_numbers,
                    watermark_text=wm,
                    watermark_on_all_pages=wm_all,
                    page_break_before_headings=pb_for_css,
                    include_toc_style=include_toc_style,
                    font_preset=font_preset,
                    typeface_style=face_style,
                )

            html_doc = HTML(string=html_content)
            css_doc = CSS(string=css_content)
            pdf_bytes = html_doc.write_pdf(stylesheets=[css_doc])

            return pdf_bytes

        except Exception as e:
            logger.error(f"PDF export failed: {e}", exc_info=True)
            raise
