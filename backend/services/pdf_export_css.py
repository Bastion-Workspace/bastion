"""
Print CSS for PDF export (WeasyPrint): article layout and book layout with named pages.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from models.api_models import PdfBookExportOptions

from services.pdf_export_css_fragments import (
    TOC_CSS_BLOCK,
    base_typography_css,
    watermark_css_block,
)
from services.pdf_font_presets import get_pdf_font_stacks


def _page_dimensions_mm(page_size: str, orientation: str) -> Tuple[str, str]:
    if page_size.lower() == "a4":
        page_width = "210mm"
        page_height = "297mm"
    else:
        page_width = "215.9mm"
        page_height = "279.4mm"
    orient = (orientation or "portrait").lower()
    if orient == "landscape":
        page_width, page_height = page_height, page_width
    return page_width, page_height


def _margin_bottom_for_numbers(
    base_bottom_mm: float, page_numbers: bool, vertical: str
) -> float:
    if not page_numbers:
        return base_bottom_mm
    if (vertical or "bottom").lower() == "top":
        return base_bottom_mm
    return max(base_bottom_mm, 28.0)


def _page_number_content_expr(fmt: str) -> str:
    f = (fmt or "n_of_total").lower().replace("-", "_")
    if f in ("n_only", "n"):
        return "counter(page)"
    return 'counter(page) " of " counter(pages)'


def _margin_box_at(
    vertical: str, horizontal: str, content_expr: str, font_size: str = "9pt"
) -> str:
    v = (vertical or "bottom").lower()
    h = (horizontal or "center").lower()
    if v not in ("top", "bottom"):
        v = "bottom"
    if h not in ("left", "center", "right"):
        h = "center"
    box = f"@top-{h}" if v == "top" else f"@bottom-{h}"
    return f"""
    {box} {{
        content: {content_expr};
        font-size: {font_size};
        color: #888;
    }}"""


def _article_page_break_rules(levels_break: List[int], book_mode: bool) -> str:
    if book_mode or not levels_break:
        return ""
    rules = ""
    for lv in levels_break:
        rules += f"""
.main-body h{lv} {{
    page-break-before: always;
}}
.main-body > h{lv}:first-child {{
    page-break-before: avoid;
}}"""
    return rules


def build_article_print_css(
    page_size: str = "letter",
    orientation: str = "portrait",
    *,
    page_numbers: bool = True,
    watermark_text: Optional[str] = None,
    watermark_on_all_pages: bool = True,
    page_break_before_headings: Optional[List[int]] = None,
    include_toc_style: bool = False,
    font_preset: str = "liberation",
    typeface_style: str = "mixed",
) -> str:
    """Original single-@page article layout."""
    body_f, head_f = get_pdf_font_stacks(font_preset, typeface_style)
    _wm = (watermark_text or "").strip()
    typo = base_typography_css(
        body_f,
        head_f,
        body_extra_css="position: relative;" if _wm else "",
    )
    page_width, page_height = _page_dimensions_mm(page_size, orientation)
    bottom_margin = "28mm" if page_numbers else "20mm"
    bottom_center = ""
    if page_numbers:
        bottom_center = """
    @bottom-center {
        content: counter(page) " of " counter(pages);
        font-size: 9pt;
        color: #888;
    }"""

    levels_break = sorted(
        {int(x) for x in (page_break_before_headings or []) if 1 <= int(x) <= 6}
    )
    page_break_rules = _article_page_break_rules(levels_break, book_mode=False)

    toc_css = TOC_CSS_BLOCK if include_toc_style else ""
    wm_css = watermark_css_block(head_f) if _wm else ""

    return f"""
@page {{
    size: {page_width} {page_height};
    margin: 20mm 20mm {bottom_margin} 20mm;
{bottom_center}
}}

{typo}
{toc_css}
{wm_css}
{page_break_rules}
"""


def _empty_margin_boxes() -> str:
    return """
    @bottom-left { content: none; }
    @bottom-center { content: none; }
    @bottom-right { content: none; }
    @top-left { content: none; }
    @top-center { content: none; }
    @top-right { content: none; }
"""


def build_book_print_css(
    page_size: str,
    orientation: str,
    book_opts: PdfBookExportOptions,
    *,
    global_page_numbers: bool,
    watermark_text: Optional[str],
    include_toc_style: bool,
    template_for_page_name: Dict[str, Tuple[bool, bool]],
    suppress_first_doc_page: bool,
    font_preset: str = "liberation",
    typeface_style: str = "mixed",
) -> str:
    """
    Book layout: default @page for TOC/anonymous flow; named @page per segment template.
    template_for_page_name maps page name -> (show_page_numbers, plain_first_page).
    """
    body_f, head_f = get_pdf_font_stacks(font_preset, typeface_style)
    _wm = (watermark_text or "").strip()
    typo = base_typography_css(
        body_f,
        head_f,
        body_extra_css="position: relative;" if _wm else "",
    )
    page_width, page_height = _page_dimensions_mm(page_size, orientation)
    mt = book_opts.margin_top_mm
    mr = book_opts.margin_right_mm
    mb_base = book_opts.margin_bottom_mm
    ml = book_opts.margin_left_mm

    fmt = book_opts.page_number_format
    vert = book_opts.page_number_vertical
    horiz = book_opts.page_number_horizontal
    content_expr = _page_number_content_expr(fmt)

    toc_css = TOC_CSS_BLOCK if include_toc_style else ""
    wm_css = watermark_css_block(head_f) if _wm else ""

    indent_css = ""
    if book_opts.indent_body_paragraphs:
        indent_css += """
.main-body.book .pdf-book-segment p { text-indent: 1.25em; }
"""
    if book_opts.no_indent_after_section_heading:
        indent_css += """
.main-body.book .pdf-book-segment h1 + p,
.main-body.book .pdf-book-segment h2 + p,
.main-body.book .pdf-book-segment h3 + p,
.main-body.book .pdf-book-segment h4 + p,
.main-body.book .pdf-book-segment h5 + p,
.main-body.book .pdf-book-segment h6 + p { text-indent: 0; }
"""

    book_segment_rules = """
.main-body.book .pdf-book-segment {
    page-break-before: always;
}
.main-body.book .pdf-book-segment.pdf-book-segment-first {
    page-break-before: auto;
}
"""

    anon_mb = _margin_bottom_for_numbers(mb_base, global_page_numbers, vert)
    anon_margin = f"{mt}mm {mr}mm {anon_mb}mm {ml}mm"
    anon_boxes = ""
    if global_page_numbers:
        anon_boxes = _margin_box_at(vert, horiz, content_expr)

    parts: List[str] = [
        f"""
@page {{
    size: {page_width} {page_height};
    margin: {anon_margin};
{anon_boxes}
}}"""
    ]

    for page_name, (show_nums, plain_first) in sorted(template_for_page_name.items()):
        effective = show_nums and global_page_numbers
        mb = _margin_bottom_for_numbers(mb_base, effective, vert)
        margin = f"{mt}mm {mr}mm {mb}mm {ml}mm"
        margin_boxes = ""
        if effective:
            margin_boxes = _margin_box_at(vert, horiz, content_expr)
        parts.append(f"""
@page {page_name} {{
    size: {page_width} {page_height};
    margin: {margin};
{margin_boxes}
}}""")
        if plain_first and effective:
            parts.append(f"""
@page {page_name} :first {{
    size: {page_width} {page_height};
    margin: {margin};
{_empty_margin_boxes()}
}}""")

    doc_first_rule = ""
    if suppress_first_doc_page and global_page_numbers:
        doc_first_rule = f"""
@page :first {{
{_empty_margin_boxes()}
}}
"""

    css = (
        "\n".join(parts)
        + doc_first_rule
        + "\n"
        + typo
        + indent_css
        + book_segment_rules
        + toc_css
        + wm_css
    )
    return css
