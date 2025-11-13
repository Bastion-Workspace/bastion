"""
EPUB Export Service

Builds an EPUB v3 package from Markdown content with options mirroring the desktop exporter.
Relies on Python stdlib zipfile for correct packaging order and basic HTML/XHTML generation.
"""

from __future__ import annotations

import io
import os
import re
import uuid
import zipfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class EpubExportService:
    def __init__(self):
        # Regexes for simple markdown parsing and resource detection
        self._heading_regex = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        self._image_regex = re.compile(r"!\[[^\]]*\]\(([^\)]+)\)")
        self._link_regex = re.compile(r"\[[^\]]+\]\(([^\)]+)\)")

    async def export_markdown_to_epub(self, content: str, options: Dict) -> bytes:
        # Normalize newlines
        content = (content or "").replace("\r\n", "\n")

        include_toc = bool(options.get("include_toc", True))
        split_on = bool(options.get("split_on_headings", True))
        split_levels = options.get("split_on_heading_levels") or [1, 2]
        metadata = options.get("metadata") or {}
        include_cover = bool(options.get("include_cover", True))
        heading_alignments = options.get("heading_alignments") or {}

        # Resolve cover path from metadata if provided
        cover_href: Optional[str] = None
        cover_src = str(metadata.get("cover") or "").strip() if metadata else ""
        if include_cover and cover_src:
            # Place under images/ with preserved extension
            ext = os.path.splitext(cover_src)[1] or ".jpg"
            cover_href = f"images/cover{ext}"

        # Convert markdown to simple HTML and split chapters
        html = self._convert_markdown_to_html(content)
        chapters = self._split_into_chapters(content, html, split_on, split_levels)
        if not chapters:
            chapters = [(metadata.get("title") or "Chapter 1", html)]

        # Build CSS honoring heading alignments
        css = self._build_css(heading_alignments)

        # Build in-memory EPUB
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            # Write mimetype first with no compression
            zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)

            # META-INF/container.xml
            container_xml = (
                """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
            )
            zf.writestr("META-INF/container.xml", container_xml)

            # OPS/style.css
            zf.writestr("OPS/style.css", css)

            # Cover XHTML and image (image is referenced by href; caller should provide data URI or external path?)
            manifest_items: List[Tuple[str, str, str]] = []  # (id, href, media-type)
            spine_ids: List[str] = []

            # Chapters XHTML
            chapter_files: List[str] = []
            chapter_ids: List[str] = []
            for idx, (title, body_html) in enumerate(chapters, start=1):
                file_name = f"chapter{idx}.xhtml"
                xhtml = self._wrap_xhtml(title, body_html)
                zf.writestr(f"OPS/{file_name}", xhtml)
                chapter_files.append(file_name)
                chap_id = f"chap{idx}"
                chapter_ids.append(chap_id)
                manifest_items.append((chap_id, file_name, "application/xhtml+xml"))

            # TOC nav.xhtml
            nav_id = None
            if include_toc:
                nav_xhtml = self._build_nav_xhtml(chapters, chapter_files)
                zf.writestr("OPS/nav.xhtml", nav_xhtml)
                nav_id = "nav"
                manifest_items.append((nav_id, "nav.xhtml", "application/xhtml+xml"))

            # Cover handling (XHTML only; image file entry if provided as path is not copied here)
            cover_id = None
            if include_cover and cover_href:
                cover_xhtml = self._build_cover_xhtml(metadata.get("title") or "Cover", cover_href)
                zf.writestr("OPS/cover.xhtml", cover_xhtml)
                cover_id = "cover"
                manifest_items.append((cover_id, "cover.xhtml", "application/xhtml+xml"))

            # content.opf
            content_opf = self._build_content_opf(
                metadata=metadata,
                manifest_items=manifest_items,
                nav_id=nav_id,
                cover_id=cover_id,
                chapter_ids=chapter_ids,
            )
            zf.writestr("OPS/content.opf", content_opf)

            # Note: For simplicity, we don't copy external resources/images in this first pass.
            # Future enhancement: parse and embed local images/assets and update manifest.

        return buf.getvalue()

    def _convert_markdown_to_html(self, markdown_text: str) -> str:
        # Basic conversions: headings, bold, italic, images, links, paragraphs
        text = markdown_text

        def repl_heading(m: re.Match) -> str:
            level = len(m.group(1))
            title = m.group(2).strip()
            return f"<h{level}>{self._escape_html(title)}</h{level}>"

        text = self._heading_regex.sub(repl_heading, text)
        # Bold and italic
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
        # Images and links
        text = re.sub(r"!\[([^\]]*)\]\(([^\)]+)\)", r"<img alt='\1' src='\2'/>", text)
        text = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"<a href='\2'>\1</a>", text)
        # Paragraphs: split on double newlines
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        paragraphs = [p if p.startswith("<h") else f"<p>{p}</p>" for p in parts]
        return "\n".join(paragraphs)

    def _split_into_chapters(self, md_text: str, html_text: str, split_on: bool, levels: List[int]) -> List[Tuple[str, str]]:
        if not split_on:
            # Use whole html as one chapter; try to infer title from first h1 or default
            m = re.search(r"<h1>(.*?)</h1>", html_text)
            title = m.group(1) if m else "Untitled"
            return [(title, html_text)]

        headings = list(self._heading_regex.finditer(md_text))
        filtered = [h for h in headings if len(h.group(1)) in set(levels or [1, 2])]
        if not filtered:
            m = re.search(r"<h1>(.*?)</h1>", html_text)
            title = m.group(1) if m else "Untitled"
            return [(title, html_text)]

        chapters: List[Tuple[str, str]] = []
        indices = [h.start() for h in filtered] + [len(md_text)]
        for i, h in enumerate(filtered):
            start = h.start()
            end = indices[i + 1]
            title = h.group(2).strip()
            md_chunk = md_text[start:end]
            html_chunk = self._convert_markdown_to_html(md_chunk)
            # Ensure heading included at top
            level = len(h.group(1))
            html_chunk = f"<h{level}>{self._escape_html(title)}</h{level}>\n" + html_chunk
            chapters.append((title, html_chunk))
        return chapters

    def _wrap_xhtml(self, title: str, body_html: str) -> str:
        return (
            """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <meta charset="utf-8" />
  <title>"""
            + self._escape_html(title)
            + """</title>
  <link rel="stylesheet" type="text/css" href="style.css" />
</head>
<body>
"""
            + body_html
            + "\n</body>\n</html>"
        )

    def _build_css(self, heading_alignments: Dict[int, str]) -> str:
        css_lines = [
            "body { font-family: serif; margin: 5%; line-height: 1.5; }",
            "h1, h2, h3, h4, h5, h6 { font-family: sans-serif; }",
            "p { margin: 0.5em 0; }",
        ]
        for level, align in heading_alignments.items():
            align = str(align).lower()
            if align in {"left", "center", "right", "justify"}:
                css_lines.append(f"h{int(level)} {{ text-align: {align}; }}")
        return "\n".join(css_lines) + "\n"

    def _build_nav_xhtml(self, chapters: List[Tuple[str, str]], chapter_files: List[str]) -> str:
        items = []
        for idx, (title, _) in enumerate(chapters):
            href = chapter_files[idx]
            items.append(f"        <li><a href=\"{href}\">{self._escape_html(title)}</a></li>")
        return (
            """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
  <meta charset="utf-8" />
  <title>Table of Contents</title>
  <link rel="stylesheet" type="text/css" href="style.css" />
</head>
<body>
  <nav epub:type="toc" id="toc">
    <h1>Table of Contents</h1>
    <ol>
"""
            + "\n".join(items)
            + """
    </ol>
  </nav>
</body>
</html>"""
        )

    def _build_cover_xhtml(self, title: str, img_href: str) -> str:
        return (
            """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <meta charset="utf-8" />
  <title>Cover</title>
  <link rel="stylesheet" type="text/css" href="style.css" />
</head>
<body>
  <section>
    <h1>"""
            + self._escape_html(title)
            + """</h1>
    <div style="text-align:center">
      <img src="""
            + img_href
            + """" alt="Cover" />
    </div>
  </section>
</body>
</html>"""
        )

    def _build_content_opf(
        self,
        *,
        metadata: Dict,
        manifest_items: List[Tuple[str, str, str]],
        nav_id: Optional[str],
        cover_id: Optional[str],
        chapter_ids: List[str],
    ) -> str:
        title = (metadata or {}).get("title") or "Untitled"
        author = (metadata or {}).get("author") or "Unknown Author"
        language = (metadata or {}).get("language") or "en"
        book_id = f"urn:uuid:{uuid.uuid4()}"
        modified = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Manifest
        manifest_lines = [
            '    <item id="css" href="style.css" media-type="text/css"/>'
        ]
        for item_id, href, media_type in manifest_items:
            props = ' properties="nav"' if item_id == nav_id else ""
            manifest_lines.append(f'    <item id="{item_id}" href="{href}" media-type="{media_type}"{props}/>')

        # Spine
        spine_lines: List[str] = []
        if nav_id:
            spine_lines.append(f'    <itemref idref="{nav_id}"/>')
        if cover_id:
            spine_lines.append(f'    <itemref idref="{cover_id}"/>')
        for cid in chapter_ids:
            spine_lines.append(f'    <itemref idref="{cid}"/>')

        return (
            """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="BookID" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:opf="http://www.idpf.org/2007/opf">
    <dc:title id="title">"""
            + self._escape_html(title)
            + """</dc:title>
    <dc:creator id="creator">"""
            + self._escape_html(author)
            + """</dc:creator>
    <dc:language>"""
            + self._escape_html(language)
            + """</dc:language>
    <dc:identifier id="BookID">"""
            + book_id
            + """</dc:identifier>
    <meta property="dcterms:modified">"""
            + modified
            + """</meta>
  </metadata>
  <manifest>
"""
            + "\n".join(manifest_lines)
            + "\n  </manifest>\n  <spine>\n"
            + "\n".join(spine_lines)
            + "\n  </spine>\n</package>"
        )

    def _escape_html(self, s: str) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )





























