"""
Large static CSS fragments for PDF print styles (WeasyPrint).
"""

from __future__ import annotations


def base_typography_css(
    body_font_stack: str,
    heading_font_stack: str,
    *,
    body_extra_css: str = "",
) -> str:
    """Body + heading font stacks; !important on heading sizes for WeasyPrint UA overrides."""
    return f"""
body {{
    font-family: {body_font_stack};
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
    margin: 0;
    padding: 0;
    {body_extra_css}
}}

html body h1,
html body h2,
html body h3,
html body h4,
html body h5,
html body h6 {{
    display: block !important;
    white-space: normal;
    overflow-wrap: break-word;
    font-family: {heading_font_stack};
    font-weight: bold;
    margin-top: 1.2em;
    margin-bottom: 0.6em;
    page-break-after: avoid;
}}

html body h1 {{
    font-size: 22pt !important;
    line-height: 1.25;
    border-bottom: 2px solid #333;
    padding-bottom: 0.3em;
}}

html body h2 {{
    font-size: 18pt !important;
    line-height: 1.28;
}}

html body h3 {{
    font-size: 15pt !important;
    line-height: 1.3;
}}

html body h4 {{
    font-size: 13pt !important;
    line-height: 1.35;
}}

html body h5 {{
    font-size: 11.5pt !important;
    line-height: 1.4;
}}

html body h6 {{
    font-size: 10.5pt !important;
    line-height: 1.4;
    font-weight: bold;
}}

/* Document (.main-body) and chat (.message-content): class + element beats body font-size inheritance */
.main-body h1,
.main-body h2,
.main-body h3,
.main-body h4,
.main-body h5,
.main-body h6,
.message-content h1,
.message-content h2,
.message-content h3,
.message-content h4,
.message-content h5,
.message-content h6 {{
    display: block !important;
    white-space: normal;
    overflow-wrap: break-word;
    font-family: {heading_font_stack};
    font-weight: bold;
    margin-top: 1.2em;
    margin-bottom: 0.6em;
    page-break-after: avoid;
}}

.main-body h1,
.message-content h1 {{
    font-size: 22pt !important;
    line-height: 1.25;
    border-bottom: 2px solid #333;
    padding-bottom: 0.3em;
}}

.main-body h2,
.message-content h2 {{
    font-size: 18pt !important;
    line-height: 1.28;
    border-bottom: none;
}}

.main-body h3,
.message-content h3 {{
    font-size: 15pt !important;
    line-height: 1.3;
}}

.main-body h4,
.message-content h4 {{
    font-size: 13pt !important;
    line-height: 1.35;
}}

.main-body h5,
.message-content h5 {{
    font-size: 11.5pt !important;
    line-height: 1.4;
}}

.main-body h6,
.message-content h6 {{
    font-size: 10.5pt !important;
    line-height: 1.4;
    font-weight: bold;
}}

.toc-wrap h2 {{
    font-family: {heading_font_stack};
    font-size: 16pt !important;
    margin-top: 0;
    border-bottom: none;
}}

p {{
    margin: 0.8em 0;
    text-align: justify;
}}

ul, ol {{
    margin: 0.8em 0;
    padding-left: 2em;
}}

li {{
    margin: 0.4em 0;
}}

code {{
    font-family: "Courier New", Courier, monospace;
    font-size: 0.85em;
    background-color: #f5f5f5;
    padding: 0.2em 0.4em;
    border-radius: 3px;
}}

pre {{
    font-family: "Courier New", Courier, monospace;
    font-size: 0.85em;
    background-color: #f5f5f5;
    padding: 1em;
    border-radius: 4px;
    page-break-inside: avoid;
}}

pre code {{
    background-color: transparent;
    padding: 0;
}}

blockquote {{
    border-left: 4px solid #ddd;
    margin: 1em 0;
    padding-left: 1em;
    color: #666;
    font-style: italic;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    table-layout: fixed;
    margin: 1em 0;
    page-break-inside: avoid;
}}

th, td {{
    border: 1px solid #ddd;
    padding: 0.6em;
    text-align: left;
    vertical-align: top;
    overflow-wrap: break-word;
    word-wrap: break-word;
}}

th {{
    background-color: #f5f5f5;
    font-weight: bold;
}}

img {{
    max-width: 100%;
    height: auto;
    page-break-inside: avoid;
}}

a {{
    color: #0366d6;
    text-decoration: none;
}}

a:hover {{
    text-decoration: underline;
}}

hr {{
    border: none;
    border-top: 1px solid #ddd;
    margin: 2em 0;
}}

p, li {{
    orphans: 3;
    widows: 3;
}}

.message-card {{
    margin: 1.5em 0;
    padding: 1em;
    border: 1px solid #e1e4e8;
    border-radius: 8px;
    background-color: #fafbfc;
    page-break-inside: avoid;
}}

.message-header {{
    font-size: 12pt;
    font-weight: bold;
    margin-bottom: 0.5em;
    color: #0366d6;
}}

.message-header.user {{
    color: #0366d6;
}}

.message-header.assistant {{
    color: #28a745;
}}

.message-timestamp {{
    font-size: 9pt;
    color: #666;
    margin-left: 0.5em;
}}

.message-content {{
    margin-top: 0.8em;
}}

.chat-header {{
    margin-bottom: 2em;
    padding-bottom: 1em;
    border-bottom: 2px solid #333;
}}

.chat-title {{
    font-family: {heading_font_stack};
    font-size: 16pt;
    font-weight: bold;
    margin-bottom: 0.3em;
}}

.chat-meta {{
    font-size: 9pt;
    color: #666;
}}
"""


def watermark_css_block(heading_font_stack: str) -> str:
    """Watermark layers only; body positioning is merged in base_typography_css when a watermark is set."""
    return f"""
.watermark {{
    font-family: {heading_font_stack};
    font-weight: bold;
    white-space: nowrap;
    pointer-events: none;
    z-index: 1000;
    isolation: isolate;
}}
.watermark-all-pages {{
    position: fixed;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%) rotate(-45deg);
    font-size: 64pt;
    color: rgba(0, 0, 0, 0.07);
}}
.watermark-first-page {{
    position: absolute;
    left: 0;
    right: 0;
    top: 40%;
    text-align: center;
    transform: rotate(-45deg);
    font-size: 64pt;
    color: rgba(0, 0, 0, 0.07);
}}
"""


TOC_CSS_BLOCK = """
.toc-wrap {
    page-break-after: always;
    margin-bottom: 2em;
}
.toc-list {
    list-style: none;
    padding-left: 0;
    margin: 0.5em 0 0 0;
}
.toc-list li {
    margin: 0.35em 0;
}
.toc-list a {
    display: flex;
    align-items: baseline;
    color: inherit;
    text-decoration: none;
}
.toc-title {
    flex: 0 1 auto;
}
.toc-leader {
    flex: 1 1 auto;
    border-bottom: 1px dotted #bbb;
    margin: 0 0.4em;
    min-width: 0.5em;
    height: 0.75em;
}
.toc-list a::after {
    content: target-counter(attr(href), page);
    flex: 0 0 auto;
    min-width: 1.5em;
    text-align: right;
}
.toc-h2 { margin-left: 1em; }
.toc-h3 { margin-left: 2em; }
.toc-h4 { margin-left: 3em; }
.toc-h5 { margin-left: 4em; }
.toc-h6 { margin-left: 5em; }
"""
