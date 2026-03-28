"""PDF font stack presets for WeasyPrint (Linux-friendly open fonts + fallbacks)."""

from __future__ import annotations

from typing import Tuple


def get_pdf_font_stacks(
    preset: str, typeface_style: str = "mixed"
) -> Tuple[str, str]:
    """
    Returns (body_font_family_css, heading_font_family_css) as CSS-ready values.

    typeface_style:
      - mixed: serif body, sans-serif headings (default)
      - serif: body and headings use the preset's serif stack
      - sans: body and headings use the preset's sans stack
    """
    p = (preset or "liberation").lower().replace("-", "_")
    style = (typeface_style or "mixed").lower().strip()
    if style not in ("serif", "sans", "mixed"):
        style = "mixed"

    presets: dict[str, Tuple[str, str]] = {
        "liberation": (
            '"Liberation Serif", "DejaVu Serif", "Times New Roman", "Noto Serif", serif',
            '"Liberation Sans", "DejaVu Sans", "Helvetica Neue", Arial, "Noto Sans", sans-serif',
        ),
        "dejavu": (
            '"DejaVu Serif", "Liberation Serif", "Times New Roman", serif',
            '"DejaVu Sans", "Liberation Sans", Helvetica, Arial, sans-serif',
        ),
        "noto": (
            '"Noto Serif", "DejaVu Serif", "Liberation Serif", serif',
            '"Noto Sans", "DejaVu Sans", "Liberation Sans", Helvetica, Arial, sans-serif',
        ),
        "times_helvetica": (
            '"Times New Roman", "Liberation Serif", Times, serif',
            '"Helvetica Neue", Helvetica, "Liberation Sans", Arial, sans-serif',
        ),
    }
    serif_f, sans_f = presets.get(p, presets["liberation"])
    if style == "serif":
        return (serif_f, serif_f)
    if style == "sans":
        return (sans_f, sans_f)
    return (serif_f, sans_f)
