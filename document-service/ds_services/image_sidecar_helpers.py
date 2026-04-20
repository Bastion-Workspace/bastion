"""Shared helpers for image *.metadata.json sidecar payloads."""

from __future__ import annotations

from typing import Any, Dict


def build_minimal_image_sidecar_metadata(
    image_filename: str, image_type: str = "photo"
) -> Dict[str, Any]:
    """Minimal sidecar dict for search indexing (disk + DB)."""
    return {
        "image_filename": image_filename,
        "image_type": image_type,
        "type": image_type,
        "has_searchable_metadata": True,
        "title": image_filename,
        "content": "",
    }
