"""Stub enhanced PDF segmentation (vendored document_service falls back on failure)."""

from typing import Any


class EnhancedPDFSegmentationService:
    def __init__(self, *args, **kwargs):
        pass

    async def initialize(self):
        return None

    async def extract_pdf_info(self, request: Any):
        class R:
            pages_extracted = 0

        return R()
