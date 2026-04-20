"""Compatibility shim — delegates to document-service implementation."""

from ds_services.image_sidecar_service import (  # noqa: F401
    ImageSidecarService,
    get_image_sidecar_service,
)

__all__ = ["ImageSidecarService", "get_image_sidecar_service"]
