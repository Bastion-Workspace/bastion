"""Document-service configuration (vendored from backend config + DS runtime fields)."""

from .settings import Settings, get_settings, settings


def validate_runtime() -> None:
    """Validate settings required for the gRPC + NER server."""
    if not settings.SPACY_MODEL or not str(settings.SPACY_MODEL).strip():
        raise ValueError("SPACY_MODEL must be set")


__all__ = ["Settings", "get_settings", "settings", "validate_runtime"]
