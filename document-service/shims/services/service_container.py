"""
Injectable stand-in for backend ``service_container``.

``document-service/main.py`` assigns ``document_service`` and ``folder_service``
after the processing pipeline is built so lazy imports inside vendored code
that call ``get_service_container()`` still resolve.

Porting contract: code copied from the backend often does ``if container.foo``.
Attributes checked that way must exist on ``_ServiceContainer`` (default
``None``); missing attributes raise ``AttributeError`` before the branch runs.
Prefer declaring optional slots here over ``getattr`` in scattered call sites.
"""

from typing import Any, Optional

_document_service: Optional[Any] = None
_folder_service: Optional[Any] = None


def set_document_services(document_service: Any, folder_service: Any) -> None:
    global _document_service, _folder_service
    _document_service = document_service
    _folder_service = folder_service


def clear_document_services() -> None:
    global _document_service, _folder_service
    _document_service = None
    _folder_service = None


class _ServiceContainer:
    """Minimal container; optional slots mirror backend for truthiness checks."""

    # Parity with backend service_container: direct_search checks this for vector share scopes.
    document_sharing_service: Optional[Any] = None
    # LangGraph / agent tools (see backend ServiceContainer.__init__).
    file_manager: Optional[Any] = None
    websocket_manager: Optional[Any] = None

    @property
    def is_initialized(self) -> bool:
        return _document_service is not None and _folder_service is not None

    @property
    def document_service(self):
        if _document_service is None:
            raise RuntimeError(
                "service_container.document_service is not set; "
                "call set_document_services() from document-service startup"
            )
        return _document_service

    @property
    def folder_service(self):
        if _folder_service is None:
            raise RuntimeError(
                "service_container.folder_service is not set; "
                "call set_document_services() from document-service startup"
            )
        return _folder_service


service_container = _ServiceContainer()


async def get_service_container() -> _ServiceContainer:
    return service_container
