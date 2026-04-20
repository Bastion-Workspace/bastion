"""Factory wiring for VECTOR_DB_BACKEND (no Qdrant connection required)."""

from unittest.mock import MagicMock

import pytest


def test_get_vector_backend_rejects_unknown():
    from service.backends.factory import get_vector_backend

    s = MagicMock()
    s.VECTOR_DB_BACKEND = "not_a_real_backend"
    with pytest.raises(ValueError, match="Unsupported VECTOR_DB_BACKEND"):
        get_vector_backend(s)


def test_get_vector_backend_accepts_qdrant():
    pytest.importorskip("qdrant_client")

    from service.backends.factory import get_vector_backend

    s = MagicMock()
    s.VECTOR_DB_BACKEND = "qdrant"
    s.QDRANT_URL = ""
    s.QDRANT_TIMEOUT = 30
    s.QDRANT_API_KEY = None
    s.EMBEDDING_DIMENSIONS = 3072
    s.HYBRID_SEARCH_ENABLED = False
    b = get_vector_backend(s)
    b.initialize()
    assert not b.is_available()
