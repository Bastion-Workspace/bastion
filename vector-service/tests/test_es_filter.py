"""Elasticsearch/OpenSearch filter DSL builder (no cluster required)."""

from service.backends.base import SparseVectorData, VectorFilterInput
from service.backends.es_backend import _es_filter_from_filters, _sparse_str_dict


def test_search_equals_and_not_equals():
    q = _es_filter_from_filters(
        [
            VectorFilterInput(field="user_id", value="u1", operator="equals", values=[]),
            VectorFilterInput(field="x", value="y", operator="not_equals", values=[]),
        ],
        "search",
    )
    assert q is not None
    assert "bool" in q
    must = q["bool"].get("must") or []
    must_not = q["bool"].get("must_not") or []
    assert any("payload.user_id" in str(m) for m in must)
    assert len(must_not) >= 1


def test_search_any_of():
    q = _es_filter_from_filters(
        [
            VectorFilterInput(
                field="tags",
                value="",
                operator="any_of",
                values=["a", "b"],
            ),
        ],
        "search",
    )
    assert q is not None
    assert "terms" in str(q)


def test_equality_mode():
    q = _es_filter_from_filters(
        [
            VectorFilterInput(field="document_id", value="d1", operator="equals", values=[]),
        ],
        "equality",
    )
    assert q is not None
    assert q["bool"]["must"][0] == {"term": {"payload.document_id": "d1"}}


def test_sparse_str_dict():
    d = _sparse_str_dict(SparseVectorData(indices=[1, 42], values=[0.5, 0.25]))
    assert d == {"1": 0.5, "42": 0.25}


def test_empty_filters():
    assert _es_filter_from_filters([], "search") is None
