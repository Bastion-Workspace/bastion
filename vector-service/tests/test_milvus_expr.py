"""Milvus filter expression builder (no running Milvus required)."""

from service.backends.base import VectorFilterInput
from service.backends.milvus_backend import _milvus_expr_from_filters


def test_search_equals_and_not_equals():
    expr = _milvus_expr_from_filters(
        [
            VectorFilterInput(field="user_id", value="u1", operator="equals", values=[]),
            VectorFilterInput(field="x", value="y", operator="not_equals", values=[]),
        ],
        "search",
    )
    assert 'payload["user_id"] == "u1"' in expr
    assert "not" in expr.lower() or "!=" in expr


def test_search_any_of():
    expr = _milvus_expr_from_filters(
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
    assert "in [" in expr
    assert '"a"' in expr


def test_equality_mode():
    expr = _milvus_expr_from_filters(
        [
            VectorFilterInput(field="document_id", value="d1", operator="equals", values=[]),
        ],
        "equality",
    )
    assert expr == '(payload["document_id"] == "d1")'


def test_escape_quotes_in_value():
    expr = _milvus_expr_from_filters(
        [
            VectorFilterInput(field="k", value='a"b', operator="equals", values=[]),
        ],
        "search",
    )
    assert '\\"' in expr or "payload" in expr
