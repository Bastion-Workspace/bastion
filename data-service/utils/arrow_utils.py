"""
Apache Arrow helpers for columnar serialization over gRPC (IPC stream format).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Union

import pyarrow as pa
import pyarrow.ipc as ipc

logger = logging.getLogger(__name__)


def _scalar_for_arrow(value: Any) -> Any:
    """Normalize asyncpg / Python values for Arrow array construction."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, time):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, memoryview):
        return bytes(value).decode("utf-8", errors="replace")
    mod = type(value).__module__
    if mod and mod != "builtins" and not isinstance(value, (str, int, float, bool, dict, list)):
        if hasattr(value, "isoformat") and callable(getattr(value, "isoformat")):
            try:
                return value.isoformat()
            except Exception:
                pass
        return str(value)
    return value


def asyncpg_rows_to_record_batch(
    rows: Sequence[Any],
    column_names: Optional[List[str]] = None,
) -> pa.RecordBatch:
    """
    Build a RecordBatch from asyncpg Records (or dict-like rows).
    """
    if not rows:
        return pa.RecordBatch.from_pydict({})

    if column_names is None:
        column_names = list(rows[0].keys())

    columns: Dict[str, List[Any]] = {c: [] for c in column_names}
    for row in rows:
        for c in column_names:
            columns[c].append(_scalar_for_arrow(row[c]))

    arrays = []
    for name in column_names:
        try:
            arrays.append(pa.array(columns[name]))
        except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
            logger.debug("Falling back to string column %s: %s", name, e)
            str_vals = [None if v is None else str(v) for v in columns[name]]
            arrays.append(pa.array(str_vals, type=pa.string()))

    return pa.record_batch(arrays, names=column_names)


def record_batch_to_ipc_bytes(batch: pa.RecordBatch) -> bytes:
    """Serialize a single RecordBatch to Arrow IPC stream bytes."""
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def empty_ipc_record_batch_bytes() -> bytes:
    """Arrow IPC bytes for an empty RecordBatch (no columns)."""
    return record_batch_to_ipc_bytes(pa.RecordBatch.from_pydict({}))


def ipc_bytes_to_record_batch(data: bytes) -> pa.RecordBatch:
    """Deserialize IPC stream bytes to a RecordBatch (first batch only)."""
    reader = ipc.open_stream(pa.py_buffer(data))
    return reader.read_next_batch()


def record_batch_to_dicts(batch: pa.RecordBatch) -> List[Dict[str, Any]]:
    """Convert RecordBatch rows to list of dicts (JSON-friendly scalars)."""
    if batch.num_rows == 0:
        return []
    pyd = batch.to_pydict()
    keys = list(pyd.keys())
    n = batch.num_rows
    out: List[Dict[str, Any]] = []
    for i in range(n):
        row = {}
        for k in keys:
            v = pyd[k][i]
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            row[k] = v
        out.append(row)
    return out


def table_page_rows_to_ipc_bytes(data_rows: List[Dict[str, Any]]) -> bytes:
    """
    Encode UI table rows (row_id, row_index, row_data, row_color, formula_data)
    as a RecordBatch with JSON columns for nested dicts.
    """
    row_ids: List[str] = []
    row_indices: List[int] = []
    row_colors: List[str] = []
    row_data_strs: List[str] = []
    formula_strs: List[str] = []

    for r in data_rows:
        row_ids.append(str(r.get("row_id", "")))
        row_indices.append(int(r.get("row_index", 0)))
        rc = r.get("row_color")
        row_colors.append("" if rc is None else str(rc))
        row_data_strs.append(json.dumps(r.get("row_data", {}), default=str))
        formula_strs.append(json.dumps(r.get("formula_data", {}), default=str))

    batch = pa.record_batch(
        [
            pa.array(row_ids, type=pa.string()),
            pa.array(row_indices, type=pa.int32()),
            pa.array(row_colors, type=pa.string()),
            pa.array(row_data_strs, type=pa.string()),
            pa.array(formula_strs, type=pa.string()),
        ],
        names=["row_id", "row_index", "row_color", "row_data_json", "formula_data_json"],
    )
    return record_batch_to_ipc_bytes(batch)


def ipc_bytes_to_table_page_rows(data: bytes) -> List[Dict[str, Any]]:
    """Decode table page Arrow payload back to row dicts."""
    batch = ipc_bytes_to_record_batch(data)
    pyd = batch.to_pydict()
    n = batch.num_rows
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        rd = pyd["row_data_json"][i]
        fd = pyd["formula_data_json"][i]
        rows.append(
            {
                "row_id": pyd["row_id"][i],
                "row_index": int(pyd["row_index"][i]),
                "row_color": pyd["row_color"][i] or None,
                "row_data": json.loads(rd) if rd else {},
                "formula_data": json.loads(fd) if fd else {},
            }
        )
    return rows
