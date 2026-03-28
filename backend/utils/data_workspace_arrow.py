"""
Decode Apache Arrow IPC payloads from the data-service gRPC API.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.ipc as ipc


def ipc_bytes_to_record_batch(data: bytes) -> pa.RecordBatch:
    reader = ipc.open_stream(pa.py_buffer(data))
    return reader.read_next_batch()


def record_batch_to_dicts(batch: pa.RecordBatch) -> List[Dict[str, Any]]:
    if batch.num_rows == 0:
        return []
    pyd = batch.to_pydict()
    keys = list(pyd.keys())
    n = batch.num_rows
    out: List[Dict[str, Any]] = []
    for i in range(n):
        row: Dict[str, Any] = {}
        for k in keys:
            v = pyd[k][i]
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            row[k] = v
        out.append(row)
    return out


def decode_query_arrow(arrow_bytes: bytes) -> List[Dict[str, Any]]:
    """Decode SQL query result Arrow IPC to row dicts."""
    return record_batch_to_dicts(ipc_bytes_to_record_batch(arrow_bytes))


def decode_native_table_arrow(arrow_bytes: bytes) -> List[Dict[str, Any]]:
    """Decode flat-typed native table Arrow batch to UI row dicts."""
    batch = ipc_bytes_to_record_batch(arrow_bytes)
    if batch.num_rows == 0:
        return []
    pyd = batch.to_pydict()
    keys = list(pyd.keys())
    n = batch.num_rows
    rows: List[Dict[str, Any]] = []
    has_rid = "row_id" in pyd
    has_ri = "row_index" in pyd
    for i in range(n):
        row_data = {k: pyd[k][i] for k in keys if k not in ("row_id", "row_index")}
        for rk, rv in list(row_data.items()):
            if hasattr(rv, "isoformat") and callable(getattr(rv, "isoformat")):
                try:
                    row_data[rk] = rv.isoformat() if rv is not None else None
                except Exception:
                    pass
        rid = pyd["row_id"][i] if has_rid else str(i)
        ri = int(pyd["row_index"][i]) if has_ri else i
        rows.append(
            {
                "row_id": rid,
                "row_index": ri,
                "row_data": row_data,
                "row_color": None,
                "formula_data": {},
            }
        )
    return rows


def decode_table_page_arrow(arrow_bytes: bytes) -> List[Dict[str, Any]]:
    """Decode GetTableData Arrow IPC to row dicts matching JSON path shape."""
    batch = ipc_bytes_to_record_batch(arrow_bytes)
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
