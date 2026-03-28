"""
Decode Apache Arrow IPC payloads from tools-service / data-service gRPC.
"""

from __future__ import annotations

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
