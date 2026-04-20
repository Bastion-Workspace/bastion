"""Lightweight vector upsert payload (no Qdrant SDK).

Used to shuttle (id, vector, payload) into VectorStoreService.insert_points(),
which serialises to dicts before sending to vector-service over gRPC.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class VectorPoint:
    id: Union[str, int]
    vector: List[float]
    payload: Dict[str, Any]
