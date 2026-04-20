"""
Registry of file paths currently being OCR'd.
Used by the file watcher to skip re-processing a document while OCR is in progress.
"""

import threading
from typing import Set

_paths: Set[str] = set()
_lock = threading.Lock()


def add(path: str) -> None:
    with _lock:
        _paths.add(path)


def remove(path: str) -> None:
    with _lock:
        _paths.discard(path)


def contains(path: str) -> bool:
    with _lock:
        return path in _paths
