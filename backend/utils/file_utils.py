"""
Simple file utilities
"""

import os


def ensure_dir_exists(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        # Best effort; caller can handle failures on write
        pass


