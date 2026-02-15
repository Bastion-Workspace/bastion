"""
Central version management for Bastion
Reads version from root VERSION file
"""

import os
from pathlib import Path

def get_version() -> str:
    """Get version from root VERSION file"""
    # Get the root directory (parent of backend/)
    root_dir = Path(__file__).parent.parent
    version_file = root_dir / "VERSION"
    
    if version_file.exists():
        with open(version_file, "r") as f:
            version = f.read().strip()
            return version
    else:
        # Fallback if VERSION file doesn't exist
        return "0.20.0"

__version__ = get_version()

