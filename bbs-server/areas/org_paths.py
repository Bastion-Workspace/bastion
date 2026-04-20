"""Resolve org file paths for refile API (paths relative to Users/<username>/)."""


def relative_user_library_path(username: str, absolute_file_path: str, filename: str) -> str:
    """
    Convert absolute library path to path relative to Users/<username>/.
    Falls back to OrgMode/<filename> when the marker is not found.
    """
    if not absolute_file_path:
        return f"OrgMode/{filename}" if filename else ""
    norm = absolute_file_path.replace("\\", "/").strip()
    needle = f"Users/{username}/"
    if needle in norm:
        return norm.split(needle, 1)[1].lstrip("/")
    if filename:
        return f"OrgMode/{filename}"
    return ""
