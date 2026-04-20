"""Stub domain detector for metadata updates."""


def get_domain_detector():
    return _D()


class _D:
    def get_domain_changes(self, old_tags, old_category, new_tags, new_category):
        return {"changed": False, "added": [], "removed": []}
