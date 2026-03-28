"""
Base command builder with allowlist validation helpers.
All CLI commands are built as list[str]; no shell or user-controlled strings.
"""
from __future__ import annotations


class BaseCommandBuilder:
    """Helpers for validating parameters against allowlists and ranges."""

    @staticmethod
    def validate_enum(value: str, allowed: set[str], param_name: str) -> str:
        v = (value or "").strip().lower()
        if v not in allowed:
            raise ValueError(f"{param_name} must be one of {sorted(allowed)}, got {value!r}")
        return v

    @staticmethod
    def validate_range(value: int, min_v: int, max_v: int, param_name: str) -> int:
        if not (min_v <= value <= max_v):
            raise ValueError(f"{param_name} must be {min_v}-{max_v}, got {value}")
        return value

    @staticmethod
    def validate_positive(value: int, param_name: str) -> int:
        if value <= 0:
            raise ValueError(f"{param_name} must be positive, got {value}")
        return value
