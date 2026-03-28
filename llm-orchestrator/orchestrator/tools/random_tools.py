"""
Random Tools - Pure in-process random number generation for playbooks and automation.

Zone 1 (orchestrator): no gRPC, no backend_tool_client. Used for A/B sampling,
delays/jitter, and reproducible randomness via optional seed.
"""

import logging
import random
from typing import Optional, Union

from pydantic import BaseModel, Field

from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


# ── I/O models for random_number_tool ────────────────────────────────────────


class RandomNumberInputs(BaseModel):
    """Required inputs for random_number_tool."""
    min_val: float = Field(description="Lower bound (inclusive for integer mode)")
    max_val: float = Field(description="Upper bound (inclusive for integer mode)")


class RandomNumberParams(BaseModel):
    """Optional parameters."""
    integer: bool = Field(default=True, description="If True, return integer; if False, return float")
    seed: Optional[Union[int, str]] = Field(default=None, description="Optional seed for reproducible randomness")


class RandomNumberOutputs(BaseModel):
    """Typed outputs for random_number_tool."""
    value: Union[int, float] = Field(description="Generated random number")
    min_val: float = Field(description="Lower bound used")
    max_val: float = Field(description="Upper bound used")
    integer: bool = Field(description="Whether result was integer")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


def random_number_tool(
    min_val: float,
    max_val: float,
    integer: bool = True,
    seed: Optional[Union[int, str]] = None,
) -> dict:
    """
    Generate a random number in [min_val, max_val]. Integer mode uses inclusive bounds.
    Optional seed for reproducibility.
    """
    if seed is not None:
        if isinstance(seed, str):
            random.seed(hash(seed) % (2**32))
        else:
            random.seed(seed)
    lo, hi = min_val, max_val
    if lo > hi:
        lo, hi = hi, lo
    if integer:
        val = random.randint(int(lo), int(hi))
    else:
        val = random.uniform(lo, hi)
    formatted = f"Random number: {val} (range {lo}–{hi}, integer={integer})"
    return {
        "value": val,
        "min_val": min_val,
        "max_val": max_val,
        "integer": integer,
        "formatted": formatted,
    }


register_action(
    name="random_number",
    category="math",
    description="Generate a random number between min and max (integer or float); optional seed for reproducibility",
    inputs_model=RandomNumberInputs,
    params_model=RandomNumberParams,
    outputs_model=RandomNumberOutputs,
    tool_function=random_number_tool,
)
