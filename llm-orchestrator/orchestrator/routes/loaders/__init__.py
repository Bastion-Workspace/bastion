"""
Route context loaders - optional loaders for complex routes (e.g. fiction context).

Used when a route sets context_loader to a dotted path; resolved at runtime if needed.
"""


async def fiction_context_loader(state: dict) -> dict:
    """Placeholder for fiction context loading. Context preparation is handled by fiction_context_subgraph."""
    return state
