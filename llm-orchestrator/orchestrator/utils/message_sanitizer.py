"""
Strip legacy injected tool-call trace prefixes from message text.

Historically, assistant history could be prefixed with
``[Tool actions: ...]\\n`` for LLM context. That text must not appear in
user-visible responses or persist back into stored message content.
"""


def strip_tool_actions_prefix(text: str) -> str:
    """
    Remove a leading ``[Tool actions: ...]`` block from *text*.

    Uses bracket depth so summaries containing ``]`` inside nested structures
    (e.g. repr) are handled correctly. If the prefix is malformed or absent,
    returns *text* unchanged.
    """
    if not text or not isinstance(text, str):
        return text or ""
    if not text.startswith("[Tool actions:"):
        return text
    depth = 0
    for i, c in enumerate(text):
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                while end < len(text) and text[end] in "\r\n":
                    end += 1
                return text[end:].lstrip()
    return text
