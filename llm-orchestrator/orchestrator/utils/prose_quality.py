from __future__ import annotations

import re
from typing import List


_BULLET_PREFIX_RE = re.compile(r"^\s*(?:-\s+|\*\s+|\d+\.\s+)")


def extract_outline_bullets(outline_text: str) -> List[str]:
    """Extract bullet-like lines from outline text (normalized minimally)."""
    if not outline_text:
        return []
    bullets: List[str] = []
    for line in outline_text.splitlines():
        if _BULLET_PREFIX_RE.match(line):
            bullets.append(line.strip())
    return bullets


def count_bullet_like_lines(text: str) -> int:
    """Count lines that look like bullets/numbered lists in generated prose."""
    if not text:
        return 0
    count = 0
    for line in text.splitlines():
        if _BULLET_PREFIX_RE.match(line):
            count += 1
    return count


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def count_outline_bullet_hits(generated_text: str, outline_bullets: List[str], sample_len: int = 80) -> int:
    """Count how many outline bullets appear (approximately) inside generated text.

    This is a heuristic: we normalize whitespace and check a prefix sample of each bullet.
    """
    if not generated_text or not outline_bullets:
        return 0

    gen_norm = " ".join(generated_text.split())
    hits = 0
    for bullet in outline_bullets:
        bullet_norm = " ".join(bullet.split())
        if len(bullet_norm) < 25:
            continue
        sample = bullet_norm[:sample_len]
        if sample in gen_norm:
            hits += 1
    return hits


