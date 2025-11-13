"""
Chapter scope utilities: detect chapters by '## Chapter XX' headings, locate
current/previous/next chapter given a cursor offset, and compute paragraph range.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re


@dataclass
class ChapterRange:
    heading_text: str
    chapter_number: Optional[int]
    start: int  # inclusive char offset
    end: int    # exclusive char offset


CHAPTER_PATTERN = re.compile(r"^##\s+Chapter\s+(\d+)\b.*$", re.MULTILINE)


def find_chapter_ranges(text: str) -> List[ChapterRange]:
    if not text:
        return []
    matches = list(CHAPTER_PATTERN.finditer(text))
    if not matches:
        return []
    ranges: List[ChapterRange] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chapter_num: Optional[int] = None
        try:
            chapter_num = int(m.group(1))
        except Exception:
            chapter_num = None
        ranges.append(ChapterRange(heading_text=m.group(0), chapter_number=chapter_num, start=start, end=end))
    return ranges


def locate_chapter_index(ranges: List[ChapterRange], cursor_offset: int) -> int:
    if cursor_offset < 0:
        return -1
    for i, r in enumerate(ranges):
        if r.start <= cursor_offset < r.end:
            return i
    return -1


def get_adjacent_chapters(ranges: List[ChapterRange], idx: int) -> Tuple[Optional[ChapterRange], Optional[ChapterRange]]:
    prev_c = ranges[idx - 1] if 0 <= idx - 1 < len(ranges) else None
    next_c = ranges[idx + 1] if 0 <= idx + 1 < len(ranges) else None
    return prev_c, next_c


def paragraph_bounds(text: str, cursor_offset: int) -> Tuple[int, int]:
    if not text:
        return 0, 0
    cursor = max(0, min(len(text), cursor_offset))
    # expand left to previous blank line or start
    left = text.rfind("\n\n", 0, cursor)
    start = left + 2 if left != -1 else 0
    # expand right to next blank line or end
    right = text.find("\n\n", cursor)
    end = right if right != -1 else len(text)
    return start, end











































