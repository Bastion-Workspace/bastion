"""
spaCy-based entity extraction for Document Service.
Loads model once, runs NER + pattern extraction, name cleaning, and deduplication.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Set

from ds_config import settings

logger = logging.getLogger(__name__)

# spaCy labels we skip entirely (low value for knowledge graph)
SKIP_LABELS: Set[str] = {
    "CARDINAL",
    "ORDINAL",
    "QUANTITY",
    "PERCENT",
    "MONEY",
    "TIME",
}

# Map spaCy NER labels to backend/Neo4j canonical types (only for labels we keep)
SPACY_LABEL_MAP: Dict[str, str] = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "FACILITY",
    "PRODUCT": "PRODUCT",
    "EVENT": "EVENT",
    "DATE": "DATE",
    "NORP": "GROUP",
    "LANGUAGE": "MISC",
    "LAW": "MISC",
    "WORK_OF_ART": "MISC",
}

# Name cleaning regexes
_LEADING_JUNK = re.compile(r"^(?:the|a|an|this|that|these|those)\s+", re.IGNORECASE)
_TRAILING_POSSESSIVE = re.compile(r"['\u2019]s?\s*$")
_TRAILING_PUNCT = re.compile(r'[.,;:!?\'")\]}>]+$')
_LEADING_PUNCT = re.compile(r'^[(\[{<\'"]+')

# Pattern extraction regexes
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
_PHONE_PATTERN = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
)
_DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
        re.IGNORECASE,
    ),
]
_TECH_KEYWORDS = [
    "Python", "JavaScript", "Java", "C++", "C#", "Ruby", "PHP", "Go", "Rust",
    "React", "Vue", "Angular", "Node.js", "Django", "Flask", "Spring",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "MongoDB", "PostgreSQL",
    "MySQL", "Redis", "Elasticsearch", "Apache", "Nginx", "Git", "GitHub",
]

NOISE_WORDS: Set[str] = {
    "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "a", "an",
    "the days", "the day", "five minutes", "ten minutes", "sixty percent", "one percent",
    "first", "second", "last", "next", "few", "many", "some", "several", "most",
}


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    confidence: float
    context: str
    source: str = "spacy"


def _clean_name(name: str) -> str:
    """Strip possessives, leading determiners, and leading/trailing punctuation."""
    name = name.strip()
    name = _TRAILING_POSSESSIVE.sub("", name)
    name = _TRAILING_PUNCT.sub("", name)
    name = _LEADING_PUNCT.sub("", name)
    name = _LEADING_JUNK.sub("", name)
    return name.strip()


class EntityExtractor:
    """Loads spaCy once; runs NER, pattern extraction, cleaning, and deduplication."""

    def __init__(self) -> None:
        self._nlp = None
        self._loaded = False

    async def initialize(self) -> None:
        """Load spaCy model (blocking load; run in executor if needed)."""
        if self._loaded:
            return
        try:
            import spacy
            model = settings.SPACY_MODEL or "en_core_web_lg"
            logger.info("Loading spaCy model: %s", model)
            self._nlp = spacy.load(model)
            self._loaded = True
            logger.info("spaCy model loaded")
        except Exception as e:
            logger.error("Failed to load spaCy model: %s", e)
            raise

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._nlp is not None

    def _normalize_type(self, label: str) -> str:
        """Map spaCy label to canonical entity type."""
        key = (label or "").strip().upper()
        return SPACY_LABEL_MAP.get(key, key if key else "MISC")

    def _extract_spacy_entities(self, text: str, doc) -> List[ExtractedEntity]:
        """Build entities from spaCy doc with name cleaning, label skip, frequency confidence."""
        name_counts: Counter = Counter()
        for ent in doc.ents:
            if not ent.text or not ent.text.strip():
                continue
            if ent.label_ in SKIP_LABELS:
                continue
            cleaned = _clean_name(ent.text)
            if not cleaned or len(cleaned) < 2 or cleaned.isdigit():
                continue
            name_counts[cleaned.lower()] += 1

        seen: Set[tuple[str, str]] = set()
        entities: List[ExtractedEntity] = []
        for ent in doc.ents:
            if not ent.text or not ent.text.strip():
                continue
            if ent.label_ in SKIP_LABELS:
                continue
            cleaned = _clean_name(ent.text)
            if not cleaned or len(cleaned) < 2 or cleaned.isdigit():
                continue
            entity_type = self._normalize_type(ent.label_)
            key = (cleaned.lower(), entity_type)
            if key in seen:
                continue
            seen.add(key)
            count = name_counts[cleaned.lower()]
            confidence = min(0.95, 0.7 + 0.05 * count)
            context = text[max(0, ent.start_char - 50) : ent.end_char + 50]
            entities.append(
                ExtractedEntity(
                    name=cleaned,
                    entity_type=entity_type,
                    confidence=confidence,
                    context=context[:500],
                    source="spacy",
                )
            )
        return entities

    def _extract_pattern_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using regex patterns (email, URL, phone, date, technology)."""
        entities: List[ExtractedEntity] = []
        try:
            for m in _EMAIL_PATTERN.finditer(text):
                entities.append(
                    ExtractedEntity(
                        name=m.group(),
                        entity_type="EMAIL",
                        confidence=0.9,
                        context="",
                        source="pattern",
                    )
                )
            for m in _URL_PATTERN.finditer(text):
                entities.append(
                    ExtractedEntity(
                        name=m.group(),
                        entity_type="URL",
                        confidence=0.9,
                        context="",
                        source="pattern",
                    )
                )
            for m in _PHONE_PATTERN.finditer(text):
                entities.append(
                    ExtractedEntity(
                        name=m.group(),
                        entity_type="PHONE",
                        confidence=0.8,
                        context="",
                        source="pattern",
                    )
                )
            for pat in _DATE_PATTERNS:
                for m in pat.finditer(text):
                    entities.append(
                        ExtractedEntity(
                            name=m.group(),
                            entity_type="DATE",
                            confidence=0.7,
                            context="",
                            source="pattern",
                        )
                    )
            for keyword in _TECH_KEYWORDS:
                pat = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
                for m in pat.finditer(text):
                    entities.append(
                        ExtractedEntity(
                            name=m.group(),
                            entity_type="TECHNOLOGY",
                            confidence=0.6,
                            context="",
                            source="pattern",
                        )
                    )
        except Exception as e:
            logger.warning("Pattern entity extraction failed: %s", e)
        return entities

    def _deduplicate(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove noise, merge by normalized name, substring dedup, cap at MAX_ENTITY_RESULTS."""
        if not entities:
            return []
        min_name_len = 3
        source_priority = {"spacy": 2, "pattern": 1}
        entity_groups: Dict[str, List[ExtractedEntity]] = {}

        for entity in entities:
            normalized = entity.name.lower().strip()
            if not normalized or len(normalized) < min_name_len:
                continue
            if normalized in NOISE_WORDS:
                continue
            if re.match(r"^[\d\s%]+$", normalized):
                continue
            if re.match(r"^\d+\s*(percent|minutes?|hours?|days?|years?)$", normalized):
                continue
            if normalized not in entity_groups:
                entity_groups[normalized] = []
            entity_groups[normalized].append(entity)

        sorted_names = sorted(entity_groups.keys(), key=len, reverse=True)
        to_remove: Set[str] = set()
        for i, longer in enumerate(sorted_names):
            if longer in to_remove:
                continue
            for shorter in sorted_names[i + 1 :]:
                if shorter in to_remove:
                    continue
                if shorter != longer and shorter in longer:
                    longer_type = entity_groups[longer][0].entity_type
                    shorter_type = entity_groups[shorter][0].entity_type
                    if longer_type == shorter_type or "MISC" in (longer_type, shorter_type):
                        to_remove.add(shorter)
        for key in to_remove:
            del entity_groups[key]

        final_entities: List[ExtractedEntity] = []
        for group in entity_groups.values():
            if not group:
                continue
            group.sort(
                key=lambda e: (e.confidence, source_priority.get(e.source, 0)),
                reverse=True,
            )
            best = group[0]
            type_counts: Dict[str, int] = {}
            for e in group:
                type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1
            best.entity_type = max(type_counts.items(), key=lambda x: x[1])[0]
            avg_conf = sum(e.confidence for e in group) / len(group)
            best.confidence = min(0.95, avg_conf)
            final_entities.append(best)

        final_entities.sort(key=lambda e: e.confidence, reverse=True)
        return final_entities[: settings.MAX_ENTITY_RESULTS]

    def extract(self, text: str, max_length: int | None = None) -> List[ExtractedEntity]:
        """Run spaCy NER + pattern extraction, then deduplicate and return."""
        if not self._nlp:
            logger.warning("spaCy model not loaded")
            return []
        max_len = max_length or settings.MAX_TEXT_LENGTH
        if len(text) > max_len:
            text = text[:max_len]
        if not text.strip():
            return []

        doc = self._nlp(text)
        spacy_entities = self._extract_spacy_entities(text, doc)
        pattern_entities = self._extract_pattern_entities(text)
        combined = spacy_entities + pattern_entities
        return self._deduplicate(combined)
