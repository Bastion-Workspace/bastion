"""
BM25 sparse vector encoder for hybrid search.

Produces sparse vectors (indices + values) suitable for Qdrant's named
sparse vector fields.  Thread-safe after fit() or load().
"""

import json
import math
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENGLISH_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "just", "me", "my", "no", "nor", "not", "of", "on",
    "or", "our", "out", "own", "s", "she", "so", "some", "t", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "to",
    "too", "up", "us", "was", "we", "were", "what", "when", "which", "who",
    "will", "with", "would", "you", "your",
})

_SPLIT_RE = re.compile(r"[a-z0-9]+")


class BM25Encoder:
    """BM25 sparse vector encoder with fit/encode/save/load."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        use_stopwords: bool = True,
        default_idf: Optional[float] = None,
    ):
        self.k1 = k1
        self.b = b
        self.use_stopwords = use_stopwords
        # token -> integer index (stable across encode calls)
        self._vocab: Dict[str, int] = {}
        # token -> IDF weight
        self._idf: Dict[str, float] = {}
        self._avgdl: float = 0.0
        self._n_docs: int = 0
        self._fitted = False
        # IDF assigned to tokens not seen during fit (rare = valuable)
        self._default_idf = default_idf

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def tokenize(self, text: str) -> List[str]:
        tokens = _SPLIT_RE.findall(text.lower())
        if self.use_stopwords:
            return [t for t in tokens if t not in ENGLISH_STOPWORDS]
        return tokens

    def fit(self, documents: List[str]) -> "BM25Encoder":
        """Build vocabulary + IDF from a corpus of document texts."""
        n = len(documents)
        if n == 0:
            logger.warning("BM25Encoder.fit called with empty corpus")
            self._fitted = True
            return self

        df: Dict[str, int] = {}
        total_len = 0
        for doc in documents:
            tokens = self.tokenize(doc)
            total_len += len(tokens)
            seen = set(tokens)
            for tok in seen:
                df[tok] = df.get(tok, 0) + 1

        self._n_docs = n
        self._avgdl = total_len / n if n else 1.0

        self._vocab = {}
        self._idf = {}
        for idx, (tok, doc_freq) in enumerate(sorted(df.items())):
            self._vocab[tok] = idx
            self._idf[tok] = math.log((n - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

        if self._default_idf is None:
            self._default_idf = math.log((n + 0.5) / 0.5 + 1.0)

        self._fitted = True
        logger.info(
            "BM25Encoder fitted: %d docs, %d vocab, avgdl=%.1f",
            n, len(self._vocab), self._avgdl,
        )
        return self

    def encode(self, text: str) -> Dict[str, List]:
        """Produce sparse vector {"indices": [...], "values": [...]}."""
        if not self._fitted:
            raise RuntimeError("BM25Encoder not fitted; call fit() or load() first")

        tokens = self.tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}

        doc_len = len(tokens)
        tf: Dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1

        indices: List[int] = []
        values: List[float] = []
        default_idf = self._default_idf or 5.0

        for tok, freq in tf.items():
            idx = self._vocab.get(tok)
            if idx is None:
                # Unknown token: assign next index and default high IDF
                idx = len(self._vocab)
                self._vocab[tok] = idx
                idf = default_idf
            else:
                idf = self._idf.get(tok, default_idf)

            avgdl = self._avgdl if self._avgdl > 0 else 1.0
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
            score = idf * numerator / denominator
            if score > 0:
                indices.append(idx)
                values.append(round(score, 6))

        return {"indices": indices, "values": values}

    def save(self, path: str) -> None:
        """Persist vocabulary and parameters to JSON."""
        data = {
            "k1": self.k1,
            "b": self.b,
            "use_stopwords": self.use_stopwords,
            "n_docs": self._n_docs,
            "avgdl": self._avgdl,
            "default_idf": self._default_idf,
            "vocab": self._vocab,
            "idf": self._idf,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info("BM25Encoder saved to %s (%d vocab)", path, len(self._vocab))

    def load(self, path: str) -> "BM25Encoder":
        """Restore vocabulary and parameters from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.k1 = data.get("k1", self.k1)
        self.b = data.get("b", self.b)
        self.use_stopwords = data.get("use_stopwords", self.use_stopwords)
        self._n_docs = data.get("n_docs", 0)
        self._avgdl = data.get("avgdl", 1.0)
        self._default_idf = data.get("default_idf")
        self._vocab = {k: int(v) for k, v in data.get("vocab", {}).items()}
        self._idf = {k: float(v) for k, v in data.get("idf", {}).items()}
        self._fitted = True
        logger.info(
            "BM25Encoder loaded from %s: %d vocab, %d docs, avgdl=%.1f",
            path, len(self._vocab), self._n_docs, self._avgdl,
        )
        return self


def build_default_english_idf(
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Encoder:
    """Build a general-purpose English IDF encoder from synthetic term frequencies.

    This produces a reasonable IDF distribution for BM25 without needing an
    actual corpus.  High-frequency English words get low IDF; rare/domain words
    get the default high IDF at encode time.
    """
    # Approximate document frequencies for common English tokens, as if from
    # a 100,000 document corpus.  Tokens not listed here get the default IDF
    # (high value = rare = informative).
    N = 100_000
    synthetic_df: Dict[str, int] = {
        "the": 95000, "be": 90000, "to": 88000, "of": 87000, "and": 86000,
        "a": 85000, "in": 84000, "that": 80000, "have": 78000, "i": 76000,
        "it": 75000, "for": 74000, "not": 72000, "on": 70000, "with": 69000,
        "he": 65000, "as": 64000, "you": 63000, "do": 62000, "at": 60000,
        "this": 58000, "but": 57000, "his": 55000, "by": 54000, "from": 53000,
        "they": 52000, "we": 51000, "say": 48000, "her": 47000, "she": 46000,
        "or": 45000, "an": 44000, "will": 43000, "my": 42000, "one": 41000,
        "all": 40000, "would": 39000, "there": 38000, "their": 37000, "what": 36000,
        "so": 35000, "up": 34000, "out": 33000, "if": 32000, "about": 31000,
        "who": 30000, "get": 29000, "which": 28000, "go": 27000, "me": 26000,
        "when": 25000, "make": 24000, "can": 23000, "like": 22000, "time": 21000,
        "no": 20000, "just": 19000, "him": 18000, "know": 17000, "take": 16000,
        "people": 15000, "into": 14000, "year": 13000, "your": 12000, "good": 11000,
        "some": 10000, "could": 9500, "them": 9000, "see": 8500, "other": 8000,
        "than": 7500, "then": 7000, "now": 6500, "look": 6000, "only": 5500,
        "come": 5000, "its": 4800, "over": 4600, "think": 4400, "also": 4200,
        "back": 4000, "after": 3800, "use": 3600, "two": 3400, "how": 3200,
        "our": 3000, "work": 2800, "first": 2600, "well": 2400, "way": 2200,
        "even": 2000, "new": 1900, "want": 1800, "because": 1700, "any": 1600,
        "give": 1500, "day": 1400, "most": 1300, "find": 1200, "here": 1100,
        "thing": 1000, "many": 950, "those": 900, "tell": 850, "very": 800,
        "when": 750, "need": 700, "call": 650, "hand": 600, "high": 550,
        "keep": 500, "last": 480, "long": 460, "great": 440, "small": 420,
        "end": 400, "put": 380, "home": 360, "read": 340, "own": 320,
        "leave": 300, "never": 280, "let": 260, "thought": 240, "city": 220,
        "far": 200, "point": 190, "write": 180, "next": 170, "begin": 160,
        "life": 150, "group": 140, "always": 130, "run": 120, "both": 110,
        "press": 100, "turn": 95, "real": 90, "might": 85, "still": 80,
        "set": 75, "close": 70, "night": 65, "hard": 60, "open": 55,
        "help": 50, "start": 48, "show": 46, "world": 44, "name": 42,
        "move": 40, "line": 38, "number": 36, "head": 34, "stand": 32,
        "play": 30, "every": 28, "change": 26, "follow": 24, "add": 22,
        "house": 20, "study": 18, "book": 16, "hear": 14, "plan": 12,
        "answer": 10, "grow": 9, "watch": 8, "talk": 7, "walk": 6,
        "search": 5, "email": 4, "send": 4, "file": 4, "document": 4,
        "create": 3, "delete": 3, "update": 3, "list": 3, "folder": 3,
        "query": 2, "database": 2, "vector": 2, "embed": 2, "index": 2,
        "api": 1, "endpoint": 1, "webhook": 1, "server": 1, "client": 1,
    }

    encoder = BM25Encoder(k1=k1, b=b, use_stopwords=True)
    encoder._n_docs = N
    encoder._avgdl = 250.0
    for idx, (tok, doc_freq) in enumerate(sorted(synthetic_df.items())):
        encoder._vocab[tok] = idx
        encoder._idf[tok] = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
    encoder._default_idf = math.log((N + 0.5) / 0.5 + 1.0)
    encoder._fitted = True
    return encoder


# ---------------------------------------------------------------------------
# Lazy-loaded singleton for the backend process
# ---------------------------------------------------------------------------

_default_encoder: Optional[BM25Encoder] = None


def get_default_bm25_encoder() -> BM25Encoder:
    """Return a singleton BM25Encoder loaded from the default IDF file.

    Falls back to a synthetic English IDF when the JSON file is unavailable
    (e.g. during development outside Docker).
    """
    global _default_encoder
    if _default_encoder is not None:
        return _default_encoder

    idf_path = os.environ.get("BM25_DEFAULT_IDF_PATH", "/app/data/bm25_default_idf.json")
    try:
        enc = BM25Encoder()
        enc.load(idf_path)
        _default_encoder = enc
        return _default_encoder
    except FileNotFoundError:
        logger.warning(
            "BM25 default IDF file not found at %s — using synthetic English IDF",
            idf_path,
        )
    except Exception as exc:
        logger.warning("Failed to load BM25 IDF from %s: %s — using synthetic English IDF", idf_path, exc)

    _default_encoder = build_default_english_idf()
    return _default_encoder
