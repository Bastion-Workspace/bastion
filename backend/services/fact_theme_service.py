"""
User fact theme clustering and theme-first retrieval with adaptive context budgets.

Uses existing user_facts.embedding vectors. No LLM calls in retrieval.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from services.database_manager.database_helpers import execute, fetch_all, fetch_one

logger = logging.getLogger(__name__)

MIN_EMBEDDED_FACTS_TO_CLUSTER = 6
THEME_MATCH_THRESHOLD = 0.30
FACT_BASE_THRESHOLD = 0.32
EPISODE_BASE_THRESHOLD = 0.28
MAX_THEMES_FOR_RETRIEVAL = 5
MAX_FACTS_CAP = 15
MAX_EPISODES_CAP = 10
MIN_FACTS_FROM_THRESHOLD_POOL = 3
MIN_EPISODES_FROM_THRESHOLD_POOL = 2
TOP_DROP_RATIO = 0.45
STEP_DROP_RATIO = 0.55
AGED_EPISODE_SCORE_MULT = 0.88

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "my",
        "your",
        "their",
        "our",
        "me",
        "him",
        "her",
        "us",
        "them",
        "not",
        "no",
        "yes",
        "with",
        "from",
        "by",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "now",
    }
)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


def _top_keywords(texts: List[str], max_keywords: int = 2) -> str:
    counts: Counter[str] = Counter()
    for t in texts:
        for w in _tokenize(t):
            if len(w) < 3 or w in _STOPWORDS:
                continue
            counts[w] += 1
    if not counts:
        return "general"
    top = [w for w, _ in counts.most_common(max_keywords)]
    return ", ".join(top) if top else "general"


def _build_theme_label(categories: List[str], values: List[str], fact_keys: List[str]) -> str:
    cat = "general"
    if categories:
        cat = Counter(categories).most_common(1)[0][0] or "general"
    kws = _top_keywords(values + fact_keys, max_keywords=2)
    label = f"{cat}: {kws}"
    return label[:255]


def adaptive_select_facts(scored: List[Tuple[float, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    scored: list of (similarity, fact) sorted by similarity descending.
    Keep facts with score >= FACT_BASE_THRESHOLD, apply drop-off after min_keep satisfied.
    """
    pool = [(s, f) for s, f in scored if s >= FACT_BASE_THRESHOLD]
    pool.sort(key=lambda x: -x[0])
    if not pool:
        return []
    top_score = pool[0][0]
    out: List[Dict[str, Any]] = []
    prev: Optional[float] = None
    for s, f in pool:
        if len(out) >= MAX_FACTS_CAP:
            break
        if len(out) >= MIN_FACTS_FROM_THRESHOLD_POOL:
            if s < top_score * TOP_DROP_RATIO:
                break
            if prev is not None and s < prev * STEP_DROP_RATIO:
                break
        out.append(f)
        prev = s
    return out


def adaptive_select_episodes(scored: List[Tuple[float, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    pool = [(s, e) for s, e in scored if s >= EPISODE_BASE_THRESHOLD]
    pool.sort(key=lambda x: -x[0])
    if not pool:
        return []
    top_score = pool[0][0]
    out: List[Dict[str, Any]] = []
    prev: Optional[float] = None
    for s, e in pool:
        if len(out) >= MAX_EPISODES_CAP:
            break
        if len(out) >= MIN_EPISODES_FROM_THRESHOLD_POOL:
            if s < top_score * TOP_DROP_RATIO:
                break
            if prev is not None and s < prev * STEP_DROP_RATIO:
                break
        out.append(e)
        prev = s
    return out


def _score_facts_against_query(
    facts: List[Dict[str, Any]], qvec: List[float]
) -> List[Tuple[float, Dict[str, Any]]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for f in facts:
        emb = f.get("embedding")
        if emb and isinstance(emb, (list, tuple)) and len(emb) == len(qvec):
            scored.append((cosine_similarity(emb, qvec), f))
        else:
            scored.append((0.0, f))
    scored.sort(key=lambda x: -x[0])
    return scored


def _score_episodes_against_query(
    episodes: List[Dict[str, Any]], qvec: List[float]
) -> List[Tuple[float, Dict[str, Any]]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for e in episodes:
        emb = e.get("embedding")
        if emb and isinstance(emb, (list, tuple)) and len(emb) == len(qvec):
            s = cosine_similarity(emb, qvec)
            if e.get("is_aged"):
                s *= AGED_EPISODE_SCORE_MULT
            scored.append((s, e))
        else:
            scored.append((0.0, e))
    scored.sort(key=lambda x: -x[0])
    return scored


def select_facts_for_query(
    facts: List[Dict[str, Any]],
    themes: List[Dict[str, Any]],
    qvec: List[float],
    *,
    use_themed_memory: bool = True,
) -> List[Dict[str, Any]]:
    """
    Theme-first fact selection with adaptive budget. Falls back to flat ranking when no themes.
    Facts without embeddings are appended after selected embedded facts if not already included.
    """
    if not facts:
        return []

    embedded = [f for f in facts if f.get("embedding")]
    no_emb = [f for f in facts if not f.get("embedding")]

    candidate_embedded = embedded
    if use_themed_memory and themes and qvec:
        theme_scored: List[Tuple[float, Dict[str, Any]]] = []
        for t in themes:
            c = t.get("centroid")
            if c and isinstance(c, (list, tuple)) and len(c) == len(qvec):
                theme_scored.append((cosine_similarity(c, qvec), t))
        theme_scored.sort(key=lambda x: -x[0])
        selected_theme_ids_list = [
            t["id"]
            for s, t in theme_scored
            if s >= THEME_MATCH_THRESHOLD and t.get("id") is not None
        ][:MAX_THEMES_FOR_RETRIEVAL]
        selected_theme_ids = set(selected_theme_ids_list)
        if selected_theme_ids:
            themed = [f for f in embedded if f.get("theme_id") in selected_theme_ids]
            if themed:
                candidate_embedded = themed

    scored = _score_facts_against_query(candidate_embedded, qvec)
    selected = adaptive_select_facts(scored)
    sel_ids = {id(f) for f in selected}
    for f in no_emb:
        if id(f) not in sel_ids:
            selected.append(f)
            sel_ids.add(id(f))
    return selected


def select_episodes_for_query(
    episodes: List[Dict[str, Any]],
    qvec: List[float],
) -> List[Dict[str, Any]]:
    if not episodes:
        return []
    scored = _score_episodes_against_query(episodes, qvec)
    selected = adaptive_select_episodes(scored)
    no_emb = [e for e in episodes if not e.get("embedding")]
    sel_ids = {id(e) for e in selected}
    for e in no_emb:
        if id(e) not in sel_ids:
            selected.append(e)
            sel_ids.add(id(e))
    return selected


async def load_themes_for_user(user_id: str) -> List[Dict[str, Any]]:
    try:
        rows = await fetch_all(
            """
            SELECT id, user_id, label, centroid, fact_count, created_at, updated_at
            FROM user_fact_themes
            WHERE user_id = $1
            """,
            user_id,
        )
        return [dict(r) for r in (rows or [])]
    except Exception as e:
        if "user_fact_themes" in str(e) or "does not exist" in str(e):
            return []
        logger.warning("load_themes_for_user failed: %s", e)
        return []


async def list_users_with_embedded_facts(min_count: int = MIN_EMBEDDED_FACTS_TO_CLUSTER) -> List[str]:
    rows = await fetch_all(
        """
        SELECT user_id, COUNT(*) AS c
        FROM user_facts
        WHERE embedding IS NOT NULL
        GROUP BY user_id
        HAVING COUNT(*) >= $1
        """,
        min_count,
    )
    return [str(r["user_id"]) for r in (rows or [])]


async def fetch_facts_for_clustering(user_id: str) -> List[Dict[str, Any]]:
    rows = await fetch_all(
        """
        SELECT id, fact_key, value, category, embedding
        FROM user_facts
        WHERE user_id = $1 AND embedding IS NOT NULL
        """,
        user_id,
    )
    out = []
    for r in rows or []:
        d = dict(r)
        emb = d.get("embedding")
        if emb and isinstance(emb, (list, tuple)) and len(emb) > 0:
            out.append(d)
    return out


async def cluster_user_facts(user_id: str) -> Dict[str, Any]:
    """
    Rebuild themes for one user from embedded facts. Clears old theme assignments first.
    """
    facts = await fetch_facts_for_clustering(user_id)
    if len(facts) < MIN_EMBEDDED_FACTS_TO_CLUSTER:
        return {"user_id": user_id, "skipped": True, "reason": "too_few_embedded", "count": len(facts)}

    X = np.array([np.asarray(f["embedding"], dtype=np.float64) for f in facts], dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        return {"user_id": user_id, "skipped": True, "reason": "single_row"}

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.55,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(X)

    await execute("UPDATE user_facts SET theme_id = NULL WHERE user_id = $1", user_id)
    await execute("DELETE FROM user_fact_themes WHERE user_id = $1", user_id)

    unique_labels = sorted(set(labels.tolist()))
    themes_created = 0
    for lab in unique_labels:
        idxs = [i for i in range(n) if labels[i] == lab]
        members = [facts[i] for i in idxs]
        centroid = np.mean(X[idxs], axis=0).tolist()
        categories = [str(m.get("category") or "general") for m in members]
        values = [str(m.get("value") or "") for m in members]
        keys = [str(m.get("fact_key") or "") for m in members]
        label_str = _build_theme_label(categories, values, keys)

        row = await fetch_one(
            """
            INSERT INTO user_fact_themes (user_id, label, centroid, fact_count, created_at, updated_at)
            VALUES ($1, $2, $3, $4, NOW(), NOW())
            RETURNING id
            """,
            user_id,
            label_str,
            centroid,
            len(members),
        )
        if not row:
            continue
        tid = row["id"]
        for m in members:
            await execute("UPDATE user_facts SET theme_id = $1 WHERE id = $2", tid, m["id"])
        themes_created += 1

    return {
        "user_id": user_id,
        "skipped": False,
        "facts_clustered": n,
        "themes_created": themes_created,
    }
