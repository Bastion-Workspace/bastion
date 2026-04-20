"""
Skill Vector Service - Semantic skill indexing and search for auto-discovery.

Maintains a single Qdrant collection "skills" with payload filtering by user_id.
Built-in skills use user_id "__builtin__". Search restricts to requesting user + builtin.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from clients.vector_service_client import get_vector_service_client

_hybrid_enabled = getattr(settings, "HYBRID_SEARCH_ENABLED", False)

logger = logging.getLogger(__name__)

SKILLS_COLLECTION_NAME = "skills"
BUILTIN_USER_ID = "__builtin__"


def _compose_embedding_text(skill: Dict[str, Any]) -> str:
    """Build natural-language text for skill vectorization (optimized for query matching)."""
    name = (skill.get("name") or skill.get("slug") or "").strip()
    description = (skill.get("description") or "").strip()
    category = (skill.get("category") or "").strip()
    tags = skill.get("tags") or []
    depends_on = skill.get("depends_on") or []
    required_tools = skill.get("required_tools") or []
    optional_tools = skill.get("optional_tools") or []
    examples = skill.get("examples") or []

    parts = [name]
    if description:
        parts.append(description)
    if category:
        parts.append(f"Category: {category}")
    if tags:
        parts.append("Use for: " + ", ".join(str(t) for t in tags))
    tool_names = [
        str(t).strip()
        for t in list(required_tools) + list(optional_tools)
        if str(t).strip()
    ]
    if tool_names:
        parts.append("Tools: " + ", ".join(dict.fromkeys(tool_names)))
    if depends_on:
        parts.append("Includes: " + ", ".join(str(d) for d in depends_on))
    ex_texts: List[str] = []
    for ex in examples[:3] if isinstance(examples, list) else []:
        if isinstance(ex, dict):
            q = (ex.get("query") or ex.get("use_case") or "").strip()
            if q:
                ex_texts.append(q)
    if ex_texts:
        parts.append("Examples: " + " | ".join(ex_texts))
    return ". ".join(parts)


async def ensure_skills_collection() -> Tuple[bool, bool]:
    """Create the skills collection if needed.

    Returns:
        (success, created_or_migrated): second flag True if collection was created or
        replaced for hybrid schema (caller should re-embed user-defined skills).
    """
    client = await get_vector_service_client(required=False)
    if not client:
        return False, False
    info = await client.get_collection_info(SKILLS_COLLECTION_NAME)
    exists = bool(info.get("success") and info.get("collection"))

    if exists and info.get("success"):
        coll = info.get("collection") or {}
        needs_recreate = False

        if _hybrid_enabled:
            schema = (coll.get("schema_type") or "").strip().lower()
            if schema != "named_hybrid":
                logger.warning(
                    "Replacing skills collection for hybrid schema (schema_type=%r)",
                    schema or "unknown",
                )
                needs_recreate = True

        existing_dims = coll.get("vector_size", 0)
        if existing_dims and existing_dims != settings.EMBEDDING_DIMENSIONS:
            logger.warning(
                "Replacing skills collection: dimension mismatch existing=%s configured=%s",
                existing_dims,
                settings.EMBEDDING_DIMENSIONS,
            )
            needs_recreate = True

        if needs_recreate:
            del_result = await client.delete_collection(SKILLS_COLLECTION_NAME)
            if not del_result.get("success"):
                logger.warning(
                    "Failed to delete skills collection for migration: %s",
                    del_result.get("error"),
                )
                return False, False
            exists = False

    if exists:
        return True, False

    result = await client.create_collection(
        collection_name=SKILLS_COLLECTION_NAME,
        vector_size=settings.EMBEDDING_DIMENSIONS,
        distance="COSINE",
        enable_sparse=_hybrid_enabled,
    )
    if result.get("success"):
        logger.info("Created skills vector collection")
        return True, True
    logger.warning("Failed to create skills collection: %s", result.get("error"))
    return False, False


async def embed_skill(skill: Dict[str, Any]) -> bool:
    """Generate embedding for a skill and upsert into the skills collection."""
    skill_id = skill.get("id")
    if not skill_id:
        logger.warning("embed_skill: skill has no id")
        return False
    client = await get_vector_service_client(required=False)
    if not client:
        logger.warning("Vector service unavailable, skipping skill embed")
        return False
    try:
        ok, _ = await ensure_skills_collection()
        if not ok:
            return False
    except Exception as e:
        logger.warning("ensure_skills_collection failed: %s", e)
        return False
    text = _compose_embedding_text(skill)
    if not text.strip():
        logger.debug("Skill %s has no text to embed", skill_id)
        return False
    try:
        vector = await client.generate_embedding(text)
    except Exception as e:
        logger.warning("generate_embedding failed for skill %s: %s", skill_id, e)
        return False
    user_id = BUILTIN_USER_ID if skill.get("is_builtin") else (skill.get("user_id") or "system")
    required_tools = skill.get("required_tools") or []
    req_conn = skill.get("required_connection_types") or []
    payload = {
        "skill_id": str(skill_id),
        "user_id": str(user_id),
        "name": (skill.get("name") or "")[:500],
        "slug": (skill.get("slug") or "")[:200],
        "category": (skill.get("category") or "")[:100],
        "is_builtin": "true" if skill.get("is_builtin") else "false",
        "required_tools_json": json.dumps(required_tools),
        "required_connection_types_json": json.dumps(list(req_conn)),
        "tags_json": json.dumps(list(skill.get("tags") or [])),
    }
    point = {"id": str(skill_id), "vector": vector, "payload": payload}
    if _hybrid_enabled:
        try:
            from services.bm25_encoder import get_default_bm25_encoder
            sv = get_default_bm25_encoder().encode(text)
            if sv and sv.get("indices"):
                point["sparse_vector"] = sv
        except Exception as e:
            logger.warning("BM25 encode failed for skill %s: %s", skill_id, e)
    result = await client.upsert_vectors(SKILLS_COLLECTION_NAME, [point])
    if not result.get("success"):
        logger.warning("upsert_vectors failed for skill %s: %s", skill_id, result.get("error"))
        return False
    logger.debug("Embedded skill %s", skill_id)
    return True


async def remove_skill_vector(skill_id: str) -> bool:
    """Delete the vector for a skill by id (point id = skill_id)."""
    client = await get_vector_service_client(required=False)
    if not client:
        return False
    result = await client.delete_vectors(
        SKILLS_COLLECTION_NAME,
        [{"field": "skill_id", "value": str(skill_id), "operator": "equals"}],
    )
    if not result.get("success"):
        logger.warning("remove_skill_vector failed for %s: %s", skill_id, result.get("error"))
        return False
    return True


def _payload_required_connection_types(payload: Dict[str, Any]) -> List[str]:
    raw = payload.get("required_connection_types_json") or "[]"
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return [str(x).strip() for x in data if x and str(x).strip()] if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _skill_connection_requirements_met(
    required_types: List[str],
    active_connection_types: Optional[List[str]],
) -> bool:
    if not required_types:
        return True
    if active_connection_types is None:
        return True
    active = {str(t).strip().lower() for t in active_connection_types if t}
    if "contacts" in active:
        active.add("email")
    if "code_platform" in active:
        active.add("github")
        active.add("gitea")
    return all((rt or "").strip().lower() in active for rt in required_types)


def _payload_tags(payload: Dict[str, Any]) -> List[str]:
    raw = payload.get("tags_json") or "[]"
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, list):
            return []
        return [str(t).strip() for t in data if t and str(t).strip()]
    except (json.JSONDecodeError, TypeError):
        return []


def _skill_payload_matches_query(query: str, payload: Dict[str, Any]) -> bool:
    """
    True if tags, name, slug, or category align with the query (keyword bypass for vector threshold).
    """
    ql = (query or "").strip().lower()
    if not ql:
        return False
    for tag in _payload_tags(payload):
        tl = tag.lower()
        if len(tl) < 2:
            continue
        if tl in ql or ql in tl:
            return True
    name = (payload.get("name") or "").strip().lower()
    slug = (payload.get("slug") or "").strip().lower()
    category = (payload.get("category") or "").strip().lower()
    for field in (name, slug, category):
        if not field:
            continue
        if ql == field or ql in field or field in ql:
            return True
    q_tokens = [x for x in ql.replace("-", " ").split() if len(x) > 1]
    if q_tokens and name:
        if all(tok in name for tok in q_tokens):
            return True
    if q_tokens and slug:
        slug_text = slug.replace("-", " ")
        if all(tok in slug_text for tok in q_tokens):
            return True
    return False


async def search_skills(
    user_id: str,
    query: str,
    limit: int = 3,
    score_threshold: float = 0.5,
    active_connection_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search skills by semantic similarity. Returns only skills visible to the user
    (user_id = requesting user or __builtin__). Results include similarity_score.
    """
    if not (query or "").strip():
        return []
    client = await get_vector_service_client(required=False)
    if not client:
        logger.warning("Vector service unavailable for skill search")
        return []
    try:
        ok, _ = await ensure_skills_collection()
        if not ok:
            return []
    except Exception as e:
        logger.warning("ensure_skills_collection failed: %s", e)
        return []
    try:
        query_vector = await client.generate_embedding(query.strip())
    except Exception as e:
        logger.warning("generate_embedding for query failed: %s", e)
        return []
    filters = [
        {
            "field": "user_id",
            "values": [user_id, BUILTIN_USER_ID],
            "operator": "any_of",
        }
    ]
    vec_limit = limit
    if active_connection_types:
        vec_limit = min(50, max(limit * 8, limit))

    sparse_query = None
    fusion = ""
    if _hybrid_enabled:
        try:
            from services.bm25_encoder import get_default_bm25_encoder
            sv = get_default_bm25_encoder().encode(query.strip())
            if sv and sv.get("indices"):
                sparse_query = sv
                fusion = "rrf"
        except Exception as e:
            logger.warning("BM25 encode failed for skill query, falling back to dense-only: %s", e)

    hits = await client.search_vectors(
        collection_name=SKILLS_COLLECTION_NAME,
        query_vector=query_vector,
        limit=vec_limit,
        score_threshold=0.0,
        filters=filters,
        sparse_query_vector=sparse_query,
        fusion_mode=fusion,
    )
    if not hits:
        unfiltered = await client.search_vectors(
            collection_name=SKILLS_COLLECTION_NAME,
            query_vector=query_vector,
            limit=5,
            score_threshold=0.0,
            filters=None,
        )
        if unfiltered:
            logger.warning(
                "search_skills: filtered search returned 0 but unfiltered found %d (filter may be wrong)",
                len(unfiltered),
            )
        else:
            logger.warning(
                "search_skills: 0 results for user_id=%s query='%s' (collection may be empty)",
                user_id,
                (query or "")[:100],
            )
    # When hybrid search is enabled, use threshold-only ranking (RRF when available; dense-only if BM25 failed).
    # Tag/name/slug bypass remains only for HYBRID_SEARCH_ENABLED=false (dense-only collections).
    if _hybrid_enabled:
        results: List[Dict[str, Any]] = []
        for hit in hits:
            payload = hit.get("payload") or {}
            req_conn = _payload_required_connection_types(payload)
            if not _skill_connection_requirements_met(req_conn, active_connection_types):
                continue
            score = float(hit.get("score") or 0.0)
            if score < float(score_threshold):
                continue
            results.append({
                "id": payload.get("skill_id") or hit.get("id"),
                "score": score,
                "name": payload.get("name"),
                "slug": payload.get("slug"),
                "category": payload.get("category"),
                "similarity_score": score,
            })
            if len(results) >= limit:
                break
        return results

    # Dense-only collections: tag/name/slug keyword bypass below cosine threshold
    boosted: List[tuple] = []
    vector_ok: List[tuple] = []
    filtered_below_threshold: List[Dict[str, Any]] = []
    qstrip = (query or "").strip()

    for hit in hits:
        payload = hit.get("payload") or {}
        req_conn = _payload_required_connection_types(payload)
        if not _skill_connection_requirements_met(req_conn, active_connection_types):
            continue
        score = float(hit.get("score") or 0.0)
        entry = {
            "id": payload.get("skill_id") or hit.get("id"),
            "score": hit.get("score", 0.0),
            "name": payload.get("name"),
            "slug": payload.get("slug"),
            "category": payload.get("category"),
            "similarity_score": hit.get("score", 0.0),
        }
        if _skill_payload_matches_query(qstrip, payload):
            boosted.append((score, entry))
        elif score >= float(score_threshold):
            vector_ok.append((score, entry))
        else:
            filtered_below_threshold.append({
                "name": payload.get("name") or "",
                "slug": payload.get("slug") or "",
                "score": round(score, 4),
            })

    boosted.sort(key=lambda x: -x[0])
    vector_ok.sort(key=lambda x: -x[0])
    merged: List[Dict[str, Any]] = []
    seen_ids = set()
    for _, entry in boosted + vector_ok:
        sid = str(entry.get("id") or "")
        if not sid or sid in seen_ids:
            continue
        seen_ids.add(sid)
        merged.append(entry)
        if len(merged) >= limit:
            break

    if filtered_below_threshold:
        logger.info(
            "search_skills: filtered below vector threshold=%s (no tag/name/slug match), sample: %s",
            score_threshold,
            filtered_below_threshold[:20],
        )

    return merged


async def sync_all_skills(user_id: Optional[str] = None, upsert_only: bool = False) -> int:
    """
    Re-embed all skills (for startup seeding). If user_id is None, sync only built-in skills.
    If user_id is set, sync that user's skills only. Returns count of skills embedded.
    When upsert_only is True, skip the delete step (use from backend post-seed to avoid wiping
    the collection that tools-service may have just populated).
    """
    from services.agent_skills_service import list_skills

    client = await get_vector_service_client(required=False)
    if not getattr(client, "_initialized", False):
        logger.warning("Vector service unavailable for sync_all_skills (client not initialized)")
        return 0
    try:
        ok, _ = await ensure_skills_collection()
        if not ok:
            return 0
    except Exception as e:
        logger.warning("ensure_skills_collection failed: %s", e)
        return 0
    if user_id is None:
        try:
            skills = await list_skills(user_id="", category=None, include_builtin=True)
            skills = [s for s in skills if s.get("is_builtin")]
        except Exception as e:
            logger.warning("list_skills (built-in) failed: %s", e)
            return 0
    else:
        try:
            skills = await list_skills(user_id=user_id, category=None, include_builtin=False)
        except Exception as e:
            logger.warning("list_skills (user %s) failed: %s", user_id, e)
            return 0
    if not skills:
        logger.warning("sync_all_skills: no skills to embed (built-in skills may not be seeded in DB)")
        return 0

    target_uid = BUILTIN_USER_ID if user_id is None else user_id
    if not upsert_only:
        try:
            result = await client.delete_vectors(
                SKILLS_COLLECTION_NAME,
                [{"field": "user_id", "value": target_uid, "operator": "equals"}],
            )
            if result.get("success") and result.get("points_deleted", 0) > 0:
                logger.info("sync_all_skills: removed %d stale/orphan vectors for user %s", result.get("points_deleted", 0), target_uid)
        except Exception as e:
            logger.warning("sync_all_skills: delete stale vectors failed (continuing): %s", e)

    texts = [_compose_embedding_text(s) for s in skills]
    try:
        vectors = await client.generate_embeddings(texts)
    except Exception as e:
        logger.warning("sync_all_skills: batch generate_embeddings failed: %s", e)
        return 0
    if len(vectors) != len(skills):
        logger.warning("sync_all_skills: embedding count mismatch (%d vs %d)", len(vectors), len(skills))
        return 0

    points = []
    for i, skill in enumerate(skills):
        skill_id = skill.get("id")
        if not skill_id:
            continue
        vec = vectors[i] if i < len(vectors) else None
        if not vec:
            continue
        uid = BUILTIN_USER_ID if skill.get("is_builtin") else (skill.get("user_id") or "system")
        required_tools = skill.get("required_tools") or []
        req_conn_batch = skill.get("required_connection_types") or []
        payload = {
            "skill_id": str(skill_id),
            "user_id": str(uid),
            "name": (skill.get("name") or "")[:500],
            "slug": (skill.get("slug") or "")[:200],
            "category": (skill.get("category") or "")[:100],
            "is_builtin": "true" if skill.get("is_builtin") else "false",
            "required_tools_json": json.dumps(required_tools),
            "required_connection_types_json": json.dumps(list(req_conn_batch)),
            "tags_json": json.dumps(list(skill.get("tags") or [])),
        }
        point = {"id": str(skill_id), "vector": vec, "payload": payload}
        if _hybrid_enabled:
            try:
                from services.bm25_encoder import get_default_bm25_encoder
                sv = get_default_bm25_encoder().encode(texts[i])
                if sv and sv.get("indices"):
                    point["sparse_vector"] = sv
            except Exception as e:
                logger.warning("BM25 encode failed for skill %s in sync: %s", skill_id, e)
        points.append(point)

    if not points:
        return 0
    result = await client.upsert_vectors(SKILLS_COLLECTION_NAME, points)
    if not result.get("success"):
        logger.warning("sync_all_skills: batch upsert_vectors failed: %s", result.get("error"))
        return 0
    count = result.get("points_stored", len(points))
    logger.info("sync_all_skills: embedded %d skills", count)
    return count
