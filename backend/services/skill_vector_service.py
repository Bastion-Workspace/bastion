"""
Skill Vector Service - Semantic skill indexing and search for auto-discovery.

Maintains a single Qdrant collection "skills" with payload filtering by user_id.
Built-in skills use user_id "__builtin__". Search restricts to requesting user + builtin.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from config import settings
from clients.vector_service_client import get_vector_service_client

logger = logging.getLogger(__name__)

SKILLS_COLLECTION_NAME = "skills"
BUILTIN_USER_ID = "__builtin__"


def _compose_embedding_text(skill: Dict[str, Any]) -> str:
    """Build natural-language text for skill vectorization (optimized for query matching)."""
    name = (skill.get("name") or skill.get("slug") or "").strip()
    description = (skill.get("description") or "").strip()
    tags = skill.get("tags") or []
    tags_str = ", ".join(str(t) for t in tags) if tags else ""
    parts = [name]
    if description:
        parts.append(description)
    if tags_str:
        parts.append(f"Use for: {tags_str}")
    return ". ".join(parts)


async def ensure_skills_collection() -> bool:
    """Create the skills collection if it does not exist."""
    client = await get_vector_service_client(required=False)
    if not client:
        return False
    info = await client.get_collection_info(SKILLS_COLLECTION_NAME)
    if info.get("success") and info.get("collection"):
        return True
    result = await client.create_collection(
        collection_name=SKILLS_COLLECTION_NAME,
        vector_size=settings.EMBEDDING_DIMENSIONS,
        distance="COSINE",
    )
    if result.get("success"):
        logger.info("Created skills vector collection")
        return True
    logger.warning("Failed to create skills collection: %s", result.get("error"))
    return False


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
        await ensure_skills_collection()
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
    payload = {
        "skill_id": str(skill_id),
        "user_id": str(user_id),
        "name": (skill.get("name") or "")[:500],
        "slug": (skill.get("slug") or "")[:200],
        "category": (skill.get("category") or "")[:100],
        "is_builtin": "true" if skill.get("is_builtin") else "false",
        "required_tools_json": json.dumps(required_tools),
    }
    point = {"id": str(skill_id), "vector": vector, "payload": payload}
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


async def search_skills(
    user_id: str,
    query: str,
    limit: int = 3,
    score_threshold: float = 0.5,
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
        await ensure_skills_collection()
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
    hits = await client.search_vectors(
        collection_name=SKILLS_COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit,
        score_threshold=0.0,
        filters=filters,
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
    results = []
    for hit in hits:
        payload = hit.get("payload") or {}
        results.append({
            "id": payload.get("skill_id") or hit.get("id"),
            "score": hit.get("score", 0.0),
            "name": payload.get("name"),
            "slug": payload.get("slug"),
            "category": payload.get("category"),
            "similarity_score": hit.get("score", 0.0),
        })
    return results


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
        await ensure_skills_collection()
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
        payload = {
            "skill_id": str(skill_id),
            "user_id": str(uid),
            "name": (skill.get("name") or "")[:500],
            "slug": (skill.get("slug") or "")[:200],
            "category": (skill.get("category") or "")[:100],
            "is_builtin": "true" if skill.get("is_builtin") else "false",
            "required_tools_json": json.dumps(required_tools),
        }
        points.append({"id": str(skill_id), "vector": vec, "payload": payload})

    if not points:
        return 0
    result = await client.upsert_vectors(SKILLS_COLLECTION_NAME, points)
    if not result.get("success"):
        logger.warning("sync_all_skills: batch upsert_vectors failed: %s", result.get("error"))
        return 0
    count = result.get("points_stored", len(points))
    logger.info("sync_all_skills: embedded %d skills", count)
    return count
