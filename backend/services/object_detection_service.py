"""
Object Detection Service - Orchestrates YOLO + CLIP detection and user annotation matching

Calls Image Vision Service for object detection (YOLO) and feature extraction (CLIP).
Matches detected regions against user-defined object annotations via ObjectEncodingService.
For large YOLO boxes, optionally chunks the bbox and runs CLIP on each cell so small
user-defined objects (e.g. a logo on the side of a truck) can match. Stores results in
detected_objects table.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from clients.image_vision_client import get_image_vision_client
from services.object_encoding_service import get_object_encoding_service
from services.database_manager.database_helpers import fetch_all, fetch_one

logger = logging.getLogger(__name__)

# Minimum side length (px) for a bbox to be chunked; each 2x2 cell will be at least half this.
MIN_SIDE_FOR_CHUNKING = 64
CHUNK_GRID_SIZE = 2

# Max semantic descriptions sent to vision (user names + optional user-typed terms) to limit CLIP cost.
MAX_SEMANTIC_DESCRIPTIONS = 50

# Minimum side length to run sub-grid refinement on other documents (avoid tiny cells).
MIN_SIDE_FOR_REFINEMENT = 32


def _chunk_bbox(
    bbox_x: int, bbox_y: int, bbox_width: int, bbox_height: int, grid: int = 2
) -> List[Dict[str, int]]:
    """Subdivide a bounding box into a grid of smaller boxes (e.g. 2x2)."""
    cells = []
    for row in range(grid):
        for col in range(grid):
            cw = bbox_width // grid
            ch = bbox_height // grid
            x = bbox_x + col * cw
            y = bbox_y + row * ch
            if col == grid - 1:
                cw = bbox_width - (grid - 1) * cw
            if row == grid - 1:
                ch = bbox_height - (grid - 1) * ch
            if cw > 0 and ch > 0:
                cells.append({"bbox_x": x, "bbox_y": y, "bbox_width": cw, "bbox_height": ch})
    return cells


class ObjectDetectionService:
    """Orchestrates object detection and user-defined object matching."""

    def __init__(self):
        self._vision_client = None
        self._object_encoding_service = None

    async def _get_vision_client(self):
        if self._vision_client is None:
            self._vision_client = await get_image_vision_client()
            await self._vision_client.initialize(required=False)
        return self._vision_client

    async def _get_object_encoding_service(self):
        if self._object_encoding_service is None:
            self._object_encoding_service = await get_object_encoding_service()
        return self._object_encoding_service

    async def _get_user_annotation_names(self, user_id: str) -> List[str]:
        """Fetch distinct object names from user's custom annotations for semantic sweep."""
        try:
            rows = await fetch_all(
                "SELECT DISTINCT object_name FROM user_object_annotations WHERE user_id = $1 AND object_name IS NOT NULL AND TRIM(object_name) != ''",
                user_id,
            )
            return [str(r["object_name"]).strip() for r in rows if r.get("object_name")] if rows else []
        except Exception as e:
            logger.warning("Failed to fetch user annotation names for semantic sweep: %s", e)
            return []

    async def _refine_bbox_subgrid(
        self,
        image_path: str,
        bbox: Dict[str, int],
        user_id: str,
        user_annotation_threshold: float,
    ) -> Dict[str, int]:
        """
        On a different document, subdivide the matching box into 2x2 and pick the cell
        that best matches the user annotation so the box centers on the actual item.
        """
        w, h = bbox.get("bbox_width", 0), bbox.get("bbox_height", 0)
        if w < MIN_SIDE_FOR_REFINEMENT or h < MIN_SIDE_FOR_REFINEMENT:
            return bbox
        vision_client = await self._get_vision_client()
        object_encoding = await self._get_object_encoding_service()
        cells = _chunk_bbox(
            bbox["bbox_x"], bbox["bbox_y"], w, h, grid=2
        )
        best_cell: Optional[Tuple[Dict[str, int], float]] = None
        for cell in cells:
            features = await vision_client.extract_object_features(
                image_path=image_path,
                bbox=cell,
                description="",
            )
            if not features or not features.get("combined_embedding"):
                continue
            matches = await object_encoding.search_similar_objects(
                query_embedding=features["combined_embedding"],
                user_id=user_id,
                top_k=1,
                similarity_threshold=user_annotation_threshold,
            )
            if matches and (best_cell is None or matches[0]["similarity_score"] > best_cell[1]):
                best_cell = (cell, matches[0]["similarity_score"])
        return best_cell[0] if best_cell else bbox

    async def detect_objects_in_image(
        self,
        document_id: str,
        image_path: str,
        user_id: str,
        class_filter: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        semantic_descriptions: Optional[List[str]] = None,
        match_user_annotations: bool = True,
        user_annotation_threshold: float = 0.75,
        chunk_large_boxes: bool = True,
        min_side_for_chunking: int = MIN_SIDE_FOR_CHUNKING,
        chunk_grid_size: int = CHUNK_GRID_SIZE,
    ) -> Dict[str, Any]:
        """
        Run object detection (YOLO + CLIP semantic sweep) and match user-defined objects.

        A semantic sweep over the image is always run using the user's custom annotation names
        (plus any optional semantic_descriptions), so custom objects (e.g. logos) are found without
        the user having to type a search term. For each YOLO detection we run CLIP on the full bbox;
        if it matches a user annotation we replace the label. For large boxes where the full crop
        did not match, we optionally chunk the bbox into a grid and run CLIP on each cell so small
        user-defined objects (e.g. a logo on the side of a truck) can match.

        Args:
            document_id: Document ID for storage.
            image_path: Path to image file.
            user_id: User ID for matching user annotations.
            class_filter: Optional YOLO class names to include.
            confidence_threshold: YOLO confidence threshold.
            semantic_descriptions: Optional extra text descriptions for CLIP (merged with user annotation names).
            match_user_annotations: If True, check each region against user annotations.
            user_annotation_threshold: Minimum similarity for user-defined match.
            chunk_large_boxes: If True, subdivide large YOLO boxes and run CLIP on each chunk.
            min_side_for_chunking: Only chunk if bbox width and height >= this (default 64).
            chunk_grid_size: Grid size for chunking (e.g. 2 = 2x2).

        Returns:
            Dict with objects (list of detections), image_width, image_height, processing_time_seconds.
        """
        vision_client = await self._get_vision_client()
        if not vision_client._initialized:
            return {
                "objects": [],
                "image_width": 0,
                "image_height": 0,
                "processing_time_seconds": 0.0,
                "error": "Image Vision Service unavailable",
            }

        # Always run semantic sweep with user's custom annotation names so custom objects (e.g. logos)
        # are found without the user having to type a search term.
        user_annotation_names = await self._get_user_annotation_names(user_id)
        combined_descriptions = list(
            set(
                [s.strip() for s in (semantic_descriptions or []) if s and str(s).strip()]
                + [n for n in user_annotation_names if n]
            )
        )
        if len(combined_descriptions) > MAX_SEMANTIC_DESCRIPTIONS:
            combined_descriptions = combined_descriptions[:MAX_SEMANTIC_DESCRIPTIONS]

        result = await vision_client.detect_objects(
            image_path=image_path,
            document_id=document_id,
            class_filter=class_filter,
            confidence_threshold=confidence_threshold,
            semantic_descriptions=combined_descriptions,
        )

        if not result:
            return {
                "objects": [],
                "image_width": 0,
                "image_height": 0,
                "processing_time_seconds": 0.0,
                "error": "Detection returned no result",
            }

        objects = list(result.get("objects", []))
        image_width = result.get("image_width", 0)
        image_height = result.get("image_height", 0)
        processing_time = result.get("processing_time_seconds", 0.0)

        if match_user_annotations and objects:
            object_encoding = await self._get_object_encoding_service()
            vision_client = await self._get_vision_client()
            replaced = []
            for obj in objects:
                bbox = {
                    "bbox_x": obj.get("bbox_x", 0),
                    "bbox_y": obj.get("bbox_y", 0),
                    "bbox_width": obj.get("bbox_width", 0),
                    "bbox_height": obj.get("bbox_height", 0),
                }
                features = await vision_client.extract_object_features(
                    image_path=image_path,
                    bbox=bbox,
                    description="",
                )
                if not features or not features.get("combined_embedding"):
                    replaced.append(obj)
                    continue
                matches = await object_encoding.search_similar_objects(
                    query_embedding=features["combined_embedding"],
                    user_id=user_id,
                    top_k=1,
                    similarity_threshold=user_annotation_threshold,
                )
                if matches:
                    best = matches[0]
                    source_doc = str((best.get("metadata") or {}).get("source_document_id") or "")
                    if source_doc == str(document_id):
                        refined = bbox
                    else:
                        refined = await self._refine_bbox_subgrid(
                            image_path, bbox, user_id, user_annotation_threshold
                        )
                    replaced.append({
                        "class_name": best["object_name"],
                        "class_id": 0,
                        "confidence": best["similarity_score"],
                        "bbox_x": refined["bbox_x"],
                        "bbox_y": refined["bbox_y"],
                        "bbox_width": refined["bbox_width"],
                        "bbox_height": refined["bbox_height"],
                        "detection_method": "user_defined",
                        "matched_description": best["object_name"],
                        "annotation_id": best.get("annotation_id"),
                    })
                    continue

                # Full bbox did not match. If bbox is large, chunk and run CLIP on each cell
                # so small user-defined objects (e.g. logo on truck) can match.
                w, h = bbox["bbox_width"], bbox["bbox_height"]
                if (
                    chunk_large_boxes
                    and w >= min_side_for_chunking
                    and h >= min_side_for_chunking
                ):
                    cells = _chunk_bbox(
                        bbox["bbox_x"], bbox["bbox_y"], w, h, grid=chunk_grid_size
                    )
                    chunk_matches: List[Tuple[Dict[str, int], Dict[str, Any]]] = []
                    for cell in cells:
                        cell_features = await vision_client.extract_object_features(
                            image_path=image_path,
                            bbox=cell,
                            description="",
                        )
                        if not cell_features or not cell_features.get("combined_embedding"):
                            continue
                        cell_matches = await object_encoding.search_similar_objects(
                            query_embedding=cell_features["combined_embedding"],
                            user_id=user_id,
                            top_k=1,
                            similarity_threshold=user_annotation_threshold,
                        )
                        if cell_matches:
                            chunk_matches.append((cell, cell_matches[0]))
                    # Keep original YOLO detection (e.g. truck).
                    replaced.append(obj)
                    # Add at most one detection per annotation_id (best-scoring chunk).
                    by_ann: Dict[Optional[int], Tuple[Dict[str, int], Dict[str, Any]]] = {}
                    for cell, best in chunk_matches:
                        ann_id = best.get("annotation_id")
                        if ann_id not in by_ann or best["similarity_score"] > by_ann[ann_id][1]["similarity_score"]:
                            by_ann[ann_id] = (cell, best)
                    for _ann_id, (cell, best) in by_ann.items():
                        source_doc = str((best.get("metadata") or {}).get("source_document_id") or "")
                        if source_doc == str(document_id):
                            refined = cell
                        else:
                            refined = await self._refine_bbox_subgrid(
                                image_path, cell, user_id, user_annotation_threshold
                            )
                        replaced.append({
                            "class_name": best["object_name"],
                            "class_id": 0,
                            "confidence": best["similarity_score"],
                            "bbox_x": refined["bbox_x"],
                            "bbox_y": refined["bbox_y"],
                            "bbox_width": refined["bbox_width"],
                            "bbox_height": refined["bbox_height"],
                            "detection_method": "user_defined",
                            "matched_description": best["object_name"],
                            "annotation_id": best.get("annotation_id"),
                        })
                else:
                    replaced.append(obj)
            objects = replaced

        # When we ran a semantic sweep for user annotation names, hide raw clip_semantic boxes
        # so only YOLO and user_defined (embedding-matched) detections are shown.
        if combined_descriptions:
            objects = [o for o in objects if o.get("detection_method") != "clip_semantic"]

        return {
            "objects": objects,
            "image_width": image_width,
            "image_height": image_height,
            "processing_time_seconds": processing_time,
        }

    async def process_detection_results(
        self,
        document_id: str,
        objects: List[Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Store detection results in detected_objects table.

        Args:
            document_id: Document ID.
            objects: List of detection dicts (class_name, confidence, bbox_*, detection_method, annotation_id).
            user_id: Optional user ID for confirmed_by.

        Returns:
            List of stored rows with id.
        """
        if not objects:
            return []

        stored = []
        for obj in objects:
            class_name = obj.get("class_name", "object")
            annotation_id = obj.get("annotation_id")
            if annotation_id is not None:
                ann_check = await fetch_one(
                    "SELECT id FROM user_object_annotations WHERE id = $1",
                    annotation_id if isinstance(annotation_id, int) else int(annotation_id),
                )
                if not ann_check:
                    annotation_id = None
            row = await fetch_one(
                """
                INSERT INTO detected_objects (
                    document_id, class_name, original_class_name, detection_method, confidence,
                    bbox_x, bbox_y, bbox_width, bbox_height, annotation_id
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id, document_id, class_name, original_class_name, detection_method, confidence,
                    bbox_x, bbox_y, bbox_width, bbox_height, annotation_id, detected_at
                """,
                document_id,
                class_name,
                class_name,
                obj.get("detection_method", "yolo"),
                float(obj.get("confidence", 0.5)),
                int(obj.get("bbox_x", 0)),
                int(obj.get("bbox_y", 0)),
                int(obj.get("bbox_width", 0)),
                int(obj.get("bbox_height", 0)),
                annotation_id,
            )
            if row:
                stored.append(dict(row))
        return stored

    async def get_detected_objects(
        self,
        document_id: str,
        include_rejected: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch detected objects for a document. By default excludes rejected (hidden) objects."""
        where = "document_id = $1"
        params: List[Any] = [document_id]
        if not include_rejected:
            where += " AND (rejected IS NULL OR rejected = FALSE)"
        try:
            rows = await fetch_all(
                f"""
                SELECT id, document_id, class_name, original_class_name, detection_method, confidence,
                       bbox_x, bbox_y, bbox_width, bbox_height, annotation_id,
                       confirmed, rejected, user_tag, detected_at, confirmed_by, confirmed_at
                FROM detected_objects
                WHERE {where}
                ORDER BY detected_at
                """,
                *params,
            )
        except Exception:
            rows = await fetch_all(
                f"""
                SELECT id, document_id, class_name, detection_method, confidence,
                       bbox_x, bbox_y, bbox_width, bbox_height, annotation_id,
                       confirmed, rejected, user_tag, detected_at, confirmed_by, confirmed_at
                FROM detected_objects
                WHERE {where}
                ORDER BY detected_at
                """,
                *params,
            )
        out = [dict(r) for r in rows] if rows else []
        for d in out:
            d.setdefault("user_tag", None)
            d.setdefault("original_class_name", None)
        return out


_object_detection_service: Optional[ObjectDetectionService] = None


async def get_object_detection_service() -> ObjectDetectionService:
    """Get or create global Object Detection Service instance."""
    global _object_detection_service
    if _object_detection_service is None:
        _object_detection_service = ObjectDetectionService()
    return _object_detection_service
