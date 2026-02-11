"""
Vision Engine - Face detection and object detection
Face detection: face_recognition library. Object detection: YOLOv8 + CLIP.
CPU-optimized.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logging.warning("face_recognition library not available")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None
    logging.warning("ultralytics (YOLO) not available")

try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    CLIPProcessor = None
    CLIPModel = None
    torch = None
    logging.warning("transformers (CLIP) not available")

from PIL import Image

logger = logging.getLogger(__name__)


def _get_vision_config():
    """Lazy load config to avoid import at module load."""
    from config.settings import settings
    return settings


class VisionEngine:
    """Handles face detection and encoding"""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize vision engine

        Args:
            device: "cpu" or "cuda" (currently only CPU is supported via face_recognition)
        """
        self.device = device
        self._initialized = False
        self._object_detection_initialized = False
        self._clip_only_initialized = False
        self.yolo_model = None
        self.clip_model = None
        self.clip_processor = None
        self.clip_embedding_dim = 512

        if not FACE_RECOGNITION_AVAILABLE:
            raise RuntimeError("face_recognition library is not installed")
    
    async def initialize(self):
        """Initialize the vision engine"""
        try:
            cfg = _get_vision_config()
            logger.info(f"Initializing Vision Engine (device: {self.device})")
            
            # Detection: HOG is faster on CPU; CNN is more accurate but much slower
            self.detection_model = getattr(cfg, "DETECTION_MODEL", "hog")
            # Encoding: num_jitters = re-samples for robust encoding; encoding_model = small vs large
            self.num_jitters = getattr(cfg, "NUM_JITTERS", 5)
            self.encoding_model = getattr(cfg, "ENCODING_MODEL", "large")
            
            if self.device == "cuda":
                logger.warning("CUDA requested but face_recognition uses CPU-only dlib. Using CPU.")
                self.device = "cpu"
            
            logger.info(
                f"Face config: detection={self.detection_model}, encoding_model={self.encoding_model}, num_jitters={self.num_jitters}"
            )
            self._initialized = True
            logger.info("Vision Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vision Engine: {e}")
            raise
    
    async def detect_faces(self, image_path: str) -> Dict[str, Any]:
        """
        Detect faces in an image and return encodings
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with:
                - faces: List of detected faces with bounding boxes and encodings
                - image_width: Image width in pixels
                - image_height: Image height in pixels
                - processing_time_seconds: Time taken to process
        """
        if not self._initialized:
            raise RuntimeError("Vision Engine not initialized")
        
        start_time = time.time()
        
        try:
            # Load image
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load image with face_recognition (uses PIL internally)
            image = face_recognition.load_image_file(str(image_path))
            
            # Get image dimensions
            pil_image = Image.open(image_path)
            image_width, image_height = pil_image.size
            
            # Detect face locations (hog = faster on CPU, cnn = more accurate)
            face_locations = face_recognition.face_locations(image, model=self.detection_model)
            
            logger.info(f"Found {len(face_locations)} face(s) in image")
            
            # Get face encodings: num_jitters averages over perturbed crops for robustness;
            # encoding_model "large" uses more landmarks for better accuracy
            face_encodings = face_recognition.face_encodings(
                image,
                face_locations,
                num_jitters=self.num_jitters,
                model=self.encoding_model,
            )
            
            # Build response
            detected_faces = []
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                # face_recognition returns (top, right, bottom, left)
                # Convert to (x, y, width, height)
                bbox_x = left
                bbox_y = top
                bbox_width = right - left
                bbox_height = bottom - top
                
                detected_faces.append({
                    "bbox_x": bbox_x,
                    "bbox_y": bbox_y,
                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height,
                    "face_encoding": encoding.tolist(),  # Convert numpy array to list
                    "confidence": 1.0  # face_recognition doesn't provide confidence scores
                })
            
            processing_time = time.time() - start_time
            
            return {
                "faces": detected_faces,
                "image_width": image_width,
                "image_height": image_height,
                "processing_time_seconds": processing_time
            }
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            raise
    
    async def match_faces(
        self,
        unknown_faces: list,  # List of face encodings
        known_identities: dict,  # {identity_name: face_encoding}
        confidence_threshold: float = 0.82
    ) -> Dict[str, Any]:
        """
        Match unknown faces against known identities
        
        Args:
            unknown_faces: List of 128-dimensional face encodings
            known_identities: Dict mapping identity names to their face encodings
            confidence_threshold: Minimum confidence (0.0-1.0) for a match
            
        Returns:
            Dict with matches list and processing time
        """
        import time
        import numpy as np
        
        start_time = time.time()
        
        logger.info(f"🔍 Matching {len(unknown_faces)} unknown faces against {len(known_identities)} known identities")
        logger.info(f"   Confidence threshold: {confidence_threshold * 100}%")
        
        try:
            matches = []
            
            for idx, unknown_encoding in enumerate(unknown_faces):
                unknown_np = np.array(unknown_encoding)
                
                best_match = None
                best_confidence = 0
                
                # Compare against all known identities
                for identity_name, known_encoding in known_identities.items():
                    known_np = np.array(known_encoding)
                    
                    # Calculate face distance (lower = more similar)
                    distance = face_recognition.face_distance([known_np], unknown_np)[0]
                    
                    # Convert distance to confidence percentage
                    # face_recognition uses 0.6 as typical threshold for "same person"
                    confidence = max(0, (1 - (distance / 0.6)) * 100)
                    
                    logger.info(f"   📊 Face #{idx} vs '{identity_name}': distance={distance:.4f}, confidence={confidence:.1f}%")
                    
                    # Track best confidence regardless of threshold
                    if confidence > best_confidence:
                        best_confidence = confidence
                        # Only mark as match if meets threshold
                        if confidence >= (confidence_threshold * 100):
                            best_match = identity_name
                
                # Add match if found
                if best_match:
                    matches.append({
                        "face_index": idx,
                        "matched_identity": best_match,
                        "confidence": round(best_confidence, 1)
                    })
                    logger.info(f"✨ Matched face #{idx} to {best_match} ({best_confidence:.1f}% confidence)")
                elif best_confidence > 0:
                    logger.info(f"❌ No match found for face #{idx} (best: {best_confidence:.1f}% < {confidence_threshold * 100:.1f}% threshold)")
                else:
                    logger.info(f"❌ No match found for face #{idx} (no known identities to compare)")
            
            processing_time = time.time() - start_time
            
            return {
                "matches": matches,
                "processing_time_seconds": processing_time
            }
            
        except Exception as e:
            logger.error(f"Face matching failed: {e}")
            raise

    async def initialize_object_detection(self) -> None:
        """Lazy-load YOLOv8 and CLIP models for object detection."""
        if self._object_detection_initialized:
            return
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics (YOLO) is not installed")
        if not CLIP_AVAILABLE:
            raise RuntimeError("transformers (CLIP) is not installed")
        try:
            cfg = _get_vision_config()
            model_name = getattr(cfg, "OBJECT_DETECTION_MODEL", "yolov8n.pt")
            clip_model_id = getattr(cfg, "CLIP_MODEL", "openai/clip-vit-base-patch32")
            logger.info(f"Loading object detection: YOLO={model_name}, CLIP={clip_model_id}")
            self.yolo_model = YOLO(model_name)
            self.clip_model = CLIPModel.from_pretrained(clip_model_id)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
            if self.device == "cuda" and torch is not None and torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
            self._object_detection_initialized = True
            logger.info("Object detection initialized (YOLO + CLIP)")
        except Exception as e:
            logger.error(f"Failed to initialize object detection: {e}")
            raise

    async def detect_objects(
        self,
        image_path: str,
        class_filter: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Detect objects using YOLO (region proposals).

        Args:
            image_path: Path to image file.
            class_filter: Optional list of COCO class names to include.
            confidence_threshold: Minimum confidence (0.0-1.0).

        Returns:
            Dict with objects (bbox, class_name, confidence), image dimensions, processing_time_seconds.
        """
        await self.initialize_object_detection()
        start_time = time.time()
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        pil_image = Image.open(image_path)
        image_width, image_height = pil_image.size
        results = self.yolo_model.predict(
            str(image_path),
            conf=confidence_threshold,
            verbose=False,
        )
        detected = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = self.yolo_model.names[class_id]
                if class_filter and class_name not in class_filter:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                detected.append({
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": conf,
                    "bbox_x": int(x1),
                    "bbox_y": int(y1),
                    "bbox_width": int(x2 - x1),
                    "bbox_height": int(y2 - y1),
                    "detection_method": "yolo",
                    "matched_description": "",
                })
        elapsed = time.time() - start_time
        return {
            "objects": detected,
            "image_width": image_width,
            "image_height": image_height,
            "processing_time_seconds": elapsed,
        }

    async def match_objects_semantically(
        self,
        image_path: str,
        regions: List[Dict[str, Any]],
        object_descriptions: List[str],
        similarity_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        Match image regions to text descriptions using CLIP.

        Args:
            image_path: Path to image file.
            regions: List of dicts with bbox_x, bbox_y, bbox_width, bbox_height.
            object_descriptions: Text descriptions to match (e.g. "Nike logo", "tribal tattoo").
            similarity_threshold: Minimum cosine similarity (CLIP logits are normalized).

        Returns:
            List of matches: region index, matched_description, confidence (0-1).
        """
        if not object_descriptions or not regions:
            return []
        await self.initialize_object_detection()
        pil_image = Image.open(image_path)
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        text_inputs = self.clip_processor(
            text=object_descriptions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        matches = []
        for idx, region in enumerate(regions):
            x = region["bbox_x"]
            y = region["bbox_y"]
            w = region["bbox_width"]
            h = region["bbox_height"]
            crop = pil_image.crop((x, y, x + w, y + h))
            if crop.size[0] < 1 or crop.size[1] < 1:
                continue
            image_inputs = self.clip_processor(images=crop, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = (image_features @ text_features.T).squeeze(0).cpu().numpy()
            probs = self._softmax(logits)
            best_idx = int(np.argmax(probs))
            score = float(probs[best_idx])
            if score >= similarity_threshold:
                matches.append({
                    "region_index": idx,
                    "matched_description": object_descriptions[best_idx],
                    "confidence": score,
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_width": w,
                    "bbox_height": h,
                })
        return matches

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    @staticmethod
    def _box_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """Compute IoU of two boxes with bbox_x, bbox_y, bbox_width, bbox_height."""
        ax1 = a["bbox_x"]
        ay1 = a["bbox_y"]
        aw = a["bbox_width"]
        ah = a["bbox_height"]
        bx1 = b["bbox_x"]
        by1 = b["bbox_y"]
        bw = b["bbox_width"]
        bh = b["bbox_height"]
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _nms_boxes(boxes: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Non-maximum suppression: keep highest-confidence boxes, suppress overlapping."""
        if not boxes:
            return []
        sorted_boxes = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
        kept = []
        for b in sorted_boxes:
            if any(VisionEngine._box_iou(b, k) >= iou_threshold for k in kept):
                continue
            kept.append(b)
        return kept

    async def find_semantic_regions(
        self,
        image_path: str,
        object_descriptions: List[str],
        chunk_size: int = 224,
        stride: Optional[int] = None,
        similarity_threshold: float = 0.25,
        max_chunks: int = 200,
        nms_iou_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Run CLIP on a grid of image chunks to find regions matching text descriptions.
        Used when semantic terms are set to find items YOLO may have missed.

        Args:
            image_path: Path to image file.
            object_descriptions: Text descriptions to match (e.g. "logo", "tattoo").
            chunk_size: Side length of square window (default 224 for CLIP).
            stride: Step between windows; default chunk_size (no overlap).
            similarity_threshold: Minimum score to keep a chunk.
            max_chunks: Cap on number of chunks to limit cost on large images.
            nms_iou_threshold: IoU threshold for NMS to merge overlapping detections.

        Returns:
            List of {bbox_x, bbox_y, bbox_width, bbox_height, matched_description, confidence}.
        """
        if not object_descriptions:
            return []
        await self.initialize_object_detection()
        if not CLIP_AVAILABLE:
            return []
        pil_image = Image.open(image_path).convert("RGB")
        w, h = pil_image.size
        stride = stride if stride is not None else chunk_size
        # Build grid of windows
        xs = list(range(0, max(1, w - chunk_size + 1), stride))
        ys = list(range(0, max(1, h - chunk_size + 1), stride))
        if not xs:
            xs = [0]
        if not ys:
            ys = [0]
        total = len(xs) * len(ys)
        if total > max_chunks:
            # Increase stride to cap chunks
            ratio = (total / max_chunks) ** 0.5
            step_x = max(1, int(len(xs) / ratio))
            step_y = max(1, int(len(ys) / ratio))
            xs = xs[::step_x] if step_x > 1 else xs
            ys = ys[::step_y] if step_y > 1 else ys
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        text_inputs = self.clip_processor(
            text=object_descriptions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        raw_matches = []
        for y in ys:
            for x in xs:
                # Clip to image bounds
                x2 = min(x + chunk_size, w)
                y2 = min(y + chunk_size, h)
                cw = x2 - x
                ch = y2 - y
                if cw < 8 or ch < 8:
                    continue
                crop = pil_image.crop((x, y, x2, y2))
                image_inputs = self.clip_processor(images=crop, return_tensors="pt").to(device)
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**image_inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    logits = (image_features @ text_features.T).squeeze(0).cpu().numpy()
                probs = self._softmax(logits)
                best_idx = int(np.argmax(probs))
                score = float(probs[best_idx])
                if score >= similarity_threshold:
                    raw_matches.append({
                        "bbox_x": x,
                        "bbox_y": y,
                        "bbox_width": cw,
                        "bbox_height": ch,
                        "matched_description": object_descriptions[best_idx],
                        "confidence": score,
                    })
        if not raw_matches:
            return []
        kept = self._nms_boxes(raw_matches, iou_threshold=nms_iou_threshold)
        logger.debug("Semantic sweep: %d raw hits, %d after NMS", len(raw_matches), len(kept))
        return kept

    async def extract_object_features(
        self,
        image_path: str,
        bbox: Dict[str, int],
        description: str,
    ) -> Dict[str, Any]:
        """
        Extract CLIP embeddings for a user-annotated region (visual + semantic + combined).

        Args:
            image_path: Path to image file.
            bbox: Dict with x, y, width, height (or bbox_x, bbox_y, bbox_width, bbox_height).
            description: User text description.

        Returns:
            Dict with visual_embedding, semantic_embedding, combined_embedding (lists), embedding_dim.
        """
        await self.initialize_object_detection()
        x = bbox.get("bbox_x", bbox.get("x", 0))
        y = bbox.get("bbox_y", bbox.get("y", 0))
        w = bbox.get("bbox_width", bbox.get("width", 0))
        h = bbox.get("bbox_height", bbox.get("height", 0))
        pil_image = Image.open(image_path)
        crop = pil_image.crop((x, y, x + w, y + h))
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        image_inputs = self.clip_processor(images=crop, return_tensors="pt").to(device)
        text_inputs = self.clip_processor(
            text=[description] if description else ["object"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            combined = (image_features + text_features) / 2
            combined = combined / combined.norm(dim=-1, keepdim=True)
        return {
            "visual_embedding": image_features[0].cpu().numpy().tolist(),
            "semantic_embedding": text_features[0].cpu().numpy().tolist(),
            "combined_embedding": combined[0].cpu().numpy().tolist(),
            "embedding_dim": self.clip_embedding_dim,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check if vision engine is healthy."""
        out = {
            "status": "healthy" if self._initialized else "unhealthy",
            "device": self.device,
            "model": getattr(self, "detection_model", "hog"),
            "detection_model": getattr(self, "detection_model", "hog"),
            "encoding_model": getattr(self, "encoding_model", "large"),
            "num_jitters": getattr(self, "num_jitters", 5),
            "face_recognition_available": FACE_RECOGNITION_AVAILABLE,
            "object_detection_initialized": getattr(self, "_object_detection_initialized", False),
            "clip_only_initialized": getattr(self, "_clip_only_initialized", False),
        }
        return out