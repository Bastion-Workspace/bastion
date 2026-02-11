"""
Image Vision Service Configuration
"""

import os
from typing import Optional

class Settings:
    """Image vision service settings from environment variables"""
    
    # Service Configuration
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "image-vision-service")
    GRPC_PORT: int = int(os.getenv("GRPC_PORT", "50056"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    SERVICE_VERSION: str = os.getenv("SERVICE_VERSION", "1.0.0")
    
    # Device Configuration
    DEVICE: str = os.getenv("DEVICE", "auto")  # "cpu", "cuda", or "auto"
    
    # Performance Tuning
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))

    # Face encoding accuracy (CPU-bound tradeoffs)
    # num_jitters: re-sample face N times (zoom/rotate/translate), average encoding. Higher = more accurate, slower.
    NUM_JITTERS: int = int(os.getenv("NUM_JITTERS", "5"))
    # encoding_model: "small" (faster) or "large" (more accurate landmarks/encoding)
    ENCODING_MODEL: str = os.getenv("ENCODING_MODEL", "large")
    # detection_model: "hog" (faster on CPU) or "cnn" (more accurate, much slower on CPU)
    DETECTION_MODEL: str = os.getenv("DETECTION_MODEL", "hog")

    # Object detection (YOLO + CLIP)
    OBJECT_DETECTION_MODEL: str = os.getenv("OBJECT_DETECTION_MODEL", "yolov8n.pt")
    CLIP_MODEL: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")

    @classmethod
    def validate(cls) -> None:
        """Validate required settings"""
        if cls.DEVICE not in ["cpu", "cuda", "auto"]:
            raise ValueError(f"DEVICE must be 'cpu', 'cuda', or 'auto', got: {cls.DEVICE}")
        if cls.ENCODING_MODEL not in ["small", "large"]:
            raise ValueError(f"ENCODING_MODEL must be 'small' or 'large', got: {cls.ENCODING_MODEL}")
        if cls.DETECTION_MODEL not in ["hog", "cnn"]:
            raise ValueError(f"DETECTION_MODEL must be 'hog' or 'cnn', got: {cls.DETECTION_MODEL}")
    
    @classmethod
    def get_device(cls) -> str:
        """Detect device (CPU or CUDA)"""
        if cls.DEVICE == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return cls.DEVICE

# Global settings instance
settings = Settings()
