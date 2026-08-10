"""Pydantic request/response models for FastAPI"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class BoundingBox(BaseModel):
    """Bounding box coordinates (XYXY format)"""
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """Single object detection"""
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: List[float]
    frame_idx: Optional[int] = None


class OCRResult(BaseModel):
    """OCR text extraction result"""
    text: str
    confidence: float
    bbox: List[List[float]]


class VerdictReasoning(BaseModel):
    """Reasoning verdict for a detection"""
    violation_type: str
    confidence: float
    reasoning: str
    evidence: List[str]
    recommended_action: str  # flag, review, allow


class FrameAnalysis(BaseModel):
    """Analysis result for a single frame"""
    frame_idx: int
    detections: List[Detection]
    ocr: List[OCRResult]
    reasoning: List[VerdictReasoning]
    timestamp: Optional[float] = None


class AnalysisRequest(BaseModel):
    """Request to analyze a video/image"""
    file_name: str
    detector_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    ocr_enabled: bool = True
    reasoning_enabled: bool = True


class AnalysisResponse(BaseModel):
    """Response with analysis results"""
    job_id: str
    status: str  # pending, processing, completed, failed
    file_name: str
    frames_analyzed: int = 0
    violations_detected: int = 0
    frames: List[FrameAnalysis] = []
    summary: Dict = {}
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    models_loaded: Dict[str, str]
    gpu_available: bool
    memory_usage_mb: float
