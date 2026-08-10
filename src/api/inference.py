"""Async inference orchestration for FastAPI backend"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import json

import torch
import cv2

from src.api.models import Detection, OCRResult, VerdictReasoning, FrameAnalysis

logger = logging.getLogger(__name__)


class InferenceOrchestrator:
    """Coordinate detection, OCR, ASR, and VLM reasoning"""

    def __init__(self):
        """Initialize all models"""
        logger.info("Initializing inference orchestrator...")

        self.gpu_available = torch.cuda.is_available()
        self.device = "cuda" if self.gpu_available else "cpu"
        self.models_status = {}
        self.latency_history = []

        # Load detector (YOLOv8n fine-tuned)
        self._load_detector()

        # Load OCR (EasyOCR)
        self._load_ocr()

        # Load ASR (Whisper) - optional for images
        self._load_asr()

        logger.info(f"✅ Orchestrator ready on {self.device}")

    def _load_detector(self):
        """Load YOLOv8n detector"""
        try:
            from ultralytics import YOLO
            from torch.serialization import add_safe_globals

            # Ultralytics uses pickle, allow it
            try:
                add_safe_globals([lambda: None])
            except:
                pass

            model_path = Path("models/best.pt")
            if not model_path.exists():
                logger.warning(f"⚠️ Fine-tuned model not found at {model_path}")
                logger.info("   Loading pretrained YOLOv8n instead...")
                self.detector = YOLO("yolov8n.pt")
                self.models_status["detector"] = "pretrained"
            else:
                torch.serialization.add_safe_globals([lambda: None])
                self.detector = YOLO(str(model_path))
                self.models_status["detector"] = "fine-tuned"

            logger.info("✓ YOLOv8n detector loaded")

        except Exception as e:
            logger.error(f"❌ Failed to load detector: {e}")
            self.detector = None
            self.models_status["detector"] = "failed"

    def _load_ocr(self):
        """Load EasyOCR"""
        try:
            import easyocr

            self.ocr_reader = easyocr.Reader(
                ["en"],
                gpu=self.gpu_available,
                model_storage_directory="models/easyocr"
            )
            logger.info("✓ EasyOCR loaded")
            self.models_status["ocr"] = "ready"

        except Exception as e:
            logger.error(f"❌ Failed to load OCR: {e}")
            self.ocr_reader = None
            self.models_status["ocr"] = "failed"

    def _load_asr(self):
        """Load Whisper ASR (optional for images)"""
        try:
            import whisper

            self.asr_model = whisper.load_model(
                "base",
                device=self.device
            )
            logger.info("✓ Whisper ASR loaded")
            self.models_status["asr"] = "ready"

        except Exception as e:
            logger.warning(f"⚠️ Whisper ASR not available: {e}")
            self.asr_model = None
            self.models_status["asr"] = "unavailable"

    def get_memory_usage(self) -> float:
        """Get GPU/CPU memory usage in MB"""
        if self.gpu_available:
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024

    def get_average_latency(self) -> float:
        """Get average inference latency"""
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history[-100:]) / len(self.latency_history[-100:])

    async def analyze_image(
        self,
        image_path: Path,
        detector_threshold: float = 0.45,
        ocr_enabled: bool = True,
        reasoning_enabled: bool = True
    ) -> Dict:
        """Analyze a single image"""

        start_time = time.time()

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        result = {
            "file": str(image_path),
            "frames": [],
            "violations_detected": 0,
            "summary": {
                "detections": 0,
                "ocr_regions": 0,
                "verdicts": 0,
                "flagged": 0,
                "flagged_percent": 0.0
            }
        }

        # Run detection
        detections = self._run_detection(image, detector_threshold)
        logger.info(f"   Detections: {len(detections)}")

        if len(detections) == 0:
            # No violations
            frame_analysis = FrameAnalysis(
                frame_idx=0,
                detections=[],
                ocr=[],
                reasoning=[],
                timestamp=0.0
            )
            result["frames"].append(frame_analysis)
            result["summary"]["detections"] = 0
            return result

        # Run OCR if enabled
        ocr_results = []
        if ocr_enabled:
            ocr_results = self._run_ocr(image)
            logger.info(f"   OCR regions: {len(ocr_results)}")

        # Generate reasoning
        verdicts = []
        if reasoning_enabled:
            for det in detections:
                ocr_text = None
                if ocr_results:
                    ocr_text = " ".join([o.text for o in ocr_results[:3]])

                verdict = self._generate_reasoning(det, ocr_text)
                verdicts.append(verdict)

        # Create frame analysis
        frame_analysis = FrameAnalysis(
            frame_idx=0,
            detections=detections,
            ocr=ocr_results,
            reasoning=verdicts,
            timestamp=0.0
        )
        result["frames"].append(frame_analysis)

        # Update summary
        result["violations_detected"] = len(verdicts) if verdicts else len(detections)
        result["summary"]["detections"] = len(detections)
        result["summary"]["ocr_regions"] = len(ocr_results)
        result["summary"]["verdicts"] = len(verdicts)
        result["summary"]["flagged"] = sum(1 for v in verdicts if v.recommended_action == "flag")
        result["summary"]["flagged_percent"] = (
            result["summary"]["flagged"] / len(verdicts) * 100
            if verdicts else 0
        )

        # Track latency
        latency = (time.time() - start_time) * 1000  # Convert to ms
        self.latency_history.append(latency)
        logger.info(f"   Latency: {latency:.1f}ms")

        return result

    def _run_detection(self, image, threshold: float = 0.45) -> List[Detection]:
        """Run YOLOv8n detection"""
        if self.detector is None:
            return []

        try:
            results = self.detector(image, conf=threshold)
            detections = []

            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()

                    # Map class ID to name
                    class_name = r.names.get(class_id, f"class_{class_id}")

                    detection = Detection(
                        class_name=class_name,
                        confidence=conf,
                        bbox_xyxy=xyxy
                    )
                    detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def _run_ocr(self, image) -> List[OCRResult]:
        """Run EasyOCR text extraction"""
        if self.ocr_reader is None:
            return []

        try:
            results = self.ocr_reader.readtext(image, detail=1)
            ocr_results = []

            for (bbox, text, conf) in results:
                ocr_result = OCRResult(
                    text=text,
                    confidence=conf,
                    bbox=bbox
                )
                ocr_results.append(ocr_result)

            return ocr_results

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return []

    def _generate_reasoning(
        self,
        detection: Detection,
        ocr_text: Optional[str] = None
    ) -> VerdictReasoning:
        """Generate reasoning verdict (rule-based)"""

        class_name = detection.class_name.lower()
        confidence = detection.confidence

        # Rule-based verdict logic
        if "weapon" in class_name:
            violation_type = "weapon"
            reasoning = f"Weapon detected with {confidence:.1%} confidence."
            if ocr_text:
                reasoning += f" Contains text: '{ocr_text[:50]}'."
            action = "flag" if confidence > 0.6 else "review"

        elif "nsfw" in class_name or "person" in class_name:
            violation_type = "nsfw"
            reasoning = f"Potentially inappropriate content ({confidence:.1%} confidence)."
            action = "review" if confidence > 0.5 else "allow"

        else:
            violation_type = "counterfeit" if confidence > 0.7 else "product"
            reasoning = f"Product detected ({confidence:.1%} confidence). Requires manual review."
            action = "review"

        return VerdictReasoning(
            violation_type=violation_type,
            confidence=min(confidence, 0.99),
            reasoning=reasoning,
            evidence=[class_name, f"confidence:{confidence:.2f}"],
            recommended_action=action
        )
