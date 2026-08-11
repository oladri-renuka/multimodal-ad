#!/usr/bin/env python3
"""
Phase 3: OCR + ASR Extraction Pipeline
Extract text from images and speech from audio using real violation detection
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ViolationAnalyzer:
    """Detect violations + extract OCR + ASR"""

    def __init__(self, checkpoint_path: Path):
        """Initialize all models"""
        logger.info("=" * 70)
        logger.info("PHASE 3: OCR + ASR EXTRACTION PIPELINE")
        logger.info("=" * 70)

        checkpoint_path = Path(checkpoint_path)

        # Load detector
        logger.info("\n1⃣ Loading YOLOv8n checkpoint...")
        try:
            from ultralytics import YOLO
            self.detector = YOLO(str(checkpoint_path))
            logger.info(f"    Loaded: {checkpoint_path.name}")
        except Exception as e:
            logger.error(f"    Failed to load detector: {e}")
            raise

        # Load OCR
        logger.info("\n2⃣ Loading EasyOCR...")
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'])
            logger.info("    EasyOCR ready (English)")
        except Exception as e:
            logger.error(f"    Failed to load OCR: {e}")
            self.ocr_reader = None

        # Load ASR
        logger.info("\n3⃣ Loading Whisper ASR...")
        try:
            import whisper
            self.asr_model = whisper.load_model("base")
            logger.info("    Whisper 'base' loaded")
        except Exception as e:
            logger.error(f"    Failed to load ASR: {e}")
            self.asr_model = None

        logger.info("\n All models loaded successfully!\n")

    def detect_violations(self, image_path: Path) -> List[Dict]:
        """Detect weapons/products in image"""
        logger.info(f" Running YOLOv8n detection on {image_path.name}...")

        try:
            results = self.detector.predict(str(image_path), conf=0.45, verbose=False)
            detections = []

            for r in results:
                for box in r.boxes:
                    detection = {
                        "class_id": int(box.cls),
                        "class_name": r.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox_xyxy": box.xyxy[0].tolist()
                    }
                    detections.append(detection)

            logger.info(f"    Found {len(detections)} violations")
            return detections

        except Exception as e:
            logger.error(f"    Detection failed: {e}")
            return []

    def extract_text(self, image_path: Path) -> List[Dict]:
        """Extract text from image using OCR"""
        if not self.ocr_reader:
            logger.warning("    OCR not available")
            return []

        logger.info(f" Extracting text from {image_path.name}...")

        try:
            results = self.ocr_reader.readtext(str(image_path))
            text_data = []

            for detection in results:
                bbox, text, confidence = detection
                text_data.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": [[float(x), float(y)] for x, y in bbox]
                })

            logger.info(f"    Extracted {len(text_data)} text regions")
            return text_data

        except Exception as e:
            logger.error(f"    OCR failed: {e}")
            return []

    def transcribe_audio(self, audio_path: Path) -> Optional[Dict]:
        """Extract speech from audio using Whisper"""
        if not self.asr_model:
            logger.warning("    ASR not available")
            return None

        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.warning(f"    Audio file not found: {audio_path}")
            return None

        logger.info(f" Transcribing {audio_path.name}...")

        try:
            result = self.asr_model.transcribe(str(audio_path), verbose=False)

            transcript = {
                "text": result['text'],
                "language": result.get('language', 'en'),
                "segments": []
            }

            if 'segments' in result:
                for seg in result['segments']:
                    transcript["segments"].append({
                        "start": seg['start'],
                        "end": seg['end'],
                        "text": seg['text']
                    })

            logger.info(f"    Transcribed: {result['text'][:80]}...")
            return transcript

        except Exception as e:
            logger.error(f"    ASR failed: {e}")
            return None

    def analyze(self, image_path: Path, audio_path: Optional[Path] = None) -> Dict:
        """Run complete analysis on image + audio"""
        logger.info(f"\n{'=' * 70}")
        logger.info(f"ANALYZING: {image_path.name}")
        logger.info(f"{'=' * 70}")

        analysis = {
            "file": str(image_path),
            "detections": self.detect_violations(image_path),
            "ocr": self.extract_text(image_path),
        }

        if audio_path:
            audio_path = Path(audio_path)
            if audio_path.exists():
                analysis["asr"] = self.transcribe_audio(audio_path)

        logger.info(f"\n{'=' * 70}")
        logger.info("SUMMARY")
        logger.info(f"{'=' * 70}")
        logger.info(f"  Detections: {len(analysis['detections'])}")
        logger.info(f"  Text regions: {len(analysis['ocr'])}")
        if analysis.get("asr"):
            logger.info(f"  Speech: '{analysis['asr']['text'][:60]}...'")

        return analysis


def main():
    """Run Phase 3 pipeline"""

    # Initialize analyzer
    checkpoint = Path("models/best.pt")
    if not checkpoint.exists():
        logger.error(f" Checkpoint not found: {checkpoint}")
        return

    analyzer = ViolationAnalyzer(checkpoint)

    # Find test images
    test_dirs = [
        Path("data/test_clips"),
        Path("data/raw/weapons/images"),
        Path("data/raw/products/images"),
    ]

    test_images = []
    for test_dir in test_dirs:
        if test_dir.exists():
            test_images.extend(list(test_dir.glob("*.jpg")))
            test_images.extend(list(test_dir.glob("*.png")))

    if not test_images:
        logger.warning(" No test images found")
        logger.info("Create test images in data/test_clips/ or use data/raw/")
        return

    logger.info(f"\n Found {len(test_images)} test images\n")

    # Create output directory
    output_dir = Path("results/phase3_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze first 5 images
    results = []
    for img_path in test_images[:5]:
        try:
            analysis = analyzer.analyze(img_path)
            results.append(analysis)
        except Exception as e:
            logger.error(f"Failed to analyze {img_path}: {e}")
            continue

    # Save results
    output_file = output_dir / "phase3_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'=' * 70}")
    logger.info("PHASE 3 COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f" Analyzed {len(results)} images")
    logger.info(f" Results saved: {output_file}\n")

    # Summary statistics
    total_detections = sum(len(r['detections']) for r in results)
    total_text = sum(len(r['ocr']) for r in results)

    logger.info(" STATISTICS:")
    logger.info(f"   Total detections: {total_detections}")
    logger.info(f"   Total text regions: {total_text}")
    logger.info(f"   Average detections/image: {total_detections/len(results):.1f}")
    logger.info(f"   Average text regions/image: {total_text/len(results):.1f}")

    logger.info(f"\n Next: Phase 4 - VLM Reasoning Layer (LLaVA)")


if __name__ == "__main__":
    main()
