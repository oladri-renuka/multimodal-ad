"""YOLOv8n inference wrapper for batch predictions on violation detection."""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from ultralytics import YOLO
import torch

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single object detection."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels

    def to_dict(self) -> Dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox": list(self.bbox),
        }


@dataclass
class ImagePrediction:
    """Predictions for a single image."""
    image_path: Path
    image_size: Tuple[int, int]
    detections: List[Detection]

    def to_dict(self) -> Dict:
        return {
            "image": str(self.image_path),
            "size": self.image_size,
            "detections": [d.to_dict() for d in self.detections],
            "num_detections": len(self.detections),
        }


class YOLOv8nInferencer:
    """Run inference with YOLOv8n on violation detection task."""

    CLASSES = {0: "weapon", 1: "nsfw", 2: "counterfeit"}

    def __init__(
        self,
        checkpoint_path: Path,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Initialize inferencer.

        Args:
            checkpoint_path: Path to .pt checkpoint (pretrained or fine-tuned)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading YOLOv8n from {checkpoint_path}")
        logger.info(f"Device: {self.device}, Conf: {conf_threshold}, IOU: {iou_threshold}")

        self.model = YOLO(str(checkpoint_path))
        self.model.to(self.device)

    def predict_image(self, image_path: Path) -> ImagePrediction:
        """
        Run inference on single image.

        Args:
            image_path: Path to input image

        Returns:
            ImagePrediction with detections
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Run inference
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        # Parse results
        detections = []
        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item()) if box.cls is not None else 0
                    confidence = float(box.conf[0].item()) if box.conf is not None else 0.0

                    # Bounding box (x1, y1, x2, y2) in pixels
                    bbox = tuple(float(x) for x in box.xyxy[0].cpu().numpy())

                    detection = Detection(
                        class_id=class_id,
                        class_name=self.CLASSES.get(class_id, "unknown"),
                        confidence=confidence,
                        bbox=bbox,
                    )
                    detections.append(detection)

            # Image size
            if hasattr(result, 'orig_shape'):
                image_size = tuple(result.orig_shape)
            else:
                image_size = (result.shape[0], result.shape[1])

            return ImagePrediction(
                image_path=image_path,
                image_size=image_size,
                detections=detections,
            )

        return ImagePrediction(
            image_path=image_path,
            image_size=(0, 0),
            detections=[],
        )

    def predict_batch(
        self,
        image_paths: List[Path],
        verbose: bool = True,
    ) -> List[ImagePrediction]:
        """
        Run inference on batch of images.

        Args:
            image_paths: List of image paths
            verbose: Print progress

        Returns:
            List of ImagePredictions
        """
        image_paths = [Path(p) for p in image_paths]

        if not image_paths:
            return []

        predictions = []
        for i, image_path in enumerate(image_paths):
            if verbose and i % 10 == 0:
                logger.info(f"Processed {i}/{len(image_paths)}")

            try:
                pred = self.predict_image(image_path)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                predictions.append(
                    ImagePrediction(image_path=image_path, image_size=(0, 0), detections=[])
                )

        logger.info(f"Completed batch inference: {len(predictions)} images")
        return predictions

    def filter_detections(
        self,
        predictions: List[ImagePrediction],
        class_id: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> List[ImagePrediction]:
        """
        Filter predictions by class or confidence.

        Args:
            predictions: List of ImagePredictions
            class_id: Filter to specific class (None = all)
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered predictions
        """
        filtered = []

        for pred in predictions:
            filtered_detections = pred.detections

            if class_id is not None:
                filtered_detections = [d for d in filtered_detections if d.class_id == class_id]

            if min_confidence is not None:
                filtered_detections = [d for d in filtered_detections if d.confidence >= min_confidence]

            filtered.append(
                ImagePrediction(
                    image_path=pred.image_path,
                    image_size=pred.image_size,
                    detections=filtered_detections,
                )
            )

        return filtered

    def nms(
        self,
        predictions: List[ImagePrediction],
        iou_threshold: Optional[float] = None,
    ) -> List[ImagePrediction]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.

        Args:
            predictions: List of ImagePredictions
            iou_threshold: IOU threshold for NMS

        Returns:
            Predictions with NMS applied
        """
        if iou_threshold is None:
            iou_threshold = self.iou_threshold

        from torchvision.ops import nms

        nmsed = []
        for pred in predictions:
            if not pred.detections:
                nmsed.append(pred)
                continue

            # Prepare tensors
            boxes = []
            scores = []
            for det in pred.detections:
                x1, y1, x2, y2 = det.bbox
                boxes.append([x1, y1, x2, y2])
                scores.append(det.confidence)

            boxes = torch.tensor(boxes, dtype=torch.float32)
            scores = torch.tensor(scores, dtype=torch.float32)

            # Apply NMS
            keep_idx = nms(boxes, scores, iou_threshold)

            # Filter detections
            nmsed_detections = [pred.detections[i] for i in keep_idx.tolist()]

            nmsed.append(
                ImagePrediction(
                    image_path=pred.image_path,
                    image_size=pred.image_size,
                    detections=nmsed_detections,
                )
            )

        return nmsed

    def statistics(self, predictions: List[ImagePrediction]) -> Dict:
        """
        Compute statistics on predictions.

        Args:
            predictions: List of ImagePredictions

        Returns:
            Statistics dictionary
        """
        total_images = len(predictions)
        total_detections = sum(len(p.detections) for p in predictions)

        class_counts = {self.CLASSES[i]: 0 for i in range(len(self.CLASSES))}
        confidence_scores = []

        for pred in predictions:
            for det in pred.detections:
                class_counts[det.class_name] += 1
                confidence_scores.append(det.confidence)

        stats = {
            "total_images": total_images,
            "total_detections": total_detections,
            "average_detections_per_image": total_detections / total_images if total_images > 0 else 0,
            "class_distribution": class_counts,
            "confidence_stats": {
                "mean": float(np.mean(confidence_scores)) if confidence_scores else 0.0,
                "min": float(np.min(confidence_scores)) if confidence_scores else 0.0,
                "max": float(np.max(confidence_scores)) if confidence_scores else 0.0,
                "std": float(np.std(confidence_scores)) if confidence_scores else 0.0,
            },
        }

        return stats


def main():
    """Example usage."""
    import sys

    checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/yolov8n_finetuned.pt")
    image_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/yolo_dataset/images/test")

    if not checkpoint.exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        sys.exit(1)

    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        sys.exit(1)

    # Initialize inferencer
    inferencer = YOLOv8nInferencer(checkpoint)

    # Run inference
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    logger.info(f"Found {len(image_paths)} images")

    predictions = inferencer.predict_batch(image_paths)

    # Statistics
    stats = inferencer.statistics(predictions)
    logger.info(f"Statistics: {stats}")

    # Show some detections
    for pred in predictions[:5]:
        logger.info(f"{pred.image_path.name}: {len(pred.detections)} detections")
        for det in pred.detections:
            logger.info(f"  - {det.class_name} ({det.confidence:.2f})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
