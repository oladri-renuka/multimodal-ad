"""Evaluate YOLOv8n detections and compute metrics."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np
from dataclasses import dataclass, asdict

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.detectors.yolov8_inferencer import ImagePrediction

logger = logging.getLogger(__name__)


@dataclass
class ClassMetrics:
    """Per-class evaluation metrics."""
    class_id: int
    class_name: str
    precision: float
    recall: float
    f1: float
    ap: float  # Average Precision

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    model_name: str
    num_images: int
    num_detections: int
    precision: float  # Micro-averaged
    recall: float
    f1: float
    map50: float  # mAP @ IOU=0.5
    false_positive_rate: float
    per_class_metrics: Dict[str, ClassMetrics]

    def to_dict(self) -> Dict:
        return {
            "model": self.model_name,
            "num_images": self.num_images,
            "num_detections": self.num_detections,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "map50": self.map50,
            "false_positive_rate": self.false_positive_rate,
            "per_class_metrics": {
                name: m.to_dict() for name, m in self.per_class_metrics.items()
            },
        }


class DetectorEvaluator:
    """Evaluate YOLOv8n predictions against ground truth."""

    def __init__(self, class_names: Dict[int, str] = None):
        self.class_names = class_names or {0: "weapon", 1: "nsfw", 2: "counterfeit"}
        self.id_to_name = self.class_names
        self.name_to_id = {v: k for k, v in self.class_names.items()}

    def evaluate(
        self,
        predictions: List[ImagePrediction],
        ground_truth: Dict[str, List[Tuple]],  # {img_path: [(class_id, bbox), ...]}
        model_name: str = "detector",
        iou_threshold: float = 0.5,
    ) -> EvaluationReport:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions: List of ImagePrediction objects
            ground_truth: Dict mapping image paths to list of (class_id, bbox) tuples
            model_name: Name of model being evaluated
            iou_threshold: IOU threshold for considering a match

        Returns:
            EvaluationReport with all metrics
        """
        logger.info("=" * 70)
        logger.info(f"EVALUATING: {model_name}")
        logger.info("=" * 70)

        # Match predictions to ground truth
        matches, false_positives, false_negatives = self._match_predictions(
            predictions,
            ground_truth,
            iou_threshold,
        )

        # Compute metrics
        precision = self._compute_precision(matches, false_positives)
        recall = self._compute_recall(matches, false_negatives)
        f1 = self._compute_f1(precision, recall)
        map50 = self._compute_map50(matches, predictions, ground_truth)
        fpr = self._compute_fpr(false_positives, len(predictions))

        # Per-class metrics
        per_class_metrics = self._compute_per_class_metrics(
            matches,
            false_positives,
            false_negatives,
        )

        report = EvaluationReport(
            model_name=model_name,
            num_images=len(predictions),
            num_detections=sum(len(p.detections) for p in predictions),
            precision=precision,
            recall=recall,
            f1=f1,
            map50=map50,
            false_positive_rate=fpr,
            per_class_metrics=per_class_metrics,
        )

        self._log_report(report)
        return report

    def _match_predictions(
        self,
        predictions: List[ImagePrediction],
        ground_truth: Dict[str, List[Tuple]],
        iou_threshold: float,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Match predictions to ground truth using IOU."""
        matches = []
        false_positives = []
        false_negatives = []

        for pred in predictions:
            img_path = str(pred.image_path)
            gt_boxes = ground_truth.get(img_path, [])

            # Track which ground truth boxes were matched
            matched_gt = set()

            # Try to match each prediction
            for detection in pred.detections:
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, (gt_class_id, gt_bbox) in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue

                    iou = self._compute_iou(detection.bbox, gt_bbox)

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    # True positive
                    gt_class_id, _ = gt_boxes[best_gt_idx]
                    matches.append({
                        "class_id": detection.class_id,
                        "confidence": detection.confidence,
                        "is_correct": detection.class_id == gt_class_id,
                        "iou": best_iou,
                    })
                    matched_gt.add(best_gt_idx)
                else:
                    # False positive
                    false_positives.append({
                        "class_id": detection.class_id,
                        "confidence": detection.confidence,
                    })

            # Unmatched ground truth = false negatives
            for gt_idx, (gt_class_id, _) in enumerate(gt_boxes):
                if gt_idx not in matched_gt:
                    false_negatives.append({"class_id": gt_class_id})

        return matches, false_positives, false_negatives

    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute Intersection over Union for two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def _compute_precision(self, matches: List, false_positives: List) -> float:
        """Compute precision."""
        correct = sum(1 for m in matches if m["is_correct"])
        total_positives = len(matches) + len(false_positives)
        return correct / total_positives if total_positives > 0 else 0.0

    def _compute_recall(self, matches: List, false_negatives: List) -> float:
        """Compute recall."""
        correct = sum(1 for m in matches if m["is_correct"])
        total_ground_truth = len(matches) + len(false_negatives)
        return correct / total_ground_truth if total_ground_truth > 0 else 0.0

    def _compute_f1(self, precision: float, recall: float) -> float:
        """Compute F1 score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _compute_map50(self, matches: List, predictions: List, ground_truth: Dict) -> float:
        """Compute mAP @ IOU=0.5 (simplified)."""
        correct = sum(1 for m in matches if m["is_correct"] and m["iou"] >= 0.5)
        total_ground_truth = sum(len(gt) for gt in ground_truth.values())
        return correct / total_ground_truth if total_ground_truth > 0 else 0.0

    def _compute_fpr(self, false_positives: List, total_negatives: int) -> float:
        """Compute false positive rate."""
        if total_negatives == 0:
            return 0.0
        return len(false_positives) / total_negatives

    def _compute_per_class_metrics(
        self,
        matches: List,
        false_positives: List,
        false_negatives: List,
    ) -> Dict[str, ClassMetrics]:
        """Compute metrics per class."""
        per_class = {}

        for class_id, class_name in self.id_to_name.items():
            class_matches = [m for m in matches if m["class_id"] == class_id]
            class_fps = [fp for fp in false_positives if fp["class_id"] == class_id]
            class_fns = [fn for fn in false_negatives if fn["class_id"] == class_id]

            correct = sum(1 for m in class_matches if m["is_correct"])

            precision = correct / (len(class_matches) + len(class_fps)) if (len(class_matches) + len(class_fps)) > 0 else 0.0
            recall = correct / (len(class_matches) + len(class_fns)) if (len(class_matches) + len(class_fns)) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            ap = correct / (len(class_matches) + len(class_fns)) if (len(class_matches) + len(class_fns)) > 0 else 0.0

            per_class[class_name] = ClassMetrics(
                class_id=class_id,
                class_name=class_name,
                precision=precision,
                recall=recall,
                f1=f1,
                ap=ap,
            )

        return per_class

    def _log_report(self, report: EvaluationReport) -> None:
        """Log evaluation report."""
        logger.info("\n" + "=" * 70)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 70)
        logger.info(f"Model: {report.model_name}")
        logger.info(f"Images: {report.num_images}, Detections: {report.num_detections}")
        logger.info(f"Precision: {report.precision:.3f}")
        logger.info(f"Recall: {report.recall:.3f}")
        logger.info(f"F1: {report.f1:.3f}")
        logger.info(f"mAP50: {report.map50:.3f}")
        logger.info(f"FPR: {report.false_positive_rate:.3f}")

        logger.info("\nPer-class metrics:")
        for class_name, metrics in report.per_class_metrics.items():
            logger.info(
                f"  {class_name}: P={metrics.precision:.3f} R={metrics.recall:.3f} F1={metrics.f1:.3f}"
            )
        logger.info("=" * 70 + "\n")

    def plot_confusion_matrix(
        self,
        predictions: List[ImagePrediction],
        ground_truth: Dict[str, List[Tuple]],
        output_path: Path,
    ) -> None:
        """Plot confusion matrix."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect predictions and ground truth
        pred_labels = []
        true_labels = []

        for pred in predictions:
            img_path = str(pred.image_path)
            gt_boxes = ground_truth.get(img_path, [])

            for detection in pred.detections:
                pred_labels.append(detection.class_id)
                # Simplified: assign to first ground truth (in reality, use matching)
                if gt_boxes:
                    true_labels.append(gt_boxes[0][0])
                else:
                    true_labels.append(-1)  # No ground truth

        if not pred_labels or not true_labels:
            logger.warning("No predictions or ground truth for confusion matrix")
            return

        # Compute confusion matrix
        cm = confusion_matrix(
            true_labels,
            pred_labels,
            labels=list(range(len(self.id_to_name))),
        )

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(self.id_to_name.values()),
            yticklabels=list(self.id_to_name.values()),
            ax=ax,
        )
        ax.set_ylabel("Ground Truth")
        ax.set_xlabel("Prediction")
        ax.set_title("Confusion Matrix")

        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved: {output_path}")
        plt.close()


def main():
    """Example usage."""
    import sys

    evaluator = DetectorEvaluator()

    # Dummy predictions and ground truth
    predictions = []  # Would load actual predictions
    ground_truth = {}  # Would load actual ground truth

    report = evaluator.evaluate(predictions, ground_truth, "test_model")
    logger.info(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
