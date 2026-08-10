"""Fine-tune YOLOv8n on real violation datasets."""

import logging
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime

from ultralytics import YOLO
import torch

# Allow loading ultralytics models in PyTorch 2.6+
try:
    from ultralytics.nn.tasks import DetectionModel
    from torch.nn.modules.container import Sequential
    torch.serialization.add_safe_globals([DetectionModel, Sequential])
except AttributeError:
    # Older PyTorch versions don't have this
    pass
except Exception as e:
    logger.debug(f"Could not configure torch safe globals: {e}")

from src.core.config import Config, DetectorConfig

logger = logging.getLogger(__name__)


class YOLOv8nTrainer:
    """Train and evaluate YOLOv8n on custom violation dataset."""

    def __init__(self, config: Config):
        self.config = config
        self.detector_config = config.detector
        self.device = torch.device(self.detector_config.device)
        self.model = None
        self.training_results = None

    def train(
        self,
        dataset_yaml: Path,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Fine-tune YOLOv8n on custom dataset.

        Args:
            dataset_yaml: Path to data.yaml (YOLO format)
            output_dir: Where to save checkpoints and logs

        Returns:
            Dictionary with training results and metrics
        """
        dataset_yaml = Path(dataset_yaml)
        output_dir = Path(output_dir or "models")
        output_dir.mkdir(parents=True, exist_ok=True)

        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

        logger.info("=" * 70)
        logger.info("PHASE 2: FINE-TUNE YOLOv8n ON REAL VIOLATION DATASET")
        logger.info("=" * 70)
        logger.info(f"Dataset: {dataset_yaml}")
        logger.info(f"Model: YOLOv8{self.detector_config.model_size}")
        logger.info(f"Epochs: {self.detector_config.epochs}")
        logger.info(f"Batch size: {self.detector_config.batch_size}")
        logger.info(f"Device: {self.detector_config.device}")
        logger.info("=" * 70)

        # Load pretrained model
        logger.info(f"Loading pretrained YOLOv8{self.detector_config.model_size}...")
        self.model = YOLO(self.detector_config.pretrained_weights)

        # Fine-tune
        logger.info("Starting fine-tuning...")
        results = self.model.train(
            # Data
            data=str(dataset_yaml),

            # Model
            model=self.model.model if self.model else None,

            # Training hyperparams
            epochs=self.detector_config.epochs,
            batch=self.detector_config.batch_size,
            imgsz=640,
            lr0=self.detector_config.learning_rate,
            lrf=0.01,  # Final learning rate
            momentum=self.detector_config.momentum,
            weight_decay=self.detector_config.weight_decay,
            warmup_epochs=self.detector_config.warmup_epochs,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,

            # Optimization
            optimizer="SGD",  # SGD or Adam
            patience=self.detector_config.early_stopping_patience,
            close_mosaic=15,

            # Device and precision
            device=self.detector_config.device,
            half=self.detector_config.half,  # FP16

            # Data loading
            workers=4,
            cache="ram",  # Cache images in RAM for speed

            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,

            # Saving
            save=True,
            save_period=10,  # Save every 10 epochs
            project=str(output_dir / "yolov8n_finetuned"),
            name="run",
            exist_ok=True,

            # Logging
            verbose=True,
            seed=42,
        )

        self.training_results = results
        logger.info("✓ Fine-tuning complete")

        # Save best checkpoint
        best_checkpoint = output_dir / "yolov8n_finetuned.pt"
        if hasattr(results, 'save_dir'):
            best_model_path = Path(results.save_dir) / "weights/best.pt"
            if best_model_path.exists():
                import shutil
                shutil.copy2(best_model_path, best_checkpoint)
                logger.info(f"✓ Best checkpoint saved: {best_checkpoint}")

        return self._summarize_results(results)

    def evaluate_pretrained(
        self,
        dataset_yaml: Path,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Evaluate pretrained YOLOv8n on test set (baseline comparison).

        Args:
            dataset_yaml: Path to data.yaml
            output_dir: Output directory

        Returns:
            Evaluation metrics
        """
        logger.info("=" * 70)
        logger.info("EVALUATING PRETRAINED YOLOv8n (BASELINE)")
        logger.info("=" * 70)

        output_dir = Path(output_dir or "models")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load pretrained model
        model = YOLO(self.detector_config.pretrained_weights)

        # Evaluate on test set
        results = model.val(
            data=str(dataset_yaml),
            device=self.detector_config.device,
            half=self.detector_config.half,
            imgsz=640,
            project=str(output_dir / "yolov8n_pretrained_eval"),
            name="run",
            exist_ok=True,
        )

        logger.info("✓ Pretrained evaluation complete")

        return self._summarize_results(results)

    def evaluate_finetuned(
        self,
        checkpoint: Path,
        dataset_yaml: Path,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Evaluate fine-tuned checkpoint on test set.

        Args:
            checkpoint: Path to fine-tuned .pt file
            dataset_yaml: Path to data.yaml
            output_dir: Output directory

        Returns:
            Evaluation metrics
        """
        logger.info("=" * 70)
        logger.info("EVALUATING FINE-TUNED YOLOv8n")
        logger.info("=" * 70)

        checkpoint = Path(checkpoint)
        output_dir = Path(output_dir or "models")

        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        # Load fine-tuned model
        model = YOLO(str(checkpoint))

        # Evaluate on test set
        results = model.val(
            data=str(dataset_yaml),
            device=self.detector_config.device,
            half=self.detector_config.half,
            imgsz=640,
            project=str(output_dir / "yolov8n_finetuned_eval"),
            name="run",
            exist_ok=True,
        )

        logger.info("✓ Fine-tuned evaluation complete")

        return self._summarize_results(results)

    def _summarize_results(self, results) -> Dict:
        """Extract key metrics from YOLO results object."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model": f"YOLOv8{self.detector_config.model_size}",
        }

        # Standard YOLO metrics
        if hasattr(results, 'box'):
            metrics["map50"] = float(results.box.map50) if hasattr(results.box, 'map50') else None
            metrics["map"] = float(results.box.map) if hasattr(results.box, 'map') else None

        if hasattr(results, 'box') and hasattr(results.box, 'p'):
            # Per-class precision/recall
            if hasattr(results.box, 'p'):
                metrics["precision"] = float(results.box.p.mean()) if hasattr(results.box.p, 'mean') else None
            if hasattr(results.box, 'r'):
                metrics["recall"] = float(results.box.r.mean()) if hasattr(results.box.r, 'mean') else None

        # Fitness
        if hasattr(results, 'fitness'):
            metrics["fitness"] = float(results.fitness)

        logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
        return metrics

    def save_config(self, output_path: Path) -> None:
        """Save detector config to YAML."""
        output_path = Path(output_path)
        self.config.to_yaml(output_path)
        logger.info(f"Config saved: {output_path}")


def main():
    """Fine-tune YOLOv8n on custom dataset."""
    import sys

    # Load config
    config = Config.from_env()

    # Get dataset YAML path
    dataset_yaml = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/yolo_dataset/data.yaml")

    if not dataset_yaml.exists():
        logger.error(f"Dataset YAML not found: {dataset_yaml}")
        logger.info("Run: python scripts/prepare_yolo_dataset.py")
        sys.exit(1)

    # Train
    trainer = YOLOv8nTrainer(config)

    # Evaluate pretrained baseline
    baseline_metrics = trainer.evaluate_pretrained(dataset_yaml)

    # Fine-tune
    training_metrics = trainer.train(dataset_yaml)

    # Evaluate fine-tuned
    finetuned_metrics = trainer.evaluate_finetuned(
        Path("models/yolov8n_finetuned.pt"),
        dataset_yaml
    )

    # Save comparison
    comparison = {
        "pretrained": baseline_metrics,
        "finetuned": finetuned_metrics,
    }

    output_path = Path("metrics_output/detector_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"✓ Metrics saved: {output_path}")

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Baseline mAP50: {baseline_metrics.get('map50', 'N/A')}")
    logger.info(f"Fine-tuned mAP50: {finetuned_metrics.get('map50', 'N/A')}")
    logger.info("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
