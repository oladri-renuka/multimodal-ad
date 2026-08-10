#!/usr/bin/env python3
"""Main training script for Phase 2: Fine-tune YOLOv8n."""

import logging
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Train YOLOv8n on violation dataset."""
    logger.info("=" * 70)
    logger.info("PHASE 2: FINE-TUNE YOLOv8n ON REAL VIOLATION DATASET")
    logger.info("=" * 70)

    # Parse arguments
    if len(sys.argv) < 2:
        logger.error("Usage: python train_yolov8n.py <dataset_yaml_path>")
        logger.error("Example: python train_yolov8n.py data/yolo_dataset/data.yaml")
        sys.exit(1)

    dataset_yaml = Path(sys.argv[1])

    if not dataset_yaml.exists():
        logger.error(f"Dataset YAML not found: {dataset_yaml}")
        logger.error("First run: python scripts/prepare_yolo_dataset.py")
        sys.exit(1)

    logger.info(f"Dataset YAML: {dataset_yaml}")

    try:
        from src.detectors.yolov8_trainer import YOLOv8nTrainer
        from src.core.config import Config
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you're in the project directory")
        sys.exit(1)

    # Load config
    config = Config.from_env()

    logger.info(f"Model: YOLOv8{config.detector.model_size}")
    logger.info(f"Device: {config.detector.device}")
    logger.info(f"Epochs: {config.detector.epochs}")
    logger.info(f"Batch size: {config.detector.batch_size}")

    # Initialize trainer
    trainer = YOLOv8nTrainer(config)

    logger.info("\n" + "-" * 70)
    logger.info("STEP 1: Evaluate pretrained baseline")
    logger.info("-" * 70)
    baseline_metrics = trainer.evaluate_pretrained(dataset_yaml)

    logger.info("\n" + "-" * 70)
    logger.info("STEP 2: Fine-tune YOLOv8n (50 epochs)")
    logger.info("-" * 70)
    training_metrics = trainer.train(dataset_yaml)

    logger.info("\n" + "-" * 70)
    logger.info("STEP 3: Evaluate fine-tuned model")
    logger.info("-" * 70)
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

    logger.info(f"\n✓ Metrics saved: {output_path}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE - RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Baseline mAP50: {baseline_metrics.get('map50', 'N/A'):.3f}")
    logger.info(f"Fine-tuned mAP50: {finetuned_metrics.get('map50', 'N/A'):.3f}")
    improvement = (finetuned_metrics.get('map50', 0) - baseline_metrics.get('map50', 0)) * 100
    logger.info(f"Improvement: +{improvement:.1f} points")
    logger.info("=" * 70)

    # Next steps
    logger.info("\nNext steps:")
    logger.info("1. View metrics: cat metrics_output/detector_metrics.json")
    logger.info("2. Generate visualizations: python scripts/evaluate_models.py data/yolo_dataset/data.yaml")
    logger.info("3. Run tests: pytest tests/test_detectors.py -v")

    return 0


if __name__ == "__main__":
    sys.exit(main())
