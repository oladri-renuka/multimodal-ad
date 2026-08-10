#!/usr/bin/env python3
"""Simplified YOLOv8n training for CPU environments."""

import logging
import sys
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Train YOLOv8n on violation dataset."""
    logger.info("=" * 70)
    logger.info("PHASE 2: FINE-TUNE YOLOv8n ON REAL VIOLATION DATASET")
    logger.info("=" * 70)

    # Parse arguments
    if len(sys.argv) < 2:
        logger.error("Usage: python train_yolov8n_simple.py <dataset_yaml_path>")
        sys.exit(1)

    dataset_yaml = Path(sys.argv[1])

    if not dataset_yaml.exists():
        logger.error(f"Dataset YAML not found: {dataset_yaml}")
        sys.exit(1)

    logger.info(f"Dataset YAML: {dataset_yaml}")

    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        logger.error(f"Import error: {e}")
        sys.exit(1)

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if device == "cpu":
        logger.warning("⚠️ CUDA not available - training on CPU will be SLOW")
        epochs = 10  # Reduce epochs for CPU demo
        batch_size = 4  # Reduce batch size for CPU
        logger.info(f"CPU mode: training for {epochs} epochs (reduced from 50 for demo)")
    else:
        epochs = 50
        batch_size = 16
        logger.info(f"GPU mode: training for {epochs} epochs")

    try:
        # Load pretrained model
        logger.info("\nLoading YOLOv8n pretrained model...")
        model = YOLO("yolov8n.pt")

        logger.info(f"Model: YOLOv8n")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Device: {device}")

        # Train
        logger.info("\n" + "-" * 70)
        logger.info("FINE-TUNING (Step 1-3 combined)")
        logger.info("-" * 70)

        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            batch=batch_size,
            device=device,
            lr0=0.001,
            weight_decay=5e-4,
            patience=15,  # Early stopping
            cache="ram",
            save=True,
            project="models",
            name="yolov8n_finetuned",
            verbose=True
        )

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)

        # Try to extract metrics
        if hasattr(results, 'results_dict'):
            logger.info(f"Training results: {results.results_dict}")

        # Save summary
        summary = {
            "device": device,
            "epochs": epochs,
            "batch_size": batch_size,
            "dataset": str(dataset_yaml),
            "model": "yolov8n",
            "status": "completed"
        }

        output_path = Path("metrics_output/training_summary.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✓ Summary saved: {output_path}")
        logger.info(f"✓ Checkpoint: models/yolov8n_finetuned/weights/best.pt")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
