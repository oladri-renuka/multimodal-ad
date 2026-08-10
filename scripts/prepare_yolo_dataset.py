#!/usr/bin/env python3
"""Prepare YOLO dataset from raw downloaded images."""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Prepare YOLO dataset from raw images."""
    logger.info("=" * 70)
    logger.info("PREPARE YOLO DATASET FROM RAW IMAGES")
    logger.info("=" * 70)

    try:
        from src.detectors.dataset_prep import prepare_from_raw
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you're in the project directory")
        sys.exit(1)

    # Paths
    raw_dir = "data/raw"
    output_dir = "data/yolo_dataset"

    # Check raw directory exists
    if not Path(raw_dir).exists():
        logger.error(f"Raw directory not found: {raw_dir}")
        sys.exit(1)

    logger.info(f"Raw directory: {raw_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Prepare dataset
    try:
        result_dir = prepare_from_raw(raw_dir, output_dir)
        logger.info(f"✓ Dataset prepared: {result_dir}")

        # Verify
        logger.info("\nVerifying dataset...")
        train_images = list((result_dir / "images/train").glob("*.jpg"))
        val_images = list((result_dir / "images/val").glob("*.jpg"))
        test_images = list((result_dir / "images/test").glob("*.jpg"))
        train_labels = list((result_dir / "labels/train").glob("*.txt"))
        val_labels = list((result_dir / "labels/val").glob("*.txt"))
        test_labels = list((result_dir / "labels/test").glob("*.txt"))

        logger.info(f"  Train: {len(train_images)} images, {len(train_labels)} labels")
        logger.info(f"  Val: {len(val_images)} images, {len(val_labels)} labels")
        logger.info(f"  Test: {len(test_images)} images, {len(test_labels)} labels")
        logger.info(f"  data.yaml: {(result_dir / 'data.yaml').exists()}")

        if (train_images and train_labels and
            val_images and val_labels and
            test_images and test_labels):
            logger.info("\n✓ Dataset preparation successful!")
            logger.info("\nNext steps:")
            logger.info(f"1. python scripts/train_yolov8n.py {result_dir}/data.yaml")
            return 0
        else:
            logger.error("\n✗ Dataset verification failed (missing images or labels)")
            return 1

    except Exception as e:
        logger.error(f"Failed to prepare dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
