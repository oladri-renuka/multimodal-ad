#!/usr/bin/env python3
"""Export already-downloaded FiftyOne datasets to YOLO format."""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def export_dataset(dataset_name, output_dir):
    """Export a FiftyOne dataset to YOLO format."""
    try:
        import fiftyone as fo
    except ImportError:
        logger.error("FiftyOne not installed")
        return False

    try:
        logger.info(f"Loading {dataset_name}...")
        dataset = fo.load_dataset(dataset_name)

        logger.info(f"Dataset contains {len(dataset)} samples")

        # Check if dataset has detections
        first_sample = dataset.first()
        if first_sample is None:
            logger.error(f"Dataset {dataset_name} is empty")
            return False

        logger.info(f"Sample info: {first_sample}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting {len(dataset)} samples to {output_path}...")

        # Export to YOLO format
        dataset.export(
            export_dir=str(output_path),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="detections"
        )

        logger.info(f"✓ Exported to {output_path}")

        # Verify
        images = list((output_path / "images").glob("**/*.jpg"))
        labels = list((output_path / "labels").glob("**/*.txt"))
        logger.info(f"  Images: {len(images)}")
        logger.info(f"  Labels: {len(labels)}")

        return len(images) > 0 and len(labels) > 0

    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info("\n" + "=" * 70)
    logger.info("EXPORTING FIFTYONE DATASETS TO YOLO FORMAT")
    logger.info("=" * 70 + "\n")

    try:
        import fiftyone as fo
    except ImportError:
        logger.error("FiftyOne not installed")
        return 1

    # List available datasets
    logger.info("Available FiftyOne datasets:")
    datasets = fo.list_datasets()
    for ds in datasets:
        logger.info(f"  - {ds}")

    results = {}

    # Try to export weapons dataset
    logger.info("\n" + "-" * 70)
    logger.info("EXPORTING WEAPONS")
    logger.info("-" * 70)
    if "open-images-v6-train-2000" in datasets:
        results["weapons"] = export_dataset(
            "open-images-v6-train-2000",
            "data/raw/weapons"
        )
    else:
        logger.warning("Weapons dataset not found")
        results["weapons"] = False

    logger.info("\n" + "=" * 70)
    logger.info("EXPORT SUMMARY")
    logger.info("=" * 70)
    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{dataset.upper()}: {status}")

    if results.get("weapons"):
        logger.info("\n✓ Export successful!")
        logger.info("\nNext steps:")
        logger.info("1. python scripts/prepare_yolo_dataset.py")
        logger.info("2. python scripts/train_yolov8n.py data/yolo_dataset/data.yaml")
        return 0
    else:
        logger.error("\n✗ Export failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
