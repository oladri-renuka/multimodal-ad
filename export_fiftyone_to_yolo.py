#!/usr/bin/env python3
"""Export FiftyOne datasets to YOLO format."""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def export_weapons_to_yolo():
    """Export weapons dataset from FiftyOne to YOLO format."""
    logger.info("=" * 70)
    logger.info("EXPORTING WEAPONS DATASET TO YOLO FORMAT")
    logger.info("=" * 70)

    try:
        import fiftyone as fo
    except ImportError:
        logger.error("FiftyOne not installed")
        return False

    try:
        # Load the FiftyOne dataset
        logger.info("Loading weapons dataset from FiftyOne...")
        dataset = fo.zoo.load_zoo_dataset(
            "open-images-v6",
            split="train",
            label_types=["detections"],
            classes=["Weapon"],
            max_samples=2000,
            shuffle=True,
            seed=42
        )

        output_dir = Path("data/raw/weapons")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting {len(dataset)} images to YOLO format...")

        # Export to YOLO format
        dataset.export(
            export_dir=str(output_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="detections"
        )

        logger.info(f"✓ Weapons exported to {output_dir}")

        # Verify
        images = list((output_dir / "images").glob("*.jpg")) if (output_dir / "images").exists() else []
        labels = list((output_dir / "labels").glob("*.txt")) if (output_dir / "labels").exists() else []
        logger.info(f"  Images: {len(images)}")
        logger.info(f"  Labels: {len(labels)}")

        return True

    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_products_to_yolo():
    """Export products dataset from FiftyOne to YOLO format."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPORTING PRODUCTS DATASET TO YOLO FORMAT")
    logger.info("=" * 70)

    try:
        import fiftyone as fo
    except ImportError:
        logger.error("FiftyOne not installed")
        return False

    try:
        # Load the FiftyOne dataset
        logger.info("Loading products dataset from FiftyOne...")
        dataset = fo.zoo.load_zoo_dataset(
            "open-images-v6",
            split="train",
            label_types=["detections"],
            classes=["Product"],
            max_samples=2000,
            shuffle=True,
            seed=42
        )

        output_dir = Path("data/raw/products")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting {len(dataset)} images to YOLO format...")

        # Export to YOLO format
        dataset.export(
            export_dir=str(output_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="detections"
        )

        logger.info(f"✓ Products exported to {output_dir}")

        # Verify
        images = list((output_dir / "images").glob("*.jpg")) if (output_dir / "images").exists() else []
        labels = list((output_dir / "labels").glob("*.txt")) if (output_dir / "labels").exists() else []
        logger.info(f"  Images: {len(images)}")
        logger.info(f"  Labels: {len(labels)}")

        return True

    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info("\nEXPORTING FIFTYONE DATASETS TO YOLO FORMAT\n")

    results = {
        "weapons": export_weapons_to_yolo(),
        "products": export_products_to_yolo(),
    }

    logger.info("\n" + "=" * 70)
    logger.info("EXPORT SUMMARY")
    logger.info("=" * 70)
    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{dataset.upper()}: {status}")

    if all(results.values()):
        logger.info("\n✓ All datasets exported successfully!")
        logger.info("\nNext steps:")
        logger.info("1. python scripts/prepare_yolo_dataset.py")
        logger.info("2. python scripts/train_yolov8n.py data/yolo_dataset/data.yaml")
        return 0
    else:
        logger.error("\n✗ Some exports failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
