#!/usr/bin/env python3
"""Download datasets from OpenImages V6 and export to local directories."""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_and_export_weapons():
    """Download weapons dataset and export to YOLO format."""
    logger.info("=" * 70)
    logger.info("DOWNLOADING & EXPORTING WEAPONS DATASET")
    logger.info("=" * 70)

    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        logger.error("FiftyOne not installed")
        return False

    try:
        logger.info("Loading weapons dataset from OpenImages V6...")
        dataset = foz.load_zoo_dataset(
            "open-images-v6",
            split="train",
            label_types=["detections"],
            classes=["Weapon"],
            max_samples=2000,
            shuffle=True,
            seed=42,
            dataset_name="weapons_dataset"
        )

        logger.info(f"✓ Loaded {len(dataset)} weapon images")

        # Create output directory
        output_dir = Path("data/raw/weapons")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting to {output_dir}...")

        # Export to YOLO format
        dataset.export(
            export_dir=str(output_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="detections"
        )

        # Verify export
        images = list((output_dir / "images").glob("**/*.jpg"))
        labels = list((output_dir / "labels").glob("**/*.txt"))

        logger.info(f"✓ Weapons exported")
        logger.info(f"  Images: {len(images)}")
        logger.info(f"  Labels: {len(labels)}")

        return len(images) > 0

    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_and_export_products():
    """Download products dataset and export to YOLO format."""
    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOADING & EXPORTING PRODUCTS DATASET")
    logger.info("=" * 70)

    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        logger.error("FiftyOne not installed")
        return False

    try:
        logger.info("Loading products dataset from OpenImages V6...")

        # Note: Need to find correct class name for products
        # Common alternatives: "Product", "Object", or specific product categories
        dataset = foz.load_zoo_dataset(
            "open-images-v6",
            split="train",
            label_types=["detections"],
            max_samples=2000,
            shuffle=True,
            seed=42,
            dataset_name="products_dataset"
        )

        logger.info(f"✓ Loaded {len(dataset)} product images")

        # Create output directory
        output_dir = Path("data/raw/products")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting to {output_dir}...")

        # Export to YOLO format
        dataset.export(
            export_dir=str(output_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="detections"
        )

        # Verify export
        images = list((output_dir / "images").glob("**/*.jpg"))
        labels = list((output_dir / "labels").glob("**/*.txt"))

        logger.info(f"✓ Products exported")
        logger.info(f"  Images: {len(images)}")
        logger.info(f"  Labels: {len(labels)}")

        return len(images) > 0

    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOAD & EXPORT REAL OPENIMAGES DATASETS")
    logger.info("=" * 70)

    results = {
        "weapons": download_and_export_weapons(),
        "products": download_and_export_products(),
    }

    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOAD & EXPORT SUMMARY")
    logger.info("=" * 70)
    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{dataset.upper()}: {status}")

    if all(results.values()):
        logger.info("\n✓ All datasets downloaded and exported!")
        logger.info("\nNext steps:")
        logger.info("1. python scripts/prepare_yolo_dataset.py")
        logger.info("2. python scripts/train_yolov8n.py data/yolo_dataset/data.yaml")
        return 0
    else:
        logger.error("\n✗ Some downloads failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
