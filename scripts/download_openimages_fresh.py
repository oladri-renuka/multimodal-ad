#!/usr/bin/env python3
"""Download OpenImages V6 data using FiftyOne and save locally."""

import logging
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_with_fiftyone(class_name, output_dir, max_samples=2000):
    """Download OpenImages V6 data with specified class."""
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        logger.error("FiftyOne not installed")
        return False

    logger.info(f"\nDownloading {class_name} images...")

    try:
        # Download to FiftyOne's cache
        dataset = foz.load_zoo_dataset(
            "open-images-v6",
            split="train",
            label_types=["detections"],
            classes=[class_name],
            max_samples=max_samples,
            shuffle=True,
            seed=42
        )

        logger.info(f"✓ Loaded {len(dataset)} {class_name} images")

        # Create output directories
        output_path = Path(output_dir)
        images_path = output_path / "images"
        labels_path = output_path / "labels"
        images_path.mkdir(parents=True, exist_ok=True)
        labels_path.mkdir(parents=True, exist_ok=True)

        # Manually copy images and extract labels
        logger.info(f"Processing and copying images to {output_path}...")

        count = 0
        for sample in dataset:
            try:
                # Get source image path (from FiftyOne cache)
                src_image_path = Path(sample.filepath)
                if not src_image_path.exists():
                    logger.warning(f"Source file not found: {src_image_path}")
                    continue

                # Copy image
                dst_image_path = images_path / src_image_path.name
                shutil.copy2(src_image_path, dst_image_path)

                # Extract labels from detections
                label_lines = []

                # Check various possible label field names
                detections = None
                for field_name in ["detections", "ground_truth", "objects", "labels"]:
                    if field_name in sample:
                        detections = sample[field_name]
                        break

                if detections is not None and hasattr(detections, 'detections'):
                    # Process each detection
                    for detection in detections.detections:
                        if hasattr(detection, 'bounding_box'):
                            bbox = detection.bounding_box
                            # YOLO format: class_id center_x center_y width height (normalized)
                            center_x = bbox[0] + bbox[2] / 2
                            center_y = bbox[1] + bbox[3] / 2
                            width = bbox[2]
                            height = bbox[3]
                            label_lines.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

                # Write label file (even if empty)
                label_path = labels_path / (src_image_path.stem + ".txt")
                with open(label_path, 'w') as f:
                    if label_lines:
                        f.write("\n".join(label_lines) + "\n")
                    else:
                        # Full image bounding box if no detections
                        f.write("0 0.5 0.5 1.0 1.0\n")

                count += 1
                if count % 100 == 0:
                    logger.info(f"  Processed {count}/{len(dataset)}...")

            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                continue

        logger.info(f"✓ Successfully exported {count} images")

        # Verify
        images = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        labels = list(labels_path.glob("*.txt"))
        logger.info(f"  Final count: {len(images)} images, {len(labels)} labels")

        return len(images) > 0

    except Exception as e:
        logger.error(f"Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info("=" * 70)
    logger.info("DOWNLOAD OPENIMAGES V6 DATA")
    logger.info("=" * 70)

    # Try common class names for weapons
    weapons_classes = ["Weapon", "Gun", "Firearm", "Rifle", "Handgun", "Knife"]
    product_classes = ["Product", "Object", "Item", "Consumer Product"]

    results = {}

    # Try to download weapons
    for class_name in weapons_classes:
        logger.info(f"\nAttempting to download class: {class_name}")
        result = download_with_fiftyone(class_name, "data/raw/weapons", max_samples=2000)
        if result:
            results["weapons"] = True
            break
        results["weapons"] = False

    # Try to download products
    for class_name in product_classes:
        logger.info(f"\nAttempting to download class: {class_name}")
        result = download_with_fiftyone(class_name, "data/raw/products", max_samples=2000)
        if result:
            results["products"] = True
            break
        results["products"] = False

    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 70)
    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{dataset.upper()}: {status}")

    if any(results.values()):
        logger.info("\n✓ Download successful!")
        logger.info("Next: python scripts/prepare_yolo_dataset.py")
        return 0
    else:
        logger.error("\n✗ All downloads failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
