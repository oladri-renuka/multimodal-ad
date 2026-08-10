#!/usr/bin/env python3
"""Extract images and labels from cached FiftyOne datasets."""

import logging
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_weapons_data():
    """Extract weapons data from FiftyOne cache."""
    logger.info("=" * 70)
    logger.info("EXTRACTING WEAPONS DATA FROM FIFTYONE")
    logger.info("=" * 70)

    try:
        import fiftyone as fo
    except ImportError:
        logger.error("FiftyOne not installed")
        return False

    try:
        # List available datasets
        datasets = fo.list_datasets()
        logger.info(f"Available FiftyOne datasets: {datasets}")

        # Try to find and load the weapons dataset
        weapons_dataset = None
        for ds_name in datasets:
            if "weapon" in ds_name.lower():
                weapons_dataset = ds_name
                break

        if not weapons_dataset:
            logger.warning("No weapons dataset found in FiftyOne cache")
            logger.info("Available datasets: " + ", ".join(datasets))
            return False

        logger.info(f"Loading dataset: {weapons_dataset}")
        dataset = fo.load_dataset(weapons_dataset)

        logger.info(f"Dataset contains {len(dataset)} samples")

        # Create output directories
        output_dir = Path("data/raw/weapons")
        images_dir = output_dir / "images"
        labels_dir = output_dir / "annotations"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Copy images and extract labels
        count = 0
        for sample in dataset:
            try:
                # Copy image
                src_path = Path(sample.filepath)
                if src_path.exists():
                    dst_path = images_dir / src_path.name
                    shutil.copy2(src_path, dst_path)

                    # Extract and save labels
                    if sample.detections is not None:
                        label_lines = []
                        for detection in sample.detections.detections:
                            # YOLO format: class_id center_x center_y width height
                            bbox = detection.bounding_box  # [x, y, width, height] normalized
                            class_id = 0  # weapons
                            label_lines.append(f"{class_id} {bbox[0] + bbox[2]/2} {bbox[1] + bbox[3]/2} {bbox[2]} {bbox[3]}")

                        label_path = labels_dir / (src_path.stem + ".txt")
                        with open(label_path, 'w') as f:
                            f.write("\n".join(label_lines))

                    count += 1
                    if count % 100 == 0:
                        logger.info(f"  Extracted {count}/{len(dataset)}...")

            except Exception as e:
                logger.debug(f"Failed to extract {sample}: {e}")
                continue

        logger.info(f"✓ Extracted {count} weapon images")
        return count > 0

    except Exception as e:
        logger.error(f"Failed to extract weapons data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info("\n" + "=" * 70)
    logger.info("EXTRACT FIFTYONE CACHED DATA")
    logger.info("=" * 70 + "\n")

    result = extract_weapons_data()

    if result:
        logger.info("\n✓ Extraction successful!")
        logger.info("Next: python scripts/prepare_yolo_dataset.py")
        return 0
    else:
        logger.error("\n✗ Extraction failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
