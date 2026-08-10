#!/usr/bin/env python3
"""Split raw YOLO-format datasets into train/val/test directories."""

import logging
import shutil
from pathlib import Path
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_dataset():
    """Split raw YOLO datasets into train/val/test."""
    logger.info("=" * 70)
    logger.info("SPLITTING YOLO DATASET INTO TRAIN/VAL/TEST")
    logger.info("=" * 70)

    raw_dir = Path("data/raw")
    output_dir = Path("data/yolo_dataset")

    # Create output structure
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Collect all images and labels
    all_images = []
    for dataset_dir in ["weapons", "products"]:
        class_dir = raw_dir / dataset_dir
        if not class_dir.exists():
            logger.warning(f"Dataset directory not found: {class_dir}")
            continue

        images_dir = class_dir / "images"
        labels_dir = class_dir / "labels"

        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            continue

        # Find all images
        image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
        logger.info(f"Found {len(image_files)} images in {dataset_dir}")

        for img_path in image_files:
            label_path = labels_dir / (img_path.stem + ".txt")
            all_images.append((img_path, label_path, dataset_dir))

    logger.info(f"Total images: {len(all_images)}")

    # Shuffle and split (70/15/15)
    random.seed(42)
    random.shuffle(all_images)

    n_train = int(len(all_images) * 0.70)
    n_val = int(len(all_images) * 0.15)

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:],
    }

    # Copy files and track statistics
    stats = {"train": {}, "val": {}, "test": {}}

    for split, samples in splits.items():
        logger.info(f"\nProcessing {split}...")
        for img_path, label_path, dataset_name in samples:
            # Copy image
            dst_img = output_dir / "images" / split / img_path.name
            shutil.copy2(img_path, dst_img)

            # Copy label (if exists)
            if label_path.exists():
                dst_label = output_dir / "labels" / split / label_path.name
                shutil.copy2(label_path, dst_label)
            else:
                # Create empty label
                dst_label = output_dir / "labels" / split / (img_path.stem + ".txt")
                dst_label.write_text("0 0.5 0.5 1.0 1.0\n")

            # Track statistics
            class_id = 0 if dataset_name == "weapons" else 1
            if class_id not in stats[split]:
                stats[split][class_id] = 0
            stats[split][class_id] += 1

    # Print statistics
    logger.info("\n" + "=" * 70)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 70)

    class_names = {0: "weapon", 1: "product"}
    for split in ["train", "val", "test"]:
        total = sum(stats[split].values())
        logger.info(f"\n{split.upper()} ({total} samples):")
        for class_id, count in sorted(stats[split].items()):
            pct = 100.0 * count / total if total > 0 else 0
            logger.info(f"  {class_names.get(class_id, f'class_{class_id}')}: {count} ({pct:.1f}%)")

    logger.info("=" * 70)

    # Create data.yaml
    import yaml
    data_yaml = {
        "path": str(output_dir.absolute()),
        "train": str((output_dir / "images/train").absolute()),
        "val": str((output_dir / "images/val").absolute()),
        "test": str((output_dir / "images/test").absolute()),
        "nc": 2,
        "names": {0: "weapon", 1: "product"}
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    logger.info(f"\n✓ Created: {yaml_path}")

    # Verify
    train_images = list((output_dir / "images/train").glob("*.*"))
    val_images = list((output_dir / "images/val").glob("*.*"))
    test_images = list((output_dir / "images/test").glob("*.*"))

    logger.info(f"\n✓ Dataset prepared successfully!")
    logger.info(f"  Train: {len(train_images)} images")
    logger.info(f"  Val: {len(val_images)} images")
    logger.info(f"  Test: {len(test_images)} images")

    return True


def main():
    try:
        success = split_dataset()
        if success:
            logger.info("\n✓ Ready for training!")
            logger.info("Next: python scripts/train_yolov8n.py data/yolo_dataset/data.yaml")
            return 0
        else:
            logger.error("Dataset splitting failed")
            return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
