"""Prepare real datasets in YOLO format for YOLOv8 training."""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
import random

logger = logging.getLogger(__name__)


class YOLODatasetPreparer:
    """Convert raw datasets to YOLO format."""

    def __init__(self, raw_dir: Path, output_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.class_names = ["weapon", "nsfw", "counterfeit"]
        self.class_id_map = {name: idx for idx, name in enumerate(self.class_names)}

    def prepare_dataset(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Path:
        """
        Prepare YOLO dataset structure from raw images.

        Creates:
        output_dir/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── data.yaml
        """
        logger.info(f"Preparing YOLO dataset: {self.raw_dir} → {self.output_dir}")

        # Create directory structure
        for split in ["train", "val", "test"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Collect all images by class
        all_samples = []
        for class_name in self.class_names:
            class_dir = self.raw_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            images_dir = class_dir / "images"
            if not images_dir.exists():
                logger.warning(f"Images directory not found: {images_dir}")
                continue

            # Find all images
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            logger.info(f"Found {len(image_files)} images for class '{class_name}'")

            for img_path in image_files:
                all_samples.append((img_path, class_name, class_dir))

        logger.info(f"Total samples: {len(all_samples)}")

        # Shuffle and split
        random.shuffle(all_samples)
        n_train = int(len(all_samples) * train_ratio)
        n_val = int(len(all_samples) * val_ratio)

        splits = {
            "train": all_samples[:n_train],
            "val": all_samples[n_train:n_train + n_val],
            "test": all_samples[n_train + n_val:],
        }

        # Copy images and create labels
        stats = {"train": {}, "val": {}, "test": {}}
        for split, samples in splits.items():
            for img_path, class_name, class_dir in samples:
                # Copy image
                dst_img = self.output_dir / "images" / split / img_path.name
                shutil.copy2(img_path, dst_img)

                # Create label file
                label_path = self.output_dir / "labels" / split / img_path.stem + ".txt"
                self._create_label_file(label_path, class_name, class_dir, img_path)

                # Track stats
                if class_name not in stats[split]:
                    stats[split][class_name] = 0
                stats[split][class_name] += 1

        # Log statistics
        self._log_statistics(stats)

        # Create data.yaml
        self._create_data_yaml()

        logger.info(f"✓ Dataset prepared: {self.output_dir}")
        return self.output_dir

    def _create_label_file(self, label_path: Path, class_name: str, class_dir: Path, img_path: Path) -> None:
        """Create YOLO format label file."""
        class_id = self.class_id_map[class_name]

        # Try to find corresponding annotation
        annotations_dir = class_dir / "annotations"
        annotation_path = annotations_dir / (img_path.stem + ".txt")

        if annotation_path.exists():
            # Copy existing YOLO annotations with updated class ID
            with open(annotation_path, 'r') as f:
                lines = f.readlines()

            # Update class ID (assume first number in line)
            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:
                    parts[0] = str(class_id)
                    updated_lines.append(" ".join(parts) + "\n")

            with open(label_path, 'w') as f:
                f.writelines(updated_lines)
        else:
            # No annotation found, create a dummy full-image box
            # YOLO format: class_id center_x center_y width height (normalized 0-1)
            logger.warning(f"No annotation found for {img_path}, using full-image box")
            with open(label_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

    def _create_data_yaml(self) -> None:
        """Create data.yaml for YOLO training."""
        data_yaml = {
            "path": str(self.output_dir),
            "train": str(self.output_dir / "images/train"),
            "val": str(self.output_dir / "images/val"),
            "test": str(self.output_dir / "images/test"),
            "nc": len(self.class_names),
            "names": {idx: name for idx, name in enumerate(self.class_names)}
        }

        yaml_path = self.output_dir / "data.yaml"
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        logger.info(f"Created: {yaml_path}")

    def _log_statistics(self, stats: Dict) -> None:
        """Log dataset statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 60)

        for split, class_counts in stats.items():
            total = sum(class_counts.values())
            logger.info(f"\n{split.upper()} ({total} samples):")
            for class_name, count in sorted(class_counts.items()):
                pct = 100.0 * count / total if total > 0 else 0
                logger.info(f"  {class_name}: {count} ({pct:.1f}%)")

        logger.info("=" * 60 + "\n")


def prepare_from_raw(
    raw_dir: str = "data/raw",
    output_dir: str = "data/yolo_dataset"
) -> Path:
    """Prepare YOLO dataset from raw directories."""
    preparer = YOLODatasetPreparer(Path(raw_dir), Path(output_dir))
    return preparer.prepare_dataset()


if __name__ == "__main__":
    import sys
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/yolo_dataset"

    logging.basicConfig(level=logging.INFO)
    prepare_from_raw(raw_dir, output_dir)
