"""Data pipeline for dataset preparation and splitting."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import random
import shutil
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class DatasetStats:
    """Dataset statistics container."""
    def __init__(self):
        self.total_samples = 0
        self.class_distribution: Dict[str, int] = defaultdict(int)
        self.split_distribution: Dict[str, Dict[str, int]] = {
            "train": defaultdict(int),
            "val": defaultdict(int),
            "test": defaultdict(int)
        }
        self.total_frames = 0
        self.total_duration_seconds = 0.0

    def add_sample(self, class_name: str, num_frames: int, duration: float, split: str = "train"):
        """Add sample statistics."""
        self.class_distribution[class_name] += 1
        self.split_distribution[split][class_name] += 1
        self.total_samples += 1
        self.total_frames += num_frames
        self.total_duration_seconds += duration

    def __repr__(self) -> str:
        lines = [
            f"Dataset Statistics:",
            f"  Total samples: {self.total_samples}",
            f"  Total frames: {self.total_frames}",
            f"  Total duration: {self.total_duration_seconds:.1f}s ({self.total_duration_seconds/3600:.2f}h)",
            f"  Class distribution:",
        ]
        for cls, count in sorted(self.class_distribution.items()):
            pct = 100.0 * count / self.total_samples if self.total_samples > 0 else 0
            lines.append(f"    {cls}: {count} ({pct:.1f}%)")

        for split_name in ["train", "val", "test"]:
            lines.append(f"  {split_name.upper()} split:")
            for cls, count in sorted(self.split_distribution[split_name].items()):
                lines.append(f"    {cls}: {count}")

        return "\n".join(lines)


class DataPipeline:
    """Handles dataset preparation, splitting, and metadata."""

    def __init__(self, dataset_dir: Path, config: "DatasetConfig"):
        self.dataset_dir = Path(dataset_dir)
        self.raw_dir = self.dataset_dir / "raw"
        self.processed_dir = self.dataset_dir / "processed"
        self.config = config

        # Create directories
        for d in [self.raw_dir, self.processed_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def register_sample(
        self,
        sample_id: str,
        video_path: str,
        class_name: str,
        num_frames: int,
        duration: float,
        metadata: Optional[Dict] = None,
        split: Optional[str] = None
    ) -> Dict:
        """
        Register a dataset sample with metadata.

        Args:
            sample_id: Unique identifier for sample
            video_path: Path to video file
            class_name: Violation class (weapon, nsfw, counterfeit)
            num_frames: Number of frames in video
            duration: Duration in seconds
            metadata: Additional metadata dict
            split: Explicit split (train/val/test). If None, will be auto-assigned

        Returns:
            Sample record dict
        """
        sample_record = {
            "id": sample_id,
            "video_path": str(video_path),
            "class": class_name,
            "num_frames": num_frames,
            "duration": duration,
            "split": split or self._assign_split(class_name),
            "metadata": metadata or {}
        }

        return sample_record

    def _assign_split(self, class_name: str) -> str:
        """Assign sample to train/val/test split ensuring class balance."""
        rand = random.random()
        if rand < self.config.train_ratio:
            return "train"
        elif rand < (self.config.train_ratio + self.config.val_ratio):
            return "val"
        else:
            return "test"

    def create_dataset_manifest(
        self,
        samples: List[Dict],
        manifest_path: Optional[Path] = None
    ) -> Dict:
        """
        Create dataset manifest with splits and statistics.

        Args:
            samples: List of sample records
            manifest_path: Path to save manifest JSON

        Returns:
            Manifest dict
        """
        manifest = {
            "version": "1.0",
            "total_samples": len(samples),
            "classes": self.config.classes,
            "splits": {"train": [], "val": [], "test": []},
            "statistics": None,
            "samples": samples
        }

        # Organize by split
        for sample in samples:
            manifest["splits"][sample["split"]].append(sample["id"])

        # Calculate statistics
        stats = DatasetStats()
        for sample in samples:
            stats.add_sample(
                sample["class"],
                sample["num_frames"],
                sample["duration"],
                sample["split"]
            )
        manifest["statistics"] = {
            "class_distribution": dict(stats.class_distribution),
            "total_frames": stats.total_frames,
            "total_duration_seconds": stats.total_duration_seconds
        }

        # Save manifest
        if manifest_path is None:
            manifest_path = self.processed_dir / "manifest.json"

        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Created dataset manifest: {manifest_path}")
        logger.info(f"Total samples: {manifest['total_samples']}")
        logger.info(f"  Train: {len(manifest['splits']['train'])}")
        logger.info(f"  Val: {len(manifest['splits']['val'])}")
        logger.info(f"  Test: {len(manifest['splits']['test'])}")

        return manifest

    def create_yolo_dataset(
        self,
        manifest: Dict,
        samples_dir: Path,
        output_dir: Path
    ) -> Path:
        """
        Create YOLO format dataset structure for training.

        Expected structure:
        dataset/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── data.yaml

        Args:
            manifest: Dataset manifest
            samples_dir: Directory containing sample videos/frames
            output_dir: Output dataset directory

        Returns:
            Path to created dataset
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create split directories
        for split in ["train", "val", "test"]:
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Create data.yaml for YOLO
        data_yaml = {
            "path": str(output_dir),
            "train": str(output_dir / "images" / "train"),
            "val": str(output_dir / "images" / "val"),
            "test": str(output_dir / "images" / "test"),
            "nc": len(self.config.classes),
            "names": {i: cls for i, cls in enumerate(self.config.classes)}
        }

        yaml_path = output_dir / "data.yaml"
        import yaml
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        logger.info(f"Created YOLO dataset config: {yaml_path}")

        return output_dir

    def load_manifest(self, manifest_path: Path) -> Dict:
        """Load dataset manifest from JSON."""
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        return manifest

    def get_split_samples(
        self,
        manifest: Dict,
        split: str
    ) -> List[Dict]:
        """Get all samples for a specific split."""
        sample_ids = set(manifest["splits"].get(split, []))
        return [s for s in manifest["samples"] if s["id"] in sample_ids]

    def print_statistics(self, manifest: Dict) -> None:
        """Print dataset statistics."""
        stats = manifest.get("statistics", {})
        logger.info("=" * 60)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total samples: {manifest['total_samples']}")

        class_dist = stats.get("class_distribution", {})
        logger.info("Class distribution:")
        for cls, count in sorted(class_dist.items()):
            pct = 100.0 * count / manifest["total_samples"]
            logger.info(f"  {cls}: {count} ({pct:.1f}%)")

        logger.info(f"Total frames: {stats.get('total_frames', 0)}")
        logger.info(f"Total duration: {stats.get('total_duration_seconds', 0):.1f}s")

        splits = manifest.get("splits", {})
        logger.info("Split distribution:")
        for split_name in ["train", "val", "test"]:
            count = len(splits.get(split_name, []))
            pct = 100.0 * count / manifest["total_samples"]
            logger.info(f"  {split_name}: {count} ({pct:.1f}%)")

        logger.info("=" * 60)


class DataAugmentationConfig:
    """Configuration for data augmentation."""
    def __init__(self):
        self.enable_flip_horizontal = True
        self.enable_flip_vertical = False
        self.brightness_range = (0.7, 1.3)
        self.contrast_range = (0.7, 1.3)
        self.saturation_range = (0.7, 1.3)
        self.hue_shift_range = (-15, 15)
        self.enable_temporal_jitter = True
        self.temporal_jitter_frames = 2


class DataAugmenter:
    """Applies augmentation to frames."""

    def __init__(self, config: DataAugmentationConfig):
        self.config = config

        try:
            import albumentations as A
            self.A = A
            self.augmentor = A.Compose([
                A.HorizontalFlip(p=0.5) if config.enable_flip_horizontal else A.NoOp(),
                A.RandomBrightnessContrast(
                    brightness_limit=config.brightness_range,
                    contrast_limit=config.contrast_range,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=config.hue_shift_range,
                    sat_shift_limit=config.saturation_range,
                    val_shift_limit=config.saturation_range,
                    p=0.5
                ),
            ])
        except ImportError:
            logger.warning("albumentations not available, augmentation disabled")
            self.augmentor = None

    def augment_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply augmentation to a single frame."""
        if self.augmentor is None:
            return frame

        try:
            augmented = self.augmentor(image=frame)
            return augmented["image"]
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}, returning original frame")
            return frame

    def augment_frames(self, frames: List[np.ndarray], times: int = 1) -> List[np.ndarray]:
        """Apply augmentation to multiple frames."""
        augmented = []
        for _ in range(times):
            for frame in frames:
                augmented.append(self.augment_frame(frame))
        return augmented
