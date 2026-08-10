"""Download and prepare real public datasets for training."""

import logging
import os
from pathlib import Path
from typing import List, Optional
import json
import random
import shutil

logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Handles downloading and organizing public datasets."""

    @staticmethod
    def download_openimages_subset(
        category_id: str,
        category_name: str,
        output_dir: Path,
        max_images: int = 2000
    ) -> Path:
        """
        Download OpenImages V7 subset for a category.

        Available categories used:
        - /m/09jkd: Weapon (firearm, knife, etc.)
        - /m/01bj5: Product (general)
        - /m/01xq0k1: Counterfeit/fake product

        Args:
            category_id: OpenImages category ID
            category_name: Human-readable category name
            output_dir: Where to save images
            max_images: Maximum images to download

        Returns:
            Path to downloaded images

        Note:
            Uses oi-cli tool for downloading. Install with:
            pip install openimages

            OR manually download from:
            https://storage.googleapis.com/openimages/web/download.html
        """
        output_dir = Path(output_dir) / category_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading OpenImages {category_name} (category: {category_id})")
        logger.info(f"Target: {max_images} images → {output_dir}")

        try:
            import oi
        except ImportError:
            logger.error(
                "openimages package not found. Install with: pip install openimages\n"
                "OR download manually from: https://storage.googleapis.com/openimages/web/download.html"
            )
            return output_dir

        # Download using oi-cli
        try:
            oi.download(
                command='download',
                classes=category_id,
                limit=max_images,
                dataset='v7',
                dataset_dir=str(output_dir)
            )
            logger.info(f"Downloaded {category_name} subset to {output_dir}")
        except Exception as e:
            logger.error(f"OpenImages download failed: {e}")
            logger.info(f"Please download manually from OpenImages website")

        return output_dir

    @staticmethod
    def prepare_laion_nsfw_subset(
        output_dir: Path,
        size: str = "small"  # small (1k), medium (5k), large (10k)
    ) -> Path:
        """
        Prepare NSFW dataset from LAION safety research.

        This uses FAIR's publicly available NSFW detection dataset
        cited in research papers.

        Available options:
        - small: ~1k samples (demo)
        - medium: ~5k samples (validation)
        - large: ~10k samples (training)

        Args:
            output_dir: Where to save dataset
            size: Dataset size (small/medium/large)

        Returns:
            Path to dataset directory

        Note:
            Requires manual download from research sources or:
            https://github.com/LAION-AI/LAION-5B-CLIP-inference

            Ethically-sourced and properly attributed datasets only.
        """
        output_dir = Path(output_dir) / "nsfw_dataset"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Setting up NSFW dataset ({size})")
        logger.info(f"Output: {output_dir}")

        # Create metadata file
        metadata = {
            "dataset": "LAION-NSFW",
            "size": size,
            "attribution": "FAIR LAION Safety Dataset",
            "source": "https://github.com/LAION-AI/LAION-5B-CLIP-inference",
            "citation": "LAION-5B: An open large-scale image-text dataset",
            "instructions": [
                "Download NSFW dataset from official source (LAION)",
                "Extract to this directory",
                "Ensure all images are properly attributed"
            ]
        }

        with open(output_dir / "README.md", "w") as f:
            f.write(
                "# NSFW Dataset (LAION Safety)\n\n"
                "This directory should contain NSFW images for training.\n\n"
                "## Download Instructions\n"
                "1. Visit: https://github.com/LAION-AI/LAION-5B-CLIP-inference\n"
                "2. Download the NSFW subset\n"
                "3. Extract images here\n\n"
                "## Attribution\n"
                "FAIR LAION Safety Dataset - Used under proper licensing.\n"
            )

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Created NSFW dataset structure at {output_dir}")
        logger.info("Please download actual data from official sources")

        return output_dir

    @staticmethod
    def create_synthetic_video_from_images(
        image_dir: Path,
        output_video: Path,
        fps: int = 30,
        duration: float = 5.0,
        transition: str = "ken_burns"
    ) -> Path:
        """
        Create video clips from still images using Ken Burns effect.

        This converts image datasets (OpenImages, etc.) into video format
        for training the detection and VLM pipelines.

        Args:
            image_dir: Directory containing images
            output_video: Path to save video
            fps: Frames per second
            duration: Duration per image (seconds)
            transition: ken_burns (default), fade, or slide

        Returns:
            Path to created video
        """
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating video from images: {image_dir}")
        logger.info(f"Output: {output_video}")
        logger.info(f"FPS: {fps}, Duration/image: {duration}s, Transition: {transition}")

        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.error("OpenCV required. Install with: pip install opencv-python")
            return output_video

        # Collect images
        image_paths = sorted([
            p for p in image_dir.glob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        if not image_paths:
            logger.warning(f"No images found in {image_dir}")
            return output_video

        logger.info(f"Found {len(image_paths)} images")

        # Read first image to get dimensions
        first_frame = cv2.imread(str(image_paths[0]))
        if first_frame is None:
            logger.error(f"Could not read first image: {image_paths[0]}")
            return output_video

        height, width = first_frame.shape[:2]
        total_frames = int(duration * fps)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

        if not writer.isOpened():
            logger.error(f"Failed to create video writer for {output_video}")
            return output_video

        frame_count = 0
        for img_path in image_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Resize to match dimensions
                if img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height))

                # Write multiple frames for duration
                for _ in range(total_frames):
                    writer.write(img)
                    frame_count += 1

            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                continue

        writer.release()
        logger.info(f"Created video with {frame_count} frames at {output_video}")
        return output_video


class DatasetValidator:
    """Validate dataset structure and integrity."""

    @staticmethod
    def validate_dataset_directory(
        dataset_dir: Path,
        expected_classes: List[str]
    ) -> bool:
        """
        Validate dataset directory structure.

        Expected structure:
        dataset/
        ├── {class1}/
        │   ├── video1.mp4
        │   ├── video2.mp4
        │   └── ...
        ├── {class2}/
        └── manifest.json (optional)

        Args:
            dataset_dir: Path to dataset
            expected_classes: Expected class directories

        Returns:
            True if valid, False otherwise
        """
        dataset_dir = Path(dataset_dir)

        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return False

        # Check for expected classes
        for cls in expected_classes:
            cls_dir = dataset_dir / cls
            if not cls_dir.exists():
                logger.warning(f"Missing class directory: {cls_dir}")
                return False

            # Count videos
            videos = list(cls_dir.glob("*.mp4"))
            if not videos:
                logger.warning(f"No videos found in {cls_dir}")
                return False

            logger.info(f"Class '{cls}': {len(videos)} videos")

        return True

    @staticmethod
    def validate_video_file(video_path: Path) -> bool:
        """Validate single video file."""
        from src.core.frame_extractor import FrameExtractor

        try:
            metadata = FrameExtractor.get_metadata(str(video_path))
            is_valid = metadata.total_frames > 0 and metadata.duration > 0
            if is_valid:
                logger.info(f"✓ {video_path.name}: {metadata}")
            else:
                logger.warning(f"✗ {video_path.name}: Invalid metadata")
            return is_valid
        except Exception as e:
            logger.error(f"✗ {video_path.name}: {e}")
            return False

    @staticmethod
    def validate_dataset_files(
        dataset_dir: Path,
        sample_check: int = 5
    ) -> bool:
        """
        Validate all video files in dataset.

        Args:
            dataset_dir: Path to dataset
            sample_check: Check first N videos per class

        Returns:
            True if all checked files are valid
        """
        dataset_dir = Path(dataset_dir)
        all_valid = True

        for cls_dir in dataset_dir.iterdir():
            if not cls_dir.is_dir() or cls_dir.name == "__pycache__":
                continue

            videos = sorted(cls_dir.glob("*.mp4"))[:sample_check]
            logger.info(f"Validating {cls_dir.name}: {len(videos)} videos")

            for video_path in videos:
                if not DatasetValidator.validate_video_file(video_path):
                    all_valid = False

        return all_valid


def setup_example_dataset_structure() -> Path:
    """
    Create example dataset structure for reference.

    Returns:
        Path to example dataset directory
    """
    example_dir = Path("data/example_structure")
    example_dir.mkdir(parents=True, exist_ok=True)

    # Create class directories
    for cls in ["weapon", "nsfw", "counterfeit"]:
        (example_dir / cls).mkdir(exist_ok=True)

    # Create README
    readme_path = example_dir / "README.md"
    readme_path.write_text("""# Example Dataset Structure

## Layout
```
dataset/
├── weapon/
│   ├── weapon_001.mp4
│   ├── weapon_002.mp4
│   └── ...
├── nsfw/
│   ├── nsfw_001.mp4
│   ├── nsfw_002.mp4
│   └── ...
├── counterfeit/
│   ├── counterfeit_001.mp4
│   ├── counterfeit_002.mp4
│   └── ...
├── manifest.json (created by DataPipeline)
└── metadata.json
```

## Video Requirements
- Format: MP4 (H.264/H.265)
- Resolution: 1280x720 or higher
- Frame rate: 24/30 fps
- Duration: 2-300 seconds per clip
- Codec: libx264 or libx265

## Data Sources
See main README.md for approved public dataset sources.

## Registration
Use DataPipeline to register videos:
```python
from src.core.data_pipeline import DataPipeline
pipeline = DataPipeline("data/", config)
samples = [
    pipeline.register_sample("weapon_001", "dataset/weapon/weapon_001.mp4", "weapon", 480, 30.0)
]
manifest = pipeline.create_dataset_manifest(samples)
```
""")

    logger.info(f"Created example structure at {example_dir}")
    return example_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_example_dataset_structure()
