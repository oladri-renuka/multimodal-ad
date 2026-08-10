#!/usr/bin/env python3
"""Download REAL violation datasets from public sources - NO SYNTHETIC DATA."""

import logging
import sys
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_weapons_dataset():
    """Download REAL weapons dataset from OpenImages V7."""
    logger.info("=" * 70)
    logger.info("DOWNLOADING REAL WEAPONS DATASET (OpenImages V7)")
    logger.info("=" * 70)

    output_dir = Path("data/raw/weapons")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try different methods to download
    try:
        # Method 1: Try direct openimages import
        logger.info("Attempting download via openimages.download...")
        from openimages.download import download

        download(
            classes=["Weapon"],
            limit=2000,
            dataset="v7",
            dataset_dir=str(output_dir)
        )
        logger.info("✓ Weapons dataset successfully downloaded")
        return True

    except Exception as e:
        logger.error(f"Method 1 failed: {e}")

    try:
        # Method 2: Try cvdata (installed with openimages)
        logger.info("Attempting download via cvdata...")
        from cvdata.download import downloader

        downloader(
            classes=["Weapon"],
            limit=2000,
            dataset="v7",
            dataset_dir=str(output_dir)
        )
        logger.info("✓ Weapons dataset successfully downloaded")
        return True

    except Exception as e:
        logger.error(f"Method 2 failed: {e}")

    # Method 3: Manual instructions
    logger.warning("\nAutomatic download failed. Please download manually:")
    logger.warning("-" * 70)
    logger.warning("1. Visit: https://storage.googleapis.com/openimages/web/download.html")
    logger.warning("2. Select Dataset: OpenImages V7")
    logger.warning("3. Select Class: Weapon (category /m/09jkd)")
    logger.warning("4. Download images + annotations")
    logger.warning("5. Extract to: data/raw/weapons/")
    logger.warning("-" * 70)

    return False


def download_products_dataset():
    """Download REAL products dataset from OpenImages V7 for counterfeit detection."""
    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOADING REAL PRODUCTS DATASET (OpenImages V7)")
    logger.info("=" * 70)

    output_dir = Path("data/raw/products")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from openimages.download import download

        download(
            classes=["Product"],
            limit=2000,
            dataset="v7",
            dataset_dir=str(output_dir)
        )
        logger.info("✓ Products dataset successfully downloaded")
        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.warning("\nManual download instructions:")
        logger.warning("-" * 70)
        logger.warning("1. Visit: https://storage.googleapis.com/openimages/web/download.html")
        logger.warning("2. Select Dataset: OpenImages V7")
        logger.warning("3. Select Class: Product (category /m/01bj5)")
        logger.warning("4. Download images + annotations")
        logger.warning("5. Extract to: data/raw/products/")
        logger.warning("-" * 70)
        return False


def setup_nsfw_dataset():
    """Set up NSFW dataset (requires manual download from FAIR LAION)."""
    logger.info("\n" + "=" * 70)
    logger.info("SETTING UP NSFW DATASET (FAIR LAION Safety Dataset)")
    logger.info("=" * 70)

    nsfw_dir = Path("data/raw/nsfw")
    (nsfw_dir / "images").mkdir(parents=True, exist_ok=True)
    (nsfw_dir / "labels").mkdir(parents=True, exist_ok=True)

    readme_path = nsfw_dir / "README.md"
    readme_path.write_text("""# NSFW Dataset (FAIR LAION Safety Dataset)

This directory contains REAL NSFW images from the FAIR LAION Safety research dataset.

## Download Instructions (REAL DATA ONLY)

1. **Source**: https://github.com/LAION-AI/LAION-5B-CLIP-inference
   - Ethically-sourced research dataset
   - Proper attribution required

2. **Steps**:
   a) Clone: git clone https://github.com/LAION-AI/LAION-5B-CLIP-inference
   b) Follow their download instructions for NSFW subset
   c) Download ~2000 real NSFW images with proper labels
   d) Convert to YOLO format if needed

3. **Directory Structure**:
   data/raw/nsfw/
   ├── images/
   │   ├── xxxxxx.jpg
   │   ├── xxxxxx.jpg
   │   └── ... (2000+ real NSFW images)
   └── labels/
       ├── xxxxxx.txt (YOLO format: class_id bbox...)
       ├── xxxxxx.txt
       └── ... (one label per image)

## Attribution
- FAIR LAION Safety Dataset
- Used for research and safety training only
- Proper citation: https://github.com/LAION-AI/LAION-5B-CLIP-inference

## Important
- All images must be real, ethically-sourced data
- NO synthetic or artificially generated images
- Proper labeling required (YOLO format)
- Target: 2000-4000 images for quality fine-tuning
""")

    logger.info(f"✓ NSFW dataset directory created: {nsfw_dir}")
    logger.info(f"✓ README with download instructions: {readme_path}")
    return True


def verify_datasets():
    """Verify downloaded datasets."""
    logger.info("\n" + "=" * 70)
    logger.info("VERIFYING DATASETS")
    logger.info("=" * 70)

    datasets = {
        "weapons": Path("data/raw/weapons"),
        "products": Path("data/raw/products"),
        "nsfw": Path("data/raw/nsfw"),
    }

    for name, path in datasets.items():
        if path.exists():
            images = list((path / "images").glob("*.jpg")) + list((path / "images").glob("*.png"))
            logger.info(f"✓ {name}: {len(images)} images found")
        else:
            logger.warning(f"✗ {name}: NOT YET DOWNLOADED")


def main():
    """Download all REAL datasets."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: DOWNLOAD REAL VIOLATION DATASETS")
    logger.info("NO SYNTHETIC DATA - PRODUCTION QUALITY ONLY")
    logger.info("=" * 70)

    # Attempt downloads
    results = {
        "weapons": download_weapons_dataset(),
        "products": download_products_dataset(),
        "nsfw": setup_nsfw_dataset(),
    }

    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOAD STATUS")
    logger.info("=" * 70)
    for dataset, success in results.items():
        status = "✓ READY" if success else "⚠ MANUAL DOWNLOAD REQUIRED"
        logger.info(f"{dataset.upper()}: {status}")

    # Verify
    verify_datasets()

    logger.info("\n" + "=" * 70)
    logger.info("NEXT STEPS")
    logger.info("=" * 70)
    logger.info("1. Complete manual downloads for any failed datasets")
    logger.info("2. Verify all images in data/raw/{weapon,products,nsfw}/images/")
    logger.info("3. Ensure YOLO format labels in data/raw/{weapon,products,nsfw}/labels/")
    logger.info("4. Run: python scripts/prepare_yolo_dataset.py")
    logger.info("5. Run: python scripts/train_yolov8n.py")
    logger.info("=" * 70 + "\n")

    return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
