#!/usr/bin/env python3
"""Inspect FiftyOne datasets and find correct field names for export."""

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def inspect_datasets():
    """Inspect available FiftyOne datasets."""
    try:
        import fiftyone as fo
    except ImportError:
        logger.error("FiftyOne not installed")
        return False

    logger.info("=" * 70)
    logger.info("INSPECTING FIFTYONE DATASETS")
    logger.info("=" * 70)

    # List datasets
    datasets = fo.list_datasets()
    logger.info(f"\nAvailable datasets ({len(datasets)}):")
    for ds in datasets:
        logger.info(f"  - {ds}")

    if not datasets:
        logger.warning("No FiftyOne datasets found")
        return False

    # Inspect first dataset with 'weapon' in name
    target_dataset = None
    for ds_name in datasets:
        if "weapon" in ds_name.lower() or "train" in ds_name.lower():
            target_dataset = ds_name
            break

    if not target_dataset:
        target_dataset = datasets[0]

    logger.info(f"\nInspecting dataset: {target_dataset}")
    dataset = fo.load_dataset(target_dataset)

    logger.info(f"Dataset info:")
    logger.info(f"  Samples: {len(dataset)}")
    logger.info(f"  Schema: {dataset.get_field_schema()}")

    # Sample a few records
    logger.info(f"\nSample records:")
    for i, sample in enumerate(dataset.limit(3)):
        logger.info(f"\nSample {i}:")
        logger.info(f"  Filepath: {sample.filepath}")
        logger.info(f"  Fields: {list(sample.keys())}")
        for key in sample.keys():
            if key != "filepath":
                logger.info(f"    {key}: {type(sample[key])}")

    return True


if __name__ == "__main__":
    inspect_datasets()
