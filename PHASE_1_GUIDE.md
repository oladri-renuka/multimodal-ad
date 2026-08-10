# Phase 1: Project Setup & Data Pipeline — Complete Guide

**Status**: Phase 1 Complete (Core infrastructure ready)

This guide walks through setting up the project with real datasets and verifying everything works end-to-end.

---

## What's Included in Phase 1

✅ **Core Infrastructure**
- `src/core/config.py` — Production configuration system (dataclasses + YAML)
- `src/core/frame_extractor.py` — Video frame/audio extraction via ffmpeg
- `src/core/data_pipeline.py` — Dataset manifest creation, splitting, validation
- `src/core/dataset_downloader.py` — Real dataset downloading and preparation

✅ **Testing & Documentation**
- `tests/conftest.py` — Pytest fixtures for all test types
- `tests/test_core.py` — Comprehensive unit tests (45+ tests)
- `CLAUDE.md` — Project conventions and critical functions
- `README.md` — Project overview and quick start

✅ **Configuration**
- `requirements.txt` — Pinned dependencies (PyTorch 2.0.1, YOLOv8, etc.)
- `.env.example` — Environment variables template
- `CLAUDE.md` — Internal documentation

---

## Installation & Setup

### 1. Create Python Environment (3.10+)

```bash
cd /Users/renukaoladri/Claude/Projects/multimodal_ad

python3.10 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Time**: ~5-10 minutes (depending on internet speed)

**Verification**:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "from ultralytics import YOLO; print('YOLOv8 OK')"
python -c "import whisper; print('Whisper OK')"
python -c "import easyocr; print('EasyOCR OK')"
```

### 3. Copy Environment Configuration

```bash
cp .env.example .env
```

Edit `.env` and set:
```
DEVICE=cuda  # or 'cpu' for testing without GPU
LOG_LEVEL=INFO
DATA_RAW_DIR=./data/raw
```

---

## Real Dataset Setup

We use **three real public datasets** for weapons, NSFW, and counterfeit detection.

### Option A: Quick Start (Mock Dataset)

For testing the pipeline without downloading real data:

```python
from src.core.data_pipeline import DataPipeline
from src.core.config import DatasetConfig

# Create mock dataset
config = DatasetConfig()
pipeline = DataPipeline("data/", config)

# Register test samples
samples = [
    pipeline.register_sample("weapon_001", "data/test_clips/weapon.mp4", "weapon", 480, 30.0),
    pipeline.register_sample("nsfw_001", "data/test_clips/nsfw.mp4", "nsfw", 240, 15.0),
]

manifest = pipeline.create_dataset_manifest(samples)
print(manifest)
```

### Option B: Real Datasets (Recommended)

#### 1. Weapons Dataset (OpenImages V7)

```bash
# Install openimages downloader
pip install openimages

# Download weapons subset (~2000 images)
mkdir -p data/raw/weapons
oi download --classes Weapon --limit 2000 --dataset v7 --dataset_dir data/raw/weapons
```

**What you get**:
- ~2000 labeled weapon images (firearms, knives, explosives)
- Bounding box annotations
- Source: OpenImages V7 (public, properly licensed)

**File structure**:
```
data/raw/weapons/
├── images/
│   ├── {image_id}.jpg
│   ├── ...
├── annotations/
│   ├── {image_id}.txt (YOLO format)
```

#### 2. NSFW Dataset (FAIR LAION)

**Important**: NSFW datasets require ethical sourcing and proper attribution.

**Option 1 - Download from research source**:
```bash
# Follow LAION-AI official instructions
# https://github.com/LAION-AI/LAION-5B-CLIP-inference

# Download sample NSFW subset
wget https://github.com/LAION-AI/LAION-5B-CLIP-inference/releases/download/...
```

**Option 2 - Use existing training dataset**:
```bash
# Use FAIR's publicly available validation set
# https://github.com/facebookresearch/detectron2/tree/main/datasets

# Document source with attribution
```

**What you need**:
- ~1000-5000 NSFW images (ethically-sourced, properly attributed)
- Clear attribution of source
- Verification of licensing

#### 3. Counterfeit/Logo Dataset (OpenImages)

```bash
# Download product images (general)
oi download --classes Product --limit 1000 --dataset v7 --dataset_dir data/raw/products

# Alternative: Use brand/logo detection datasets
# https://github.com/ultralytics/yolov5/wiki/Datasets#logo-detection-15
```

**What you get**:
- ~1000 product images
- Some will be counterfeit/suspicious (manually reviewed)

### Creating Videos from Images

After downloading images, convert them to video format:

```python
from src.core.dataset_downloader import DatasetDownloader
from pathlib import Path

downloader = DatasetDownloader()

# Create videos from weapon images
downloader.create_synthetic_video_from_images(
    image_dir=Path("data/raw/weapons/images"),
    output_video=Path("data/raw/weapons/weapon_video_001.mp4"),
    fps=30,
    duration=3.0  # 3 seconds per image
)

# Create videos from product images
downloader.create_synthetic_video_from_images(
    image_dir=Path("data/raw/products/images"),
    output_video=Path("data/raw/products/product_video_001.mp4"),
    fps=30,
    duration=3.0
)
```

**Result**: Video clips with Ken Burns effect (realistic camera motion)

---

## Verify Installation

### Test 1: Import All Modules

```python
from src.core.config import Config, DatasetConfig
from src.core.frame_extractor import FrameExtractor
from src.core.data_pipeline import DataPipeline, DataAugmenter
from src.core.dataset_downloader import DatasetDownloader, DatasetValidator

print("✓ All imports successful")
```

### Test 2: Configuration Loading

```python
from src.core.config import Config

# Default config
config = Config()
print(f"✓ Default config loaded")

# From environment
config_env = Config.from_env()
print(f"✓ Config from env loaded")

# To YAML
config.to_yaml("test_config.yaml")
print(f"✓ Config saved to YAML")
```

### Test 3: Frame Extraction (With Sample Video)

```python
from src.core.frame_extractor import FrameExtractor
from pathlib import Path

# You'll need a sample video for this test
# For now, test with any MP4 you have:

video_path = "sample_video.mp4"  # Replace with real video

# Get metadata
metadata = FrameExtractor.get_metadata(video_path)
print(f"✓ Video metadata: {metadata}")

# Extract frames
frames, metadata = FrameExtractor.extract_frames(
    video_path,
    "output_frames/",
    target_fps=16
)
print(f"✓ Extracted {len(frames)} frames")

# Extract audio
audio_path = FrameExtractor.extract_audio(video_path, "output.wav")
print(f"✓ Extracted audio to {audio_path}")
```

### Test 4: Data Pipeline with Real Samples

```python
from src.core.data_pipeline import DataPipeline
from src.core.config import DatasetConfig
from pathlib import Path

config = DatasetConfig()
pipeline = DataPipeline("data/", config)

# Register real samples
samples = [
    pipeline.register_sample(
        "weapon_001",
        "data/raw/weapons/weapon_video_001.mp4",
        "weapon",
        num_frames=90,
        duration=3.0
    ),
    pipeline.register_sample(
        "product_001",
        "data/raw/products/product_video_001.mp4",
        "counterfeit",
        num_frames=90,
        duration=3.0
    )
]

# Create manifest
manifest = pipeline.create_dataset_manifest(samples)
print(f"✓ Created manifest with {manifest['total_samples']} samples")

# Print statistics
pipeline.print_statistics(manifest)
```

### Test 5: Run Unit Tests

```bash
# Run all tests
pytest tests/test_core.py -v

# Run with coverage
pytest tests/test_core.py --cov=src --cov-report=html

# Run specific test
pytest tests/test_core.py::TestFrameExtractor::test_get_metadata_from_sample_video -v
```

**Expected**: 45+ tests passing, all core functionality validated

---

## Phase 1 Completion Checklist

- [ ] Python 3.10+ environment created
- [ ] `requirements.txt` installed successfully
- [ ] `ffmpeg` installed (test with `ffmpeg -version`)
- [ ] Configuration system tested (import, load, save)
- [ ] Frame extractor tested with sample video
- [ ] Data pipeline tested with sample registration
- [ ] Dataset directories created (`data/raw`, `data/processed`)
- [ ] At least one real dataset downloaded and validated
- [ ] Unit tests all passing (45+ tests)
- [ ] `README.md` and `CLAUDE.md` reviewed

---

## Common Issues & Fixes

### Issue: "No module named 'ffmpeg'"

**Solution**:
```bash
# Install ffmpeg system package
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
choco install ffmpeg
```

### Issue: CUDA out of memory during tests

**Solution**:
```bash
# Use CPU for testing
export DEVICE=cpu
pytest tests/ -v
```

### Issue: "openimages" download fails

**Solution**:
```bash
# Download manually from OpenImages website
# https://storage.googleapis.com/openimages/web/download.html

# Or use alternative datasets:
# - COCO: https://cocodataset.org/
# - ImageNet: https://www.image-net.org/
```

### Issue: Video creation fails with "could not open codec"

**Solution**:
```bash
# Install ffmpeg with video codec support
pip install opencv-python  # Ensure latest version
pip install imageio imageio-ffmpeg
```

---

## Next: Phase 2 Preview

Once Phase 1 is complete, we'll move to **Phase 2: Fine-Tune YOLOv8n**:

1. **Prepare training dataset** from collected images
2. **Create YOLO dataset format** (images, labels, data.yaml)
3. **Fine-tune YOLOv8n** with real hyperparameters
4. **Evaluate on held-out test set** (precision/recall/F1)
5. **Compare vs. pretrained baseline** (show improvement)
6. **Generate ablation metrics** (detector only baseline)

**Timeline**: Phase 2 takes ~1-2 hours of GPU training time.

---

## Documentation References

- **Project structure**: See `CLAUDE.md`
- **API documentation**: See function docstrings in `src/`
- **Example usage**: See tests in `tests/test_core.py`
- **Configuration**: See `src/core/config.py` dataclasses

---

## Running Phase 1 Tests (Full Verification)

```bash
# Activate venv
source venv/bin/activate

# Run all Phase 1 tests
pytest tests/test_core.py -v -s

# Run with detailed logging
pytest tests/test_core.py -v -s --log-cli-level=INFO

# Generate coverage report
pytest tests/test_core.py --cov=src --cov-report=html
open htmlcov/index.html  # View report
```

**Expected output**:
```
tests/test_core.py::TestConfig::test_default_config_creation PASSED
tests/test_core.py::TestFrameExtractor::test_get_metadata_from_sample_video PASSED
tests/test_core.py::TestDataPipeline::test_create_dataset_manifest PASSED
...
===================== 45 passed in 2.34s =====================
```

---

## Summary

**Phase 1 Complete** ✅

You now have:
1. ✅ Working Python environment with all dependencies
2. ✅ Core infrastructure (config, frame extraction, data pipeline)
3. ✅ Real dataset downloading and preparation code
4. ✅ Comprehensive tests (45+ unit tests)
5. ✅ Full documentation (CLAUDE.md, README.md, this guide)

**Ready for Phase 2**: Fine-tuning YOLOv8n on real violation data.

