# Multimodal Content Safety Reviewer — Project Documentation

## Project Overview

Production-grade content safety system for video/ad analysis across visual, textual, and audio modalities. Designed for TikTok Trust & Safety (weapons, NSFW) and Business Integrity (counterfeit detection).

**Key Features:**
- Fine-tuned YOLOv8n object detector (weapons, NSFW, counterfeit)
- OCR (EasyOCR) + ASR (Whisper-small) for text and speech extraction
- VLM reasoning layer (LLaVA-1.5 7B quantized) for explainable verdicts
- FastAPI backend + Streamlit demo
- Production metrics: precision/recall/F1/FPR per component, ablation studies

---

## Project Structure

```
src/
├── core/
│   ├── config.py            # Dataclass-based configuration system
│   ├── frame_extractor.py   # Video → frames/audio (ffmpeg)
│   ├── data_pipeline.py     # Dataset creation, splitting, manifests
│   └── pipeline.py          # Main inference orchestration (TBD)
├── detectors/
│   ├── yolov8_trainer.py    # Fine-tuning loop (TBD)
│   └── yolov8_inferencer.py # Inference wrapper (TBD)
├── ocr/
│   └── ocr_extractor.py     # EasyOCR wrapper (TBD)
├── asr/
│   └── whisper_extractor.py # Whisper wrapper (TBD)
├── vlm/
│   ├── vlm_reasoner.py      # LLaVA inference + quantization (TBD)
│   ├── quantization.py      # 4-bit quantization setup (TBD)
│   └── prompting.py         # Structured prompt templates (TBD)
├── metrics/
│   ├── evaluator.py         # Precision/recall/F1/confusion matrix (TBD)
│   └── ablation.py          # Ablation study runner (TBD)
├── api/
│   ├── app.py               # FastAPI application (TBD)
│   ├── models.py            # Pydantic request/response models (TBD)
│   └── inference.py         # Async inference queue (TBD)
└── demo/
    ├── app.py               # Streamlit demo (TBD)
    └── frontend.html        # Simple HTML demo (TBD)

configs/
├── detector_config.yaml     # YOLOv8 hyperparams (TBD)
├── vlm_config.yaml          # VLM settings (TBD)
└── pipeline_config.yaml     # Full pipeline config (TBD)

data/
├── raw/                     # Downloaded public datasets
├── processed/               # Train/val/test splits
└── test_clips/              # Demo test videos

notebooks/
├── 01_eda.ipynb             # Dataset exploration (TBD)
├── 02_detector_ablation.ipynb # Model comparison (TBD)
└── 03_metrics_report.ipynb  # Final metrics & ablation (TBD)

tests/
├── test_detectors.py        # Unit tests for detection (TBD)
├── test_ocr.py              # Unit tests for OCR (TBD)
├── test_asr.py              # Unit tests for ASR (TBD)
├── test_vlm.py              # Unit tests for VLM (TBD)
├── test_api.py              # Integration tests for API (TBD)
└── conftest.py              # Pytest fixtures

docker/
├── Dockerfile               # Production container (TBD)
└── docker-compose.yml       # Local dev setup (TBD)

models/                       # (TBD) Checkpoints directory
├── yolov8n_finetuned.pt
└── llava_quantized.pt

logs/                        # Training logs, inference logs
```

---

## Configuration

### Loading Configuration

```python
from src.core.config import Config

# Option 1: From YAML
config = Config.from_yaml("configs/detector_config.yaml")

# Option 2: From environment variables
config = Config.from_env()

# Option 3: Defaults
config = Config()
```

### Key Configuration Classes

- **DatasetConfig**: Paths, splits (70/15/15), target FPS (16), classes
- **DetectorConfig**: YOLOv8 hyperparams, thresholds, device settings
- **VLMConfig**: Model name, quantization (4-bit), token limits
- **APIConfig**: Host/port, upload limits, queue settings

---

## Data Pipeline

### Dataset Registration

```python
from src.core.data_pipeline import DataPipeline
from src.core.config import DatasetConfig

config = DatasetConfig()
pipeline = DataPipeline("data/", config)

# Register a sample
sample = pipeline.register_sample(
    sample_id="weapon_001",
    video_path="data/raw/weapon.mp4",
    class_name="weapon",
    num_frames=480,
    duration=30.0
)

# Create manifest with splits
manifest = pipeline.create_dataset_manifest(samples=[sample, ...])
```

### Dataset Classes

- **weapon**: Firearms, knives, explosives
- **nsfw**: Nudity, sexual content
- **counterfeit**: Fake products, counterfeit logos, fraud

---

## Frame Extraction

```python
from src.core.frame_extractor import FrameExtractor

# Get metadata
metadata = FrameExtractor.get_metadata("video.mp4")
print(metadata)  # VideoMetadata(width=1920, height=1080, fps=30.0, ...)

# Extract frames at 16 FPS
frames, metadata = FrameExtractor.extract_frames(
    "video.mp4",
    "output_frames/",
    target_fps=16
)

# Extract audio
audio_path = FrameExtractor.extract_audio("video.mp4", "output_audio.wav")
```

---

## Model Paths & Checkpoints

All model checkpoints should be stored in `models/` directory:

```
models/
├── yolov8n_finetuned.pt      # Fine-tuned YOLOv8n checkpoint
└── llava_quantized.pt        # Quantized LLaVA checkpoint (if saved)
```

Default checkpoint paths in config:
- `detector.checkpoint_path`: `./models/yolov8n_finetuned.pt`
- `vlm.checkpoint_path`: `./models/llava_quantized.pt` (usually loaded from HuggingFace)

---

## Logging

All modules use `logging` with logger names matching module paths:

```python
import logging
logger = logging.getLogger(__name__)

# Will produce: "INFO: src.detectors.yolov8_trainer — Starting training..."
logger.info("Starting training...")
```

Configure logging level in `.env`:
```
LOG_LEVEL=INFO
LOG_DIR=./logs
```

---

## Critical Functions (Phase 1 Complete)

### Completed in Phase 1
1. **src/core/config.py**: Full configuration system with dataclasses + YAML
2. **src/core/frame_extractor.py**: Video → frames/audio via ffmpeg
3. **src/core/data_pipeline.py**: Dataset manifest creation, splitting, augmentation

### Key Functions
- `Config.from_yaml()` — Load config from YAML
- `Config.from_env()` — Load config from environment
- `FrameExtractor.extract_frames()` — Extract video frames at target FPS
- `FrameExtractor.extract_audio()` — Extract audio from video
- `DataPipeline.register_sample()` — Register a dataset sample
- `DataPipeline.create_dataset_manifest()` — Create dataset splits + stats
- `DataAugmenter.augment_frame()` — Apply augmentation

---

## Dependencies & Versions

Pinned in `requirements.txt`:
- **PyTorch 2.0.1** (GPU/CPU)
- **YOLOv8** (ultralytics 8.0.168)
- **Whisper** (openai-whisper 20230314)
- **EasyOCR** (easyocr 1.7.0)
- **LLaVA** (via transformers 4.33.2 + bitsandbytes 0.41.1)
- **FastAPI** (0.103.0)
- **Streamlit** (1.28.1)

Install:
```bash
pip install -r requirements.txt
```

---

## Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key variables:
- `DATA_RAW_DIR`: Path to raw dataset
- `DETECTOR_CHECKPOINT`: Path to fine-tuned YOLOv8
- `DEVICE`: cuda or cpu
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR
- `API_PORT`: FastAPI port (default 8000)

---

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_detectors.py -v
```

With coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

---

## Naming Conventions

### Modules
- Snake_case for files: `yolov8_trainer.py`, `frame_extractor.py`
- PascalCase for classes: `FrameExtractor`, `DetectorConfig`

### Functions
- Snake_case: `extract_frames()`, `get_metadata()`
- Private functions: Leading underscore `_assign_split()`

### Variables
- Snake_case: `target_fps`, `confidence_threshold`
- Constants: UPPER_CASE: `DEFAULT_BATCH_SIZE = 16`

### Data
- Sample IDs: `{class}_{number}`, e.g., `weapon_001`, `nsfw_042`
- Frame filenames: `frame_{index:06d}.jpg`
- Manifest files: `manifest.json`, `data.yaml` (YOLO format)

---

## Common Tasks

### Add a new configuration field
Edit `src/core/config.py`, add field to relevant dataclass:
```python
@dataclass
class DetectorConfig:
    new_field: str = "default_value"
```

### Extract frames from a video
```python
from src.core.frame_extractor import FrameExtractor
frames, metadata = FrameExtractor.extract_frames("video.mp4", "output/", target_fps=16)
```

### Register dataset samples
```python
from src.core.data_pipeline import DataPipeline
pipeline = DataPipeline("data/", config)
samples = [
    pipeline.register_sample("id1", "path1", "weapon", 480, 30.0),
    pipeline.register_sample("id2", "path2", "nsfw", 240, 15.0),
]
manifest = pipeline.create_dataset_manifest(samples)
```

---

## Next Phases (TBD)

- **Phase 2**: Fine-tune YOLOv8n on violation dataset
- **Phase 3**: Implement OCR + ASR extraction
- **Phase 4**: Integrate VLM reasoning layer
- **Phase 5**: Build metrics pipeline + ablation study
- **Phase 6**: FastAPI backend + Docker
- **Phase 7**: Streamlit demo + documentation

---

## References

- **YOLOv8**: https://docs.ultralytics.com/
- **LLaVA**: https://github.com/haotian-liu/LLaVA
- **Whisper**: https://github.com/openai/whisper
- **EasyOCR**: https://github.com/JaidedAI/EasyOCR
- **FastAPI**: https://fastapi.tiangolo.com/
- **Streamlit**: https://streamlit.io/

