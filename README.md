# Multimodal Content Safety Reviewer

A production-grade content safety system for detecting policy violations (weapons, NSFW, counterfeit content) across visual, textual, and audio modalities in video/image uploads.

## Overview

This system combines fine-tuned object detection (YOLOv8n), optical character recognition (EasyOCR), automatic speech recognition (Whisper), and rule-based reasoning to provide explainable content moderation verdicts.

**Key Features:**
- Fine-tuned YOLOv8n detector: mAP50 0.78 (+140% over baseline)
- Real-time frame extraction and batch processing
- OCR text extraction with confidence scores
- ASR audio transcription and alignment
- Rule-based violation reasoning with context awareness
- Async FastAPI backend with result polling
- Docker containerization for cloud deployment

## Architecture

```
Input (Image/Video)
        ↓
[Frame Extraction] → Extract frames at target FPS
        ↓
┌─────────────────────────────────────────┐
│ Parallel Analysis Per Frame:            │
│ • YOLOv8n Detection (weapons/NSFW)      │
│ • EasyOCR Text Extraction               │
│ • Whisper ASR (audio track)             │
└─────────────────────────────────────────┘
        ↓
[Rule-Based Reasoning] → Context-aware verdicts
        ↓
[JSON Report] → Structured violations + bboxes
```

## Installation

### Requirements
- Python 3.9+
- PyTorch 2.0+ (CPU or CUDA)
- FFmpeg (for video processing)
- System libraries: libgl1, libxcb1, libXext, libXrender

### Setup

```bash
# Clone repository
git clone <repo-url>
cd multimodal-ad

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
export LIBGL_ALWAYS_INDIRECT=1  # For headless graphics
```

### Docker

```bash
# Build image
docker build -f docker/Dockerfile -t multimodal-ad:latest .

# Run container
docker run -p 8000:8000 \
  -e LIBGL_ALWAYS_INDIRECT=1 \
  multimodal-ad:latest
```

## Usage

### API Endpoints

**POST /analyze** — Submit file for analysis
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@image.jpg"

# Returns:
{
  "job_id": "abc123",
  "status": "processing",
  "file_name": "image.jpg"
}
```

**GET /results/{job_id}** — Poll for results
```bash
curl http://localhost:8000/results/abc123

# Returns (on completion):
{
  "job_id": "abc123",
  "status": "completed",
  "violations_detected": 1,
  "frames": [
    {
      "frame_idx": 0,
      "detections": [
        {
          "class_name": "weapon",
          "confidence": 0.89,
          "bbox_xyxy": [x1, y1, x2, y2]
        }
      ],
      "reasoning": [
        {
          "violation_type": "weapon",
          "confidence": 0.89,
          "reasoning": "Weapon detected with 89% confidence. Requires context review...",
          "recommended_action": "review"
        }
      ]
    }
  ]
}
```

**GET /health** — System health check
```bash
curl http://localhost:8000/health
```

### Web Demo

Open browser to `http://localhost:8000` for interactive demo with:
- Drag-drop file upload
- Real-time analysis
- Visual results with bounding boxes
- Detailed violation breakdowns

## Model Performance

| Metric | Pretrained | Fine-tuned |
|--------|-----------|-----------|
| mAP50 | 0.32 | 0.78 |
| Precision | 0.62 | 0.80 |
| Recall | 0.48 | 0.75 |
| F1-Score | 0.54 | 0.77 |
| FPR | 0.12 | 0.08 |

**Dataset:** 6,000+ real images across 3 violation classes (weapons, NSFW, counterfeit)

## Configuration

Edit `configs/` for model parameters:

- `detector_config.yaml` — YOLOv8 hyperparameters, thresholds
- `pipeline_config.yaml` — FPS, batch sizes, timeouts

Key settings:
```yaml
detector:
  confidence_threshold: 0.45
  iou_threshold: 0.5
  device: "cuda"  # or "cpu"

pipeline:
  target_fps: 2
  max_frames: 500
  inference_timeout: 120
```

## Deployment

### AWS EC2

```bash
# On t3.medium instance (2GB RAM minimum)
ssh -i key.pem ec2-user@<ip>

# Setup
git clone <repo>
cd multimodal-ad
pip install -r requirements.txt
export LIBGL_ALWAYS_INDIRECT=1

# Run
python3 -m src.api.app
```

Access: `http://<public-ip>:8000`

### Railway / Cloud Platform

Environment variables:
```
LIBGL_ALWAYS_INDIRECT=1
LOG_LEVEL=INFO
API_PORT=8000
```

Ensure volume ≥ 20GB and instance has ≥ 2GB RAM.

## Project Structure

```
multimodal-ad/
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI application
│   │   ├── models.py           # Pydantic request/response models
│   │   └── inference.py        # Orchestration + reasoning
│   ├── core/
│   │   ├── config.py           # Configuration system
│   │   ├── frame_extractor.py  # Video frame extraction
│   │   └── data_pipeline.py    # Dataset utilities
│   ├── detectors/
│   │   └── yolov8_inferencer.py # Detection wrapper
│   ├── ocr/
│   │   └── ocr_extractor.py    # EasyOCR wrapper
│   └── asr/
│       └── whisper_extractor.py # Whisper wrapper
├── models/
│   ├── best.pt                 # Fine-tuned YOLOv8n (5.9MB)
│   └── easyocr/                # OCR model cache
├── configs/
│   ├── detector_config.yaml
│   └── pipeline_config.yaml
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Train/val/test splits
│   └── test_clips/             # Demo test media
├── tests/                      # Unit tests
├── requirements.txt            # Dependencies
├── CLAUDE.md                   # Developer conventions
└── README.md                   # This file
```

## Development

### Running Tests

```bash
pytest tests/ -v
pytest tests/test_detectors.py -v --cov=src/detectors
```

### Fine-tuning on Custom Data

```python
from src.detectors.yolov8_trainer import train_detector

config = DetectorConfig(
    dataset_path="data/custom_dataset/data.yaml",
    epochs=50,
    batch_size=16,
    device="cuda"
)

best_model = train_detector(config)
```

Dataset must be in YOLO format:
```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

## Limitations & Known Issues

1. **GPU Memory:** Full model stack requires ≥ 4GB VRAM. CPU inference is slow (~5s/frame).
2. **Context Sensitivity:** Weapons in legitimate contexts (military, training videos) may flag false positives. Designed for human-in-the-loop review, not autonomous enforcement.
3. **Audio Processing:** Whisper requires separate audio track; silent videos skip ASR.
4. **Latency:** Async processing: image ~1s, 30s video ~60-120s depending on FPS extraction.

## Troubleshooting

**libGL.so.1 not found:**
```bash
export LIBGL_ALWAYS_INDIRECT=1
```

**Out of memory on inference:**
- Reduce `target_fps` in config
- Use CPU-only torch
- Deploy to larger instance

**Port 8000 in use:**
```bash
lsof -i :8000
kill -9 <PID>
```

## References

- **YOLOv8**: https://docs.ultralytics.com/
- **EasyOCR**: https://github.com/JaidedAI/EasyOCR
- **Whisper**: https://github.com/openai/whisper
- **FastAPI**: https://fastapi.tiangolo.com/

## License

Proprietary. Dataset sourced from OpenImages V7 (CC-BY-2.0).

## Contact

For questions or deployment support, contact the team.
