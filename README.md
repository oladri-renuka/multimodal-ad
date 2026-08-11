# Multimodal Content Safety Reviewer

A content safety system for detecting policy violations (weapons, NSFW, counterfeit content) across visual, textual, and audio modalities in video/image uploads.

## System Architecture

```
Input (Image/Video)
        ↓
[Frame Extraction] → Extract frames at target FPS (configurable)
        ↓
┌─────────────────────────────────────────┐
│ Per-Frame Parallel Analysis:            │
│ • YOLOv8n Detection (fine-tuned)        │
│ • EasyOCR Text Extraction               │
│ • Whisper ASR (audio track)             │
└─────────────────────────────────────────┘
        ↓
[Rule-Based Reasoning] → Violation classification
        ↓
[JSON Report] → Structured output with bboxes
```

## Components

### 1. Object Detection (YOLOv8n)

Fine-tuned on real violation dataset (6,000+ images).

**Performance on Test Set:**

| Metric | Baseline | Fine-tuned | Δ |
|--------|----------|-----------|---|
| mAP50 | 0.32 | 0.78 | +144% |
| Precision | 0.62 | 0.80 | +29% |
| Recall | 0.48 | 0.75 | +56% |
| F1 | 0.54 | 0.77 | +43% |

**Real-World Results:**

- Training batch (annotated, 7 weapons): 6/7 detected (57-81% confidence)
- Real weapon images: 1-3 detections per image (54-71% confidence)

**Model Details:**
- Architecture: YOLOv8n (3.2M parameters)
- Input: 640x640 RGB
- Output: Bounding boxes + class + confidence
- Inference: ~30ms per frame (CPU), ~5ms (GPU)
- Classes: weapon, nsfw, counterfeit

### 2. Optical Character Recognition (EasyOCR)

Extracts text regions and confidence scores from frames.

- Language: English (configurable)
- Output: Text + bounding boxes + confidence per region
- Used for context filtering (e.g., weapon in training video labeled "Combat Training")

### 3. Automatic Speech Recognition (Whisper)

Extracts and transcribes audio from video.

- Model: whisper-base (~140MB)
- Languages: Multilingual (automatic detection)
- Alignment: Maps transcript segments to frame indices
- Optional: Can skip for silent/image-only inputs

### 4. Rule-Based Reasoning

Combines detector + OCR + ASR outputs into structured verdicts.

```python
if class_name == "weapon":
    action = "review"  # Requires human verification
    reasoning = f"Weapon detected with {confidence:.1%} confidence. Requires context review."
    
elif class_name == "nsfw":
    action = "flag" if confidence > 0.7 else "review"
    
else:  # counterfeit
    action = "review"
```

Verdicts: `["flag", "review", "allow"]`

## API

### POST /analyze

Submit file for analysis.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@image.jpg"
```

Response:
```json
{
  "job_id": "abc123",
  "status": "processing",
  "file_name": "image.jpg"
}
```

### GET /results/{job_id}

Poll for results (completed results cached for 1 hour).

```bash
curl http://localhost:8000/results/abc123
```

Response (on completion):
```json
{
  "job_id": "abc123",
  "status": "completed",
  "violations_detected": 2,
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
      "ocr": [
        {
          "text": "Combat Training",
          "confidence": 0.95,
          "bbox": [[x, y], ...]
        }
      ],
      "reasoning": [
        {
          "violation_type": "weapon",
          "confidence": 0.89,
          "reasoning": "Weapon detected with 89% confidence. Requires context review.",
          "recommended_action": "review",
          "evidence": ["weapon", "confidence:0.89"]
        }
      ]
    }
  ]
}
```

### GET /health

System status and model state.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": {
    "detector": "fine-tuned",
    "ocr": "ready",
    "asr": "ready"
  },
  "gpu_available": false,
  "memory_usage_mb": 1487.45
}
```

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+ (CPU or CUDA)
- OpenCV 5.0+
- FFmpeg (for video processing)

### System Dependencies (Linux/EC2)

```bash
sudo yum install libglvnd-glx libxcb libXext libXrender -y
# or on Ubuntu:
sudo apt-get install libgl1 libsm6 libxext6 libxrender-dev -y
```

### Python Setup

```bash
git clone https://github.com/oladri-renuka/multimodal-ad.git
cd multimodal-ad

pip install -r requirements.txt

# GPU (optional)
pip install torch==2.0.0 torchvision==0.15 --index-url https://download.pytorch.org/whl/cu118

# CPU (default)
pip install torch==2.0.0 torchvision==0.15 --index-url https://download.pytorch.org/whl/cpu
```

## Usage

### Local Server

```bash
export LIBGL_ALWAYS_INDIRECT=1  # Required on headless systems
python3 -m src.api.app
```

Server starts on `http://localhost:8000`

Open `http://localhost:8000` in browser for web UI.

### Configuration

Edit `configs/pipeline_config.yaml`:

```yaml
detector:
  confidence_threshold: 0.45
  iou_threshold: 0.5
  device: "cpu"  # "cuda" if GPU available

pipeline:
  target_fps: 2  # Extract frames at 2 FPS for 30s video = 60 frames
  max_frames: 500
  inference_timeout_sec: 120
```

### Programmatic Usage

```python
from src.api.inference import InferenceOrchestrator
from pathlib import Path

orch = InferenceOrchestrator()

result = orch.analyze_image(
    Path("test_image.jpg"),
    detector_threshold=0.45,
    ocr_enabled=True,
    reasoning_enabled=True
)

print(result["violations_detected"])
print(result["frames"][0]["detections"])
```

## Project Structure

```
src/
├── api/
│   ├── app.py              # FastAPI server
│   ├── models.py           # Pydantic models
│   └── inference.py        # Orchestration + reasoning
├── core/
│   ├── config.py           # Configuration system
│   ├── frame_extractor.py  # Video → frames + audio
│   └── data_pipeline.py    # Dataset utilities
├── detectors/
│   └── yolov8_inferencer.py
├── ocr/
│   └── ocr_extractor.py
└── asr/
    └── whisper_extractor.py

models/
├── best.pt                 # Fine-tuned YOLOv8n (5.9MB)
└── easyocr/                # OCR model cache

configs/
├── detector_config.yaml
└── pipeline_config.yaml

docker/
├── Dockerfile
└── docker-compose.yml

tests/
├── test_detectors.py
├── test_ocr.py
├── test_asr.py
└── conftest.py
```

## Training

### Fine-tuning YOLOv8n

```python
from src.detectors.yolov8_trainer import train_detector
from src.core.config import DetectorConfig

config = DetectorConfig(
    dataset_path="data/yolo_dataset/data.yaml",
    epochs=50,
    batch_size=16,
    device="cuda"
)

best_checkpoint = train_detector(config)
```

Dataset format (YOLO):
```
data/yolo_dataset/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── data.yaml
```

## Testing

```bash
# All tests
pytest tests/ -v

# Specific component
pytest tests/test_detectors.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Known Issues

1. **Confidence Threshold Sensitivity:** Default 0.45 threshold may miss low-confidence violations. Adjust per use case.

2. **Context Loss:** Detector flags weapons regardless of context (military training, historical footage, etc.). Verdicts set to "review" to require human verification.

3. **Audio Processing:** Whisper inference is slow (~5-10s per 30s video). Can be disabled for image-only inputs.

4. **Memory Usage:** Full model stack on CPU uses ~1.5GB RAM. GPU mode (CUDA) requires 4GB+ VRAM.

5. **Video Frame Extraction:** Target FPS (default 2) dramatically affects analysis time. 30s video at 2 FPS = 60 frames ~30s processing.

## References

- YOLOv8 Docs: https://docs.ultralytics.com/
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- Whisper: https://github.com/openai/whisper
- FastAPI: https://fastapi.tiangolo.com/

## Dataset Attribution

- Weapons: OpenImages V7 (CC-BY-2.0)
- NSFW: FAIR LAION Safety Dataset
- Counterfeit: OpenImages Products category
