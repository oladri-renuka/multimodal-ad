# Phase 2: YOLOv8n Fine-Tuning on Kaggle T4 GPU

**Status**: ✅ Dataset ready (3,292 real images, 763 MB)  
**GPU**: Kaggle T4 (16 GB VRAM, free)  
**Training Time**: ~1-2 hours (50 epochs)  
**Expected Results**: mAP50 improvement from 0.32 → 0.78

---

## Step 1: Prepare Kaggle Dataset

### Option A: Upload via Kaggle Website (Recommended)

1. **Create Kaggle Dataset**
   - Go to: https://www.kaggle.com/settings/account
   - Click "Create" → "Dataset"
   - Upload `data/yolo_dataset.tar.gz` (763 MB)
   - Title: `multimodal-violation-detector-yolo`
   - Description: "3,292 real weapon/product detection images in YOLO format"
   - Make it **Public** so you can use in notebooks
   - Wait for upload to complete (~5-10 min)

2. **Note the Dataset Slug**
   - After upload, you'll see: `username/multimodal-violation-detector-yolo`
   - Copy this for Step 3

### Option B: Upload via Kaggle CLI

```bash
# Install Kaggle CLI
pip install kaggle

# Setup authentication (requires kaggle.json from account)
# Place ~/.kaggle/kaggle.json (get from https://www.kaggle.com/settings/account)

# Create dataset
kaggle datasets init -p data/yolo_dataset

# Edit data/yolo_dataset/dataset-metadata.json:
{
  "title": "Multimodal Violation Detector YOLO",
  "id": "multimodal-violation-detector-yolo",
  "licenses": [{"name": "CC0-1.0"}],
  "keywords": ["yolo", "object-detection", "weapons", "products"],
  "resources": [],
  "collaborators": [],
  "data": []
}

# Upload
cd data/yolo_dataset
kaggle datasets create --public
```

---

## Step 2: Create Kaggle Notebook

1. Go to: https://www.kaggle.com/code
2. Click **"Create"** → **"Notebook"**
3. Select **Python** kernel
4. Enable GPU:
   - Click **⚙️ Settings** (top right)
   - Under "Accelerator", select **"GPU T4"**
   - Save settings

---

## Step 3: Add Dataset to Notebook

1. In notebook, click **"+ Add Input"** (left sidebar)
2. Search for: `multimodal-violation-detector-yolo`
3. Click to add dataset
4. You'll see it mounted at: `/kaggle/input/multimodal-violation-detector-yolo/`

---

## Step 4: Run Training Script

**Copy the entire content of:**
```
scripts/kaggle_training_notebook.py
```

**Into your Kaggle notebook** as separate cells (or paste all at once, then "Split Cell" to separate)

**Alternatively, paste this minimal version:**

```python
# CELL 1: Install & Check GPU
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.117"])
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# CELL 2: Verify dataset
from pathlib import Path
dataset = Path("/kaggle/input/multimodal-violation-detector-yolo")
print(f"Train: {len(list((dataset / 'images/train').glob('*.*')))} images")
print(f"Val: {len(list((dataset / 'images/val').glob('*.*')))} images")
print(f"Test: {len(list((dataset / 'images/test').glob('*.*')))} images")

# CELL 3: Fine-tune
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model.train(
    data=str(dataset / "data.yaml"),
    epochs=50,
    batch=16,
    device=0,
    lr0=0.001,
    weight_decay=5e-4,
    patience=15,
    cache="ram",
    save=True,
    project="/kaggle/working/models",
    name="yolov8n_finetuned",
    verbose=True,
    plots=True
)

# CELL 4: Check results
import json
results_dir = Path("/kaggle/working/models/yolov8n_finetuned")
print(f"\n✅ Checkpoint: {results_dir / 'weights/best.pt'}")
print(f"Files: {list(results_dir.iterdir())}")
```

**Then click "Run All" ▶️**

---

## Step 5: Download Checkpoint

After training completes (~1-2 hours):

1. In notebook, click **"Output"** tab (right side)
2. Navigate to: `models/yolov8n_finetuned/`
3. Download these files:
   - ✅ `weights/best.pt` (6.3 MB) — **Main checkpoint**
   - ✅ `results.csv` — Training metrics
   - ✅ `confusion_matrix.png` — Class confusion matrix
   - ✅ `PR_curve.png`, `F1_curve.png` — Performance curves

4. Save to local project:
   ```bash
   # Local machine
   mkdir -p models/yolov8n_finetuned/weights
   # Download best.pt from Kaggle → models/yolov8n_finetuned/weights/
   ```

---

## Step 6: Verify Checkpoint Locally

```bash
# Back on your machine
python -c "
from pathlib import Path
from ultralytics import YOLO

checkpoint = Path('models/yolov8n_finetuned/weights/best.pt')
if checkpoint.exists():
    print(f'✅ Checkpoint loaded: {checkpoint}')
    model = YOLO(str(checkpoint))
    print(f'✅ Model ready for inference or further training')
else:
    print('❌ Checkpoint not found')
"
```

---

## Expected Training Output

```
Ultralytics 8.4.117 🚀 Python-3.13 torch-2.12.1
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss     Instances       Size
       1/50      5.2G      1.234      0.654      0.892           234       640
       2/50      5.2G      1.089      0.589      0.823           234       640
      ...
      50/50      5.2G      0.234      0.123      0.189           234       640

50 epochs completed in 1.2 hours.
Results saved to: /kaggle/working/models/yolov8n_finetuned/
```

---

## Expected Results

After 50 epochs on T4:

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|-----------|-------------|
| **mAP50** | 0.32 | 0.78 | +0.46 (+144%) |
| **Precision** | 0.62 | 0.80 | +0.18 |
| **Recall** | 0.48 | 0.75 | +0.27 |
| **F1** | 0.54 | 0.77 | +0.23 |

---

## Troubleshooting

### "Dataset not found" error
→ Make sure you added the dataset to the notebook (Step 3)

### "CUDA out of memory"
→ Reduce batch size: `batch=8` instead of 16
→ Or use CPU: `device="cpu"` (slower)

### "Training too slow"
→ Check GPU is enabled: Settings → Accelerator → GPU T4
→ Check memory: Run `!nvidia-smi`

### Upload dataset too large (>2GB)
→ Split into multiple uploads or compress further:
```bash
tar -czf yolo_dataset.tar.gz data/yolo_dataset/
```

---

## What's Next (Phase 3+)

Once checkpoint is downloaded:

```bash
# Phase 3: OCR + ASR Extraction
python scripts/extract_ocr_asr.py models/yolov8n_finetuned/weights/best.pt

# Phase 4: VLM Reasoning
python scripts/run_vlm_reasoning.py

# Phase 5: Full Metrics Pipeline
pytest tests/test_detectors.py -v
```

---

## Dataset Contents

```
data/yolo_dataset/
├── images/
│   ├── train/      (2,304 images)
│   ├── val/        (493 images)
│   └── test/       (495 images)
├── labels/
│   ├── train/      (2,304 .txt files, YOLO format)
│   ├── val/        (493 .txt files)
│   └── test/       (495 .txt files)
└── data.yaml       (class mappings, paths)

Total: 3,292 real images (no synthetic data)
Classes: weapon (1,646), product (1,646)
Annotations: YOLO normalized format
```

---

## Quick Reference

| Item | Link/Path |
|------|-----------|
| **Kaggle Code** | https://www.kaggle.com/code |
| **Dataset Upload** | https://www.kaggle.com/datasets |
| **Notebook Kernel Settings** | GPU: T4 |
| **Training Time** | ~1-2 hours (50 epochs) |
| **Output Path** | `/kaggle/working/models/yolov8n_finetuned/` |
| **Checkpoint Size** | 6.3 MB |
| **Local Save Path** | `models/yolov8n_finetuned/weights/best.pt` |

---

## Success Checklist

- [ ] Dataset uploaded to Kaggle (multimodal-violation-detector-yolo)
- [ ] Notebook created with GPU T4 enabled
- [ ] Dataset added to notebook
- [ ] Training script running
- [ ] Training completes (~1-2 hours)
- [ ] Checkpoint downloaded to local machine
- [ ] Checkpoint loads successfully with YOLO()
- [ ] Ready for Phase 3 (OCR + ASR)

---

**Estimated Total Time**: 2-3 hours (including upload + training + download)

**Need Help?** Check Kaggle notebook error logs or run `!nvidia-smi` to verify GPU setup
