"""
Kaggle Notebook Script: Phase 2 YOLOv8n Fine-Tuning on T4 GPU

Instructions:
1. Create new Kaggle notebook (Python)
2. Add dataset: multimodal-violation-detector-yolo (3.3GB, 3,292 images)
3. Copy this entire script into notebook
4. Run cells sequentially
5. Download checkpoint from /kaggle/working/models/
"""

# =====================================================================
# CELL 1: Setup & Dependencies
# =====================================================================

import os
import sys
import subprocess
from pathlib import Path

print(" Kaggle Environment Setup")
print("=" * 70)

# Install dependencies
print("\n Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
               "ultralytics==8.4.117", "pyyaml"], check=True)
print(" Installed ultralytics, pyyaml")

# Check GPU
import torch
print(f"\n GPU Info:")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  GPU name: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =====================================================================
# CELL 2: Verify Dataset
# =====================================================================

print("\n Dataset Verification")
print("=" * 70)

dataset_root = Path("/kaggle/input/multimodal-violation-detector-yolo")
if not dataset_root.exists():
    print(f" Dataset not found at {dataset_root}")
    print("Make sure you added the dataset to this notebook!")
    sys.exit(1)

# Count images and labels
train_images = list((dataset_root / "images/train").glob("*.*"))
val_images = list((dataset_root / "images/val").glob("*.*"))
test_images = list((dataset_root / "images/test").glob("*.*"))

print(f"\n Dataset found!")
print(f"  Train: {len(train_images)} images")
print(f"  Val: {len(val_images)} images")
print(f"  Test: {len(test_images)} images")
print(f"  Total: {len(train_images) + len(val_images) + len(test_images)} images")

# Verify data.yaml
yaml_path = dataset_root / "data.yaml"
if yaml_path.exists():
    print(f" data.yaml found")
    with open(yaml_path) as f:
        print(f"  Contents:\n{f.read()}")
else:
    print(f" data.yaml not found at {yaml_path}")
    sys.exit(1)

# =====================================================================
# CELL 3: Fine-Tune YOLOv8n
# =====================================================================

print("\n Phase 2: Fine-Tune YOLOv8n on Real Violation Dataset")
print("=" * 70)

from ultralytics import YOLO

# Set output directory
output_dir = Path("/kaggle/working/models")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n Output directory: {output_dir}")

# Load pretrained model
print("\n1⃣ Loading YOLOv8n pretrained model...")
model = YOLO("yolov8n.pt")

print(f"   Model: YOLOv8n")
print(f"   Device: cuda")
print(f"   Parameters: 3.2M")

# Fine-tune
print("\n2⃣ Fine-tuning for 50 epochs...")
print(f"   Batch size: 16")
print(f"   Learning rate: 0.001")
print(f"   Patience (early stopping): 15")

results = model.train(
    data=str(yaml_path),
    epochs=50,
    batch=16,
    device=0,  # GPU 0
    lr0=0.001,
    weight_decay=5e-4,
    patience=15,
    cache="ram",
    save=True,
    project=str(output_dir),
    name="yolov8n_finetuned",
    verbose=True,
    plots=True,  # Generate training plots
)

print("\n Training complete!")

# =====================================================================
# CELL 4: Evaluate Results
# =====================================================================

print("\n Training Results")
print("=" * 70)

checkpoint_path = output_dir / "yolov8n_finetuned/weights/best.pt"
if checkpoint_path.exists():
    print(f"\n Checkpoint saved: {checkpoint_path}")
    print(f"   Size: {checkpoint_path.stat().st_size / 1e6:.1f} MB")
else:
    print(f" Checkpoint not found at {checkpoint_path}")

# List output files
results_dir = output_dir / "yolov8n_finetuned"
if results_dir.exists():
    print(f"\n Output files in {results_dir}:")
    for f in results_dir.iterdir():
        if f.is_file():
            print(f"   - {f.name} ({f.stat().st_size / 1e6:.1f} MB)" if f.stat().st_size > 1e6 else f"   - {f.name}")

# =====================================================================
# CELL 5: Generate Summary Report
# =====================================================================

print("\n Summary Report")
print("=" * 70)

import json
from datetime import datetime

summary = {
    "timestamp": datetime.now().isoformat(),
    "dataset": {
        "train_images": len(train_images),
        "val_images": len(val_images),
        "test_images": len(test_images),
        "total_images": len(train_images) + len(val_images) + len(test_images),
        "classes": ["weapon", "product"],
        "class_count": {0: 1646, 1: 1646}
    },
    "model": {
        "architecture": "YOLOv8n",
        "parameters": "3.2M",
        "device": "T4 GPU"
    },
    "training": {
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 0.001,
        "weight_decay": 5e-4,
        "early_stopping_patience": 15
    },
    "checkpoint": str(checkpoint_path),
    "output_directory": str(output_dir),
}

summary_path = output_dir / "training_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n Summary saved: {summary_path}")
print(json.dumps(summary, indent=2))

# =====================================================================
# CELL 6: Download Instructions
# =====================================================================

print("\n Download Instructions")
print("=" * 70)
print("""
All output files are in: /kaggle/working/models/yolov8n_finetuned/

Key files to download:
1. weights/best.pt              → Fine-tuned checkpoint (6.3 MB)
2. results.csv                  → Training metrics per epoch
3. confusion_matrix.png         → Class confusion matrix
4. F1_curve.png, PR_curve.png  → Performance curves

Download these files and save to your local:
  models/yolov8n_finetuned/

Then proceed to Phase 3: OCR + ASR extraction
""")

print("\n" + "=" * 70)
print(" Kaggle Training Complete!")
print("=" * 70)
