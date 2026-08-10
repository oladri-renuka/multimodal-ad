# Phase 2: Download REAL Violation Datasets

**NO SYNTHETIC DATA. PRODUCTION QUALITY ONLY.**

This guide provides exact commands and links to download real, ethically-sourced violation datasets for fine-tuning YOLOv8n.

---

## Dataset 1: Weapons Detection (OpenImages V7)

### What You're Downloading
- **2000-4000 real weapon images** with bounding box annotations
- Categories: firearms, knives, explosives, weapons
- Source: OpenImages V7 (public, properly licensed)
- Format: YOLO (class_id, bbox coordinates, normalized 0-1)

### Download Steps

**Option A: Web Download (Recommended)**
1. Visit: https://storage.googleapis.com/openimages/web/download.html
2. Select:
   - Dataset: OpenImages V7
   - What to download: Images
   - Classes: Weapon (category `/m/09jkd`)
   - License: Include all licenses
   - Split: Train + Validation + Test
3. Click "Download"
4. Extract to: `data/raw/weapons/`

**Directory structure after extraction:**
```
data/raw/weapons/
├── images/
│   ├── xxxxxxx.jpg
│   ├── xxxxxxx.jpg
│   └── ... (2000+ real weapon images)
└── annotations/
    ├── xxxxxxx.txt  (YOLO format)
    ├── xxxxxxx.txt
    └── ... (matching labels)
```

**Option B: Command Line (if available)**
```bash
# Install OpenImages CLI tool
pip install oi-dataset

# Download weapons (replace with actual CLI command for your system)
oi-download --classes Weapon --limit 2000 --dataset v7 --dataset_dir data/raw/weapons/
```

### Verification
```bash
# Check downloaded images
ls data/raw/weapons/images/ | wc -l  # Should show 2000+
ls data/raw/weapons/annotations/ | wc -l  # Should match image count
```

---

## Dataset 2: Products/Counterfeit Detection (OpenImages V7)

### What You're Downloading
- **2000-4000 real product images**
- Used for counterfeit/suspicious product detection
- Source: OpenImages V7
- Format: YOLO with class labels

### Download Steps

1. Visit: https://storage.googleapis.com/openimages/web/download.html
2. Select:
   - Dataset: OpenImages V7
   - What to download: Images
   - Classes: Product (category `/m/01bj5`)
   - License: Include all licenses
3. Click "Download"
4. Extract to: `data/raw/products/`

### Verification
```bash
ls data/raw/products/images/ | wc -l  # Should show 2000+
ls data/raw/products/annotations/ | wc -l
```

---

## Dataset 3: NSFW/Explicit Content (FAIR LAION Safety Dataset)

### What You're Downloading
- **2000-4000 real NSFW images** from FAIR LAION Safety research dataset
- Ethically-sourced, properly attributed
- Used for research and safety model training
- Format: Images + YOLO-format labels

### Download Steps

**Official Source:**
- Repository: https://github.com/LAION-AI/LAION-5B-CLIP-inference
- Dataset: LAION-5B Safety Subset

**Steps:**
1. Visit: https://github.com/LAION-AI/LAION-5B-CLIP-inference
2. Follow their download instructions for NSFW safety dataset
3. Download ~2000-4000 images with labels
4. Convert to YOLO format if needed (class_id, bbox coordinates)
5. Extract to: `data/raw/nsfw/`

### Directory Structure
```
data/raw/nsfw/
├── images/
│   ├── xxxxxxx.jpg
│   ├── xxxxxxx.jpg
│   └── ... (2000+ real NSFW images)
└── labels/
    ├── xxxxxxx.txt  (YOLO format)
    ├── xxxxxxx.txt
    └── ... (matching labels)
```

### Conversion to YOLO Format (if needed)
If labels are not in YOLO format, convert them:
```python
# Example: Convert bounding box to YOLO format
# YOLO format: class_id center_x center_y width height (all normalized 0-1)

def convert_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height, class_id):
    """Convert absolute bbox to YOLO normalized format."""
    center_x = ((xmin + xmax) / 2) / img_width
    center_y = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return f"{class_id} {center_x} {center_y} {width} {height}"
```

### Attribution
```
Dataset: FAIR LAION-5B Safety Dataset
Source: https://github.com/LAION-AI/LAION-5B-CLIP-inference
Citation: Follow their citation guidelines
License: Check their license terms
Usage: Research and safety model training
```

---

## Combining Datasets

After downloading all three:

```bash
# Verify structure
tree data/raw/  # or: ls -la data/raw/

# Expected:
# data/raw/
# ├── weapons/
# │   ├── images/ (2000+ .jpg)
# │   └── annotations/ (2000+ .txt)
# ├── products/
# │   ├── images/ (2000+ .jpg)
# │   └── annotations/ (2000+ .txt)
# └── nsfw/
#     ├── images/ (2000+ .jpg)
#     └── labels/ (2000+ .txt)

# Total: 6000-12000 real images across 3 classes
```

---

## Next Steps (After Download)

Once all datasets are downloaded and verified:

```bash
# 1. Prepare YOLO dataset format
python scripts/prepare_yolo_dataset.py

# 2. Train YOLOv8n
python scripts/train_yolov8n.py

# 3. Generate metrics and ablation study
python scripts/evaluate_models.py
```

---

## Troubleshooting

### Issue: OpenImages website download is slow
**Solution**: Use a download manager (wget, curl) to resume interrupted downloads
```bash
wget -c "https://openimages.download/url" -O data/raw/weapons/image.jpg
```

### Issue: YOLO format labels are different
**Solution**: Convert using the provided script or custom conversion logic
```bash
python scripts/convert_annotations_to_yolo.py \
  --input-dir data/raw/weapons/annotations \
  --output-dir data/raw/weapons/annotations_yolo
```

### Issue: Missing image or annotation files
**Solution**: Verify completeness
```bash
python -c "
import os
img_dir = 'data/raw/weapons/images'
ann_dir = 'data/raw/weapons/annotations'
imgs = set(os.path.splitext(f)[0] for f in os.listdir(img_dir))
anns = set(os.path.splitext(f)[0] for f in os.listdir(ann_dir))
missing = imgs - anns
print(f'Missing annotations: {len(missing)}')
"
```

---

## Verification Checklist

- [ ] Weapons: 2000+ images in `data/raw/weapons/images/`
- [ ] Weapons: Matching .txt labels in `data/raw/weapons/annotations/`
- [ ] Products: 2000+ images in `data/raw/products/images/`
- [ ] Products: Matching .txt labels in `data/raw/products/annotations/`
- [ ] NSFW: 2000+ images in `data/raw/nsfw/images/`
- [ ] NSFW: Matching .txt labels in `data/raw/nsfw/labels/`
- [ ] All labels in YOLO format (class_id bbox_x bbox_y bbox_w bbox_h)
- [ ] Total: 6000-12000 real images

---

## Important Notes

**DO NOT USE SYNTHETIC DATA**
- All images must be real, ethically-sourced data
- No artificially generated or composite images
- Proper attribution and licensing required

**QUALITY OVER SPEED**
- Downloads may take 1-2 hours
- This is expected for production-grade datasets
- Quality metrics depend on real, diverse data

**REPRODUCIBILITY**
- Keep downloaded datasets in `data/raw/`
- Document which sources you used
- Save download URLs for reproducibility

---

## Timeline

- **Download time**: 30-60 min (parallel, depending on internet)
- **Dataset preparation**: 10-15 min
- **Fine-tuning**: 1-2 hrs (A100) or 4-6 hrs (RTX 3090)
- **Metrics & ablation**: 15-30 min
- **Total Phase 2**: 2.5-4 hours (A100) or 5-8 hours (RTX 3090)

---

**Status**: Download datasets now, then run training scripts.  
**Next Command**: `python scripts/prepare_yolo_dataset.py`
