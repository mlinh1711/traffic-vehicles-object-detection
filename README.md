# Traffic Vehicles Object Detection (YOLO Experiments)

This repository contains a complete **traffic vehicle object detection** workflow implemented in a single Jupyter Notebook. The notebook covers environment setup, dataset sanity checks, label cleaning, training multiple YOLO configurations, validating models on the validation split, and selecting the best model based on mAP.

Dataset (Kaggle): https://www.kaggle.com/datasets/saumyapatel/traffic-vehicles-object-detection

Notebook code reference: :contentReference[oaicite:0]{index=0}

---

## What this project does

The notebook implements an end to end pipeline:

1. Install and import required libraries (Ultralytics YOLO, OpenCV, etc.)
2. Create `data.yaml` for YOLO training
3. EDA + sanity checks:
   - Count images and labels per split
   - Check class distribution
   - Validate label format and ranges
   - Visualize sample images with bounding boxes
4. Clean the dataset:
   - Remove invalid label rows (if any)
   - Create a cleaned dataset folder at `/kaggle/working/traffic_dataset_clean`
   - Re-generate `data.yaml` pointing to the cleaned dataset
5. Train and compare multiple models:
   - YOLOv8n baseline
   - YOLOv8s (higher capacity, higher resolution)
   - YOLOv9m (larger model)
6. Validate each model on the validation set and export a comparison table
7. Automatically select the best model based on **mAP50-95** and save it as `best_overall.pt`
8. Run inference on the test split using the best model and save visualized predictions

---

## Classes

The model detects 7 classes:

- Car
- Number Plate
- Blur Number Plate
- Two Wheeler
- Auto
- Bus
- Truck

---

## Results summary (Validation)

Validation split stats (from the notebook output):
- Images: 185
- Instances: 1980
- Classes: 7

The notebook evaluates 3 experiments and prints a sorted comparison table by **mAP50-95**:

| Experiment        | imgsz | Precision | Recall | mAP50  | mAP50-95 |
|------------------|------:|----------:|-------:|-------:|---------:|
| y8s_896          |   896 | 0.812     | 0.774  | 0.852  | 0.607    |
| y9m_896          |   896 | 0.723     | 0.799  | 0.821  | 0.543    |
| y8n_baseline_736 |   736 | 0.804     | 0.694  | 0.777  | 0.506    |

Best model selected by the notebook:
- **Exp:** `y8s_896`
- **Best weights alias:** `/kaggle/working/best_overall.pt`
- **Image size:** 896

Note: the notebook also saves a CSV: `/kaggle/working/val_compare_models.csv`

---

## Repository structure

```

traffic-vehicles-object-detection/
├── traffic-vehicles-object-detection.ipynb
├── requirements.txt
├── .gitignore
└── README.md

````

- `traffic-vehicles-object-detection.ipynb`: full pipeline (EDA, cleaning, training, validation, best-model selection, test prediction)
- `requirements.txt`: dependencies
- `.gitignore`: prevents large datasets, videos, and generated artifacts from being pushed to GitHub

---

## Dataset note (important)

The dataset is not included in this repo because GitHub blocks large files (for example, video files can exceed 100MB per file). Please download it from Kaggle using the link above and place it locally (or in Kaggle workspace) to run the notebook.

---

## Requirements (based on the notebook)

Recommended: Python 3.9+ (the notebook log shows it was run on Python 3.11)

Install dependencies:
```bash
pip install -r requirements.txt
````

Your `requirements.txt` should include the libraries actually imported/installed in the notebook:

* ultralytics
* opencv-python
* pillow
* imagehash
* numpy
* pandas
* matplotlib
* PyYAML

Tip: `ultralytics` pulls PyTorch automatically. If you want GPU support locally, install the correct PyTorch build for your CUDA version.

---

## How to run

### Option A: Kaggle (recommended)

1. Create a Kaggle notebook
2. Add the Kaggle dataset to the notebook (as an input)
3. Upload this repository notebook or copy the code
4. Run cells in order

### Option B: Local machine

1. Clone repo
2. Install requirements
3. Download dataset from Kaggle and extract it
4. Update dataset path in the notebook if needed
5. Run notebook top to bottom

---

## Outputs produced by the notebook

During training and evaluation, Ultralytics will create experiment folders under `/kaggle/working/`, for example:

* `/kaggle/working/y8n_baseline_736/`
* `/kaggle/working/y8s_896/`
* `/kaggle/working/y9m_896/`

The notebook also creates:

* `/kaggle/working/val_compare_models.csv`
* `/kaggle/working/best_overall.pt`
* `/kaggle/working/best_selection.json`
* `/kaggle/working/test_with_boxes/` (predicted test images with bounding boxes)

These outputs are intentionally not pushed to GitHub to keep the repo lightweight.

---

