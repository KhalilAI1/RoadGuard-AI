# 🛣️ RoadGuard AI — Road Damage Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![License](https://img.shields.io/badge/License-MIT-green)

An AI-powered road damage detection system using YOLOv8 and computer vision to automatically identify and classify road surface defects from images.

---

## 📌 Overview

Road infrastructure deterioration is a critical safety and economic issue worldwide. **RoadGuard AI** addresses this by automating the detection and classification of road damage using deep learning — eliminating the need for costly manual inspections.

The system detects **5 types of road damage**:

| Class | Type | Description |
|-------|------|-------------|
| D00 | Longitudinal Crack | Cracks running parallel to the road direction |
| D10 | Transverse Crack | Cracks running perpendicular to the road direction |
| D20 | Alligator Crack | Interconnected cracks forming a mesh pattern |
| D40 | Pothole | Bowl-shaped holes in the road surface |
| D50 | Surface Deterioration | Asphalt layer wear and surface degradation |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.641 |
| mAP@0.5:0.95 | 0.348 |
| Precision | 0.682 |
| Recall | 0.593 |

**Per-class performance:**

| Class | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| D00 | 0.670 | 0.549 | 0.596 |
| D10 | 0.661 | 0.560 | 0.610 |
| D20 | 0.703 | 0.624 | 0.684 |
| D40 | 0.717 | 0.759 | 0.798 |
| D50 | 0.660 | 0.474 | 0.519 |

---

## 📂 Repository Structure

```
RoadGuard-AI/
├── data/
│   ├── raw/                  # RDD2020 dataset (YOLO format)
│   │   ├── train/
│   │   ├── valid/
│   │   ├── test/
│   │   └── data.yaml
│   └── augmented/            # Augmented images for D00 & D50
├── notebooks/
│   └── 01_EDA.ipynb          # Exploratory Data Analysis
├── src/
│   ├── train.py              # Training script
│   ├── predict.py            # Inference script
│   └── augment.py            # Data augmentation pipeline
├── app/
│   └── demo.py               # Gradio demo application
├── configs/
│   └── config.yaml           # Training configuration
├── results/
│   └── visualizations/       # Output images and charts
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10
- CUDA-compatible GPU (recommended: 4GB+ VRAM)
- Anaconda or Miniconda

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/RoadGuard-AI.git
cd RoadGuard-AI

# Create and activate environment
conda create -n roadguard python=3.10 -y
conda activate roadguard

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Download Dataset

Download the RDD2020 dataset in YOLOv8 format from [Roboflow Universe](https://universe.roboflow.com) and place it in `data/raw/`.

---

## 💻 Usage

### Data Augmentation

```bash
python src/augment.py
```

Applies geometric and photometric augmentations on D00 and D50 classes to address class imbalance.

### Training

```bash
python src/train.py
```

Fine-tunes YOLOv8m on the road damage dataset. Configure hyperparameters in `configs/config.yaml`.

### Inference

```bash
python src/predict.py
```

### Gradio Demo

```bash
python app/demo.py
```

Open your browser at `http://127.0.0.1:7860` and upload a road image to detect damage.

---

## 🛠️ Built With

- [YOLOv8](https://docs.ultralytics.com/) — Object detection model
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [Albumentations](https://albumentations.ai/) — Data augmentation
- [OpenCV](https://opencv.org/) — Image processing
- [Gradio](https://gradio.app/) — Demo interface

---

## 🔬 Methodology

1. **EDA** — Analyzed class distribution across 26,660 images from 3 countries
2. **Baseline Training** — Fine-tuned YOLOv8m for 50 epochs on RDD2020
3. **Error Analysis** — Identified weak performance on D00 and D50 classes
4. **Augmentation** — Applied targeted augmentation (flip, rotate, crop, brightness, rain) on underperforming classes
5. **Evaluation** — Assessed model using mAP@0.5, Precision, Recall, and Confusion Matrix

---

## ⚠️ Challenges & Limitations

- **Class Imbalance:** D00 (18K samples) vs D50 (4.6K samples) — addressed with targeted augmentation
- **Domain Shift:** Model trained on daytime images; performance degrades on nighttime/rainy conditions
- **Background Confusion:** D00 and D50 occasionally confused with background

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Khalil** — AI Engineering Student & Deep Learning Trainee at DeepTorch
