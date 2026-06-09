# ai-project
land classification using remote sensing- use EfficientNet-B3 on Sentinel-2 optical (RGB) imagery
# EuroSAT Optical Land Classifier — EfficientNet-B3

**B.Tech 4th Semester | CSET301 AIML | 2025**

Full-stack web application for land use classification using EfficientNet-B3 trained on Sentinel-2 optical (RGB) imagery. Achieves **98.48% accuracy** and **98.44% macro F1** on 2,430 test samples.

## Project Structure

```
eurosat_effb3_v2/
├── app.py                                  ← Flask backend (PyTorch inference)
├── requirements.txt
├── README.md
├── templates/
│   └── index.html                          ← Frontend
└── models/
    └── efficientnet_b3_optical_best.pth    ← Trained weights (~45 MB)
```

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Model Details

| Property | Value |
|---|---|
| Architecture | EfficientNet-B3 (timm · ImageNet pretrained) |
| Input | RGB 224×224 — Sentinel-2 Bands B4, B3, B2 |
| Head | BatchNorm1d(1536) → Linear(1536→512) → SiLU → Dropout(0.3) → Linear(512→10) |
| Parameters | ~12.2 Million |
| Accuracy | **98.48%** (2,430 test samples) |
| F1-Score | **98.44%** macro |

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web frontend |
| GET | `/api/status` | Server health |
| POST | `/api/classify` | Classify image |
| GET | `/api/classes` | Class names |

```bash
curl -X POST http://localhost:5000/api/classify -F "image=@patch.png"
```

*EuroSAT · EfficientNet-B3 · B.Tech CSET301 AIML 2025*
