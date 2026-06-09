"""
EuroSAT Land Classification — Flask Backend (EfficientNet-B3 Optical Only)
===========================================================================
Architecture matches notebook exactly:
  timm EfficientNet-B3 backbone (1536-d features)
  Head: BatchNorm1d(1536) → Linear(1536→512) → SiLU → Dropout(0.3) → Linear(512→10)
  Input: 224×224 RGB (Bands B4, B3, B2)
  Results: 98.48% accuracy · 98.44% macro F1 · 2,430 test samples

Usage:
  1. pip install -r requirements.txt
  2. python app.py
  3. Open http://localhost:5000
"""

import os, io, logging, traceback
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 10
CLASS_NAMES = [
    'AnnualCrop','Forest','HerbaceousVegetation','Highway',
    'Industrial','Pasture','PermanentCrop','Residential','River','SeaLake'
]
CLASS_DESCRIPTIONS = {
    'AnnualCrop':           'Annual agricultural fields',
    'Forest':               'Dense tree cover',
    'HerbaceousVegetation': 'Grasslands and shrubs',
    'Highway':              'Roads and highways',
    'Industrial':           'Factories and warehouses',
    'Pasture':              'Open grazing land',
    'PermanentCrop':        'Orchards and vineyards',
    'Residential':          'Urban housing areas',
    'River':                'Water bodies - rivers',
    'SeaLake':              'Sea and lake water bodies',
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'efficientnet_b3_optical_best.pth')
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Inference transform — 224x224 as in the notebook
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ── Model: exactly matches notebook build_efficientnet() ─────────────────────
class EfficientNetClassifier(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def build_efficientnet(num_classes=10):
    if not TIMM_AVAILABLE:
        raise ImportError("timm is required. Run: pip install timm")
    backbone = timm.create_model(
        'efficientnet_b3',
        pretrained=False,
        num_classes=0,
        global_pool='avg'
    )
    feature_dim = backbone.num_features  # 1536
    classifier = nn.Sequential(
        nn.BatchNorm1d(feature_dim),
        nn.Linear(feature_dim, 512),
        nn.SiLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return EfficientNetClassifier(backbone, classifier)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Weights not found at '{MODEL_PATH}'. "
            "Place efficientnet_b3_optical_best.pth in models/."
        )
    logger.info(f"Loading EfficientNet-B3 from {MODEL_PATH} ...")
    model = build_efficientnet(num_classes=NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Loaded — {params:,} parameters — device: {DEVICE}")
    return model


try:
    optical_model = load_model()
    MODEL_READY   = True
except Exception as e:
    logger.error(f"  Could not load model: {e}")
    optical_model = None
    MODEL_READY   = False


def run_inference(pil_img):
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    tensor = val_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = optical_model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    idx  = int(np.argmax(probs))
    return {
        'prediction':    CLASS_NAMES[idx],
        'confidence':    round(float(probs[idx]), 4),
        'description':   CLASS_DESCRIPTIONS[CLASS_NAMES[idx]],
        'probabilities': {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)},
        'model':         'EfficientNet-B3 Optical',
        'input':         'RGB - Bands B4, B3, B2',
        'device':        str(DEVICE),
    }


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    return jsonify({
        'status':      'ok' if MODEL_READY else 'model_not_loaded',
        'model_ready': MODEL_READY,
        'model':       'EfficientNet-B3 Optical',
        'device':      str(DEVICE),
        'classes':     CLASS_NAMES,
    })

@app.route('/api/classify', methods=['POST'])
def classify():
    if not MODEL_READY:
        return jsonify({'error': 'Model not loaded. Check models/efficientnet_b3_optical_best.pth.'}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'No image. Send as form field "image".'}), 400
    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'Empty filename.'}), 400
    try:
        pil_img = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({'error': f'Cannot open image: {e}'}), 400
    try:
        result = run_inference(pil_img)
        logger.info(f"-> {result['prediction']} ({result['confidence']*100:.1f}%)")
        return jsonify(result)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Inference failed: {e}'}), 500

@app.route('/api/classes')
def api_classes():
    return jsonify({'classes': CLASS_NAMES, 'descriptions': CLASS_DESCRIPTIONS})

if __name__ == '__main__':
    logger.info("EuroSAT EfficientNet-B3 Optical -> http://localhost:5000")
    logger.info(f"   Device: {DEVICE}  |  Model ready: {MODEL_READY}")
    app.run(host='0.0.0.0', port=5000, debug=True)
