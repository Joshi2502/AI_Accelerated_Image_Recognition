# app.py
import os
import io
import json
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import torch
from PIL import Image

from model import build_resnet50
from inference import prepare_transform, predict_probs
from train import train_model, load_labels, save_labels

# ---------------------------
# Config
# ---------------------------
PORT = int(os.getenv("PORT", "8080"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
DEFAULT_VERSION = os.getenv("MODEL_VERSION", "v1")
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Speedups (cuDNN autotune on fixed-size inputs)
torch.backends.cudnn.benchmark = True

# ---------------------------
# Globals
# ---------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB upload limit

current_model = None
current_version = None
labels = None
transform = prepare_transform()  # ImageNet transforms (Resize/Crop/Normalize)

def model_paths(version: str):
    """Return filesystem paths for a model version."""
    vdir = MODEL_DIR / version
    return {
        "dir": vdir,
        "weights": vdir / "model.pt",
        "labels": vdir / "labels.json",
        "meta": vdir / "meta.json",
    }

def load_model_version(version: str) -> None:
    """Load a model version into global GPU memory."""
    global current_model, current_version, labels
    paths = model_paths(version)
    if not paths["weights"].exists():
        raise FileNotFoundError(f"Model weights not found for version '{version}': {paths['weights']}")

    # Load labels (class mapping)
    labels_map = load_labels(paths["labels"])
    num_classes = len(labels_map)

    # Build & load state
    model = build_resnet50(num_classes=num_classes, pretrained=False)
    state = torch.load(paths["weights"], map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()

    # Optional compile (PyTorch 2.x)
    try:
        model = torch.compile(model)  # type: ignore
    except Exception:
        pass

    # Warmup for stable latency (build kernels/caches)
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
        dummy = torch.randn(1, 3, 224, 224, device=DEVICE)
        for _ in range(8):
            _ = model(dummy)

    current_model = model
    current_version = version
    labels = labels_map
    app.logger.info(f"Loaded model version '{version}' on device={DEVICE}")

# ---------------------------
# Routes
# ---------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "version_loaded": current_version
    })

@app.route("/load_model", methods=["POST"])
def load_model_endpoint():
    """
    Body JSON:
    {
    "version": "v1"
    }
    """
    data = request.get_json(force=True)
    version = data.get("version")
    if not version:
        return jsonify({"error": "Missing 'version' in request body"}), 400
    try:
        load_model_version(version)
        return jsonify({"ok": True, "loaded_version": version})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        app.logger.exception("Failed to load model")
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """
    multipart/form-data:
    image: file
    top_k: optional int (default 1)
    """
    if current_model is None or labels is None:
        return jsonify({"error": "No model loaded. Call /load_model first or ensure default loads on startup."}), 400

    if "image" not in request.files:
        return jsonify({"error": "Missing image file (form field 'image')"}), 400

    file = request.files["image"]
    _ = secure_filename(file.filename)  # sanitized if you want to save; here we read into memory
    image_bytes = file.read()

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image or unsupported format"}), 400

    top_k = int(request.form.get("top_k", 1))
    if top_k < 1:
        top_k = 1

    probs = predict_probs(current_model, img, transform, DEVICE, amp=(DEVICE.type == "cuda"))
    topk = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)

    idx_to_label = {v: k for k, v in labels.items()}
    results = []
    for i in range(topk.indices.shape[-1]):
        cls_idx = int(topk.indices[0, i].item())
        results.append({
            "class_index": cls_idx,
            "class_label": idx_to_label.get(cls_idx, str(cls_idx)),
            "probability": float(topk.values[0, i].item())
        })

    return jsonify({
        "version": current_version,
        "results": results
    })

@app.route("/train", methods=["POST"])
def train():
    """
    Body JSON:
    {
    "data_dir": "/path/to/ImageFolder(root with train/val subdirs)",
    "version": "v2",
    "epochs": 3,
    "batch_size": 64,
    "lr": 0.0003,
    "freeze_backbone": true
    }
    """
    payload = request.get_json(force=True)
    data_dir = payload.get("data_dir")
    version = payload.get("version", "v_new")
    epochs = int(payload.get("epochs", 3))
    batch_size = int(payload.get("batch_size", 64))
    lr = float(payload.get("lr", 3e-4))
    freeze_backbone = bool(payload.get("freeze_backbone", True))

    if not data_dir or not Path(data_dir).exists():
        return jsonify({"error": "Invalid or missing 'data_dir' (expects ImageFolder with train/val)"}), 400

    paths = model_paths(version)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    try:
        metrics, class_to_idx = train_model(
            data_dir=data_dir,
            out_weights_path=str(paths["weights"]),
            device=DEVICE,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            freeze_backbone=freeze_backbone,
            num_workers=NUM_WORKERS
        )
        # Save labels mapping for inference
        save_labels(paths["labels"], class_to_idx)
        with open(paths["meta"], "w") as f:
            json.dump({
                "version": version,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "freeze_backbone": freeze_backbone,
                "metrics": metrics
            }, f, indent=2)

        # (Optional) auto-load the newly trained model
        load_model_version(version)

        return jsonify({
            "ok": True,
            "version": version,
            "metrics": metrics,
            "classes": list(class_to_idx.keys())
        })
    except Exception as e:
        app.logger.exception("Training failed")
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Boot
# ---------------------------
if __name__ == "__main__":
    app.logger.info(f"Starting server on device={DEVICE} (cuda_available={torch.cuda.is_available()})")
    # Try to load default version if present
    try:
        load_model_version(DEFAULT_VERSION)
    except Exception as e:
        app.logger.warning(f"Default model '{DEFAULT_VERSION}' not loaded: {e}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
