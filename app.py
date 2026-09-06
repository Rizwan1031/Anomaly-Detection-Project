"""
Unsupervised Anomaly Detection in Industrial Inspection
---------------------------------------------------------
Flask backend.

Workflow:
1. Upload a large batch of NORMAL/GOOD product images -> /upload_train
2. Click "Train Model" -> extracts pretrained deep features from every
   normal image and stores them as a "memory bank" (this is what makes
   it "unsupervised" - no defect labels are needed)
3. Upload TEST images (mix of good/defective) -> /upload_test
4. Backend compares each test image's local patch features to their
   nearest match in the memory bank. Patches unlike anything "normal"
   get a high distance score -> anomaly. A heatmap + anomaly score is
   returned, with sensitivity adjustable at detection time.
"""

import os
import io
import json
import base64
import shutil
from datetime import datetime

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

from model import (
    train_autoencoder,
    run_inference,
    MODEL_PATH,
    THRESHOLD_PATH,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "static", "uploads", "train")
TEST_DIR = os.path.join(BASE_DIR, "static", "uploads", "test")
RESULTS_DIR = os.path.join(BASE_DIR, "static", "results")

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB, for 1000+ images

for d in [TRAIN_DIR, TEST_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    train_count = len([f for f in os.listdir(TRAIN_DIR) if allowed_file(f)])
    test_count = len([f for f in os.listdir(TEST_DIR) if allowed_file(f)])
    model_ready = os.path.exists(MODEL_PATH)
    return render_template(
        "index.html",
        train_count=train_count,
        test_count=test_count,
        model_ready=model_ready,
    )


@app.route("/upload_train", methods=["POST"])
def upload_train():
    """Bulk upload of normal/good images used for unsupervised training.
    Accepts hundreds/thousands of files in one multipart request
    (frontend batches them to avoid browser/network limits)."""
    files = request.files.getlist("images")
    saved = 0
    for f in files:
        if f and allowed_file(f.filename):
            fname = secure_filename(f.filename)
            # avoid collisions when many files share names across folders
            fname = f"{datetime.now().strftime('%H%M%S%f')}_{fname}"
            f.save(os.path.join(TRAIN_DIR, fname))
            saved += 1
    total = len([f for f in os.listdir(TRAIN_DIR) if allowed_file(f)])
    return jsonify({"saved": saved, "total_train_images": total})


@app.route("/upload_test", methods=["POST"])
def upload_test():
    files = request.files.getlist("images")
    saved = 0
    for f in files:
        if f and allowed_file(f.filename):
            fname = secure_filename(f.filename)
            fname = f"{datetime.now().strftime('%H%M%S%f')}_{fname}"
            f.save(os.path.join(TEST_DIR, fname))
            saved += 1
    total = len([f for f in os.listdir(TEST_DIR) if allowed_file(f)])
    return jsonify({"saved": saved, "total_test_images": total})


@app.route("/train", methods=["POST"])
def train():
    epochs = int(request.form.get("epochs", 20))
    img_files = [
        os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR) if allowed_file(f)
    ]
    if len(img_files) < 10:
        return jsonify({"error": "Upload at least 10 good/normal images before training."}), 400

    try:
        history = train_autoencoder(img_files, epochs=epochs)
    except Exception as e:
        return jsonify({"error": f"Training crashed: {type(e).__name__}: {e}"}), 500

    return jsonify({"status": "trained", "images_used": len(img_files), "history": history})


@app.route("/detect", methods=["POST"])
def detect():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet. Train it first."}), 400

    img_files = [
        os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if allowed_file(f)
    ]
    if not img_files:
        return jsonify({"error": "No test images uploaded."}), 400

    sensitivity = float(request.form.get("sensitivity", 90))
    try:
        results = run_inference(img_files, RESULTS_DIR, sensitivity=sensitivity)
    except Exception as e:
        return jsonify({"error": f"Detection crashed: {type(e).__name__}: {e}"}), 500

    return jsonify({"results": results})


@app.route("/results/<path:filename>")
def get_result(filename):
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/reset", methods=["POST"])
def reset():
    """Clear uploaded images + trained model to start a fresh run."""
    for d in [TRAIN_DIR, TEST_DIR, RESULTS_DIR]:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    for p in [MODEL_PATH, THRESHOLD_PATH]:
        if os.path.exists(p):
            os.remove(p)
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
