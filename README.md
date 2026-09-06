# Unsupervised Anomaly Detection in Industrial Inspection

CV mini project: detects defective products using **pretrained deep
feature comparison** (a simplified PatchCore-style approach) — no
labeled defect data required (unsupervised).

## How it works (v3 — feature-based)

1. Upload normal/defect-free training images (bulk upload, folder upload supported).
2. Click "Train Model" — this runs every training image through a
   **pretrained ResNet-18** (trained on millions of real photos) up to
   an early-middle layer, extracting a grid of local "patch feature
   vectors" per image. These vectors are collected into a **memory
   bank** representing "what normal looks like."
3. Upload test images (mix of good + defective).
4. For each test image, every patch's feature vector is compared to
   its nearest match in the memory bank. Patches that look like
   nothing in the bank (a crack, scratch, wrong shape) get a large
   distance — this becomes the anomaly heatmap, blended onto the real
   image.
5. Adjust the **Sensitivity slider** in Step 4 to tune strictness
   without retraining — different materials have different natural
   visual noise, so this lets you calibrate per-category live.

### Why this replaced the earlier autoencoder version

An autoencoder that tries to *redraw* the image and measures pixel
error gets confused by benign surface variation — e.g. a printed
number rotated differently on each capsule, or natural leather grain
— and can let real defects slip through as "normal." Comparing
pretrained deep features instead is far more robust to that kind of
noise, because the features already encode meaningful shape/texture
information rather than raw pixel values.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

The first run downloads pretrained ResNet-18 weights (~45MB) — needs
internet once, then it's cached and never downloaded again.

## Project structure

```
anomaly_project/
├── app.py              # Flask routes (upload, train, detect)
├── model.py             # Feature-bank building + inference logic
├── requirements.txt
├── templates/
│   └── index.html       # Upload UI + results gallery + sensitivity slider
├── static/
│   ├── style.css
│   ├── script.js         # chunked bulk upload, drag & drop, training/detection calls
│   ├── uploads/train/    # your normal images land here
│   ├── uploads/test/     # test images land here
│   └── results/          # generated heatmap overlays
```

## Notes for the project report

- **Dataset**: works well with MVTec-AD-style industrial datasets
  (bottles, cables, screws, tiles, capsules, leather, metal_nut,
  etc.) — one product category at a time.
- **Method**: simplified PatchCore — pretrained ResNet-18 (ImageNet
  weights) truncated after layer2, patch-level nearest-neighbor
  distance to a memory bank of normal patches, top-5%-worst-patch
  aggregation for the image-level anomaly score.
- **Unsupervised aspect**: the memory bank is built from zero defect
  labels; anomaly detection emerges purely from feature-distance to
  what's "normal," at test time on unseen images.
- **Adjustable sensitivity**: the anomaly threshold is a percentile of
  the training set's own score distribution, chosen live via a
  slider rather than fixed at training time — this is what lets one
  pipeline generalize across very different materials (smooth
  capsules vs. richly textured leather) without separate tuning code
  per category.
- **Scaling to 1000+ images**: the frontend uploads in batches of 40
  files per request, and training streams images through a PyTorch
  DataLoader rather than holding them all in memory.

