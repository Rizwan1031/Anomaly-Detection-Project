"""
Unsupervised Anomaly Detection - v3 (feature-based, PatchCore-style)
----------------------------------------------------------------------
Why this replaces the autoencoder approach:

The v1/v2 autoencoder tried to REDRAW each image and measured pixel
reconstruction error. That approach struggles when normal images
themselves vary a lot in superficial ways (a printed number rotated
differently on each capsule, natural leather grain, lighting) - the
model ends up reacting to that surface noise instead of real defects,
and cracks/scratches can slip through as "normal."

This version instead:
1. Runs every image through a PRETRAINED ResNet-18 (trained on millions
   of real photos) up to an early-middle layer. This gives a grid of
   "patch feature vectors" - each vector describes the local visual
   texture/shape at one small region of the image, in a much richer,
   more meaningful way than raw pixels.
2. Builds a "memory bank" of these patch vectors from ONLY your normal
   training images - this is the "library of what normal looks like."
3. For a test image, each patch's feature vector is compared to its
   NEAREST match in that memory bank. A normal patch finds a close
   match (small distance). A genuinely defective patch (crack,
   scratch, wrong shape) looks different from anything in the memory
   bank, so its nearest match is far away (large distance).
4. The distance map becomes the anomaly heatmap, upsampled and blended
   onto the real image.

This is a simplified version of PatchCore, a well-known state-of-the-art
method for exactly this kind of industrial defect detection, and is
considerably more robust to benign visual variation than a plain
autoencoder.

Note: the first time this runs, PyTorch downloads pretrained ResNet-18
weights (~45MB) - an internet connection is needed once, then it's
cached locally and never downloaded again.
"""

import os
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tvmodels

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "feature_bank.pt")   # kept as MODEL_PATH name for app.py compatibility
THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold.json")

IMG_SIZE = 224  # standard ResNet input size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATCHES_KEPT_PER_IMAGE = 120   # random patch subsample per training image
BANK_MAX_SIZE = 25000          # cap total memory bank size for speed
TOP_K_FRACTION = 0.05          # image score = mean distance of worst 5% of patches

_display_transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE))])
_feature_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_extractor = None


def _get_extractor():
    """A ResNet-18 truncated after layer2 - deep enough to capture
    meaningful shape/texture, shallow enough to run fast on a CPU."""
    global _extractor
    if _extractor is None:
        resnet = tvmodels.resnet18(weights=tvmodels.ResNet18_Weights.IMAGENET1K_V1)
        truncated = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2,
        )
        truncated.eval()
        for p in truncated.parameters():
            p.requires_grad = False
        _extractor = truncated.to(DEVICE)
    return _extractor


class ImageFolderDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return _feature_transform(img)


def _extract_patches(batch):
    """batch: (B, 3, 224, 224) normalized tensor.
    Returns patch feature vectors (B, N, C) and the spatial grid H, W
    (e.g. 28x28 patches, 128-dim each, for a 224x224 input)."""
    model = _get_extractor()
    with torch.no_grad():
        fmap = model(batch.to(DEVICE))  # (B, C, H, W)
    B, C, H, W = fmap.shape
    patches = fmap.permute(0, 2, 3, 1).reshape(B, H * W, C)
    return patches.cpu(), H, W


def _top_k_score(dist_flat, k_fraction=TOP_K_FRACTION):
    k = max(1, int(len(dist_flat) * k_fraction))
    worst = np.partition(dist_flat, -k)[-k:]
    return float(worst.mean())


def train_autoencoder(file_list, epochs=None, batch_size=8, lr=None):
    """Builds the normal-patch memory bank. `epochs`/`lr` are accepted
    for compatibility with the existing app.py call signature but are
    not used - this method doesn't do gradient training, it just
    collects reference features from your normal images."""
    dataset = ImageFolderDataset(file_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    bank_chunks = []
    grid_h, grid_w = None, None
    for batch in loader:
        patches, H, W = _extract_patches(batch)
        grid_h, grid_w = H, W
        flat = patches.reshape(-1, patches.shape[-1])  # (B*N, C)
        n_keep = min(flat.shape[0], PATCHES_KEEP_PER_IMAGE_TOTAL(batch.shape[0]))
        idx = torch.randperm(flat.shape[0])[:n_keep]
        bank_chunks.append(flat[idx])

    bank = torch.cat(bank_chunks, dim=0)
    if bank.shape[0] > BANK_MAX_SIZE:
        idx = torch.randperm(bank.shape[0])[:BANK_MAX_SIZE]
        bank = bank[idx]

    torch.save({"bank": bank, "grid_h": grid_h, "grid_w": grid_w}, MODEL_PATH)

    # Score the training set itself against the bank to calibrate the
    # sensitivity slider's percentile lookup later.
    scores = []
    for batch in DataLoader(dataset, batch_size=batch_size, num_workers=0):
        patches, H, W = _extract_patches(batch)
        for i in range(patches.shape[0]):
            d = torch.cdist(patches[i], bank)      # (N_patches, bank_size)
            min_d, _ = d.min(dim=1)
            scores.append(_top_k_score(min_d.numpy()))

    scores = np.array(scores)
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({
            "train_scores": scores.tolist(),
            "train_mean": float(scores.mean()),
            "train_std": float(scores.std()),
        }, f)

    # Fake a "history" entry so the existing frontend (which reads
    # history[-1].loss) still displays something meaningful.
    history = [{"epoch": 1, "loss": round(float(scores.mean()), 6)}]
    return history


def PATCHES_KEEP_PER_IMAGE_TOTAL(batch_size):
    return PATCHES_KEPT_PER_IMAGE * batch_size


def _jet_colormap(norm_err):
    r = np.clip(1.5 - np.abs(4 * norm_err - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * norm_err - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * norm_err - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def _make_overlay(orig_img_arr, err_map, alpha=0.55):
    norm = err_map - err_map.min()
    if norm.max() > 0:
        norm = norm / norm.max()
    norm = norm ** 0.5

    heat_rgb = (_jet_colormap(norm) * 255).astype(np.float32)
    base = orig_img_arr.astype(np.float32)
    mix = np.clip(norm * 1.3, 0, 1)[..., None] * alpha
    blended = base * (1 - mix) + heat_rgb * mix
    return Image.fromarray(blended.astype(np.uint8))


def run_inference(file_list, results_dir, sensitivity=90):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Feature bank not found - train the model first.")

    bank_data = torch.load(MODEL_PATH, map_location=DEVICE)
    bank = bank_data["bank"]
    grid_h, grid_w = bank_data["grid_h"], bank_data["grid_w"]

    with open(THRESHOLD_PATH) as f:
        thresh_data = json.load(f)
    threshold = float(np.percentile(thresh_data["train_scores"], sensitivity))

    results = []
    for path in file_list:
        fname = os.path.basename(path)
        img = Image.open(path).convert("RGB")
        x = _feature_transform(img).unsqueeze(0)
        patches, H, W = _extract_patches(x)
        p = patches[0]  # (N, C)

        d = torch.cdist(p, bank)
        min_d, _ = d.min(dim=1)
        flat = min_d.numpy()
        score = _top_k_score(flat)
        is_anomaly = score > threshold

        heat_small = flat.reshape(H, W).astype(np.float32)
        heat_img_small = Image.fromarray(heat_small)
        heat_upsampled = np.array(heat_img_small.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))

        orig_resized = np.array(_display_transform(img))
        overlay_img = _make_overlay(orig_resized, heat_upsampled)

        overlay_name = f"overlay_{fname}.png"
        overlay_img.resize((300, 300), Image.NEAREST).save(os.path.join(results_dir, overlay_name))

        orig_name = f"orig_{fname}.png"
        Image.fromarray(orig_resized).resize((300, 300)).save(os.path.join(results_dir, orig_name))

        results.append({
            "filename": fname,
            "score": round(score, 6),
            "threshold": round(threshold, 6),
            "is_anomaly": bool(is_anomaly),
            "heatmap_url": f"/results/{overlay_name}",
            "original_url": f"/results/{orig_name}",
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results
