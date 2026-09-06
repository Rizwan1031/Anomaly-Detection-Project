# Unsupervised Anomaly Detection in Industrial Inspection

A computer vision mini project that detects defective products in
industrial inspection images — without ever training on a single
labeled defect. The model learns what "normal" looks like from
defect-free images alone, using pretrained deep visual features
(inspired by PatchCore), and flags anything that doesn't match at
inference time. Includes a Flask web interface for bulk image upload,
model training, and interactive defect visualization with adjustable
detection sensitivity.

**Tech stack:** Python, Flask, PyTorch, torchvision (ResNet-18), PIL
**Dataset compatibility:** MVTec AD (bottle, capsule, screw, leather, metal_nut, etc.)
