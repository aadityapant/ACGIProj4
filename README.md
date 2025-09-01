# MeshCNN 🧠 — Edge-based Convolutions on Triangle Meshes

Geometry-aware deep learning for 3D **<classification | segmentation>** directly on **triangle meshes**.  
Implements **edge-based convolutions** with **edge-collapse pooling** (and ablations) plus a clean, config-driven training pipeline.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=fff)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=fff)](#)
[![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=fff)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

> Built around MeshCNN (Hanocka et al., 2019). Extended with **custom features** and **pooling ablations**, and evaluated in **CSE 570 — Advanced Computer Graphics (Project 4)**.  
> Colab (results): <https://colab.research.google.com/drive/1EwDCo-P8YSiKRTdh7XYmhvGkbdAceQv8?usp=sharing>  
> Source: <https://github.com/aadityapant/ACGIProj4>

---

## ✨ What’s in this repo

- **EdgeConv** on mesh **edges** (not voxels or point clouds)  
- **Coordinate-based feature extractor** (centroid/midpoint features for main edge + 4 neighbors)  
- **Pooling ablation**: variant **without pooling/unpooling** using global average pooling at the end  
- **Reproducible experiments** with YAML configs, deterministic seeds, and TensorBoard logs  
- **Preprocessing utilities** for normalization, cleaning, and cached edge graphs

---

## 🔬 Key results (from course project)

- **Original implementation**: **100% accuracy**, **1,320,558 params**, **~1.7 GB GPU**  
- **Centroid/coordinate feature variant**: **87.5% accuracy** (peaks at 100% early in training) with same params/memory  
- **No pooling/unpooling** (global average pool head): **100% accuracy**  

> See the report for details and code snippets of the modifications. (Project: CSE 570, Advanced Computer Graphics, 30 Nov 2023)

---

## 🧱 Modifications (what we changed)

### 1) Coordinate Feature Extraction (`mesh_prepare.py`)
We added a new extractor **`coordinate_feature(mesh)`** that computes the midpoint of each edge and of its up to 4 GEMM neighbors, assembling a **5-channel** feature tensor indexed by edge. This replaces the original dihedral/angle/ratio set in the experiment branch.

**High level:**
- Channel 0: mean of the edge midpoint’s XYZ  
- Channels 1–4: mean(midpoint) of each neighboring edge (or 0 if absent)

Hooked in via a modified **`extract_features(mesh)`** to return `coordinate_feature(...)`.

### 2) Remove Pooling/Unpooling (`networks.py`)
We ablated mesh **pool/unpool** layers and used **global average pooling** after the final conv:
- Commented out `MeshPool` calls  
- Added `AvgPool1d` over the last temporal dimension, then `fc1 → fc2` head

This isolates the contribution of pooling to quality/compute and yielded **100% accuracy** in our setup.

---

## 🗂️ Project structure

experiment: meshcnn-<dataset>-<task>
device: cuda
seed: 42
epochs: 200
batch_size: 16
lr: 0.001
optimizer: adam
amp: true

model:
  name: edgecnn
  width: 64
  depth: 4
  dropout: 0.2
  use_unpool: false  # set true for seg models with unpool

data:
  root: data/processed/<DATASET_NAME>
  task: <classification|segmentation>
  num_classes: <N>
  augment:
    rotate: true
    jitter: true
    scale: true
    edge_flip: false

    | Variant                  | Metric   | Score | Notes                          |
| ------------------------ | -------- | ----- | ------------------------------ |
| Original implementation  | Accuracy | 100%  | 1,320,558 params; \~1.7 GB GPU |
| Centroid features (ours) | Accuracy | 87.5% | Peaks at 100% early            |
| No pool/unpool (ours)    | Accuracy | 100%  | Global avg pooling head        |

🧠 What I learned

Building a mesh processing pipeline (normalize, edge graphs, GEMM neighbors)

Designing topology-aware features (centroid/midpoint statistics)

Understanding the effect of pooling/unpooling vs global pooling

Training engineering: mixed precision, logging, and ablation studies

Reproducibility and experiment hygiene with configs + seeds

📈 Ideas to push further

Best-model checkpointing (min loss / max acc)

Early stopping + extend max epochs (training currently capped)

Richer features: combine original (angles/ratios) + centroid set (→ 10 channels)

Hyperparameter search and k-fold CV

Interpretability: epoch-wise loss/accuracy plots; per-class IoU

👥 Authors

Aditya Pant — Code, Report

Faizan Khan — Code, Report

🧾 Citation

H. Hanocka, A. Hertz, R. Fish, R. Giryes, S. Funkhouser, D. Cohen-Or.
“MeshCNN: A Network with an Edge.” TOG 38(4), 2019.
https://ranahanocka.github.io/MeshCNN/

If you use this repo, please also cite the MeshCNN paper.
