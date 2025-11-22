# Hybrid Spatio-Temporal YOLO  
Robust Night-Time Road Object Detection with CLAHE, Gamma Correction and ConvLSTM

This repository implements a **Hybrid Spatio-Temporal YOLO** model that combines:

- Low-light image enhancement (CLAHE + Gamma)
- Multi-frame temporal modeling with ConvLSTM
- Multi-branch feature fusion for rich spatial context
- A YOLO-style detection head for real-time performance

It is designed for **night-time road scenes** (vehicles, pedestrians, road objects) under:
- very low illumination,
- strong headlight glare,
- varying exposure and noise.

---

## 🔍 High-Level Idea

Instead of treating each frame independently, this model:

1. Takes a **short video sequence** (e.g., 8 frames) from a dashcam / CCTV.
2. Enhances each frame using **CLAHE** and **gamma correction** to recover structure from dark regions.
3. Extracts multi-scale features via a YOLO-like backbone.
4. Passes the temporal feature sequence through **ConvLSTM** layers to learn motion and temporal consistency.
5. Uses **multi-branch convolutional fusion** (different kernel sizes) to capture local and global context.
6. Runs a YOLO-style detection head to predict bounding boxes and classes for the **final frame** of the sequence.

This combination yields smoother, more stable, and more accurate detections in difficult night-time conditions than:

- single-frame YOLO models, or  
- temporal models without enhancement.

---

## 📐 Architecture Overview

Conceptual pipeline:

1. Input: sequence of T frames, e.g. T = 8, each of size 640×640.
2. Per-frame enhancement: CLAHE + gamma.
3. Per-frame feature extraction: YOLO-style backbone → {P3, P4, P5 feature maps}.
4. Temporal modeling: ConvLSTM applied at each scale across time.
5. Multi-branch fusion: parallel convolutions (1×1, 3×3, 5×5, 7×7) + concatenation + residual.
6. Detection head: YOLO-style head predicts boxes, objectness, and class probabilities.

---
<img width="1453" height="711" alt="image" src="https://github.com/user-attachments/assets/46a9ee00-98d5-403c-9c70-b8038034ce6c" />

## 📂 Repository Structure (Suggested)

```text
hybrid-yolo/
├─ data/
│  ├─ raw/                    # original videos / frames
│  ├─ processed/              # pre-split sequences, enhanced frames (optional cache)
│  ├─ annotations/            # COCO-style JSON annotations
│  └─ configs/
│     └─ dataset.yaml         # paths, class names, etc.
│
├─ src/
│  ├─ preprocess.py           # CLAHE + gamma enhancement helpers
│  ├─ dataset.py              # sequence dataset + dataloader
│  ├─ model/
│  │  ├─ backbone_yolo.py     # YOLO-style backbone
│  │  ├─ convlstm.py          # ConvLSTM modules
│  │  ├─ fusion_blocks.py     # multi-branch fusion blocks
│  │  └─ hybrid_yolo.py       # full Hybrid Spatio-Temporal model
│  ├─ loss.py                 # detection loss + temporal loss
│  ├─ train.py                # training loop
│  ├─ eval.py                 # evaluation scripts (mAP, temporal stability)
│  └─ infer.py                # single video / folder inference
│
├─ experiments/
│  ├─ default_config.yaml     # hyperparameters, paths, training settings
│  └─ ablations/              # optional configs for experiments
│
├─ scripts/
│  ├─ split_video_to_frames.py
│  ├─ make_sequences.py       # build T-frame sequences
│  └─ visualize_results.py
│
├─ requirements.txt
└─ README.md
