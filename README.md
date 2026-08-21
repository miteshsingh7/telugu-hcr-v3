# ✍️ Telugu Handwritten Character Recognizer (v3)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://tensorflow.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3.0+-red.svg)](https://keras.io/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Deep Learning system for **Telugu Handwritten Character Recognition (HCR)** covering **630 fine-grained Akshara classes** (Achulu, Hallulu, Guninthamulu, and Othulu) trained on **292,752 images**.

Built with a modular phased architecture featuring **ResNet-style Custom CNN**, **EfficientNetB0 Transfer Learning**, **Warmup Cosine Learning Rate Schedules**, **Canonical Preprocessing**, and an interactive **Streamlit Web Application**.

---

## 🎯 Verified Benchmark Performance

Evaluated on the held-out test set (10% stratified split across all 630 categories with canonical preprocessing):

| Model / Checkpoint | Input Resolution | Top-1 Accuracy (Exact Match) | Top-3 Accuracy | Top-5 Accuracy | Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline Checkpoint (`telugu_v3_best.keras`)** | `96×96 (1-channel)` | **`74.20%`** | **`93.60%`** | **`96.80%`** | Verified (Canonical Eval) |
| **Fine-Tuned Checkpoint (v3 on Kaggle GPU)** | `96×96 (1-channel)` | **`85.64%`** | **`97.11%`** | **`98.64%`** | Verified (Kaggle P100) |

- **Top-1 Exact Match:** **74.20% / 85.64%** on the full 630-class space (over **460× to 535× better than random chance** of $1/630 \approx 0.16\%$).
- **Top-3 Accuracy:** **93.60% / 97.11%** (The true character glyph is within the top 3 suggestions in >93% of test samples).
- **Top-5 Accuracy:** **96.80% / 98.64%** (Near-perfect candidate retrieval across all 630 classes).

---

## 🏛️ System Architecture

```
                       ┌────────────────────────────────────────────────────────┐
                       │  Phase 0: Fast Audit & Stratified 80/10/10 Split       │
                       │  292,752 Images across 630 Classes (Indexed in < 3s)   │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │
                                                  ▼
                        ┌──────────────────────────────────────────────────────┐
                        │      Phase 1: Dual-Track Competitive Baselines       │
                        ├──────────────────────────┬───────────────────────────┤
                        │ Track A: EfficientNetB0  │ Track B: Custom ResNet    │
                        │ (Transfer Learning, 128) │ (6-Block Handwriting CNN) │
                        └─────────────┬────────────┴─────────────┬─────────────┘
                                      │                          │
                                      ▼                          ▼
                       ┌────────────────────────────────────────────────────────┐
                       │  Phase 2: Confused Pairs & Fine-Tuning Optimization    │
                       │  Cosine Decay + Label Smoothing + Regularization       │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │
                                                  ▼
                       ┌────────────────────────────────────────────────────────┐
                       │  Phase 3: Soft-Voting Ensemble Engine                  │
                       │  Multi-Model Probability Combination                   │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │
                                                  ▼
                       ┌────────────────────────────────────────────────────────┐
                       │  Phase 4: Interactive Streamlit Web Application        │
                       │  Draw Character -> Live Top-3 Glyphs + Confidence Bars │
                       └────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
telugu-hcr-v3/
├── app.py                      # Interactive Streamlit Web Application
├── demo/
│   └── app.py                  # Streamlit application copy
├── configs/
│   ├── base.yaml               # Shared hyperparameters & data settings
│   ├── track_a_efficientnet.yaml # EfficientNetB0 config (128x128, 3-channel)
│   ├── track_b_custom_cnn.yaml # Custom ResNet CNN config (96x96, 1-channel)
│   └── ensemble.yaml           # Soft-voting config
├── scripts/
│   └── train_kaggle.py         # High-throughput Kaggle P100 training script
├── src/
│   ├── checkpointing.py        # Model checkpoint manager
│   ├── train.py                # Unified CLI training script
│   ├── evaluate.py             # Top-1/3/5 metrics & confusion matrix extractor
│   ├── infer_tta.py            # Test-Time Augmentation inference engine
│   ├── data/
│   │   ├── preprocess.py       # Dual-mode pure-TF and NumPy canonical preprocessor
│   │   ├── audit.py            # Fast dataset integrity scanner
│   │   ├── split.py            # Stratified 80/10/10 data partitioner
│   │   ├── dataset.py          # High-performance tf.data pipeline
│   │   ├── augmentation.py     # Character-safe affine transforms (bounds-calibrated)
│   │   └── telugu_unicode.py   # 630-Class Unicode glyph mapper & root aggregator
│   └── models/
│       ├── backbone.py         # EfficientNetB0 with WarmupCosineDecay
│       ├── custom_cnn.py       # Custom CNN architecture with parametrized blocks
│       ├── hierarchical.py     # Coarse-to-fine clustering
│       └── ensemble.py         # Soft-voting ensemble module
├── requirements.txt            # Python dependencies
└── README.md                   # Complete documentation
```

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/miteshsingh/telugu-hcr-v3.git
cd telugu-hcr-v3
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

### 3. Run Data Audit & Stratified Split
```bash
python -m src.data.audit --data_dir data/ --output reports/
python -m src.data.split --data_dir data/ --output outputs/
```

### 4. Train Models

#### Local Dry-Run / Smoke Test:
```bash
python src/train.py --config configs/track_b_custom_cnn.yaml --dry-run
```

#### High-Throughput Kaggle P100 Training:
```bash
python scripts/train_kaggle.py --data-dir /kaggle/input/telugu-handwritten-character-dataset --config configs/track_b_custom_cnn.yaml
```

### 5. Evaluate Checkpoint
```bash
python src/evaluate.py --model checkpoints/telugu_v3_best.keras --data outputs/test.csv --label-map outputs/label_map.json --config configs/track_b_custom_cnn.yaml --output reports/
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
