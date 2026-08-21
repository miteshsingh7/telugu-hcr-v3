# ✍️ Telugu Handwritten Character Recognizer (v3)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://tensorflow.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3.0+-red.svg)](https://keras.io/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Deep Learning system for **Telugu Handwritten Character Recognition (HCR)** covering **630 fine-grained Akshara classes** (Achulu, Hallulu, Guninthamulu, and Othulu) trained on **292,752 images**.

Built with a modular phased architecture featuring **ResNet-style Custom CNN**, **EfficientNetB0 Transfer Learning**, **Warmup Cosine Learning Rate Schedules**, **Character-Safe Augmentations**, **Test-Time Augmentation (TTA)**, and an interactive **Streamlit Web Application**.

---

## 🎯 Verified Benchmark Performance

Evaluated on the held-out test set (10% stratified split across all 630 categories):

| Model / Configuration | Top-1 Accuracy (Exact Match) | Top-3 Accuracy | Top-5 Accuracy |
|---|:---:|:---:|:---:|
| **Baseline Model** | `72.85%` | `93.11%` | `96.64%` |
| **Fine-Tuned Model (v3)** | **`85.64%`** | **`97.11%`** | **`98.64%`** |
| **System Benchmark (+ TTA)** | **`78.39%`** | **`93.54%`** | **`96.45%`** |

- **Top-1 Exact Match:** Reached **85.64%** on the full 630-class space (over **535× better than random guess** of $1/630 \approx 0.16\%$).
- **Top-3 Accuracy:** **97.11%** (In 97 out of 100 images, the true glyph is within the top 3 suggestions).
- **Top-5 Accuracy:** **98.64%** (Near-perfect candidate retrieval across 630 classes).

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
                        │ (Transfer Learning)      │ (80-Layer Handwriting CNN)│
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
                       │  Phase 3: Soft-Voting Ensemble + Test-Time Aug (TTA)   │
                       │  Multi-Pass Jittered Inference -> Final Verification   │
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
│   └── app.py                  # Demo application copy
├── configs/
│   ├── base.yaml               # Shared hyperparameters & data settings
│   ├── track_a_efficientnet.yaml # EfficientNetB0 config
│   ├── track_b_custom_cnn.yaml # Custom ResNet CNN config
│   └── ensemble.yaml           # Soft-voting & TTA config
├── notebooks/
│   └── telugu_hcr_kaggle.ipynb # Standalone self-contained Kaggle notebook
├── src/
│   ├── checkpointing.py        # Model checkpoint manager
│   ├── train.py                # Unified CLI training script
│   ├── evaluate.py             # Top-1/3/5 metrics & confusion matrix extractor
│   ├── infer_tta.py            # Test-Time Augmentation inference engine
│   ├── data/
│   │   ├── audit.py            # Fast dataset integrity scanner
│   │   ├── split.py            # Stratified 80/10/10 data partitioner
│   │   ├── dataset.py          # High-performance tf.data pipeline
│   │   ├── augmentation.py     # Character-safe affine transforms
│   │   └── telugu_unicode.py   # 630-Class Unicode glyph mapper
│   └── models/
│       ├── backbone.py         # EfficientNetB0 with WarmupCosineDecay
│       ├── custom_cnn.py       # 80-layer Custom ResNet architecture
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
python -m src.data.audit --data_dir /path/to/Test1 --output reports/
python -m src.data.split --data_dir /path/to/Test1 --output outputs/
```

### 4. Train Models (CLI)
```bash
# Train Track A (EfficientNetB0)
python -m src.train --config configs/track_a_efficientnet.yaml

# Train Track B (Custom ResNet CNN)
python -m src.train --config configs/track_b_custom_cnn.yaml
```

### 5. Evaluate
```bash
python -m src.evaluate --model checkpoints/telugu_v3_best.keras --data outputs/test.csv --label-map outputs/label_map.json --config configs/base.yaml
```

---

## 🌐 Deployment to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. Select this repository and set `Main file path` to `app.py`.
4. Click **Deploy**.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
