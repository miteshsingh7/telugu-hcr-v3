"""Script to generate a self-contained, standalone Kaggle notebook."""

import base64
import io
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent

# Create in-memory zip of configs and src
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for folder in ["configs", "src"]:
        for p in (root / folder).rglob("*"):
            if p.is_file() and not p.name.startswith("."):
                rel = p.relative_to(root)
                z.write(p, arcname=str(rel))

b64_zip = base64.b64encode(buf.getvalue()).decode("utf-8")

cells = []

def add_md(text: str):
    lines = [l + "\n" for l in text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    })

def add_code(code_str: str):
    lines = [l + "\n" for l in code_str.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })

add_md("""# Telugu Handwritten Character Recognizer (v3)
**Target: >=90% Top-1 Accuracy | Platform: Kaggle GPU (P100 / T4) | Framework: TensorFlow / Keras**

> **How to Run:**
> 1. In the right sidebar under **Notebook Options** -> **Accelerator**, choose **GPU P100** or **GPU T4 x2**.
> 2. Under **Input Data**, click **+ Add Input** and attach your Telugu Handwritten Dataset (e.g. `data-telugu-handwritten`).
> 3. Click **Run All**.""")

add_code(f"""# 1. Unpack Codebase (Automatic - Zero Manual Code Uploads)
import os
import sys
import base64
import io
import zipfile
from pathlib import Path

WORK_DIR = Path("/kaggle/working/telugu-hcr-v3")
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(str(WORK_DIR))
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

B64_ZIP = "{b64_zip}"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(B64_ZIP)), "r") as z:
    z.extractall(WORK_DIR)

print("Codebase successfully unpacked to:", WORK_DIR)
print("Modules available:", [p.name for p in (WORK_DIR / "src").glob("*.py")])""")

add_code("""# 2. Environment & Kaggle P100 Grappler Workaround
import tensorflow as tf

# Workaround for TF Grappler bug causing gradient corruption on Kaggle P100
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("GPU Detected:", [g.name for g in gpus])
else:
    print("WARNING: No GPU detected! Please enable GPU Accelerator in the right sidebar.")""")

add_code("""# 3. Instant Auto-Detect Dataset Directory in /kaggle/input (0.01s)
import os
from pathlib import Path

DATA_DIR = None
input_root = "/kaggle/input"

# Search top directory levels only (stops immediately when Test1 is found)
for root, dirs, _ in os.walk(input_root):
    if "Test1" in dirs:
        DATA_DIR = os.path.join(root, "Test1")
        break
    if any(d.lower() in {"guninthamulu", "achulu", "hallulu", "othulu"} for d in dirs):
        DATA_DIR = root
        break
    depth = len(os.path.relpath(root, input_root).split(os.sep))
    if depth >= 3:
        dirs.clear()

if not DATA_DIR:
    # Fallback to direct path
    DATA_DIR = "/kaggle/input/data-telugu-handwritten"

print(f"🎯 Dataset root detected in milliseconds: {DATA_DIR}")

OUTPUT_DIR = str(WORK_DIR / "outputs")
REPORTS_DIR = str(WORK_DIR / "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)""")

add_code("""# 4. Phase 0: Data Audit & Stratified 80/10/10 Split
from src.data.audit import run_audit
from src.data.split import create_splits

print("=== Phase 0: Running Data Audit ===")
audit_results = run_audit(data_dir=DATA_DIR, output_dir=REPORTS_DIR)

print("=== Phase 0: Creating Stratified Splits (80/10/10) ===")
split_stats = create_splits(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
)""")

add_code("""# 5. Phase 1 Track A: EfficientNetB0 Transfer Learning
from src.train import train

print("=== Training Track A: EfficientNetB0 (Head -> Fine-tuning) ===")
history_a = train("configs/track_a_efficientnet.yaml", resume=True)""")

add_code("""# 6. Phase 1 Track B: Lean Custom CNN Baseline
print("=== Training Track B: Lean Custom CNN ===")
history_b = train("configs/track_b_custom_cnn.yaml", resume=True)""")

add_code("""# 7. Evaluate Baseline Models on Held-Out Validation Set
from src.evaluate import evaluate

print("=== Evaluating Track A on Validation Set ===")
results_a = evaluate(
    model_path=str(WORK_DIR / "checkpoints/track_a_efficientnet/best_model/model.keras"),
    data_csv=f"{OUTPUT_DIR}/val.csv",
    label_map_path=f"{OUTPUT_DIR}/label_map.json",
    config_path="configs/track_a_efficientnet.yaml",
    output_dir=f"{REPORTS_DIR}/track_a_val"
)

val_acc = results_a["top1"]
print(f"Track A Validation Top-1 Accuracy: {val_acc * 100:.2f}%")

if val_acc >= 0.90:
    print("Decision Gate Cleared: >=90% Top-1 Accuracy! Proceeding to Phase 3.")
else:
    print("Under 90%: Check reports/track_a_val/top_confused_pairs.csv for Phase 2 clustering.")""")

add_code("""# 8. Phase 3: Soft-Voting Ensemble + Test-Time Augmentation (TTA)
# Untouched test set evaluated ONCE for the final official report score
import json
import yaml
import numpy as np
import tensorflow as tf
from src.data.dataset import build_dataset
from src.infer_tta import ensemble_predict_with_tta

with open("configs/ensemble.yaml", "r") as f:
    ensemble_cfg = yaml.safe_load(f)
with open(f"{OUTPUT_DIR}/label_map.json", "r") as f:
    label_map = json.load(f)

test_ds = build_dataset(f"{OUTPUT_DIR}/test.csv", label_map, ensemble_cfg, training=False)

model_paths = [
    str(WORK_DIR / "checkpoints/track_a_efficientnet/best_model/model.keras"),
    str(WORK_DIR / "checkpoints/track_b_custom_cnn/best_model/model.keras")
]
models = [tf.keras.models.load_model(p, compile=False) for p in model_paths if Path(p).exists()]

if models:
    tta_cfg = ensemble_cfg.get("ensemble", {}).get("tta", {})
    print(f"Running Ensemble ({len(models)} models) with TTA on held-out test set...")
    test_preds = ensemble_predict_with_tta(models, test_ds, tta_cfg, num_augmentations=5)
    
    y_pred = np.argmax(test_preds, axis=1)
    y_true_onehot = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_true = np.argmax(y_true_onehot, axis=1)
    
    final_test_acc = float(np.mean(y_true == y_pred))
    print(f"FINAL TEST ACCURACY (Ensemble + TTA): {final_test_acc * 100:.2f}%")
else:
    print("No trained models found to ensemble.")""")

notebook = {
    "nbformat": 4,
    "nbformat_minor": 2,
    "metadata": {
        "accelerator": "GPU",
        "language_info": {"name": "python"}
    },
    "cells": cells
}

(root / "notebooks").mkdir(parents=True, exist_ok=True)
out_path = root / "notebooks/telugu_hcr_kaggle.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Successfully generated standalone notebook with {len(cells)} cells at: {out_path}")
