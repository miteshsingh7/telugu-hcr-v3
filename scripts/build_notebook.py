"""Script to generate a self-contained, standalone Kaggle notebook for Telugu HCR v3."""

import base64
import io
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent

# Create in-memory zip of configs and src
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for folder in ["configs", "src", "scripts"]:
        for p in (root / folder).rglob("*"):
            if p.is_file() and not p.name.startswith(".") and not p.name.endswith(".pyc"):
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
**Architecture: Multi-Task Grapheme Decomposition & Pretrained Transfer Learning**
**Platform: Kaggle GPU (P100 / T4) | Framework: TensorFlow / Keras 3**

> **How to Run:**
> 1. In the right sidebar under **Notebook Options** -> **Accelerator**, choose **GPU P100** or **GPU T4 x2**.
> 2. Under **Input Data**, click **+ Add Input** and attach the Telugu Handwritten Dataset (e.g. `data-telugu-handwritten`).
> 3. Click **Run All** (Total runtime: ~25-30 minutes).""")

add_code(f"""# 1. Unpack Full Source Code & Configuration
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

print("✅ Codebase unpacked successfully to:", WORK_DIR)
print("📦 Available modules:", [p.name for p in (WORK_DIR / "src").glob("*.py")])""")

add_code("""# 2. Environment Setup & GPU Verification
import tensorflow as tf

# Mixed precision for 2.5x speedup on Tensor Cores
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("🚀 GPU Active:", [g.name for g in gpus])
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("⚡ Mixed precision (mixed_float16) active.")
    except Exception as e:
        print("Note on mixed precision:", e)
else:
    print("⚠️ WARNING: No GPU detected! Please enable GPU in the right sidebar under Accelerator.")""")

add_code("""# 3. Auto-Discover Dataset Directory in /kaggle/input (Instant 0.01s)
import os
from pathlib import Path

DATA_DIR = None
input_root = "/kaggle/input"

for root, dirs, _ in os.walk(input_root):
    if "Test1" in dirs:
        DATA_DIR = os.path.join(root, "Test1")
        break
    if any(d.lower() in {"guninthamulu", "achulu", "hallulu", "othulu"} for d in dirs):
        DATA_DIR = root
        break

if not DATA_DIR:
    DATA_DIR = "/kaggle/input/data-telugu-handwritten/Final Dataset of Telugu Handwritten Chararcters/Test1"

print(f"🎯 Dataset root detected: {DATA_DIR}")

OUTPUT_DIR = str(WORK_DIR / "outputs")
REPORTS_DIR = str(WORK_DIR / "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)""")

add_code("""# 4. Generate Grapheme Decomposition Maps (Base 52, Mod 16, Vattu 36)
from src.data.decomposition import export_grapheme_maps
maps = export_grapheme_maps(str(WORK_DIR / "outputs/grapheme_maps.json"))
print(f"✅ Grapheme decomposition maps ready: {maps['num_base_classes']} Base Aksharas, {maps['num_modifier_classes']} Vowel Signs, {maps['num_vattu_classes']} Conjuncts.")""")

add_code("""# 5. Train Pre-Trained Multi-Task Model (MobileNetV2 + Focal Loss, ~25 Mins)
from scripts.train_pretrained_multitask import run_training

print("=== Starting Pretrained MobileNetV2 Multi-Task Training ===")
history = run_training(
    head_epochs=4,
    finetune_epochs=24,
    batch_size=128,
    img_size=128,
)""")

add_code("""# 6. Copy Model to Output for 1-Click Download
import shutil

checkpoint_src = WORK_DIR / "checkpoints/multitask_mobilenet_best.keras"
checkpoint_dest = Path("/kaggle/working/multitask_mobilenet_best.keras")

if checkpoint_src.exists():
    shutil.copy(checkpoint_src, checkpoint_dest)
    print("✅ Model copied to main working output:", checkpoint_dest)
    print(f"📦 File size: {checkpoint_dest.stat().st_size / (1024*1024):.2f} MB")

from IPython.display import FileLink
display(FileLink(r'multitask_mobilenet_best.keras'))""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = root / "notebooks/telugu_hcr_kaggle.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Generated clean, self-contained Kaggle notebook at:", out_path)
