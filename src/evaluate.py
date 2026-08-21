import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import yaml

from src.data.dataset import build_dataset

logger = logging.getLogger(__name__)

def top_k_accuracy(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 5) -> float:
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
    correct = np.any(top_k_preds == y_true[:, np.newaxis], axis=1)
    return float(np.mean(correct))

def extract_confused_pairs(
    cm: np.ndarray,
    class_names: List[str],
    top_k: int = 20,
) -> pd.DataFrame:
    n_classes = len(class_names)
    pairs = []

    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            c_ij = int(cm[i, j])
            c_ji = int(cm[j, i])
            total = c_ij + c_ji
            if total > 0:
                pairs.append({
                    "class_a": class_names[i],
                    "class_b": class_names[j],
                    "count_a_pred_b": c_ij,
                    "count_b_pred_a": c_ji,
                    "total_confusion": total,
                })

    df = pd.DataFrame(pairs)
    if not df.empty:
        df = df.sort_values("total_confusion", ascending=False).head(top_k)
    return df

def evaluate(
    model_path: str,
    data_csv: str,
    label_map_path: str,
    config_path: str,
    output_dir: str = "reports/",
    sample_limit: int = None,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "_base_" in config:
        base_path = Path(config_path).parent / config["_base_"]
        with open(base_path, "r") as f:
            base_config = yaml.safe_load(f)
        config = {**base_config, **config}

    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    idx_to_class = {v: k for k, v in label_map.items()}
    class_names = [idx_to_class[i] for i in range(len(label_map))]

    print(f"[Eval] Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    print(f"[Eval] Building dataset from {data_csv} with canonical preprocessing...")
    dataset = build_dataset(data_csv, label_map, config, training=False)

    y_true_list = []
    with open(data_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true_list.append(int(row["label_idx"]))
    y_true = np.array(y_true_list)

    if sample_limit:
        batch_size = config.get("batch_size", 64)
        num_batches = max(1, sample_limit // batch_size)
        dataset = dataset.take(num_batches)
        y_true = y_true[: num_batches * batch_size]

    print(f"[Eval] Running model predictions over {len(y_true)} samples...")
    t0 = time.time()
    y_pred_proba = model.predict(dataset)
    y_pred_proba = y_pred_proba[:len(y_true)]
    y_pred = np.argmax(y_pred_proba, axis=1)
    eval_duration = time.time() - t0

    top1 = float(np.mean(y_true == y_pred))
    top3 = top_k_accuracy(y_true, y_pred_proba, k=3)
    top5 = top_k_accuracy(y_true, y_pred_proba, k=5)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_map))))
    confused_pairs = extract_confused_pairs(cm, class_names, top_k=20)
    confused_pairs.to_csv(out_dir / "top_confused_pairs.csv", index=False)

    timestamp = datetime.now().isoformat()
    results = {
        "timestamp": timestamp,
        "model_path": model_path,
        "data_csv": data_csv,
        "top1_accuracy": round(top1 * 100, 2),
        "top3_accuracy": round(top3 * 100, 2),
        "top5_accuracy": round(top5 * 100, 2),
        "num_samples": len(y_true),
        "num_classes": len(label_map),
        "eval_seconds": round(eval_duration, 2),
    }

    report_md = out_dir / "evaluation_report.md"
    with open(report_md, "w") as f:
        f.write("# Model Evaluation Benchmark Report\n\n")
        f.write(f"- **Timestamp:** `{timestamp}`\n")
        f.write(f"- **Model Checkpoint:** `{model_path}`\n")
        f.write(f"- **Evaluation Dataset:** `{data_csv}` ({len(y_true)} samples)\n")
        f.write(f"- **Classes Evaluated:** {len(label_map)}\n")
        f.write(f"- **Top-1 Accuracy:** **{top1 * 100:.2f}%**\n")
        f.write(f"- **Top-3 Accuracy:** **{top3 * 100:.2f}%**\n")
        f.write(f"- **Top-5 Accuracy:** **{top5 * 100:.2f}%**\n")
        f.write(f"- **Evaluation Time:** {eval_duration:.2f}s\n\n")
        f.write("### Top Confused Pairs:\n\n")
        f.write(confused_pairs.to_markdown(index=False))
        f.write("\n")

    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "="*50)
    print(f"🏆 BENCHMARK EVALUATION RESULTS ({timestamp}):")
    print(f"  • Evaluated Samples: {len(y_true)}")
    print(f"  • Top-1 Accuracy: {top1 * 100:.2f}%")
    print(f"  • Top-3 Accuracy: {top3 * 100:.2f}%")
    print(f"  • Top-5 Accuracy: {top5 * 100:.2f}%")
    print("="*50 + "\n")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="reports/")
    parser.add_argument("--sample-limit", type=int, default=None)
    args = parser.parse_args()

    evaluate(args.model, args.data, args.label_map, args.config, args.output, sample_limit=args.sample_limit)
