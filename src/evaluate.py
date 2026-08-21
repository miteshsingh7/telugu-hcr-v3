import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

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
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    idx_to_class = {v: k for k, v in label_map.items()}
    class_names = [idx_to_class[i] for i in range(len(label_map))]

    model = tf.keras.models.load_model(model_path, compile=False)
    dataset = build_dataset(data_csv, label_map, config, training=False)

    y_true_list = []
    with open(data_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true_list.append(int(row["label_idx"]))
    y_true = np.array(y_true_list)

    y_pred_proba = model.predict(dataset)
    y_pred = np.argmax(y_pred_proba, axis=1)

    top1 = float(np.mean(y_true == y_pred))
    top3 = top_k_accuracy(y_true, y_pred_proba, k=3)
    top5 = top_k_accuracy(y_true, y_pred_proba, k=5)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_map))))
    confused_pairs = extract_confused_pairs(cm, class_names, top_k=20)
    confused_pairs.to_csv(out_dir / "top_confused_pairs.csv", index=False)

    results = {
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "num_samples": len(y_true),
        "num_classes": len(label_map),
    }

    report_md = out_dir / "evaluation_report.md"
    with open(report_md, "w") as f:
        f.write("# Evaluation Report\n\n")
        f.write(f"- **Top-1 Accuracy:** {top1 * 100:.2f}%\n")
        f.write(f"- **Top-3 Accuracy:** {top3 * 100:.2f}%\n")
        f.write(f"- **Top-5 Accuracy:** {top5 * 100:.2f}%\n\n")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="reports/")
    args = parser.parse_args()

    evaluate(args.model, args.data, args.label_map, args.config, args.output)
