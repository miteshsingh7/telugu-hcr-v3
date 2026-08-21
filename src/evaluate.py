"""Evaluation & analysis for Telugu HCR models.

Computes top-k accuracy, per-class metrics, confusion matrix, and
extracts the most confused class pairs (feeds Phase 2A directly).

Usage:
    python -m src.evaluate --model checkpoints/best/model.keras \\
        --data outputs/val.csv --label-map outputs/label_map.json \\
        --config configs/track_a_efficientnet.yaml
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, confusion_matrix as sk_confusion_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import build_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading (duplicated here for standalone CLI use)
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config with _base_ inheritance."""
    path = Path(config_path)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if "_base_" in config:
        base_path = path.parent / config["_base_"]
        with open(base_path, "r") as f:
            base_config = yaml.safe_load(f)

        def _deep_merge(base: dict, override: dict) -> dict:
            for k, v in override.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k] = _deep_merge(base[k], v)
                else:
                    base[k] = v
            return base

        config = _deep_merge(base_config, config)
        config.pop("_base_", None)

    return config


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def top_k_accuracy(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 5) -> float:
    """Calculates top-k accuracy.

    Args:
        y_true: 1D array of true class indices.
        y_pred_proba: 2D array of predicted probabilities (samples × classes).
        k: The k value for top-k accuracy.

    Returns:
        Top-k accuracy as a float.
    """
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
    correct = np.any(top_k_preds == y_true[:, np.newaxis], axis=1)
    return float(np.mean(correct))


def extract_confused_pairs(
    cm: np.ndarray, class_names: List[str], top_k: int = 20
) -> pd.DataFrame:
    """Extracts the top K most confused class pairs from a confusion matrix.

    Args:
        cm: Confusion matrix array (N × N).
        class_names: List of class names.
        top_k: Number of top confused pairs to return.

    Returns:
        DataFrame with columns: class_a, class_b, count_a_pred_as_b,
        count_b_pred_as_a, total_confusion.
    """
    pairs = []
    n = len(class_names)
    for i in range(n):
        for j in range(i + 1, n):
            confusion_ij = int(cm[i, j])
            confusion_ji = int(cm[j, i])
            total = confusion_ij + confusion_ji
            if total > 0:
                pairs.append(
                    {
                        "class_a": class_names[i],
                        "class_b": class_names[j],
                        "count_a_pred_as_b": confusion_ij,
                        "count_b_pred_as_a": confusion_ji,
                        "total_confusion": total,
                    }
                )

    df = pd.DataFrame(pairs)
    if not df.empty:
        df = df.sort_values(by="total_confusion", ascending=False).head(top_k)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_path: str,
    data_csv: str,
    label_map_path: str,
    config_path: str,
    output_dir: str = "reports/",
) -> dict:
    """Evaluates a model and generates reports.

    Args:
        model_path: Path to the ``.keras`` model file.
        data_csv: Path to the evaluation data CSV (val.csv or test.csv).
        label_map_path: Path to the label map JSON.
        config_path: Path to the configuration YAML.
        output_dir: Directory to save report artefacts.

    Returns:
        Dictionary containing computed metrics.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Label map
    with open(label_map_path, "r") as f:
        label_map: Dict[str, int] = json.load(f)

    idx_to_class = {int(v): k for k, v in label_map.items()}
    num_classes = len(label_map)
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Load config and build evaluation dataset
    config = _load_config(config_path)
    dataset = build_dataset(data_csv, label_map, config, training=False)

    # ------------------------------------------------------------------
    # Generate predictions and collect true labels
    # ------------------------------------------------------------------
    logger.info("Generating predictions…")
    y_pred_proba = model.predict(dataset)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # The dataset yields one-hot labels → convert to class indices
    y_true_onehot = np.concatenate([y.numpy() for _, y in dataset], axis=0)
    y_true = np.argmax(y_true_onehot, axis=1)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    top1 = float(np.mean(y_true == y_pred))
    top3 = top_k_accuracy(y_true, y_pred_proba, k=3)
    top5 = top_k_accuracy(y_true, y_pred_proba, k=5)

    logger.info(f"Top-1 Acc: {top1:.4f} | Top-3: {top3:.4f} | Top-5: {top5:.4f}")

    # Classification report
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    macro_precision = report_dict["macro avg"]["precision"]
    macro_recall = report_dict["macro avg"]["recall"]
    macro_f1 = report_dict["macro avg"]["f1-score"]

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    cm = sk_confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    np.save(out_path / "confusion_matrix.npy", cm)

    # Heatmap (only practical for ≤ ~60 classes)
    if num_classes <= 60:
        fig_size = max(8, num_classes * 0.25)
        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(cm, annot=(num_classes <= 30), fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(out_path / "confusion_matrix.png", dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # Top confused pairs
    # ------------------------------------------------------------------
    confused_pairs_df = extract_confused_pairs(cm, class_names, top_k=30)
    confused_pairs_df.to_csv(out_path / "top_confused_pairs.csv", index=False)

    # ------------------------------------------------------------------
    # Per-class accuracy (sorted worst → best)
    # ------------------------------------------------------------------
    per_class_total = cm.sum(axis=1)
    per_class_correct = cm.diagonal()
    with np.errstate(divide="ignore", invalid="ignore"):
        class_acc = np.where(per_class_total > 0, per_class_correct / per_class_total, 0.0)

    class_acc_df = pd.DataFrame({"class": class_names, "accuracy": class_acc, "support": per_class_total})
    class_acc_df = class_acc_df.sort_values(by="accuracy")
    class_acc_df.to_csv(out_path / "per_class_accuracy.csv", index=False)

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    report_md = f"""# Evaluation Report

## Overall Metrics

| Metric | Value |
|---|---|
| **Top-1 Accuracy** | {top1:.4f} |
| **Top-3 Accuracy** | {top3:.4f} |
| **Top-5 Accuracy** | {top5:.4f} |
| **Macro Precision** | {macro_precision:.4f} |
| **Macro Recall** | {macro_recall:.4f} |
| **Macro F1** | {macro_f1:.4f} |

## Top Confused Pairs

{confused_pairs_df.to_markdown(index=False) if not confused_pairs_df.empty else "No confused pairs found."}

## Worst-Performing Classes (bottom 10)

{class_acc_df.head(10).to_markdown(index=False)}
"""

    with open(out_path / "evaluation_report.md", "w") as f:
        f.write(report_md)
    logger.info(f"Evaluation report saved to {out_path / 'evaluation_report.md'}")

    return {
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate Telugu HCR Model")
    parser.add_argument("--model", required=True, help="Path to .keras model")
    parser.add_argument("--data", required=True, help="CSV path (val.csv or test.csv)")
    parser.add_argument("--label-map", required=True, help="Label map JSON")
    parser.add_argument("--config", required=True, help="Config YAML")
    parser.add_argument("--output", default="reports/", help="Output directory for reports")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    evaluate(args.model, args.data, args.label_map, args.config, args.output)
