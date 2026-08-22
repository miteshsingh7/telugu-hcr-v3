import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import tensorflow as tf
import yaml

from src.data.augmentation import build_tta_augmentation_fn
from src.data.dataset import build_dataset

logger = logging.getLogger(__name__)

def predict_with_tta(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    tta_config: Dict[str, Any] = None,
    num_augmentations: int = 5,
    normalize_mode: str = None,
    num_channels: int = None,
) -> np.ndarray:
    tta_config = tta_config or {}
    
    # Auto-detect channels and normalization from model if not explicitly provided
    if num_channels is None:
        num_channels = model.input_shape[-1] if len(model.input_shape) == 4 else 1
    if normalize_mode is None:
        normalize_mode = "imagenet" if num_channels == 3 else "rescale"

    tta_fn = build_tta_augmentation_fn(
        tta_config,
        normalize_mode=normalize_mode,
        num_channels=num_channels,
    )
    all_predictions = []

    for batch in dataset:
        if isinstance(batch, (tuple, list)):
            x_batch = batch[0]
        else:
            x_batch = batch

        pred_orig = model(x_batch, training=False)
        if isinstance(pred_orig, dict):
            pred_orig = pred_orig.get("base_output", list(pred_orig.values())[0])
        pred_orig = pred_orig.numpy() if hasattr(pred_orig, "numpy") else pred_orig
        pred_sum = pred_orig.copy()

        for _ in range(num_augmentations):
            x_aug = tf.map_fn(tta_fn, x_batch)
            pred_aug = model(x_aug, training=False)
            if isinstance(pred_aug, dict):
                pred_aug = pred_aug.get("base_output", list(pred_aug.values())[0])
            pred_aug = pred_aug.numpy() if hasattr(pred_aug, "numpy") else pred_aug
            pred_sum += pred_aug

        pred_avg = pred_sum / (num_augmentations + 1)
        all_predictions.append(pred_avg)

    return np.concatenate(all_predictions, axis=0)


def ensemble_predict_with_tta(
    models: List[tf.keras.Model],
    dataset: tf.data.Dataset,
    tta_config: Dict[str, Any] = None,
    num_augmentations: int = 5,
    weights: List[float] = None,
) -> np.ndarray:
    """Computes Test-Time-Augmented ensemble predictions across multiple models."""
    tta_config = tta_config or {}
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    else:
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

    ensemble_preds = None
    for model, w in zip(models, weights):
        m_preds = predict_with_tta(model, dataset, tta_config, num_augmentations)
        if ensemble_preds is None:
            ensemble_preds = w * m_preds
        else:
            ensemble_preds += w * m_preds

    return ensemble_preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="reports/")
    parser.add_argument("--num-aug", type=int, default=5)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(args.label_map, "r") as f:
        label_map = json.load(f)

    model = tf.keras.models.load_model(args.model, compile=False)
    dataset = build_dataset(args.test_data, label_map, config, training=False)
    tta_config = config.get("ensemble", {}).get("tta", {})

    preds = predict_with_tta(model, dataset, tta_config, num_augmentations=args.num_aug)
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "test_predictions.npy", preds)
