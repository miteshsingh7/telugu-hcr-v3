"""Test-time augmentation (TTA) and ensemble inference for Telugu HCR.

Generates multiple augmented views of each test image, averages the
softmax predictions across views (and optionally across models), and
exports the final predictions.

Usage:
    python -m src.infer_tta --models checkpoints/a/model.keras checkpoints/b/model.keras \\
        --test-data outputs/test.csv --label-map outputs/label_map.json \\
        --config configs/track_a_efficientnet.yaml
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm

from src.data.dataset import build_dataset
from src.data.augmentation import build_tta_augmentation_fn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading (duplicated for standalone CLI use)
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
# TTA prediction
# ---------------------------------------------------------------------------

def predict_with_tta(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    tta_config: Dict[str, Any],
    num_augmentations: int = 5,
) -> np.ndarray:
    """Predicts with test-time augmentation.

    For every batch, the original images and ``num_augmentations``
    augmented copies are each passed through the model, and the softmax
    outputs are averaged.

    Args:
        model: Trained Keras model.
        dataset: A batched ``tf.data.Dataset`` yielding ``(images, labels)``
            or just ``images``.
        tta_config: TTA configuration dict (passed to
            ``build_tta_augmentation_fn``).
        num_augmentations: Number of augmented views per image.

    Returns:
        Averaged prediction array of shape ``(N, num_classes)``.
    """
    tta_fn = build_tta_augmentation_fn(tta_config)
    all_preds: List[np.ndarray] = []

    for batch in tqdm(dataset, desc="TTA Inference"):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch

        # Original prediction
        batch_preds = model(x, training=False).numpy().astype(np.float64)

        # Augmented predictions
        for _ in range(num_augmentations):
            aug_x = tta_fn(x)  # works on batched tensors
            aug_preds = model(aug_x, training=False).numpy().astype(np.float64)
            batch_preds += aug_preds

        batch_preds /= num_augmentations + 1
        all_preds.append(batch_preds)

    return np.concatenate(all_preds, axis=0)


def ensemble_predict_with_tta(
    models: List[tf.keras.Model],
    dataset: tf.data.Dataset,
    tta_config: Dict[str, Any],
    num_augmentations: int = 5,
) -> np.ndarray:
    """Ensemble of multiple models with TTA.

    Each model independently produces TTA-averaged predictions, which are
    then averaged across all models (soft-voting).

    Args:
        models: List of trained Keras models.
        dataset: Batched ``tf.data.Dataset``.
        tta_config: TTA configuration dict.
        num_augmentations: Number of augmented views per image.

    Returns:
        Averaged ensemble prediction array of shape ``(N, num_classes)``.
    """
    if not models:
        raise ValueError("Models list cannot be empty.")

    ensemble_preds = None
    for i, model in enumerate(models):
        logger.info(f"TTA predictions for model {i + 1}/{len(models)}")
        preds = predict_with_tta(model, dataset, tta_config, num_augmentations)

        if ensemble_preds is None:
            ensemble_preds = preds
        else:
            ensemble_preds += preds

    return ensemble_preds / len(models)


# ---------------------------------------------------------------------------
# CLI — final test evaluation (test set touched ONCE)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="TTA / Ensemble inference for Telugu HCR")
    parser.add_argument("--models", nargs="+", required=True, help="One or more .keras model paths")
    parser.add_argument("--test-data", required=True, help="Path to test CSV")
    parser.add_argument("--label-map", required=True, help="Label map JSON")
    parser.add_argument("--config", required=True, help="Config YAML")
    parser.add_argument("--output", default="reports/", help="Output directory")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA (plain ensemble)")
    parser.add_argument("--num-aug", type=int, default=5, help="Number of TTA augmentations")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    config = _load_config(args.config)

    with open(args.label_map, "r") as f:
        label_map = json.load(f)

    dataset = build_dataset(args.test_data, label_map, config, training=False)

    logger.info(f"Loading {len(args.models)} model(s)…")
    models = [tf.keras.models.load_model(m, compile=False) for m in args.models]

    if args.no_tta:
        logger.info("TTA disabled — plain ensemble prediction.")
        preds = None
        for m in models:
            p = m.predict(dataset)
            preds = p if preds is None else preds + p
        preds /= len(models)
    else:
        tta_cfg = config.get("ensemble", {}).get("tta", {})
        preds = ensemble_predict_with_tta(models, dataset, tta_cfg, num_augmentations=args.num_aug)

    np.save(out_path / "test_predictions.npy", preds)
    y_pred = np.argmax(preds, axis=1)

    # Also collect true labels and report accuracy
    y_true_onehot = np.concatenate([y.numpy() for _, y in dataset], axis=0)
    y_true = np.argmax(y_true_onehot, axis=1)

    top1 = float(np.mean(y_true == y_pred))
    logger.info(f"Final Test Top-1 Accuracy: {top1:.4f}")
    logger.info(f"Predictions saved to {out_path / 'test_predictions.npy'}")
