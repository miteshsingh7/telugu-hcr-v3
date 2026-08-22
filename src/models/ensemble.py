"""Ensemble Inference Module for Telugu Handwritten Character Recognition.

Supports Soft-Voting Ensembles across homogeneous and heterogeneous neural architectures
with dynamic per-model shape adaptation and normalization.
"""

from typing import List, Optional, Any, Union, Callable
import numpy as np
import tensorflow as tf
from PIL import Image

from src.data.preprocess import tf_canonical_preprocess, numpy_canonical_preprocess


class SoftVotingEnsemble:
    """Soft-voting ensemble classifier supporting multi-resolution backbones."""

    def __init__(
        self,
        models: List[tf.keras.Model],
        weights: Optional[List[float]] = None,
    ):
        self.models = models
        if weights is not None:
            total = float(sum(weights))
            self.weights = [w / total for w in weights]
        else:
            self.weights = [1.0 / len(models)] * len(models)

    def _adapt_input_for_model(self, x: Any, model: tf.keras.Model) -> np.ndarray:
        """Dynamically shapes and normalizes input tensor matching model.input_shape."""
        shape = model.input_shape
        img_size = shape[1] if shape[1] is not None else 96
        num_channels = shape[-1] if shape[-1] is not None else 1
        normalize_mode = "imagenet" if num_channels == 3 else "rescale"

        if isinstance(x, (Image.Image, np.ndarray)) and (not isinstance(x, np.ndarray) or x.ndim in (2, 3)):
            tensor, _ = numpy_canonical_preprocess(
                x,
                img_size=img_size,
                num_channels=num_channels,
                normalize_mode=normalize_mode,
            )
            return tensor
        return x

    def predict_proba(self, x: Any) -> np.ndarray:
        """Computes weighted probability average across all ensemble members."""
        preds = []
        for model in self.models:
            x_adapted = self._adapt_input_for_model(x, model)
            if isinstance(x_adapted, tf.data.Dataset):
                p = model.predict(x_adapted, verbose=0)
            else:
                p = model(x_adapted, training=False).numpy()
            
            # If multi-task dictionary output, select base output
            if isinstance(p, dict):
                p = p.get("base_output", list(p.values())[0])
            preds.append(p)

        weighted_preds = sum(w * p for w, p in zip(self.weights, preds))
        return weighted_preds

    def predict(self, x: Any) -> np.ndarray:
        """Returns class index with highest ensemble probability."""
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=-1)

    def predict_from_csv(
        self,
        csv_path: str,
        label_map: dict,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Evaluates heterogeneous models on a CSV manifest by building per-model datasets."""
        from src.data.dataset import build_dataset
        
        preds = []
        for model in self.models:
            shape = model.input_shape
            img_size = shape[1] if shape[1] is not None else 96
            num_channels = shape[-1] if shape[-1] is not None else 1
            normalize_mode = "imagenet" if num_channels == 3 else "rescale"

            cfg = {
                "image_size": img_size,
                "num_channels": num_channels,
                "normalize_mode": normalize_mode,
                "batch_size": batch_size,
            }
            ds = build_dataset(csv_path, label_map, cfg, training=False)
            p = model.predict(ds, verbose=0)
            if isinstance(p, dict):
                p = p.get("base_output", list(p.values())[0])
            preds.append(p)

        weighted_preds = sum(w * p for w, p in zip(self.weights, preds))
        return weighted_preds
