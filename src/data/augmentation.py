"""Character-safe augmentations for Telugu handwriting.

All transforms are designed to preserve character identity:
  - NO horizontal flip (would produce invalid characters)
  - Rotation capped at ±10° (training) / ±8° (TTA)
  - Shift capped at ±10% (training) / ±8% (TTA)
"""

from typing import Callable, Dict, Any, List, Tuple

import tensorflow as tf


def build_augmentation_fn(config: Dict[str, Any]) -> Callable:
    """Builds a training augmentation function from config.

    Args:
        config: Augmentation configuration dict.  Expected keys:
            ``rotation_range`` (degrees), ``width_shift``, ``height_shift``,
            ``zoom_range``, ``brightness_range`` (list [lo, hi] or float max_delta),
            ``contrast_range`` (list [lo, hi]).

    Returns:
        A function ``(image, label) -> (augmented_image, label)``.
    """
    layers: List[tf.keras.layers.Layer] = []

    if config.get("rotation_range", 0) > 0:
        factor = config["rotation_range"] / 360.0
        layers.append(
            tf.keras.layers.RandomRotation(factor=factor, fill_mode="constant")
        )

    w_shift = config.get("width_shift", 0)
    h_shift = config.get("height_shift", 0)
    if w_shift > 0 or h_shift > 0:
        layers.append(
            tf.keras.layers.RandomTranslation(
                height_factor=h_shift, width_factor=w_shift, fill_mode="constant"
            )
        )

    if config.get("zoom_range", 0) > 0:
        z = config["zoom_range"]
        layers.append(
            tf.keras.layers.RandomZoom(height_factor=(-z, z), fill_mode="constant")
        )

    # Parse brightness — could be a list [lo, hi] or a float max_delta
    brightness_range = config.get("brightness_range", None)
    if isinstance(brightness_range, (list, tuple)):
        # Convert [0.9, 1.1] → max_delta = 0.1
        brightness_delta = (brightness_range[1] - brightness_range[0]) / 2.0
    elif isinstance(brightness_range, (int, float)):
        brightness_delta = float(brightness_range)
    else:
        brightness_delta = 0.0

    contrast_range = config.get("contrast_range", None)

    def augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply augmentations to a single image."""
        aug = tf.expand_dims(image, 0)
        for layer in layers:
            aug = layer(aug, training=True)
        aug = tf.squeeze(aug, 0)

        if brightness_delta > 0:
            aug = tf.image.random_brightness(aug, max_delta=brightness_delta)

        if contrast_range is not None:
            lo, hi = contrast_range
            aug = tf.image.random_contrast(aug, lower=lo, upper=hi)

        aug = tf.clip_by_value(aug, 0.0, 1.0)
        return aug, label

    return augment


def build_tta_augmentation_fn(config: Dict[str, Any]) -> Callable:
    """Builds a lighter augmentation function for test-time augmentation.

    Args:
        config: TTA configuration dict. Expected keys:
            ``rotation_range`` (default 8), ``shift_range`` (default 0.08).

    Returns:
        A function ``(image) -> augmented_image`` (no label).
    """
    rot = config.get("rotation_range", 8)
    shift = config.get("shift_range", 0.08)

    tta_pipeline = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(factor=rot / 360.0, fill_mode="constant"),
            tf.keras.layers.RandomTranslation(
                height_factor=shift, width_factor=shift, fill_mode="constant"
            ),
        ]
    )

    def augment(image: tf.Tensor) -> tf.Tensor:
        """Apply light augmentation to a single image (or batch)."""
        needs_expand = len(image.shape) == 3
        x = tf.expand_dims(image, 0) if needs_expand else image
        x = tta_pipeline(x, training=True)
        return tf.squeeze(x, 0) if needs_expand else x

    return augment


def generate_tta_views(
    image: tf.Tensor, tta_fn: Callable, num_augmentations: int = 5
) -> tf.Tensor:
    """Generates K augmented views of a single image plus the original.

    Args:
        image: A single image tensor ``(H, W, C)``.
        tta_fn: Augmentation function from ``build_tta_augmentation_fn``.
        num_augmentations: Number of augmented copies.

    Returns:
        Stacked tensor of shape ``(num_augmentations + 1, H, W, C)``
        where index 0 is the original.
    """
    views = [image]
    for _ in range(num_augmentations):
        views.append(tta_fn(image))
    return tf.stack(views, axis=0)
