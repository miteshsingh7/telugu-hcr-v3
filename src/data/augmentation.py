"""Character-Safe Augmentation Pipeline for Telugu Handwritten Character Recognition.

Supports:
1. Rescale mode (fill_value = 1.0) for [0, 1] normalized grayscale/RGB models.
2. ImageNet mode (fill_value = +131.32) for zero-mean-centered ImageNet models.
3. Multi-task target dictionaries and single-tensor label formats.
"""

from typing import Callable, Dict, Any, Tuple, Union
import tensorflow as tf


def get_background_fill_value(normalize_mode: str = "rescale", num_channels: int = 1) -> float:
    """Computes the exact numerical background fill value for white paper."""
    if normalize_mode == "imagenet":
        # White (255.0) minus ImageNet RGB mean (123.68, 116.78, 103.94) ≈ +131.32
        return 131.32
    else:
        # Rescale mode [0.0, 1.0]: White background = 1.0
        return 1.0


def build_augmentation_fn(
    config: Dict[str, Any] = None,
    normalize_mode: str = "rescale",
    num_channels: int = 1,
) -> Callable:
    """Builds a character-safe training augmentation function with exact fill value."""
    config = config or {}
    rotation_range = config.get("rotation_range", 4)
    width_shift = config.get("width_shift", 0.04)
    height_shift = config.get("height_shift", 0.04)
    zoom_range = config.get("zoom_range", 0.04)
    fill_val = get_background_fill_value(normalize_mode, num_channels)

    aug_model = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(
            factor=rotation_range / 360.0,
            fill_mode="constant",
            fill_value=fill_val,
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=height_shift,
            width_factor=width_shift,
            fill_mode="constant",
            fill_value=fill_val,
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-zoom_range, zoom_range),
            fill_mode="constant",
            fill_value=fill_val,
        ),
    ])

    def augment(image: tf.Tensor, targets: Union[tf.Tensor, Dict[str, tf.Tensor]]) -> Tuple[tf.Tensor, Any]:
        augmented_image = aug_model(image, training=True)
        return augmented_image, targets

    return augment


def build_tta_augmentation_fn(
    config: Dict[str, Any] = None,
    normalize_mode: str = "rescale",
    num_channels: int = 1,
) -> Callable:
    """Builds a test-time augmentation function."""
    config = config or {}
    rotation_range = config.get("rotation_range", 3)
    shift_range = config.get("shift_range", 0.03)
    fill_val = get_background_fill_value(normalize_mode, num_channels)

    tta_model = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(
            factor=rotation_range / 360.0,
            fill_mode="constant",
            fill_value=fill_val,
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=shift_range,
            width_factor=shift_range,
            fill_mode="constant",
            fill_value=fill_val,
        ),
    ])

    def augment_tta(image: tf.Tensor) -> tf.Tensor:
        return tta_model(image, training=True)

    return augment_tta
