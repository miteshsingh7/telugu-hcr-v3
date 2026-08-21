import math
from typing import Callable, Dict, Any, Tuple
import tensorflow as tf

def build_augmentation_fn(config: Dict[str, Any]) -> Callable:
    rotation_range = config.get("rotation_range", 4)
    width_shift = config.get("width_shift", 0.04)
    height_shift = config.get("height_shift", 0.04)
    zoom_range = config.get("zoom_range", 0.04)

    aug_model = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=rotation_range / 360.0, fill_mode="constant", fill_value=1.0),
        tf.keras.layers.RandomTranslation(height_factor=height_shift, width_factor=width_shift, fill_mode="constant", fill_value=1.0),
        tf.keras.layers.RandomZoom(height_factor=(-zoom_range, zoom_range), fill_mode="constant", fill_value=1.0),
    ])

    def augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        augmented_image = aug_model(image, training=True)
        return augmented_image, label

    return augment

def build_tta_augmentation_fn(config: Dict[str, Any]) -> Callable:
    rotation_range = config.get("rotation_range", 3)
    shift_range = config.get("shift_range", 0.03)

    tta_model = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=rotation_range / 360.0, fill_mode="constant", fill_value=1.0),
        tf.keras.layers.RandomTranslation(height_factor=shift_range, width_factor=shift_range, fill_mode="constant", fill_value=1.0),
    ])

    def augment_tta(image: tf.Tensor) -> tf.Tensor:
        return tta_model(image, training=True)

    return augment_tta
