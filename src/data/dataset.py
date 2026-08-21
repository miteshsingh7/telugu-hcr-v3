"""TensorFlow dataset pipeline for Telugu HCR.

Builds a ``tf.data.Dataset`` from CSV split manifests with image loading,
preprocessing, optional augmentation, and batching.
"""

import csv
from pathlib import Path
from typing import Dict, Any

import numpy as np
import tensorflow as tf

from .augmentation import build_augmentation_fn


def get_class_weights(csv_path: str) -> Dict[int, float]:
    """Computes balanced class weights from a split CSV.

    Uses the formula: weight_c = total / (num_classes * count_c).

    Args:
        csv_path: Path to the CSV manifest.

    Returns:
        Dictionary mapping class index to weight.
    """
    counts: Dict[int, int] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["label_idx"])
            counts[idx] = counts.get(idx, 0) + 1

    total = sum(counts.values())
    num_classes = len(counts)

    return {idx: total / (num_classes * count) for idx, count in counts.items()}


def get_num_classes(label_map: Dict[str, int]) -> int:
    """Returns the number of classes from a label map."""
    return len(label_map)


def build_dataset(
    csv_path: str,
    label_map: Dict[str, int],
    config: Dict[str, Any],
    training: bool = True,
) -> tf.data.Dataset:
    """Builds a ``tf.data.Dataset`` from a CSV manifest.

    The CSV must have columns ``filepath``, ``label_idx``, ``class_name``.

    Args:
        csv_path: Path to the CSV split manifest.
        label_map: Dictionary mapping class names to indices.
        config: Full experiment configuration dictionary.  Expected keys at
            the root level: ``image_size``, ``num_channels``,
            ``normalize_mode``, ``batch_size``, ``augmentation`` (dict).
        training: Whether this dataset is for training (enables shuffle and
            augmentation).

    Returns:
        A batched and prefetched ``tf.data.Dataset`` yielding
        ``(image, one_hot_label)`` tuples.
    """
    paths = []
    labels = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row["filepath"])
            labels.append(int(row["label_idx"]))

    num_classes = len(label_map)
    image_size = config.get("image_size", 128)
    num_channels = config.get("model", {}).get("num_channels", config.get("num_channels", 3))
    normalize_mode = config.get("normalize_mode", "rescale")
    batch_size = config.get("batch_size", 64)

    def process_path(file_path: tf.Tensor, label: tf.Tensor):
        """Read, decode, resize, and normalize a single image."""
        img_raw = tf.io.read_file(file_path)
        img = tf.io.decode_image(img_raw, channels=num_channels, expand_animations=False)
        img.set_shape([None, None, num_channels])

        img = tf.image.resize(img, [image_size, image_size])
        img = tf.cast(img, tf.float32)

        if normalize_mode == "imagenet" and num_channels == 3:
            # ImageNet mean/std normalization (channels-last)
            mean = tf.constant([123.68, 116.779, 103.939])
            img = img - mean
        else:  # "rescale" — default
            img = img / 255.0

        label_oh = tf.one_hot(label, num_classes)
        return img, label_oh

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(buffer_size=min(len(paths), 10000))

    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        aug_config = config.get("augmentation", {})
        if aug_config:
            aug_fn = build_augmentation_fn(aug_config)
            dataset = dataset.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
