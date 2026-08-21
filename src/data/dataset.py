import csv
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf

from src.data.augmentation import build_augmentation_fn

logger = logging.getLogger(__name__)

def get_class_weights(csv_path: str) -> Dict[int, float]:
    counts = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["label_idx"])
            counts[idx] = counts.get(idx, 0) + 1

    total = sum(counts.values())
    num_classes = len(counts)
    return {idx: total / (num_classes * count) for idx, count in counts.items()}

def get_num_classes(label_map: Dict[str, int]) -> int:
    return len(label_map)

def build_dataset(
    csv_path: str,
    label_map: Dict[str, int],
    config: Dict[str, Any],
    training: bool = True,
) -> tf.data.Dataset:
    image_size = config.get("image_size", 128)
    num_channels = config.get("num_channels", 3)
    if "model" in config and isinstance(config["model"], dict) and "num_channels" in config["model"]:
        num_channels = config["model"]["num_channels"]
    normalize_mode = config.get("normalize_mode", "imagenet")
    batch_size = config.get("batch_size", 64)
    num_classes = len(label_map)

    filepaths = []
    labels = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepaths.append(row["filepath"])
            labels.append(int(row["label_idx"]))

    if not filepaths:
        raise ValueError(f"No samples found in {csv_path}")

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    def load_and_preprocess_image(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        img_raw = tf.io.read_file(path)
        img = tf.io.decode_image(img_raw, channels=0, expand_animations=False)
        img = tf.cond(
            tf.shape(img)[-1] >= 3,
            lambda: tf.image.rgb_to_grayscale(img[:, :, :3]) if num_channels == 1 else img[:, :, :3],
            lambda: tf.repeat(img, 3, axis=-1) if num_channels == 3 else img
        )
        img.set_shape([None, None, num_channels])
        img = tf.image.resize(img, [image_size, image_size])
        img = tf.cast(img, tf.float32)

        if normalize_mode == "imagenet" and num_channels == 3:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
            img = img - mean
        else:
            img = img / 255.0

        label_one_hot = tf.one_hot(label, depth=num_classes)
        return img, label_one_hot

    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        aug_config = config.get("augmentation", {})
        aug_fn = build_augmentation_fn(aug_config)
        dataset = dataset.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=min(len(filepaths), 10000))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
