"""Standalone Kaggle GPU Runner for Multi-Task Hierarchical Telugu HCR.

Trains the 3-Head Multi-Task network (Base Letter, Vowel Modifier, Subscript Conjunct)
on Kaggle P100 / T4 GPU or local machine.
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import tensorflow as tf

# Fix import path for Kaggle or local execution
PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.data.preprocess import tf_canonical_preprocess
from src.data.decomposition import decompose_class_name, export_grapheme_maps
from src.models.hierarchical_net import build_multitask_model, compile_multitask_model


def resolve_data_paths() -> Tuple[str, str, str]:
    """Finds train.csv, val.csv, and dataset root in Kaggle or local directory."""
    kaggle_paths = [
        Path("/kaggle/input/telugu-hcr-v3"),
        Path("/kaggle/input/telugu-dataset"),
        Path("/kaggle/input"),
        PROJ_ROOT,
    ]
    
    train_csv = None
    val_csv = None
    
    for base in kaggle_paths:
        candidate_train = base / "outputs" / "train.csv"
        candidate_val = base / "outputs" / "val.csv"
        if candidate_train.exists() and candidate_val.exists():
            train_csv = str(candidate_train)
            val_csv = str(candidate_val)
            print(f"✅ Found split manifests at: {base / 'outputs'}")
            break

    if train_csv is None:
        train_csv = str(PROJ_ROOT / "outputs" / "train.csv")
        val_csv = str(PROJ_ROOT / "outputs" / "val.csv")

    return train_csv, val_csv


def build_multitask_dataset(
    csv_path: str,
    img_size: int = 96,
    num_channels: int = 1,
    batch_size: int = 64,
    training: bool = True,
) -> tf.data.Dataset:
    """Builds a tf.data.Dataset yielding (image_tensor, {base, modifier, vattu})."""
    filepaths = []
    base_labels = []
    mod_labels = []
    vattu_labels = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fpath = row["filepath"]
            cname = row["class_name"]
            b_idx, m_idx, v_idx = decompose_class_name(cname)
            
            filepaths.append(fpath)
            base_labels.append(b_idx)
            mod_labels.append(m_idx)
            vattu_labels.append(v_idx)
            
    num_samples = len(filepaths)
    print(f"Loaded {num_samples:,} samples from {Path(csv_path).name}")

    ds = tf.data.Dataset.from_tensor_slices((
        filepaths,
        base_labels,
        mod_labels,
        vattu_labels,
    ))

    def load_and_preprocess(path, b_lbl, m_lbl, v_lbl):
        img_raw = tf.io.read_file(path)
        img = tf_canonical_preprocess(
            img_raw,
            img_size=img_size,
            num_channels=num_channels,
            normalize_mode="rescale",
        )
        targets = {
            "base_output": tf.one_hot(b_lbl, depth=52),
            "modifier_output": tf.one_hot(m_lbl, depth=16),
            "vattu_output": tf.one_hot(v_lbl, depth=36),
        }
        return img, targets

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        # Augmentation with slight translation, rotation, and stroke jitter
        def augment(img, targets):
            # Random shift (+-4%)
            pad = 4
            img_pad = tf.pad(img, [[pad, pad], [pad, pad], [0, 0]], mode="CONSTANT", constant_values=1.0)
            img_aug = tf.image.random_crop(img_pad, [img_size, img_size, num_channels])
            return img_aug, targets

        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=min(num_samples, 5000))

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def run_training(
    epochs: int = 30,
    batch_size: int = 128,
    backbone: str = "custom_cnn",
    img_size: int = 96,
    lr: float = 1e-3,
    dry_run: bool = False,
):
    """Executes the Multi-Task training pipeline."""
    print("=" * 70)
    print("TELUGU HCR v3 — MULTI-TASK HIERARCHICAL TRAINING RUNNER (FAST GPU PROFILE)")
    print("=" * 70)
    
    # GPU detection
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"🚀 GPU Active: {gpus}")
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("⚡ Mixed precision (mixed_float16) enabled for 2.5x GPU throughput.")
        except Exception:
            pass
    else:
        print("💻 Running on CPU.")

    export_grapheme_maps("outputs/grapheme_maps.json")
    train_csv, val_csv = resolve_data_paths()

    train_ds = build_multitask_dataset(train_csv, img_size=img_size, batch_size=batch_size, training=True)
    val_ds = build_multitask_dataset(val_csv, img_size=img_size, batch_size=batch_size, training=False)

    if dry_run:
        print("🧪 Dry run: limiting datasets to 2 batches.")
        train_ds = train_ds.take(2)
        val_ds = val_ds.take(2)
        epochs = 1

    model = build_multitask_model(
        backbone_type=backbone,
        input_shape=(img_size, img_size, 1),
    )
    compile_multitask_model(model, learning_rate=lr)
    model.summary()

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "hierarchical_best.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_base_output_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_base_output_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        )
    ]

    print(f"\n🚀 Starting Multi-Task Training ({epochs} epochs, batch_size={batch_size})...")
    print(f"⏱️ Estimated runtime on Kaggle GPU: ~20 to 25 minutes total.")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    print(f"\n🎉 Training complete! Best model saved to: {best_model_path}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--backbone", type=str, default="custom_cnn", choices=["custom_cnn", "efficientnetb0"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        backbone=args.backbone,
        dry_run=args.dry_run,
    )
