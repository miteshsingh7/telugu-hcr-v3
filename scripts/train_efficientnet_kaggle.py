"""Standalone Kaggle GPU Runner for Track A: Transfer Learning (MobileNetV2 / EfficientNet).

Trains pre-trained ImageNet backbones with 2-phase fine-tuning on Kaggle GPU.
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.data.preprocess import tf_canonical_preprocess


def build_transfer_track_a(
    backbone: str = "mobilenetv2",
    num_classes: int = 630,
    img_size: int = 128,
) -> tf.keras.Model:
    """Builds a pre-trained transfer learning model for 630 Telugu Akshara classes."""
    inputs = layers.Input(shape=(img_size, img_size, 3), name="image_input")

    if backbone == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(img_size, img_size, 3),
        )
    elif backbone == "efficientnetb0":
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(img_size, img_size, 3),
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    base_model.trainable = False

    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.35, name="drop1")(x)
    x = layers.Dense(512, activation="relu", name="dense1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(0.35, name="drop2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"telugu_{backbone}")
    return model


def build_transfer_dataset(
    csv_path: str,
    img_size: int = 128,
    batch_size: int = 64,
    training: bool = True,
) -> tf.data.Dataset:
    filepaths = []
    labels = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepaths.append(row["filepath"])
            labels.append(int(row["label_idx"]))

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    def load_and_preprocess(path, label):
        img_raw = tf.io.read_file(path)
        img = tf_canonical_preprocess(
            img_raw,
            img_size=img_size,
            num_channels=3,
            normalize_mode="imagenet",
        )
        return img, tf.one_hot(label, depth=630)

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        def augment(img, label):
            pad = 6
            img_pad = tf.pad(img, [[pad, pad], [pad, pad], [0, 0]], mode="CONSTANT", constant_values=1.0)
            img_aug = tf.image.random_crop(img_pad, [img_size, img_size, 3])
            return img_aug, label

        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=min(len(filepaths), 5000))

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def run_transfer_training(
    backbone: str = "mobilenetv2",
    head_epochs: int = 6,
    finetune_epochs: int = 24,
    batch_size: int = 64,
    img_size: int = 128,
    dry_run: bool = False,
):
    print("=" * 70)
    print(f"TELUGU HCR v3 — TRACK A: {backbone.upper()} TRANSFER LEARNING RUNNER")
    print("=" * 70)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"🚀 GPU Active: {gpus}")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        print("💻 Running on CPU.")

    train_csv = "outputs/train.csv"
    val_csv = "outputs/val.csv"

    train_ds = build_transfer_dataset(train_csv, img_size=img_size, batch_size=batch_size, training=True)
    val_ds = build_transfer_dataset(val_csv, img_size=img_size, batch_size=batch_size, training=False)

    if dry_run:
        print("🧪 Dry run: 1 batch only.")
        train_ds = train_ds.take(2)
        val_ds = val_ds.take(2)
        head_epochs = 1
        finetune_epochs = 0

    model = build_transfer_track_a(backbone=backbone, num_classes=630, img_size=img_size)

    # Phase 1: Train classification head
    print(f"\n--- Phase 1: Training Classification Head ({head_epochs} epochs) ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=head_epochs)

    # Phase 2: Fine-tune backbone
    if finetune_epochs > 0:
        print(f"\n--- Phase 2: Fine-Tuning Backbone ({finetune_epochs} epochs) ---")
        base_net = model.get_layer(backbone)
        base_net.trainable = True
        # Unfreeze top 30 layers
        for layer in base_net.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
        )

        checkpoint_path = f"checkpoints/track_a_{backbone}_best.keras"
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=6,
                restore_best_weights=True,
            ),
        ]
        model.fit(train_ds, validation_data=val_ds, epochs=finetune_epochs, callbacks=callbacks)
        print(f"\n🎉 Track A complete! Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="mobilenetv2", choices=["mobilenetv2", "efficientnetb0"])
    parser.add_argument("--head-epochs", type=int, default=6)
    parser.add_argument("--finetune-epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_transfer_training(
        backbone=args.backbone,
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
