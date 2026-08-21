"""Standalone Kaggle / GPU Training Script for Telugu Handwritten Character Recognition.

Designed for high-throughput GPU training on Kaggle P100 / T4:
- GIL-free tf.data loading with tf_canonical_preprocess
- Mixed precision float16 execution
- Warmup cosine decay learning rate schedule
- Robust per-epoch checkpointing with auto-resume
- Final test set evaluation and artifact packaging
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import tensorflow as tf
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import create_splits
from src.data.dataset import build_dataset
from src.models.custom_cnn import build_custom_cnn, compile_custom_cnn
from src.models.backbone import WarmupCosineDecay
from src.checkpointing import CheckpointManager, CheckpointCallback
from src.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Kaggle P100 / GPU Training Runner")
    parser.add_argument("--data-dir", default="/kaggle/input/telugu-handwritten-character-dataset", help="Path to unzipped dataset root")
    parser.add_argument("--config", default="configs/track_b_custom_cnn.yaml", help="Path to experiment config")
    parser.add_argument("--output-dir", default="outputs/", help="Directory for split manifests and reports")
    parser.add_argument("--checkpoint-dir", default="checkpoints/", help="Directory for saved model checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Run a single-batch smoke test")
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if "_base_" in config:
        base_path = Path(args.config).parent / config["_base_"]
        with open(base_path, "r") as f:
            base_config = yaml.safe_load(f)
        merged = {**base_config, **config}
        del merged["_base_"]
        config = merged

    # Set random seed
    seed = config.get("seed", 42)
    tf.keras.utils.set_random_seed(seed)

    # Enable Mixed Precision for GPU speedup
    if config.get("mixed_precision", True):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[Setup] Mixed precision (mixed_float16) enabled.")

    # Workaround for TF layout optimizer bug on P100
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Check if splits exist, else generate them
    train_csv = out_path / "train.csv"
    if not train_csv.exists() and os.path.exists(args.data_dir):
        print(f"[Data] Generating stratified splits from {args.data_dir}...")
        create_splits(args.data_dir, output_dir=args.output_dir, project_root=".")

    with open(out_path / "label_map.json", "r") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    print(f"[Data] Total classes: {num_classes}")

    # Build tf.data datasets
    train_ds = build_dataset(str(out_path / "train.csv"), label_map, config, training=True)
    val_ds = build_dataset(str(out_path / "val.csv"), label_map, config, training=False)
    test_ds = build_dataset(str(out_path / "test.csv"), label_map, config, training=False)

    if args.dry_run:
        print("[Setup] Dry-run mode: using 1 batch per epoch.")
        train_ds = train_ds.take(1)
        val_ds = val_ds.take(1)

    # Build model
    img_size = config.get("image_size", 96)
    num_channels = config.get("num_channels", 1)
    input_shape = (img_size, img_size, num_channels)

    model = build_custom_cnn(num_classes=num_classes, input_shape=input_shape, config=config["model"])
    model.summary()

    # Learning rate schedule
    training_cfg = config.get("training", {})
    epochs = 1 if args.dry_run else training_cfg.get("epochs", 80)
    initial_lr = training_cfg.get("initial_lr", 1e-3)
    weight_decay = training_cfg.get("weight_decay", 1e-4)
    label_smoothing = config.get("label_smoothing", 0.05)

    steps_per_epoch = 1 if args.dry_run else max(1, 234000 // config.get("batch_size", 64))
    total_steps = epochs * steps_per_epoch
    warmup_steps = training_cfg.get("warmup_epochs", 5) * steps_per_epoch

    lr_schedule = WarmupCosineDecay(initial_lr, warmup_steps, total_steps)
    compile_custom_cnn(model, lr=lr_schedule, weight_decay=weight_decay, label_smoothing=label_smoothing)

    # Callbacks
    ckpt_manager = CheckpointManager(args.checkpoint_dir, config.get("experiment_name", "track_b_custom_cnn"))
    ckpt_callback = CheckpointCallback(ckpt_manager, config)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=training_cfg.get("early_stopping_patience", 10),
        restore_best_weights=True
    )

    print(f"\n[Train] Starting training for {epochs} epochs on {device_name()}...")
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[ckpt_callback, early_stopping]
    )
    elapsed = time.time() - t0
    print(f"\n[Train] Training complete in {elapsed/60:.2f} minutes.")

    # Save final best model
    best_model_path = Path(args.checkpoint_dir) / "track_b_best.keras"
    model.save(str(best_model_path))
    print(f"[Artifact] Saved final model to {best_model_path}")

    # Evaluate on held-out test set
    eval_results = evaluate(
        model_path=str(best_model_path),
        data_csv=str(out_path / "test.csv"),
        label_map_path=str(out_path / "label_map.json"),
        config_path=args.config,
        output_dir="reports/"
    )
    print(f"\n[Benchmark] Test Set Results: {eval_results}")


def device_name():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return f"GPU ({gpus[0].name})"
    return "CPU"


if __name__ == "__main__":
    main()
