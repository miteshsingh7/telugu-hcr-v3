"""Unified training script for Telugu HCR — Track A (transfer) and Track B (custom CNN).

Usage:
    python -m src.train --config configs/track_a_efficientnet.yaml
    python -m src.train --config configs/track_b_custom_cnn.yaml --no-resume
    python -m src.train --config configs/track_a_efficientnet.yaml --dry-run
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import yaml
import tensorflow as tf

from src.checkpointing import CheckpointManager, CheckpointCallback
from src.data.dataset import build_dataset, get_class_weights
from src.models.backbone import (
    build_transfer_model,
    freeze_backbone,
    unfreeze_top_layers,
    compile_model as compile_transfer_model,
    WarmupCosineDecay,
)
from src.models.custom_cnn import build_custom_cnn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading with _base_ inheritance
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML config, resolving ``_base_`` inheritance.

    If the config contains a ``_base_`` key, the base YAML is loaded first
    and the experiment config is deep-merged on top.

    Args:
        config_path: Path to the experiment YAML file.

    Returns:
        Merged configuration dictionary.
    """
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
# Main training entry-point
# ---------------------------------------------------------------------------

def train(config_path: str, resume: bool = True, dry_run: bool = False) -> dict:
    """Run training for Track A or Track B based on the config.

    Args:
        config_path: Path to the experiment YAML config.
        resume: If ``True``, resume from the latest checkpoint.
        dry_run: If ``True``, run a single batch to verify the pipeline.

    Returns:
        Dictionary of training history.
    """
    config = load_config(config_path)

    # ------------------------------------------------------------------
    # 1. Global setup
    # ------------------------------------------------------------------
    seed = config.get("seed", 42)
    tf.keras.utils.set_random_seed(seed)

    if config.get("mixed_precision", False):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision (mixed_float16) enabled.")

    # Grappler layout-optimizer bug workaround (Kaggle P100)
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

    gpus = tf.config.list_physical_devices("GPU")
    logger.info(f"GPUs detected: {[g.name for g in gpus]}" if gpus else "No GPUs detected — running on CPU.")

    # ------------------------------------------------------------------
    # 2. Data
    # ------------------------------------------------------------------
    output_dir = Path(config["output_dir"])
    train_csv = str(output_dir / "train.csv")
    val_csv = str(output_dir / "val.csv")
    label_map_path = output_dir / "label_map.json"

    with open(label_map_path, "r") as f:
        label_map: Dict[str, int] = json.load(f)
    num_classes = len(label_map)
    logger.info(f"Loaded label map with {num_classes} classes.")

    train_ds = build_dataset(train_csv, label_map, config, training=True)
    val_ds = build_dataset(val_csv, label_map, config, training=False)

    # Compute steps_per_epoch from the dataset cardinality (fall back to
    # a config override if the dataset is infinite / unknown).
    cardinality = tf.data.experimental.cardinality(train_ds).numpy()
    steps_per_epoch = int(cardinality) if cardinality > 0 else config["training"].get("steps_per_epoch", 500)
    logger.info(f"Steps per epoch: {steps_per_epoch}")

    # Class weights
    class_weight = None
    if config["training"].get("class_weight", False):
        class_weight = get_class_weights(train_csv)
        logger.info(f"Using class weights ({len(class_weight)} classes).")

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    model_cfg = config["model"]
    model_type = model_cfg["type"]

    if model_type == "transfer":
        backbone_name = model_cfg.get("backbone", "efficientnetb0")
        img_size = config.get("image_size", 128)
        num_ch = config.get("num_channels", 3)
        model = build_transfer_model(
            num_classes=num_classes,
            backbone=backbone_name,
            input_shape=(img_size, img_size, num_ch),
            config=model_cfg,
        )
    elif model_type == "custom_cnn":
        img_size = config.get("image_size", 128)
        num_ch = model_cfg.get("num_channels", config.get("num_channels", 1))
        # Extract filter list from conv_blocks (list of dicts or list of ints)
        conv_blocks_raw = model_cfg.get("conv_blocks", [32, 64, 128, 128, 256, 256])
        if conv_blocks_raw and isinstance(conv_blocks_raw[0], dict):
            filters = [b["filters"] for b in conv_blocks_raw]
        else:
            filters = conv_blocks_raw
        cnn_config = {"conv_blocks": filters}
        cnn_config.update({k: v for k, v in model_cfg.get("head", {}).items()})
        model = build_custom_cnn(
            num_classes=num_classes,
            input_shape=(img_size, img_size, num_ch),
            config=cnn_config,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.summary(print_fn=logger.info)

    # ------------------------------------------------------------------
    # 4. Checkpointing & auto-resume
    # ------------------------------------------------------------------
    chkpt_dir = config.get("checkpoint_dir", "checkpoints")
    experiment_name = config.get("experiment_name", "exp")
    manager = CheckpointManager(
        checkpoint_dir=chkpt_dir,
        experiment_name=experiment_name,
        kaggle_dataset_slug=config.get("kaggle_dataset_slug"),
    )

    initial_epoch = 0
    if resume:
        state = manager.load_latest_checkpoint(model)
        if state is not None:
            initial_epoch = state["epoch"]
            logger.info(f"Resumed from epoch {initial_epoch}")
        else:
            logger.info("No checkpoint found — starting from scratch.")

    # ------------------------------------------------------------------
    # 5. Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        CheckpointCallback(manager, config),
        tf.keras.callbacks.TensorBoard(log_dir=f"logs/{experiment_name}"),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["training"].get("early_stopping_patience", 10),
            restore_best_weights=True,
        ),
    ]

    # ------------------------------------------------------------------
    # Dry-run: one batch forward + backward, then exit
    # ------------------------------------------------------------------
    if dry_run:
        logger.info("DRY RUN — running 1 training step to verify pipeline.")
        label_smoothing = config.get("label_smoothing", 0.1)
        weight_decay = config["training"].get("weight_decay", 1e-4)
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=weight_decay),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
            metrics=["accuracy"],
        )
        model.fit(train_ds, epochs=1, steps_per_epoch=1)
        logger.info("Dry run complete — pipeline is functional.")
        return {"dry_run": True}

    # ------------------------------------------------------------------
    # 6. Training
    # ------------------------------------------------------------------
    history: Dict[str, Any] = {}
    label_smoothing = config.get("label_smoothing", 0.1)
    weight_decay = config["training"].get("weight_decay", 1e-4)
    warmup_epochs = config["training"].get("warmup_epochs", 5)

    if model_type == "transfer":
        # ---- Track A: two-phase training ----
        head_epochs = config["training"]["head_epochs"]
        finetune_epochs = config["training"]["finetune_epochs"]
        total_epochs = head_epochs + finetune_epochs

        # Phase 1 — Train head only (backbone frozen)
        if initial_epoch < head_epochs:
            logger.info("=== Phase 1: Training head only (backbone frozen) ===")
            freeze_backbone(model)

            head_lr = config["training"]["head_lr"]
            phase1_total_steps = head_epochs * steps_per_epoch
            lr_schedule = WarmupCosineDecay(
                initial_learning_rate=head_lr,
                warmup_steps=warmup_epochs * steps_per_epoch,
                total_steps=phase1_total_steps,
            )

            model.compile(
                optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay),
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
                metrics=["accuracy"],
            )

            hist = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=head_epochs,
                initial_epoch=initial_epoch,
                class_weight=class_weight,
                callbacks=callbacks,
            )
            history["phase1"] = hist.history
            initial_epoch = head_epochs

        # Phase 2 — Fine-tune top 1/3 of backbone
        if initial_epoch < total_epochs:
            logger.info("=== Phase 2: Fine-tuning top 1/3 of backbone ===")
            fraction = config["model"].get("unfreeze_fraction", 0.33)
            unfreeze_top_layers(model, fraction=fraction)

            finetune_lr = config["training"]["finetune_lr"]
            phase2_total_steps = finetune_epochs * steps_per_epoch
            lr_schedule = WarmupCosineDecay(
                initial_learning_rate=finetune_lr,
                warmup_steps=min(warmup_epochs, 3) * steps_per_epoch,
                total_steps=phase2_total_steps,
            )

            model.compile(
                optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay),
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
                metrics=["accuracy"],
            )

            hist = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=total_epochs,
                initial_epoch=initial_epoch,
                class_weight=class_weight,
                callbacks=callbacks,
            )
            history["phase2"] = hist.history

    else:
        # ---- Track B: single-phase training ----
        epochs = config["training"]["epochs"]
        initial_lr = config["training"]["initial_lr"]

        if initial_epoch < epochs:
            logger.info("=== Track B: Single-phase training ===")
            total_steps = epochs * steps_per_epoch
            lr_schedule = WarmupCosineDecay(
                initial_learning_rate=initial_lr,
                warmup_steps=warmup_epochs * steps_per_epoch,
                total_steps=total_steps,
            )

            model.compile(
                optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay),
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
                metrics=["accuracy"],
            )

            hist = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                initial_epoch=initial_epoch,
                class_weight=class_weight,
                callbacks=callbacks,
            )
            history["train"] = hist.history

    logger.info("Training complete.")
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train Telugu HCR Model")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 batch to verify pipeline")
    args = parser.parse_args()

    train(args.config, resume=not args.no_resume, dry_run=args.dry_run)
