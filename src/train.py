import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import tensorflow as tf
import yaml

from src.checkpointing import CheckpointCallback, CheckpointManager
from src.data.dataset import build_dataset, get_class_weights
from src.models.backbone import (
    WarmupCosineDecay,
    build_transfer_model,
    compile_model,
    freeze_backbone,
    unfreeze_top_layers,
)
from src.models.custom_cnn import build_custom_cnn, compile_custom_cnn

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "_base_" in config:
        base_path = Path(config_path).parent / config["_base_"]
        with open(base_path, "r") as f:
            base_config = yaml.safe_load(f)
        merged = {**base_config, **config}
        del merged["_base_"]
        config = merged

    return config

def train(config_path: str, resume: bool = True, dry_run: bool = False) -> Dict[str, Any]:
    config = load_config(config_path)

    seed = config.get("seed", 42)
    tf.keras.utils.set_random_seed(seed)

    if config.get("mixed_precision", True):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    if not config.get("layout_optimizer", False):
        tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

    output_dir = Path(config.get("output_dir", "outputs/"))
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints/"))
    experiment_name = config.get("experiment_name", "experiment")

    with open(output_dir / "label_map.json", "r") as f:
        label_map = json.load(f)

    num_classes = len(label_map)
    train_csv = str(output_dir / "train.csv")
    val_csv = str(output_dir / "val.csv")

    train_ds = build_dataset(train_csv, label_map, config, training=True)
    val_ds = build_dataset(val_csv, label_map, config, training=False)

    if dry_run:
        train_ds = train_ds.take(1)
        val_ds = val_ds.take(1)

    model_type = config.get("model", {}).get("type", "transfer")
    ckpt_manager = CheckpointManager(str(checkpoint_dir), experiment_name)
    ckpt_callback = CheckpointCallback(ckpt_manager, config)

    if model_type == "transfer":
        backbone_name = config["model"].get("backbone", "efficientnetb0")
        img_size = config.get("image_size", 128)
        num_channels = config.get("num_channels", 3)
        input_shape = (img_size, img_size, num_channels)

        model = build_transfer_model(
            num_classes=num_classes,
            backbone=backbone_name,
            input_shape=input_shape,
            config=config["model"],
        )

        training_cfg = config.get("training", {})
        head_epochs = 1 if dry_run else training_cfg.get("head_epochs", 8)
        head_lr = training_cfg.get("head_lr", 1e-3)
        weight_decay = training_cfg.get("weight_decay", 1e-4)
        label_smoothing = config.get("label_smoothing", 0.1)

        freeze_backbone(model)
        compile_model(model, lr=head_lr, weight_decay=weight_decay, label_smoothing=label_smoothing)
        model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, callbacks=[ckpt_callback])

        finetune_epochs = 1 if dry_run else training_cfg.get("finetune_epochs", 30)
        finetune_lr = training_cfg.get("finetune_lr", 1e-4)
        unfreeze_fraction = config["model"].get("unfreeze_fraction", 0.33)
        unfreeze_top_layers(model, fraction=unfreeze_fraction)

        total_steps = finetune_epochs * 100
        warmup_steps = training_cfg.get("warmup_epochs", 5) * 100
        lr_schedule = WarmupCosineDecay(finetune_lr, warmup_steps, total_steps)

        compile_model(model, lr=lr_schedule, weight_decay=weight_decay, label_smoothing=label_smoothing)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=training_cfg.get("early_stopping_patience", 10),
            restore_best_weights=True,
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=head_epochs + finetune_epochs,
            initial_epoch=head_epochs,
            callbacks=[ckpt_callback, early_stopping],
        )

    elif model_type == "custom_cnn":
        img_size = config.get("image_size", 96)
        num_channels = config.get("num_channels", 1)
        input_shape = (img_size, img_size, num_channels)

        model = build_custom_cnn(
            num_classes=num_classes,
            input_shape=input_shape,
            config=config["model"],
        )

        training_cfg = config.get("training", {})
        epochs = 1 if dry_run else training_cfg.get("epochs", 80)
        initial_lr = training_cfg.get("initial_lr", 1e-3)
        weight_decay = training_cfg.get("weight_decay", 1e-4)
        label_smoothing = config.get("label_smoothing", 0.05)

        total_steps = epochs * 100
        warmup_steps = training_cfg.get("warmup_epochs", 5) * 100
        lr_schedule = WarmupCosineDecay(initial_lr, warmup_steps, total_steps)

        compile_custom_cnn(model, lr=lr_schedule, weight_decay=weight_decay, label_smoothing=label_smoothing)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=training_cfg.get("early_stopping_patience", 10),
            restore_best_weights=True,
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[ckpt_callback, early_stopping],
        )

    return history.history if hasattr(history, "history") else {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train(args.config, resume=not args.no_resume, dry_run=args.dry_run)
