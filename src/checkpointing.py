import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import tensorflow as tf

logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, checkpoint_dir: str, experiment_name: str):
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.experiment_name = experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_acc = -1.0
        self.best_epoch = -1

    def save_checkpoint(
        self,
        model: tf.keras.Model,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any] = None,
    ) -> Path:
        epoch_dir = self.checkpoint_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        model_path = epoch_dir / "model.keras"
        model.save(str(model_path))

        val_acc = metrics.get("val_accuracy", metrics.get("val_acc", 0.0))
        val_loss = metrics.get("val_loss", 0.0)

        is_best = val_acc > self.best_val_acc
        if is_best:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            best_dir = self.checkpoint_dir / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(best_dir / "model.keras"))

        state = {
            "epoch": epoch,
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss),
            "best_val_acc": float(self.best_val_acc),
            "best_epoch": self.best_epoch,
            "is_best": is_best,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(epoch_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        return model_path

class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, manager: CheckpointManager, config: Dict[str, Any] = None):
        super().__init__()
        self.manager = manager
        self.config = config

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        logs = logs or {}
        self.manager.save_checkpoint(self.model, epoch + 1, logs, self.config)
