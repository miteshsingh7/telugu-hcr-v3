import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import datetime
import tensorflow as tf
import logging

try:
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    logging.warning("Kaggle API not found. push_to_kaggle will be skipped.")

class CheckpointManager:
    """Manages robust checkpointing for training, including Kaggle Dataset syncing.

    Attributes:
        checkpoint_dir: Base directory for checkpoints.
        experiment_name: Name of the current experiment.
        kaggle_dataset_slug: Optional Kaggle dataset slug (e.g., 'username/dataset-name') to push to.
    """

    def __init__(self, checkpoint_dir: str, experiment_name: str, kaggle_dataset_slug: str = None):
        """Initializes the CheckpointManager.

        Args:
            checkpoint_dir: Directory where checkpoints are saved.
            experiment_name: Name of the experiment.
            kaggle_dataset_slug: Slug for Kaggle dataset to upload checkpoints to.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.kaggle_dataset_slug = kaggle_dataset_slug
        self.exp_dir = self.checkpoint_dir / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')

    def save_checkpoint(
        self,
        model: tf.keras.Model,
        epoch: int,
        optimizer: tf.keras.optimizers.Optimizer,
        metrics: Dict[str, float],
        config: Dict[str, Any]
    ) -> None:
        """Saves a checkpoint for the current epoch.

        Args:
            model: The Keras model to save.
            epoch: The current epoch number.
            optimizer: The optimizer used for training.
            metrics: Dictionary of current metrics (must contain val_acc and val_loss if tracking best).
            config: Training configuration dictionary.
        """
        epoch_dir = self.exp_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(exist_ok=True, parents=True)

        # Save model (weights + optimizer state in .keras format)
        model_path = epoch_dir / "model.keras"
        model.save(model_path)

        # Update best metrics
        val_acc = metrics.get('val_accuracy', metrics.get('val_acc', 0.0))
        val_loss = metrics.get('val_loss', float('inf'))
        
        is_best_acc = val_acc > self.best_val_acc
        is_best_loss = val_loss < self.best_val_loss

        if is_best_acc:
            self.best_val_acc = val_acc
        if is_best_loss:
            self.best_val_loss = val_loss

        # Save training state
        state = {
            "epoch": epoch,
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "current_lr": float(optimizer.learning_rate.numpy()) if hasattr(optimizer, 'learning_rate') else None,
            "seed": config.get('seed', None),
            "config": config,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": metrics
        }
        
        with open(epoch_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=4)
            
        # Copy to best if it's the best validation accuracy
        if is_best_acc:
            best_dir = self.exp_dir / "best_model"
            best_dir.mkdir(exist_ok=True, parents=True)
            model.save(best_dir / "model.keras")
            with open(best_dir / "training_state.json", "w") as f:
                json.dump(state, f, indent=4)

    def load_latest_checkpoint(self, model: tf.keras.Model) -> Optional[Dict[str, Any]]:
        """Loads the latest checkpoint if available.

        Args:
            model: The model to load weights into.

        Returns:
            Dictionary containing the training state, or None if no checkpoint found.
        """
        epochs = []
        for d in self.exp_dir.glob("epoch_*"):
            if d.is_dir() and (d / "model.keras").exists() and (d / "training_state.json").exists():
                try:
                    ep = int(d.name.split("_")[-1])
                    epochs.append((ep, d))
                except ValueError:
                    continue

        if not epochs:
            return None

        epochs.sort(key=lambda x: x[0])
        latest_epoch_dir = epochs[-1][1]

        # Load weights
        model.load_weights(latest_epoch_dir / "model.keras")

        # Load state
        with open(latest_epoch_dir / "training_state.json", "r") as f:
            state = json.load(f)

        self.best_val_acc = state.get("best_val_acc", 0.0)
        self.best_val_loss = state.get("best_val_loss", float('inf'))

        return state

    def get_best_model_path(self) -> Optional[Path]:
        """Returns the path to the best model checkpoint."""
        best_model_path = self.exp_dir / "best_model" / "model.keras"
        if best_model_path.exists():
            return best_model_path
        return None

    def push_to_kaggle(self) -> None:
        """Pushes the experiment directory to Kaggle as a versioned dataset."""
        if not KAGGLE_AVAILABLE or not self.kaggle_dataset_slug:
            logging.info("Kaggle push skipped: Kaggle API not available or no slug provided.")
            return

        metadata_path = self.exp_dir / "dataset-metadata.json"
        
        # Create dataset metadata if it doesn't exist
        if not metadata_path.exists():
            parts = self.kaggle_dataset_slug.split('/')
            if len(parts) != 2:
                logging.error("Invalid kaggle_dataset_slug format. Expected 'username/dataset-name'.")
                return
                
            username, dataset_name = parts
            metadata = {
                "title": dataset_name,
                "id": self.kaggle_dataset_slug,
                "licenses": [{"name": "CC0-1.0"}]
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
        
        try:
            # Try to create, if exists it will fail, then we create new version
            kaggle.api.dataset_create_new(folder=str(self.exp_dir), dir_mode='zip')
            logging.info(f"Created new Kaggle dataset: {self.kaggle_dataset_slug}")
        except Exception as e:
            try:
                kaggle.api.dataset_create_version(
                    folder=str(self.exp_dir),
                    version_notes=f"Update {datetime.datetime.now().isoformat()}",
                    dir_mode='zip'
                )
                logging.info(f"Pushed new version to Kaggle dataset: {self.kaggle_dataset_slug}")
            except Exception as e2:
                logging.error(f"Failed to push to Kaggle: {e2}")

class CheckpointCallback(tf.keras.callbacks.Callback):
    """Keras Callback for checkpointing using CheckpointManager."""
    
    def __init__(self, manager: CheckpointManager, config: Dict[str, Any]):
        """Initializes the CheckpointCallback.
        
        Args:
            manager: Initialized CheckpointManager instance.
            config: Training configuration dictionary.
        """
        super().__init__()
        self.manager = manager
        self.config = config

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of an epoch to save a checkpoint."""
        logs = logs or {}
        self.manager.save_checkpoint(
            model=self.model,
            epoch=epoch + 1,  # 1-indexed for clarity in folders
            optimizer=self.model.optimizer,
            metrics=logs,
            config=self.config
        )
        
        # Optionally push to Kaggle every N epochs or on best
        if (epoch + 1) % self.config.get('kaggle_push_freq', 5) == 0:
            self.manager.push_to_kaggle()
