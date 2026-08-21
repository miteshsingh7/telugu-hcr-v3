import math
import tensorflow as tf
from typing import Dict, Any, Tuple, Optional

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with linear warmup and cosine decay."""
    
    def __init__(self, initial_learning_rate: float, warmup_steps: int, total_steps: int):
        """Initializes the WarmupCosineDecay schedule.
        
        Args:
            initial_learning_rate: The peak learning rate after warmup.
            warmup_steps: Number of warmup steps.
            total_steps: Total number of steps.
        """
        super().__init__()
        self.initial_learning_rate = tf.cast(initial_learning_rate, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        
    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        
        # Warmup phase
        warmup_lr = self.initial_learning_rate * (step / tf.maximum(self.warmup_steps, 1.0))
        
        # Cosine decay phase
        decay_steps = tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        decay_step = tf.minimum(tf.maximum(step - self.warmup_steps, 0.0), decay_steps)
        cosine_decay = 0.5 * (1.0 + tf.math.cos(tf.constant(math.pi) * decay_step / decay_steps))
        decay_lr = self.initial_learning_rate * cosine_decay
        
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: decay_lr)
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "initial_learning_rate": float(self.initial_learning_rate),
            "warmup_steps": float(self.warmup_steps),
            "total_steps": float(self.total_steps),
        }

def get_lr_schedule(total_steps: int, initial_lr: float, warmup_epochs: int, steps_per_epoch: int) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """Creates a learning rate schedule with linear warmup and cosine decay.
    
    Args:
        total_steps: Total number of training steps.
        initial_lr: Peak learning rate after warmup.
        warmup_epochs: Number of epochs for linear warmup.
        steps_per_epoch: Number of steps in one epoch.
        
    Returns:
        A LearningRateSchedule instance.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    return WarmupCosineDecay(initial_lr, warmup_steps, total_steps)

def build_transfer_model(num_classes: int, backbone: str = 'efficientnetb0', input_shape: tuple = (128, 128, 3), config: dict = None) -> tf.keras.Model:
    """Builds a transfer learning model for Phase 1 Track A.
    
    Args:
        num_classes: Number of output classes.
        backbone: The backbone architecture ('efficientnetb0' or 'mobilenetv3large').
        input_shape: Shape of the input images.
        config: Configuration dictionary for the head.
        
    Returns:
        A tf.keras.Model instance.
    """
    if config is None:
        config = {}
        
    head_config = config.get("head", {})
    dropout_1 = head_config.get("dropout_1", 0.3)
    dense_units = head_config.get("dense_units", 512)
    dropout_2 = head_config.get("dropout_2", 0.3)
    
    inputs = tf.keras.Input(shape=input_shape)
    
    if backbone == 'efficientnetb0':
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    elif backbone == 'mobilenetv3large':
        base_model = tf.keras.applications.MobileNetV3Large(include_top=False, weights='imagenet', input_tensor=inputs)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
        
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_1)(x)
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def freeze_backbone(model: tf.keras.Model) -> None:
    """Freezes all backbone layers of the model.
    
    Args:
        model: The model to modify.
    """
    for layer in model.layers:
        if 'global_average_pooling2d' in layer.name:
            break
        layer.trainable = False

def unfreeze_top_layers(model: tf.keras.Model, fraction: float = 0.33) -> None:
    """Unfreezes the top fraction of backbone layers.
    
    Args:
        model: The model to modify.
        fraction: The fraction of top layers to unfreeze.
    """
    base_layers = []
    for layer in model.layers:
        if 'global_average_pooling2d' in layer.name:
            break
        base_layers.append(layer)
        
    num_unfreeze = int(len(base_layers) * fraction)
    for layer in base_layers[-num_unfreeze:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

def compile_model(model: tf.keras.Model, lr: float, weight_decay: float = 1e-4, label_smoothing: float = 0.1, phase: str = 'head') -> None:
    """Compiles the model with AdamW and CategoricalCrossentropy.
    
    Args:
        model: The model to compile.
        lr: Learning rate or learning rate schedule.
        weight_decay: Weight decay factor for AdamW.
        label_smoothing: Label smoothing factor.
        phase: Training phase string (for logging/clarity).
    """
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
