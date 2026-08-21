import math
from typing import Dict, Any, Tuple
import tensorflow as tf

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.initial_lr = float(initial_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        decay_steps = tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        decay_step = tf.minimum(tf.maximum(step - self.warmup_steps, 0.0), decay_steps)
        cosine_decay = 0.5 * (1.0 + tf.math.cos(math.pi * decay_step / decay_steps))
        decay_lr = self.initial_lr * cosine_decay
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: decay_lr)

    def get_config(self) -> Dict[str, Any]:
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }

def build_transfer_model(
    num_classes: int,
    backbone: str = "efficientnetb0",
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    config: Dict[str, Any] = None,
) -> tf.keras.Model:
    if config is None:
        config = {}

    head_config = config.get("head", {})
    dropout_1 = head_config.get("dropout_1", 0.3)
    dense_units = head_config.get("dense_units", 512)
    dropout_2 = head_config.get("dropout_2", 0.3)

    inputs = tf.keras.Input(shape=input_shape)

    if backbone.lower() == "efficientnetb0":
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
    elif backbone.lower() == "mobilenetv3large":
        base_model = tf.keras.applications.MobileNetV3Large(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_1)(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def freeze_backbone(model: tf.keras.Model) -> None:
    for layer in model.layers[:-5]:
        layer.trainable = False

def unfreeze_top_layers(model: tf.keras.Model, fraction: float = 0.33) -> None:
    total_layers = len(model.layers)
    num_to_unfreeze = int(total_layers * fraction)
    for layer in model.layers[:-num_to_unfreeze]:
        layer.trainable = False
    for layer in model.layers[-num_to_unfreeze:]:
        layer.trainable = True

def compile_model(
    model: tf.keras.Model,
    lr: Any,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
) -> None:
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=weight_decay,
    )
    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing,
    )
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )
