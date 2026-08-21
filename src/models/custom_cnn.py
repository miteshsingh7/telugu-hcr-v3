from typing import Dict, Any, Tuple
import tensorflow as tf
from src.models.backbone import WarmupCosineDecay

def build_custom_cnn(
    num_classes: int,
    input_shape: Tuple[int, ...] = (96, 96, 1),
    config: Dict[str, Any] = None,
) -> tf.keras.Model:
    if config is None:
        config = {}

    filters = config.get("conv_blocks", [32, 64, 128, 128, 256, 256])

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for f in filters:
        x = tf.keras.layers.Conv2D(f, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def compile_custom_cnn(
    model: tf.keras.Model,
    lr: Any,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
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
