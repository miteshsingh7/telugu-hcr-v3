"""Multi-Task Hierarchical Neural Network Architecture for Telugu HCR.

Decomposes the 630-class problem into 3 multi-task classification heads:
- Head 1 (Base Akshara): 52 classes
- Head 2 (Gunintham Modifier): 16 classes
- Head 3 (Othulu Conjunct): 36 classes
"""

from typing import Dict, Any, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_multitask_model(
    backbone_type: str = "custom_cnn",
    input_shape: Tuple[int, int, int] = (96, 96, 1),
    num_base_classes: int = 52,
    num_modifier_classes: int = 16,
    num_vattu_classes: int = 36,
    dropout_rate: float = 0.35,
) -> tf.keras.Model:
    """Builds a Multi-Task Deep Neural Network with shared feature extraction.
    
    Args:
        backbone_type: 'custom_cnn', 'efficientnetb0', or 'mobilenetv3'
        input_shape: (H, W, C)
        num_base_classes: 52
        num_modifier_classes: 16
        num_vattu_classes: 36
        dropout_rate: Dropout fraction before prediction heads
        
    Returns:
        tf.keras.Model with 3 named outputs: 'base_output', 'modifier_output', 'vattu_output'.
    """
    inputs = layers.Input(shape=input_shape, name="image_input")

    if backbone_type == "custom_cnn":
        # 6-layer Conv-BN-ReLU Block with progressive channel expansion
        x = inputs
        filters = [32, 64, 128, 128, 256, 256]
        for idx, f in enumerate(filters):
            x = layers.Conv2D(f, 3, padding="same", name=f"conv_{idx+1}")(x)
            x = layers.BatchNormalization(name=f"bn_{idx+1}")(x)
            x = layers.ReLU(name=f"relu_{idx+1}")(x)
            if idx in (0, 1, 3, 5):
                x = layers.MaxPooling2D(2, 2, name=f"pool_{idx+1}")(x)

        shared_features = layers.GlobalAveragePooling2D(name="shared_gap")(x)
        shared_features = layers.Dropout(dropout_rate, name="shared_dropout")(shared_features)
        
    elif backbone_type == "efficientnetb0":
        # Handle 1-channel to 3-channel adaptation for pre-trained weights
        if input_shape[-1] == 1:
            x_rgb = layers.Concatenate(name="repeat_channels")([inputs, inputs, inputs])
        else:
            x_rgb = inputs

        base_net = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_tensor=x_rgb,
        )
        base_net.trainable = True
        shared_features = layers.GlobalAveragePooling2D(name="shared_gap")(base_net.output)
        shared_features = layers.Dropout(dropout_rate, name="shared_dropout")(shared_features)

    else:
        raise ValueError(f"Unknown backbone: {backbone_type}")

    # Shared dense representation
    shared_dense = layers.Dense(384, activation="relu", name="shared_dense")(shared_features)
    shared_dense = layers.Dropout(0.25, name="shared_dense_drop")(shared_dense)

    # 1. Base Akshara Head (Weight: 1.0)
    base_dense = layers.Dense(192, activation="relu", name="base_dense")(shared_dense)
    base_output = layers.Dense(num_base_classes, activation="softmax", dtype="float32", name="base_output")(base_dense)

    # 2. Gunintham Modifier Head (Weight: 0.5)
    mod_dense = layers.Dense(96, activation="relu", name="mod_dense")(shared_dense)
    mod_output = layers.Dense(num_modifier_classes, activation="softmax", dtype="float32", name="modifier_output")(mod_dense)

    # 3. Othulu Conjunct Head (Weight: 0.5)
    vattu_dense = layers.Dense(96, activation="relu", name="vattu_dense")(shared_dense)
    vattu_output = layers.Dense(num_vattu_classes, activation="softmax", dtype="float32", name="vattu_output")(vattu_dense)

    model = models.Model(
        inputs=inputs,
        outputs=[base_output, mod_output, vattu_output],
        name=f"telugu_multitask_{backbone_type}"
    )
    return model


def compile_multitask_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
) -> tf.keras.Model:
    """Compiles the multi-task model with calibrated loss weights."""
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    
    losses = {
        "base_output": tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        "modifier_output": tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        "vattu_output": tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
    }
    
    loss_weights = {
        "base_output": 1.0,
        "modifier_output": 0.5,
        "vattu_output": 0.5,
    }
    
    metrics = {
        "base_output": ["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
        "modifier_output": ["accuracy"],
        "vattu_output": ["accuracy"],
    }
    
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    return model
