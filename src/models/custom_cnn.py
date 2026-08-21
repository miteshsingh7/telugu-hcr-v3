import tensorflow as tf
from typing import Dict, Any, Tuple, Optional

def build_custom_cnn(num_classes: int, input_shape: tuple = (128, 128, 1), config: dict = None) -> tf.keras.Model:
    """Builds a lean custom CNN model for Phase 1 Track B.
    
    Args:
        num_classes: Number of output classes.
        input_shape: Shape of the input images.
        config: Configuration dictionary for the CNN.
        
    Returns:
        A tf.keras.Model instance.
    """
    if config is None:
        config = {}
        
    filters = config.get("conv_blocks", [32, 64, 128, 128, 256, 256])
    
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    
    for f in filters:
        x = tf.keras.layers.Conv2D(f, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)
        
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def compile_model(model: tf.keras.Model, lr: float, weight_decay: float = 1e-4, label_smoothing: float = 0.1) -> None:
    """Compiles the custom CNN model with AdamW and CategoricalCrossentropy.
    
    Args:
        model: The model to compile.
        lr: Learning rate or learning rate schedule.
        weight_decay: Weight decay factor for AdamW.
        label_smoothing: Label smoothing factor.
    """
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
