import tensorflow as tf
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Phase 2A: Confusion-driven coarse-to-fine
# ---------------------------------------------------------------------------

def build_confusion_groups(confusion_matrix: np.ndarray, n_groups: int = 30) -> Dict[int, List[int]]:
    """Builds confusion groups by clustering classes based on confusion matrix.
    
    Args:
        confusion_matrix: A 2D numpy array representing the confusion matrix.
        n_groups: The number of clusters to form.
        
    Returns:
        A dictionary mapping group ID to a list of class indices.
    """
    similarity = confusion_matrix + confusion_matrix.T
    np.fill_diagonal(similarity, 0)
    
    row_sums = similarity.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    similarity = similarity / row_sums
    
    distance = 1.0 - similarity
    
    clustering = AgglomerativeClustering(n_clusters=n_groups, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(distance)
    
    groups = {}
    for class_idx, group_id in enumerate(labels):
        groups.setdefault(int(group_id), []).append(class_idx)
        
    return groups

def build_coarse_classifier(n_groups: int, backbone: str = 'efficientnetb0', input_shape: tuple = (128, 128, 3)) -> tf.keras.Model:
    """Builds a coarse classifier to predict the confusion group.
    
    Args:
        n_groups: Number of confusion groups.
        backbone: The backbone architecture.
        input_shape: Shape of the input images.
        
    Returns:
        A tf.keras.Model instance.
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    if backbone == 'efficientnetb0':
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    elif backbone == 'mobilenetv3large':
        base_model = tf.keras.applications.MobileNetV3Large(include_top=False, weights='imagenet', input_tensor=inputs)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
        
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_groups, activation='softmax', dtype='float32')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def build_specialist_classifier(n_classes: int, backbone: str = 'efficientnetb0', input_shape: tuple = (128, 128, 3)) -> tf.keras.Model:
    """Builds a specialist classifier for a single confusion group.
    
    Args:
        n_classes: Number of classes in the confusion group.
        backbone: The backbone architecture.
        input_shape: Shape of the input images.
        
    Returns:
        A tf.keras.Model instance.
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    if backbone == 'efficientnetb0':
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    elif backbone == 'mobilenetv3large':
        base_model = tf.keras.applications.MobileNetV3Large(include_top=False, weights='imagenet', input_tensor=inputs)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
        
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', dtype='float32')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

class CoarseToFinePipeline:
    """Pipeline for coarse-to-fine classification."""
    
    def __init__(self, coarse_model: tf.keras.Model, specialist_models: Dict[int, tf.keras.Model], group_mapping: Dict[int, List[int]]):
        """Initializes the CoarseToFinePipeline.
        
        Args:
            coarse_model: The coarse classifier model.
            specialist_models: A dictionary mapping group ID to specialist models.
            group_mapping: A dictionary mapping group ID to a list of original class indices.
        """
        self.coarse_model = coarse_model
        self.specialist_models = specialist_models
        self.group_mapping = group_mapping
        
    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predicts the original class indices for the given images.
        
        Args:
            images: Input images.
            
        Returns:
            A numpy array of predicted original class indices.
        """
        predictions = []
        for i in range(len(images)):
            image = images[i:i+1]
            coarse_pred = self.coarse_model.predict(image, verbose=0)
            group_id = int(np.argmax(coarse_pred[0]))
            
            specialist_model = self.specialist_models[group_id]
            fine_pred = specialist_model.predict(image, verbose=0)
            fine_class_idx = int(np.argmax(fine_pred[0]))
            
            original_class_idx = self.group_mapping[group_id][fine_class_idx]
            predictions.append(original_class_idx)
            
        return np.array(predictions)
        
    def predict_with_confidence(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts the original class indices and returns confidence scores.
        
        Args:
            images: Input images.
            
        Returns:
            A tuple of (predictions, confidences).
        """
        predictions = []
        confidences = []
        for i in range(len(images)):
            image = images[i:i+1]
            coarse_pred = self.coarse_model.predict(image, verbose=0)
            group_id = int(np.argmax(coarse_pred[0]))
            coarse_conf = np.max(coarse_pred[0])
            
            specialist_model = self.specialist_models[group_id]
            fine_pred = specialist_model.predict(image, verbose=0)
            fine_class_idx = int(np.argmax(fine_pred[0]))
            fine_conf = np.max(fine_pred[0])
            
            original_class_idx = self.group_mapping[group_id][fine_class_idx]
            predictions.append(original_class_idx)
            confidences.append(coarse_conf * fine_conf)
            
        return np.array(predictions), np.array(confidences)

# ---------------------------------------------------------------------------
# Phase 2B: Grapheme decomposition
# ---------------------------------------------------------------------------

def decompose_labels(label_map: Dict[int, str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Decomposes Unicode labels into base graphemes and modifiers.
    
    Args:
        label_map: Dictionary mapping class names to indices (or indices to names).
        
    Returns:
        A tuple of (base_map, modifier_map) dictionaries mapping string parts to indices.
    """
    bases = set()
    modifiers = set(["none"])
    
    # Extract string names whether keys or values
    class_names = []
    for k, v in label_map.items():
        if isinstance(k, str) and not k.isdigit():
            class_names.append(k)
        elif isinstance(v, str):
            class_names.append(v)
        else:
            class_names.append(str(k))
    
    for label in class_names:
        chars = list(label)
        if len(chars) > 0:
            bases.add(chars[0])
            if len(chars) > 1:
                modifiers.add("".join(chars[1:]))
            else:
                modifiers.add("none")
                
    base_map = {base: i for i, base in enumerate(sorted(bases))}
    modifier_map = {mod: i for i, mod in enumerate(sorted(modifiers))}
    
    return base_map, modifier_map

def build_two_head_model(n_bases: int, n_modifiers: int, backbone: str = 'efficientnetb0', input_shape: tuple = (128, 128, 3)) -> tf.keras.Model:
    """Builds a model with two heads for base and modifier prediction.
    
    Args:
        n_bases: Number of base grapheme classes.
        n_modifiers: Number of modifier classes.
        backbone: The backbone architecture.
        input_shape: Shape of the input images.
        
    Returns:
        A tf.keras.Model instance.
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    if backbone == 'efficientnetb0':
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    elif backbone == 'mobilenetv3large':
        base_model = tf.keras.applications.MobileNetV3Large(include_top=False, weights='imagenet', input_tensor=inputs)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
        
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Base head
    base_x = tf.keras.layers.Dropout(0.3)(x)
    base_x = tf.keras.layers.Dense(512, activation='relu')(base_x)
    base_x = tf.keras.layers.Dropout(0.3)(base_x)
    base_out = tf.keras.layers.Dense(n_bases, activation='softmax', dtype='float32', name='base_head')(base_x)
    
    # Modifier head
    mod_x = tf.keras.layers.Dropout(0.3)(x)
    mod_x = tf.keras.layers.Dense(256, activation='relu')(mod_x)
    mod_x = tf.keras.layers.Dropout(0.3)(mod_x)
    mod_out = tf.keras.layers.Dense(n_modifiers, activation='softmax', dtype='float32', name='modifier_head')(mod_x)
    
    model = tf.keras.Model(inputs=inputs, outputs=[base_out, mod_out])
    return model
