import tensorflow as tf
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List, Optional

class SoftVotingEnsemble:
    """An ensemble that averages softmax outputs across multiple models."""
    
    def __init__(self, models: List[tf.keras.Model], weights: Optional[List[float]] = None):
        """Initializes the SoftVotingEnsemble.
        
        Args:
            models: A list of trained tf.keras.Model instances.
            weights: Optional list of weights for each model.
        """
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
            
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predicts class probabilities by averaging model outputs.
        
        Args:
            x: Input data.
            
        Returns:
            A numpy array of averaged probabilities.
        """
        # Get shape from the first model's single prediction
        sample_pred = self.models[0].predict(x[:1], verbose=0)
        probas = np.zeros((len(x), sample_pred.shape[1]), dtype=np.float32)
        
        for model, weight in zip(self.models, self.weights):
            probas += model.predict(x, verbose=0) * weight
            
        return probas
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts class indices.
        
        Args:
            x: Input data.
            
        Returns:
            A numpy array of predicted class indices.
        """
        probas = self.predict_proba(x)
        return np.argmax(probas, axis=1)


class StackingEnsemble:
    """An ensemble that trains a meta-learner on top of base model predictions."""
    
    def __init__(self, models: List[tf.keras.Model], meta_learner=None):
        """Initializes the StackingEnsemble.
        
        Args:
            models: A list of trained tf.keras.Model instances.
            meta_learner: An optional meta-learner (defaults to LogisticRegression).
        """
        self.models = models
        if meta_learner is None:
            self.meta_learner = LogisticRegression(max_iter=1000)
        else:
            self.meta_learner = meta_learner
            
    def _get_meta_features(self, x: np.ndarray) -> np.ndarray:
        """Extracts predictions from all models to use as meta-features.
        
        Args:
            x: Input data.
            
        Returns:
            A concatenated numpy array of predictions.
        """
        predictions = []
        for model in self.models:
            pred = model.predict(x, verbose=0)
            predictions.append(pred)
        return np.concatenate(predictions, axis=1)
        
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fits the meta-learner on the base model predictions.
        
        Args:
            x: Input data.
            y: Target labels (indices).
        """
        meta_features = self._get_meta_features(x)
        self.meta_learner.fit(meta_features, y)
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts class indices using the meta-learner.
        
        Args:
            x: Input data.
            
        Returns:
            A numpy array of predicted class indices.
        """
        meta_features = self._get_meta_features(x)
        return self.meta_learner.predict(meta_features)

def load_models_from_configs(model_paths: List[str]) -> List[tf.keras.Model]:
    """Loads multiple model checkpoints for ensembling.
    
    Args:
        model_paths: A list of file paths to saved models.
        
    Returns:
        A list of loaded tf.keras.Model instances.
    """
    models = []
    for path in model_paths:
        model = tf.keras.models.load_model(path, compile=False)
        models.append(model)
    return models
