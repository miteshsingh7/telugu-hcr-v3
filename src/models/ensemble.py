from typing import List, Optional, Any
import numpy as np
import tensorflow as tf

class SoftVotingEnsemble:
    def __init__(self, models: List[tf.keras.Model], weights: Optional[List[float]] = None):
        self.models = models
        if weights is not None:
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            self.weights = [1.0 / len(models)] * len(models)

    def predict_proba(self, x: Any) -> np.ndarray:
        preds = []
        for model in self.models:
            if isinstance(x, tf.data.Dataset):
                p = model.predict(x, verbose=0)
            else:
                p = model(x, training=False).numpy()
            preds.append(p)

        weighted_preds = sum(w * p for w, p in zip(self.weights, preds))
        return weighted_preds

    def predict(self, x: Any) -> np.ndarray:
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)
