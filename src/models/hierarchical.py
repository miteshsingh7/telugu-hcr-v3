from typing import Dict, List, Tuple
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import tensorflow as tf

def build_confusion_groups(confusion_matrix: np.ndarray, n_groups: int = 30) -> Dict[int, List[int]]:
    norm_cm = confusion_matrix.astype(float)
    row_sums = norm_cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm_cm = norm_cm / row_sums

    affinity_matrix = (norm_cm + norm_cm.T) / 2.0
    clustering = AgglomerativeClustering(
        n_clusters=n_groups,
        metric="precomputed",
        linkage="average",
    )
    dist_matrix = 1.0 - affinity_matrix
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0.0, 1.0)

    group_labels = clustering.fit_predict(dist_matrix)
    groups = {}
    for class_idx, group_id in enumerate(group_labels):
        groups.setdefault(int(group_id), []).append(class_idx)

    return groups

def decompose_labels(label_map: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, int]]:
    base_chars = {}
    modifiers = {}

    for cls_name, idx in label_map.items():
        clean_cls = cls_name.replace("/", "__")
        parts = clean_cls.split("__")
        base = parts[1] if len(parts) > 1 else parts[0]
        mod = parts[2] if len(parts) > 2 else "none"

        if base not in base_chars:
            base_chars[base] = len(base_chars)
        if mod not in modifiers:
            modifiers[mod] = len(modifiers)

    return base_chars, modifiers
