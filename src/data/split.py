import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF"
}

def extract_class_name(file_path: str, data_root: Path) -> str:
    parts = Path(file_path).parts
    if "Test1" in parts:
        idx = parts.index("Test1")
        return "__".join(parts[idx + 1 : -1])
    
    parent = Path(file_path).parent
    try:
        rel = parent.relative_to(data_root)
        rel_parts = rel.parts
        if "Test1" in rel_parts:
            idx = rel_parts.index("Test1")
            return "__".join(rel_parts[idx + 1 :])
        return "__".join(rel_parts)
    except ValueError:
        return parent.name

def create_splits(
    data_dir: str,
    output_dir: str = "outputs/",
    project_root: str = ".",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    min_samples_per_class: int = 3,
) -> Dict[str, Any]:
    t0 = time.time()
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    proj_path = Path(project_root).resolve()
    data_path = Path(data_dir).resolve()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    class_groups = defaultdict(list)
    total_found = 0

    for root, _, files in os.walk(data_path):
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext in VALID_EXTENSIONS:
                abs_fpath = os.path.join(root, f)
                try:
                    rel_fpath = os.path.relpath(abs_fpath, start=proj_path)
                except ValueError:
                    rel_fpath = abs_fpath
                
                cls_name = extract_class_name(abs_fpath, data_path)
                if cls_name and cls_name not in ("Guninthamulu", "hallulu", "achulu", "othulu"):
                    class_groups[cls_name].append(rel_fpath)
                    total_found += 1

    if total_found == 0:
        raise ValueError(f"No images found with valid extensions in {data_dir}")

    # Load canonical 630 class names if exists, else sort unique
    class_names_path = out_path / "class_names.json"
    if class_names_path.exists():
        with open(class_names_path, "r") as f:
            canonical_classes = json.load(f)
        unique_classes = [c for c in canonical_classes if c in class_groups]
        # Append any remaining
        for c in sorted(class_groups.keys()):
            if c not in unique_classes:
                unique_classes.append(c)
    else:
        unique_classes = sorted(list(class_groups.keys()))

    label_map = {cls: idx for idx, cls in enumerate(unique_classes)}
    with open(out_path / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=4)
        
    with open(out_path / "class_names.json", "w") as f:
        json.dump(unique_classes, f, indent=4)

    rng = np.random.RandomState(seed)
    train_data, val_data, test_data = [], [], []
    warnings_list = []

    for cls in unique_classes:
        paths = class_groups[cls]
        n_samples = len(paths)
        cls_idx = label_map[cls]

        if n_samples < min_samples_per_class:
            warnings_list.append(f"Class '{cls}' has only {n_samples} samples. Moved to train.")
            train_data.extend([(p, cls_idx, cls) for p in paths])
            continue

        shuffled = list(paths)
        rng.shuffle(shuffled)

        n_val = int(round(n_samples * val_ratio))
        n_test = int(round(n_samples * test_ratio))

        if n_samples >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n_test)
            if n_val + n_test >= n_samples:
                n_val = 1
                n_test = 1

        n_train = n_samples - (n_val + n_test)
        if n_train <= 0:
            n_train = n_samples
            n_val = 0
            n_test = 0

        train_p = shuffled[:n_train]
        val_p = shuffled[n_train : n_train + n_val]
        test_p = shuffled[n_train + n_val :]

        train_data.extend([(p, cls_idx, cls) for p in train_p])
        val_data.extend([(p, cls_idx, cls) for p in val_p])
        test_data.extend([(p, cls_idx, cls) for p in test_p])

    for data, fname in [(train_data, "train.csv"), (val_data, "val.csv"), (test_data, "test.csv")]:
        with open(out_path / fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label_idx", "class_name"])
            writer.writerows(data)

    elapsed = time.time() - t0
    stats = {
        "num_classes": len(unique_classes),
        "total_images": total_found,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "time_seconds": round(elapsed, 2),
        "warnings": warnings_list,
    }
    return stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="outputs/")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = create_splits(
        data_dir=args.data_dir,
        output_dir=args.output,
        project_root=args.project_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"Splits generated successfully: {stats}")
