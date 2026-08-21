"""High-performance Stratified Data Splitting Module for Telugu HCR.

Fast one-pass directory scanner using os.scandir/os.walk to index 300K+ images
in < 5 seconds and generate stratified train/val/test split manifests.

Usage:
    python -m src.data.split --data_dir /kaggle/input/.../Test1 --output outputs/
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List
import warnings

import numpy as np

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF"
}


def extract_class_name(file_path: str, data_root: Path) -> str:
    """Extracts unique class identifier from path, preserving multi-level sections."""
    parts = Path(file_path).parts
    if "Test1" in parts:
        idx = parts.index("Test1")
        return "/".join(parts[idx + 1 : -1])
    
    parent = Path(file_path).parent
    try:
        rel = parent.relative_to(data_root)
        rel_str = str(rel)
        if rel_str == ".":
            return parent.name
        # If relative path contains Test1, take everything after it
        rel_parts = rel.parts
        if "Test1" in rel_parts:
            idx = rel_parts.index("Test1")
            return "/".join(rel_parts[idx + 1 :])
        return rel_str
    except ValueError:
        return parent.name


def create_splits(
    data_dir: str,
    output_dir: str = "outputs/",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    min_samples_per_class: int = 3,
) -> Dict[str, Any]:
    """Creates stratified train/val/test splits in seconds for large datasets.

    Args:
        data_dir: Root dataset directory.
        output_dir: Output directory for train.csv, val.csv, test.csv, label_map.json.
        train_ratio: Proportion for train (default 0.8).
        val_ratio: Proportion for val (default 0.1).
        test_ratio: Proportion for test (default 0.1).
        seed: Random seed for shuffling.
        min_samples_per_class: Minimum samples required per class.

    Returns:
        Dictionary with split statistics.
    """
    t0 = time.time()
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    data_path = Path(data_dir).resolve()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"⚡ Scanning dataset at: {data_path} ...", flush=True)

    class_groups = defaultdict(list)
    total_found = 0

    # Fast single-pass walk
    for root, _, files in os.walk(data_path):
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext in VALID_EXTENSIONS:
                full_path = os.path.join(root, f)
                cls_name = extract_class_name(full_path, data_path)
                class_groups[cls_name].append(full_path)
                total_found += 1

    if total_found == 0:
        raise ValueError(f"No images found with valid extensions in {data_dir}")

    t_scan = time.time() - t0
    unique_classes = sorted(list(class_groups.keys()))
    print(f"✅ Found {total_found:,} images across {len(unique_classes)} classes in {t_scan:.2f}s!", flush=True)

    # Build label mapping
    label_map = {cls: idx for idx, cls in enumerate(unique_classes)}
    with open(out_path / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=4)

    rng = np.random.RandomState(seed)
    train_data, val_data, test_data = [], [], []
    warnings_list = []

    print("⚡ Generating stratified splits...", flush=True)
    for cls in unique_classes:
        paths = class_groups[cls]
        n_samples = len(paths)
        cls_idx = label_map[cls]

        if n_samples < min_samples_per_class:
            msg = f"Class '{cls}' has only {n_samples} samples (< {min_samples_per_class}). All moved to train."
            warnings_list.append(msg)
            train_data.extend([(p, cls_idx, cls) for p in paths])
            continue

        # Shuffle deterministically
        shuffled = list(paths)
        rng.shuffle(shuffled)

        n_val = int(round(n_samples * val_ratio))
        n_test = int(round(n_samples * test_ratio))

        # Ensure at least 1 sample in train if possible
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

    # Save CSV manifests in parallel / fast stream
    print("⚡ Writing CSV split manifests...", flush=True)
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

    print("\n" + "=" * 45, flush=True)
    print("🎯 STRATIFIED SPLIT SUMMARY (COMPLETE)", flush=True)
    print("=" * 45, flush=True)
    print(f"Total Classes: {stats['num_classes']:,}")
    print(f"Total Images:  {stats['total_images']:,}")
    print(f"Train samples: {stats['train_size']:,} ({stats['train_size']/total_found*100:.1f}%)")
    print(f"Val samples:   {stats['val_size']:,} ({stats['val_size']/total_found*100:.1f}%)")
    print(f"Test samples:  {stats['test_size']:,} ({stats['test_size']/total_found*100:.1f}%)")
    print(f"⚡ Completed in {elapsed:.2f} seconds!")
    print("=" * 45 + "\n", flush=True)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Stratified Splitting for Telugu HCR")
    parser.add_argument("--data_dir", required=True, help="Path to data directory")
    parser.add_argument("--output", default="outputs/", help="Output directory for manifests")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio (default 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Val ratio (default 0.1)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test ratio (default 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = parser.parse_args()

    create_splits(
        data_dir=args.data_dir,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
