import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image, ImageStat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF"
}

def extract_class_name(file_path: str, data_root: Path) -> str:
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
        rel_parts = rel.parts
        if "Test1" in rel_parts:
            idx = rel_parts.index("Test1")
            return "/".join(rel_parts[idx + 1 :])
        return rel_str
    except ValueError:
        return parent.name

def run_audit(data_dir: str, output_dir: str = "reports/", sample_check_size: int = 5000) -> Dict[str, Any]:
    t0 = time.time()
    data_path = Path(data_dir).resolve()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    inventory = defaultdict(int)
    all_image_paths: List[str] = []

    for root, _, files in os.walk(data_path):
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext in VALID_EXTENSIONS:
                full_path = os.path.join(root, f)
                all_image_paths.append(full_path)
                cls_name = extract_class_name(full_path, data_path)
                inventory[cls_name] += 1

    total_images = len(all_image_paths)
    if total_images == 0:
        raise ValueError(f"No valid image files found in {data_dir}")

    num_classes = len(inventory)
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(total_images, size=min(sample_check_size, total_images), replace=False)

    corrupted_files = []
    channel_inconsistencies = {}
    degenerate_images = []

    for idx in sample_indices:
        img_path = all_image_paths[idx]
        try:
            with Image.open(img_path) as img:
                if img.mode not in ["L", "RGB", "RGBA", "1"]:
                    channel_inconsistencies[img_path] = img.mode
                stat = ImageStat.Stat(img)
                if any(std < 5.0 for std in stat.stddev):
                    degenerate_images.append(img_path)
        except Exception as e:
            corrupted_files.append((img_path, str(e)))

    writer_files = [str(p) for p in data_path.glob("*writer*")] + [str(p) for p in data_path.glob("*metadata*")]
    has_writer_ids = len(writer_files) > 0
    writer_details = f"Found {len(writer_files)} metadata files." if has_writer_ids else "No writer metadata files found."

    results = {
        "total_classes": num_classes,
        "total_samples": total_images,
        "class_inventory": dict(inventory),
        "corrupted_files": corrupted_files,
        "channel_inconsistencies": channel_inconsistencies,
        "degenerate_images": degenerate_images,
        "audit_sample_checked": len(sample_indices),
        "writer_metadata": {
            "has_writer_ids": has_writer_ids,
            "details": writer_details,
        },
        "time_seconds": round(time.time() - t0, 2),
    }

    _generate_report(results, out_path)
    return results

def _generate_report(results: Dict[str, Any], output_dir: Path) -> None:
    counts = sorted(list(results["class_inventory"].values()))

    plt.figure(figsize=(10, 5))
    plt.plot(counts, color="royalblue", linewidth=2)
    plt.fill_between(range(len(counts)), counts, color="royalblue", alpha=0.3)
    plt.title(f"Class Distribution Across {results['total_classes']} Classes", fontsize=12)
    plt.xlabel("Classes (sorted by sample count)")
    plt.ylabel("Number of Samples")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / "class_distribution.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    underrepresented = {k: v for k, v in results["class_inventory"].items() if v < 20}
    report_path = output_dir / "audit_report.md"
    with open(report_path, "w") as f:
        f.write("# Phase 0 — Data Audit Report\n\n")
        f.write(f"- **Total Classes:** {results['total_classes']:,}\n")
        f.write(f"- **Total Images:** {results['total_samples']:,}\n")
        f.write(f"- **Min Samples/Class:** {min(counts):,}\n")
        f.write(f"- **Max Samples/Class:** {max(counts):,}\n")
        f.write(f"- **Average Samples/Class:** {sum(counts)/len(counts):.1f}\n\n")
        f.write("## Class Distribution\n\n")
        f.write("![Class Distribution](./class_distribution.png)\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="reports/")
    args = parser.parse_args()
    run_audit(data_dir=args.data_dir, output_dir=args.output)
