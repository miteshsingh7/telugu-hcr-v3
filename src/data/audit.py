"""Fast Data Audit Module for Phase 0 (optimized for 300K+ images).

Performs:
  - Instant class inventory & sample count histogram in < 2s
  - Sampled image corruption & channel consistency checks
  - Low variance / blank image detection
  - Summary report with class distribution plot

Usage:
    python -m src.data.audit --data_dir /kaggle/input/.../Test1 --output reports/
"""

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
    """Extracts unique class identifier from path."""
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
    """Runs a fast, comprehensive data audit on datasets with up to 300K+ images.

    Args:
        data_dir: Root dataset directory.
        output_dir: Directory to save audit report and distribution plot.
        sample_check_size: Number of images to deeply verify (checks corruption/variance).

    Returns:
        Dictionary of audit results.
    """
    t0 = time.time()
    data_path = Path(data_dir).resolve()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"⚡ Indexing dataset at: {data_path} ...", flush=True)

    inventory = defaultdict(int)
    all_image_paths: List[str] = []

    # Fast single-pass scan
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

    t_index = time.time() - t0
    num_classes = len(inventory)
    print(f"✅ Indexed {total_images:,} images across {num_classes} classes in {t_index:.2f}s!", flush=True)

    # Sample deep check for corruption and degenerate images
    print(f"⚡ Auditing sample of {min(sample_check_size, total_images):,} images for integrity...", flush=True)
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

    # Check for writer metadata
    writer_files = [str(p) for p in data_path.glob("*writer*")] + [str(p) for p in data_path.glob("*metadata*")]
    has_writer_ids = len(writer_files) > 0
    writer_details = f"Found {len(writer_files)} metadata files." if has_writer_ids else "No writer metadata files found (stratified split applied)."

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

    # Generate visual plot and markdown report
    print("⚡ Generating audit report & distribution chart...", flush=True)
    _generate_report(results, out_path)

    print("\n" + "=" * 45, flush=True)
    print("🎯 DATA AUDIT COMPLETE", flush=True)
    print("=" * 45, flush=True)
    print(f"Total Classes: {num_classes:,}")
    print(f"Total Images:  {total_images:,}")
    print(f"Corrupted:     {len(corrupted_files)}")
    print(f"Degenerate:    {len(degenerate_images)}")
    print(f"Time Taken:    {results['time_seconds']}s")
    print(f"Report Saved:  {out_path / 'audit_report.md'}")
    print("=" * 45 + "\n", flush=True)

    return results


def _generate_report(results: Dict[str, Any], output_dir: Path) -> None:
    """Generates markdown report and distribution plot."""
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

        f.write("### Classes with < 20 Samples\n")
        if underrepresented:
            for cls, cnt in list(underrepresented.items())[:20]:
                f.write(f"- `{cls}`: {cnt} samples\n")
        else:
            f.write("✅ None — all classes have $\\ge 20$ samples.\n")
        f.write("\n")

        f.write("## Data Quality (Sampled Integrity Check)\n")
        f.write(f"- **Images Sampled:** {results['audit_sample_checked']:,}\n")
        f.write(f"- **Corrupted Images:** {len(results['corrupted_files'])}\n")
        f.write(f"- **Channel Inconsistencies:** {len(results['channel_inconsistencies'])}\n")
        f.write(f"- **Low-Variance / Degenerate Images:** {len(results['degenerate_images'])}\n\n")

        f.write("## Writer Metadata\n")
        f.write(f"- **Found:** {results['writer_metadata']['has_writer_ids']}\n")
        f.write(f"- **Details:** {results['writer_metadata']['details']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Data Audit for Telugu HCR")
    parser.add_argument("--data_dir", required=True, help="Path to data directory")
    parser.add_argument("--output", default="reports/", help="Output directory for reports")
    args = parser.parse_args()

    run_audit(data_dir=args.data_dir, output_dir=args.output)
