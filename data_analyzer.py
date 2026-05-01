import os
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom

from utils import build_mask_from_rles


# ---------------------------
# PATHS
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "images" / "dicom-images-train"
CSV_PATH = PROJECT_ROOT / "data" / "train-rle.csv"


# ---------------------------
# CONFIG
# ---------------------------
PROGRESS_EVERY = 500
MAX_SECONDS = 20 * 60   # stop after 20 minutes if something goes wrong
MAX_SAMPLES = None      # set to an int like 1000 for quick testing


# ---------------------------
# CSV LOAD
# ---------------------------
def load_data():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    required = {"ImageId", "EncodedPixels"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}. Found: {list(df.columns)}")

    grouped = df.groupby("ImageId")["EncodedPixels"].apply(list)
    return grouped


# ---------------------------
# BUILD IMAGE INDEX ONCE
# ---------------------------
def build_image_map():
    image_map = {}

    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith(".dcm"):
                image_id = f[:-4]
                image_map[image_id] = os.path.join(root, f)

    print(f"Indexed images: {len(image_map)}")
    return image_map


# ---------------------------
# SAFE DICOM LOAD
# ---------------------------
def load_image(path):
    try:
        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array
        if image is None:
            return None
        return image
    except Exception:
        return None


# ---------------------------
# BUILD MASK
# ---------------------------
def build_mask(rles, shape):
    """
    Returns:
      - valid negative mask (all zeros) when rles are only '-1'
      - decoded binary mask for positive samples
      - None if decoding fails
    """
    try:
        return build_mask_from_rles(rles, shape)
    except Exception:
        return None


# ---------------------------
# ANALYZE
# ---------------------------
def analyze():
    start_time = time.monotonic()

    print("CSV exists:", os.path.exists(CSV_PATH))
    print("DATA_DIR exists:", os.path.exists(DATA_DIR))

    grouped = load_data()
    image_map = build_image_map()

    total = 0
    pos = 0
    neg = 0
    sizes = []

    skip_counts = Counter()

    for i, (image_id, rles) in enumerate(grouped.items()):
        # runtime guard
        elapsed = time.monotonic() - start_time
        if elapsed > MAX_SECONDS:
            print(f"\nStopped early: time limit reached ({MAX_SECONDS} seconds).")
            break

        if MAX_SAMPLES is not None and total >= MAX_SAMPLES:
            print(f"\nStopped early: sample limit reached ({MAX_SAMPLES}).")
            break

        if i % PROGRESS_EVERY == 0:
            print(f"Processed CSV rows: {i}")

        path = image_map.get(image_id)
        if path is None:
            skip_counts["missing_file"] += 1
            continue

        image = load_image(path)
        if image is None:
            skip_counts["dicom_load_fail"] += 1
            continue

        if image.ndim != 2:
            skip_counts["bad_image_shape"] += 1
            continue

        mask = build_mask(rles, image.shape)
        if mask is None:
            skip_counts["mask_decode_fail"] += 1
            continue

        if mask.shape != image.shape:
            skip_counts["shape_mismatch"] += 1
            continue

        total += 1

        if mask.sum() == 0:
            neg += 1
        else:
            pos += 1
            sizes.append(int(mask.sum()))

    print("\n===== DATASET STATS =====")
    print("Total valid images:", total)
    print("Positive:", pos)
    print("Negative:", neg)

    if sizes:
        print("\nMask size stats:")
        print("Min:", min(sizes))
        print("Max:", max(sizes))
        print("Mean:", int(np.mean(sizes)))

    print("\n===== SKIP REASONS =====")
    for k, v in skip_counts.items():
        print(f"{k}: {v}")


# ---------------------------
# VISUALIZE
# ---------------------------
def visualize_samples(n=5):
    grouped = load_data()
    image_map = build_image_map()

    shown = 0

    for image_id, rles in grouped.items():
        path = image_map.get(image_id)
        if path is None:
            continue

        image = load_image(path)
        if image is None:
            continue

        mask = build_mask(rles, image.shape)
        if mask is None:
            continue

        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Mask")
        plt.imshow(mask, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        shown += 1
        if shown >= n:
            break


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    analyze()
    visualize_samples(3)
