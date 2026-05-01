from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom

from utils import decode_rle


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "train-rle.csv"
TRAIN_DIR = ROOT / "images" / "dicom-images-train"


def load_image(path):
    ds = pydicom.dcmread(path)
    image = ds.pixel_array.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return np.stack([image] * 3, axis=-1)


def main():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    encoded = df["EncodedPixels"].astype(str).str.strip()
    df = df[(encoded != "-1") & (encoded != "nan")].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No positive samples found.")

    image_map = {}
    for path in TRAIN_DIR.rglob("*.dcm"):
        image_map[path.stem] = path

    row = df.iloc[0]
    image_id = row["ImageId"]
    image_path = image_map.get(image_id)
    if image_path is None:
        raise FileNotFoundError(f"Image not found for {image_id}")

    image = load_image(image_path)
    mask = decode_rle(str(row["EncodedPixels"]).strip(), image.shape[:2])

    print(f"image={image_id}")
    print(f"mask_pixels={int(mask.sum())}")
    print(f"mask_values={np.unique(mask)}")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("DICOM Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
    plt.title("Decoded Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
