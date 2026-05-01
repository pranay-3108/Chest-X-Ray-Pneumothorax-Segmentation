from pathlib import Path

import pandas as pd
import pydicom
from PIL import Image

from utils import build_mask_from_rles


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "train-rle.csv"
DICOM_DIR = ROOT / "images" / "dicom-images-train"
SAVE_IMG_DIR = ROOT / "aligned_dataset" / "images"
SAVE_MASK_DIR = ROOT / "aligned_dataset" / "masks"


def normalize_image(image):
    image = image.astype("float32")
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return (image * 255).astype("uint8")


def main():
    SAVE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_MASK_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    grouped = df.groupby("ImageId")["EncodedPixels"].apply(list)
    image_map = {}
    for path in DICOM_DIR.rglob("*.dcm"):
        image_map[path.stem] = path

    done = 0
    skipped = 0

    for image_id, rles in grouped.items():
        dicom_path = image_map.get(image_id)
        if dicom_path is None:
            skipped += 1
            continue

        try:
            ds = pydicom.dcmread(dicom_path)
            image = normalize_image(ds.pixel_array)
            mask = build_mask_from_rles(rles, image.shape) * 255

            Image.fromarray(image).save(SAVE_IMG_DIR / f"{image_id}.png")
            Image.fromarray(mask).save(SAVE_MASK_DIR / f"{image_id}.png")
            done += 1

            if done % 200 == 0:
                print(f"done={done} skipped={skipped}")
        except Exception as error:
            print(f"failed: {image_id} | {error}")
            skipped += 1

    print("export finished")
    print(f"done={done}")
    print(f"skipped={skipped}")


if __name__ == "__main__":
    main()
