from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

from utils import build_mask_from_rles


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "train-rle.csv"
DICOM_DIR = ROOT / "images" / "dicom-images-train"


class SIIMDataset(Dataset):
    def __init__(self, csv_path=None, dicom_dir=None):
        self.csv_path = Path(csv_path) if csv_path is not None else CSV_PATH
        self.dicom_dir = Path(dicom_dir) if dicom_dir is not None else DICOM_DIR

        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.strip()
        self.grouped = df.groupby("ImageId")["EncodedPixels"].apply(list)

        self.image_map = {}
        for path in self.dicom_dir.rglob("*.dcm"):
            self.image_map[path.stem] = path

        self.ids = [image_id for image_id in self.grouped.index if image_id in self.image_map]
        if not self.ids:
            raise RuntimeError(f"No matching images found in {self.dicom_dir}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        path = self.image_map[image_id]
        rles = self.grouped[image_id]

        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        mask = build_mask_from_rles(rles, image.shape).astype(np.float32)

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.from_numpy(image), torch.from_numpy(mask)
