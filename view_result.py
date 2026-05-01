import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import pydicom

from utils import build_mask_from_rles

DATA_DIR = "images/dicom-images-train"
CSV_PATH = "data/train-rle.csv"


# load CSV
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

grouped = df.groupby("ImageId")["EncodedPixels"].apply(list)

# build image map (fast lookup)
image_map = {}
for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith(".dcm"):
            image_map[f[:-4]] = os.path.join(root, f)


# pick random sample
image_id = random.choice(list(grouped.keys()))
rles = grouped[image_id]

path = image_map.get(image_id)

# load image
dcm = pydicom.dcmread(path)
image = dcm.pixel_array

# build mask
mask = build_mask_from_rles(rles, image.shape)


# plot
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("X-ray")
plt.imshow(image, cmap='gray')

plt.subplot(1,3,2)
plt.title("Mask")
plt.imshow(mask, cmap='gray')

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(image, cmap='gray')
plt.imshow(mask, alpha=0.4)

plt.tight_layout()
plt.savefig("outputs/result.png")
plt.show()
