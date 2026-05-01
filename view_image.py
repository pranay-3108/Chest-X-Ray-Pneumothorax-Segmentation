import os
import pydicom
import matplotlib.pyplot as plt

DATA_DIR = "images/stage_2_images"

# find one .dcm file
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith(".dcm"):
            path = os.path.join(root, f)
            print("Opening:", path)

            dcm = pydicom.dcmread(path)
            image = dcm.pixel_array

            print("Shape:", image.shape)

            plt.imshow(image, cmap='gray')
            plt.title(f)
            plt.axis("off")
            plt.show()

            exit()