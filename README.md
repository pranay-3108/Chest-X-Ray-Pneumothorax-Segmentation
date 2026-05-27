# SIIM-ACR Pneumothorax Segmentation

Preprocessing and segmentation pipeline for the SIIM-ACR Pneumothorax dataset using PyTorch.

The project focuses mainly on dataset preparation, mask generation, and training pipeline setup for lung collapse segmentation from chest X-ray DICOM images.


# Dataset

SIIM-ACR Pneumothorax Segmentation Dataset

* 10k+ DICOM chest X-rays
* RLE encoded masks in CSV format
* Positive and negative samples
* Nested DICOM folder structure

---

# Pipeline

```text
DICOM Images + CSV Labels
            ↓
ImageId ↔ File Mapping
            ↓
RLE Mask Decoding
            ↓
Aligned Image-Mask Dataset
            ↓
PyTorch Dataset Loader
            ↓
U-Net Training Pipeline
```

---

# Features

* RLE mask decoding
* ImageId to DICOM path mapping
* Indexed lookup for faster loading
* Negative sample handling
* PNG mask generation
* DICOM preprocessing
* Segmentation dataset loader

---

# Project Structure

```text
build_dataset.py   → dataset preprocessing
data_loader.py     → PyTorch dataset loader
utils.py           → RLE encode/decode utilities
test_rle_decode.py → mask visualization
train.py           → training pipeline
model.py           → U-Net model
eval.py            → evaluation
```

---

# Tech Stack

* Python
* PyTorch
* NumPy
* Pandas
* pydicom
* Matplotlib

---

# Status

* Dataset preprocessing completed
* Mask alignment validated
* Training pipeline setup completed
* Moving to model training and evaluation
