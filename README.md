# Chest X-Ray Pneumothorax Segmentation

Student experiment around the SIIM-ACR Pneumothorax Segmentation dataset.

This repository is mainly about the data side of the problem: reading DICOM files, decoding RLE masks, checking alignment, and building a usable image-mask pipeline before model training.

## What This Project Does

- reads chest X-ray DICOM files
- decodes pneumothorax masks from run-length encoded labels
- maps `ImageId` values to the real image files
- prepares image and mask pairs for segmentation experiments
- includes small utilities for validation and visualization

## Dataset

Input dataset:

- SIIM-ACR Pneumothorax Segmentation
- chest X-ray DICOM images
- CSV file with RLE mask annotations
- both positive and negative cases

This repo currently includes the label CSV structure and the preprocessing scripts used to organize the dataset.

## What Was Tested

The strongest completed part of this project is preprocessing and mask generation.

Completed work:

- RLE decoding
- image-to-label mapping
- negative sample handling
- dataset inspection scripts
- basic visualization utilities for images and masks

Not yet fully completed in public code:

- `train.py`
- `model.py`
- `eval.py`

So this repo should be read as a segmentation pipeline setup repo, not as a finished benchmarking repo.

## Pipeline

```text
DICOM image
  -> ImageId lookup
  -> RLE mask decoding
  -> image/mask pair creation
  -> dataset loading
  -> segmentation training experiments
```

## Project Structure

```text
build_dataset.py    build aligned image-mask dataset
data_loader.py      PyTorch dataset loader
data_analyzer.py    dataset inspection and checks
utils.py            RLE encode/decode helpers
test_rle_decode.py  mask visualization checks
view_result.py      inspect image + mask pairing
```

## Tech Stack

- Python
- PyTorch
- NumPy
- Pandas
- pydicom
- Matplotlib
- Pillow

## Results So Far

Current progress is strongest on preprocessing correctness rather than final model performance.

Publicly visible results in this repo today:

- DICOM loading works
- RLE masks can be decoded and visualized
- image-mask alignment pipeline is in place

Still missing from the public repo:

- final segmentation predictions
- benchmark metrics such as Dice score / IoU
- exported visualization screenshots

## Why The Repo Looks Like This

I kept the unfinished parts public instead of pretending the model side was complete.

That means this repository shows real progress on the setup and debugging phase, but it is still incomplete as an end-to-end segmentation project.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Example utilities:

```bash
python test_rle_decode.py
python data_analyzer.py
```

## Next Improvements

- add one or two real mask visualization screenshots to the README
- complete and document the training path
- report one honest baseline metric once training is stable
