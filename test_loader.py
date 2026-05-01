from data_loader import SIIMDataset


def main():
    dataset = SIIMDataset()
    image, mask = dataset[0]

    print(f"total={len(dataset)}")
    print(f"image_shape={tuple(image.shape)}")
    print(f"mask_shape={tuple(mask.shape)}")
    print(f"mask_pixels={float(mask.sum())}")


if __name__ == "__main__":
    main()
