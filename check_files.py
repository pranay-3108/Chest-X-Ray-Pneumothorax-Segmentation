import os

train_dir = r"C:\Users\prana\vision-segmentation\data\train"

files = os.listdir(train_dir)

print("Total files:", len(files))
print("Sample files:", files[:10])