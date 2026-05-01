# debug_mask.py
import pandas as pd

df = pd.read_csv(r"C:\Users\prana\vision-segmentation\data\train-rle.csv")
df.columns = df.columns.str.strip()

print(df['EncodedPixels'].value_counts().head(10))