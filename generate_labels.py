import os
import pandas as pd
import random

DATASET_PATH = r"C:\Users\a\Documents\cough_screening_system\coughz"

files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".wav")]

print("Found files:", files)

labels = ["healthy", "asthma", "copd"]

data = []

for f in files:
    label = random.choice(labels)
    data.append([f, label])

df = pd.DataFrame(data, columns=["filename", "label"])
df.to_csv("cough_labels.csv", index=False)

print("cough_labels.csv generated successfully.")