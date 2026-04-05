import os
import pandas as pd
from tqdm import tqdm
from features import extract_features

DATASET_PATH = "data"

rows = []

labels = ["starters", "middleground", "peak"]
all_files = [
    (label, file)
    for label in labels
    for file in os.listdir(os.path.join(DATASET_PATH, label))
    if file.endswith(".mp3")
]

for label, file in tqdm(all_files, desc="Extracting features", unit="track"):
    path = os.path.join(DATASET_PATH, label, file)

    try:
        features = extract_features(path)
        rows.append([*features, label])
    except Exception as e:
        tqdm.write(f"Error with {file}: {e}")

df = pd.DataFrame(rows)
df.to_csv("dataset.csv", index=False)
