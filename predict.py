import os
import joblib
import pandas as pd
from tqdm import tqdm
from features import extract_features

PREDICT_FOLDER = "predict"

model = joblib.load("model.pkl")

files = [f for f in os.listdir(PREDICT_FOLDER) if f.endswith(".mp3")]

if not files:
    print("No MP3 files found in the 'predict' folder.")
    exit(0)

results = []

for file in tqdm(files, desc="Predicting", unit="track"):
    path = os.path.join(PREDICT_FOLDER, file)
    try:
        feature_names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else None
        features = pd.DataFrame(extract_features(path).reshape(1, -1), columns=feature_names)
        prediction = model.predict(features)[0]
        confidence = max(model.predict_proba(features)[0])
        results.append((file, prediction, f"{confidence:.2%}"))
    except Exception as e:
        results.append((file, "error", str(e)))

col_widths = [
    max(len("File"), max(len(r[0]) for r in results)),
    max(len("Prediction"), max(len(r[1]) for r in results)),
    max(len("Confidence"), max(len(r[2]) for r in results)),
]

def row(cols):
    return "  ".join(str(c).ljust(w) for c, w in zip(cols, col_widths))

separator = "  ".join("-" * w for w in col_widths)

print()
print(row(["File", "Prediction", "Confidence"]))
print(separator)
for r in results:
    print(row(r))
