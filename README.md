# Music Classifier

Classifies MP3 tracks into three categories — **starters**, **middleground**, and **peak** — using a Random Forest model trained on audio features (tempo, energy, brightness, noisiness, MFCCs) extracted from the middle 3 minutes of each track.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
data/
  starters/       # Training MP3s — starter tracks
  middleground/   # Training MP3s — middleground tracks
  peak/           # Training MP3s — peak tracks
predict/          # MP3s to classify
features.py       # Audio feature extraction
build_dataset.py  # Extracts features from data/ into dataset.csv
train.py          # Trains the model and saves model.pkl
predict.py        # Classifies tracks in predict/
```

## Usage

### 1. Prepare training data

Place your labeled MP3 files into the corresponding folders inside `data/`:

- `data/starters/`
- `data/middleground/`
- `data/peak/`

### 2. Build the dataset

```bash
python build_dataset.py
```

Extracts audio features from all MP3s in `data/` and saves them to `dataset.csv`.

### 3. Train the model

```bash
python train.py
```

Trains a Random Forest classifier and saves it to `model.pkl`. Prints a classification report with precision, recall, and F1-score.

### 4. Predict

Place the MP3 files you want to classify into the `predict/` folder, then run:

```bash
python predict.py
```

Outputs a table with the filename, predicted category, and confidence score:

```
File              Prediction    Confidence
----------------  ------------  ----------
track_a.mp3       peak          91.40%
track_b.mp3       starters      78.20%
track_c.mp3       middleground  65.00%
```
