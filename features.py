import librosa
import numpy as np

def extract_features(file_path):
    duration = librosa.get_duration(path=file_path)
    offset = max(0.0, (duration - 180) / 2)
    load_duration = min(180.0, duration)
    y, sr = librosa.load(file_path, offset=offset, duration=load_duration)  # middle 3 minutes

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Energy
    rms = np.mean(librosa.feature.rms(y=y))

    # Spectral centroid (brightness)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Zero-crossing rate (noisiness)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # MFCCs (very important)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    return np.hstack([tempo, rms, centroid, zcr, mfcc_mean])
