import librosa
import numpy as np
import joblib

model = joblib.load("models/wet_dry_model.pkl")

def extract_features(segment, sr):

    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)

    rms = np.mean(librosa.feature.rms(y=segment))
    zcr = np.mean(librosa.feature.zero_crossing_rate(segment))
    centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr))

    features = np.hstack([
        mfcc_means,
        rms,
        zcr,
        centroid,
        bandwidth,
        rolloff
    ])

    return features.reshape(1, -1)


def classify_wet_dry(segment, sr):

    features = extract_features(segment, sr)
    prediction = model.predict(features)[0]

    return "wet" if prediction == 1 else "dry"


def compute_wetness(cough_segments, sr):

    wet_count = 0
    dry_count = 0

    for segment in cough_segments:
        result = classify_wet_dry(segment, sr)

        if result == "wet":
            wet_count += 1
        else:
            dry_count += 1

    total = wet_count + dry_count
    wetness_percentage = (wet_count / total) * 100 if total > 0 else 0

    return wet_count, dry_count, wetness_percentage