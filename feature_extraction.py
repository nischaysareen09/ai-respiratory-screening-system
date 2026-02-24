import librosa
import numpy as np

def extract_features(segment, sr):

    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc)

    rms = librosa.feature.rms(y=segment)
    energy = np.mean(rms)

    zcr = librosa.feature.zero_crossing_rate(segment)
    zcr_mean = np.mean(zcr)

    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    spectral_centroid = np.mean(centroid)

    return {
        "mfcc_mean": mfcc_mean,
        "energy": energy,
        "zcr": zcr_mean,
        "spectral_centroid": spectral_centroid
    }


def aggregate_features(cough_segments, sr):

    if len(cough_segments) == 0:
        return {
            "mfcc_mean": 0,
            "energy": 0,
            "zcr": 0,
            "spectral_centroid": 0
        }

    all_features = []

    for segment in cough_segments:
        feats = extract_features(segment, sr)
        all_features.append(feats)

    aggregated = {}
    for key in all_features[0].keys():
        aggregated[key] = np.mean([f[key] for f in all_features])

    return aggregated