import joblib
import numpy as np

model = joblib.load("models/disease_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

def create_feature_vector(audio_features, questionnaire):

    gender_encoded = 1 if questionnaire["gender"] == "Female" else 0

    return np.array([
        audio_features["mfcc_mean"],
        audio_features["energy"],
        audio_features["zcr"],
        audio_features["spectral_centroid"],
        questionnaire["age"],
        gender_encoded,
        int(questionnaire["smoking"]),
        int(questionnaire["wheezing"]),
        int(questionnaire["mucus"]),
        int(questionnaire["difficulty_breathing"])
    ]).reshape(1, -1)


def predict_disease(feature_vector):
    probabilities = model.predict_proba(feature_vector)[0]
    classes = label_encoder.classes_

    results = {}

    for i, cls in enumerate(classes):
        results[cls] = float(probabilities[i])

    return results