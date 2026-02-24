import os
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# -----------------------------
# PATHS
# -----------------------------
DATASET_PATH = r"C:\Users\a\Documents\cough_screening_system\coughz"
CSV_PATH = r"C:\Users\a\Documents\cough_screening_system\cough_labels.csv"

os.makedirs("models", exist_ok=True)

df = pd.read_csv(CSV_PATH)

# -----------------------------
# AUDIO FEATURE EXTRACTION
# -----------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc)

    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    return np.array([mfcc_mean, rms, zcr, centroid])


# -----------------------------
# BUILD DATASET
# -----------------------------
X = []
y_labels = []

for _, row in df.iterrows():
    filename = row["filename"]
    label = row["label"]

    file_path = os.path.join(DATASET_PATH, filename)

    if os.path.exists(file_path):

        audio_features = extract_features(file_path)

        # Synthetic questionnaire features (for training only)
        age = np.random.randint(18, 70)
        gender = np.random.randint(0, 2)  # 0 = Male, 1 = Female
        smoking = np.random.randint(0, 2)
        wheezing = np.random.randint(0, 2)
        mucus = np.random.randint(0, 2)
        difficulty_breathing = np.random.randint(0, 2)

        combined_features = np.hstack([
            audio_features,
            age,
            gender,
            smoking,
            wheezing,
            mucus,
            difficulty_breathing
        ])

        X.append(combined_features)
        y_labels.append(label)

X = np.array(X)
y_labels = np.array(y_labels)

print("Total samples:", len(X))
print("Feature shape:", X.shape)   # MUST be (20, 10)

# -----------------------------
# ENCODE LABELS
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# EVALUATE
# -----------------------------
predictions = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "models/disease_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("\nModel saved successfully.")