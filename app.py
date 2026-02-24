import streamlit as st
import traceback

from audio_processing import load_audio, detect_cough_segments, count_coughs
from feature_extraction import aggregate_features
from cough_classifier import compute_wetness
from disease_model import create_feature_vector, predict_disease

st.set_page_config(page_title="AI Respiratory Screening", layout="wide")

st.title("🫁 AI Respiratory Screening System")

st.markdown("Upload a cough recording and complete the questionnaire for analysis.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Cough WAV File", type=["wav"])

with col2:
    st.subheader("Patient Information")
    age = st.number_input("Age", 1, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.checkbox("Smoking")
    wheezing = st.checkbox("Wheezing")
    mucus = st.checkbox("Mucus")
    difficulty_breathing = st.checkbox("Difficulty Breathing")

if st.button("Run Screening"):

    try:
        if uploaded_file is None:
            st.error("Please upload a WAV file.")
        else:
            y, sr = load_audio(uploaded_file)

            cough_segments = detect_cough_segments(y, sr)
            cough_count = count_coughs(cough_segments)

            audio_features = aggregate_features(cough_segments, sr)
            wet_count, dry_count, wetness = compute_wetness(cough_segments, sr)

            questionnaire = {
                "age": age,
                "gender": gender,
                "smoking": smoking,
                "wheezing": wheezing,
                "mucus": mucus,
                "difficulty_breathing": difficulty_breathing
            }

            feature_vector = create_feature_vector(audio_features, questionnaire)
            predictions = predict_disease(feature_vector)

            st.subheader("📊 Cough Analysis")
            st.write(f"Total Cough Events: **{cough_count}**")
            st.write(f"Wet Coughs: **{wet_count}**")
            st.write(f"Dry Coughs: **{dry_count}**")
            st.write(f"Wetness Percentage: **{wetness:.2f}%**")

            st.subheader("🩺 Disease Risk Assessment")

            for disease, prob in predictions.items():
                percentage = int(prob * 100)

                if percentage < 30:
                    color = "🟢 Low"
                elif percentage < 60:
                    color = "🟡 Medium"
                else:
                    color = "🔴 High"

                st.markdown(f"### {disease.upper()}")
                st.progress(percentage)
                st.write(f"Risk Level: {color} ({percentage}%)")

            st.info("⚠ This is an AI-based screening tool. Please consult a healthcare professional for medical advice.")

    except Exception:
        st.error("Something went wrong.")
        st.text(traceback.format_exc())