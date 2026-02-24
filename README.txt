🫁 AI Cough-Based Respiratory Screening System

Overview
This project implements a multi-modal AI system for respiratory risk screening using cough audio analysis and questionnaire data.

Features
- Cough segmentation
- Acoustic feature extraction (MFCC, RMS, ZCR, Spectral Centroid)
- Wet/Dry cough classification
- Questionnaire-based feature fusion
- Random Forest disease prediction
- Probability-based risk assessment
- Streamlit web interface

Tech Stack
- Python
- Streamlit
- Librosa
- Scikit-learn
- NumPy / Pandas

How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
