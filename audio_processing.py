import librosa
import numpy as np
import soundfile as sf
import io

def load_audio(uploaded_file):
    bytes_data = uploaded_file.read()
    audio_buffer = io.BytesIO(bytes_data)
    y, sr = sf.read(audio_buffer)
    return y, sr

def load_audio(file):
    y, sr = librosa.load(file, sr=None)
    return y, sr


def detect_cough_segments(y, sr, top_db=30, min_duration=0.2):
    intervals = librosa.effects.split(y, top_db=top_db)

    cough_segments = []

    for start, end in intervals:
        duration = (end - start) / sr
        if duration >= min_duration:  # ignore very small noise bursts
            segment = y[start:end]
            cough_segments.append(segment)

    return cough_segments


def count_coughs(cough_segments):
    return len(cough_segments)