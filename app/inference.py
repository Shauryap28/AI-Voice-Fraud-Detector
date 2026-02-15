# app/inference.py

import os
import numpy as np
import joblib
from app.feature_extractor import extract_wav2vec_features
from app.utils import preprocess_audio



# -------------------------
# Label map
# -------------------------
LABEL_MAP = {
    0: "HUMAN",
    1: "AI_GENERATED"
}



# -------------------------
# Load scaler, PCA, classifier
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
pca = joblib.load(os.path.join(MODEL_DIR, "pca.joblib"))
clf = joblib.load(os.path.join(MODEL_DIR, "classifier.joblib"))






# -------------------------
# Build final feature vector
# -------------------------
def build_final_features(wav2vec_feats: np.ndarray) -> np.ndarray:
    feats = wav2vec_feats.reshape(1, -1)

    # scale
    feats_scaled = scaler.transform(feats)

    # PCA
    feats_reduced = pca.transform(feats_scaled)

    return feats_reduced.astype(np.float32)


# -------------------------
# Main inference function
# -------------------------
def predict_voice(audio_base64: str):
    # 1. Preprocess audio
    waveform, sr = preprocess_audio(audio_base64)

    # 2. Extract wav2vec features
    wav_feats = extract_wav2vec_features(waveform, sr)

    # 3. Build final features
    final_features = build_final_features(wav_feats)

    # 4. Predict
    ai_prob = float(clf.predict_proba(final_features)[0][1])

    if ai_prob >= 0.5:
        classification = "AI_GENERATED"
        confidence_score = ai_prob
    else:
        classification = "HUMAN"
        confidence_score = 1.0 - ai_prob



    return classification, float(confidence_score)
