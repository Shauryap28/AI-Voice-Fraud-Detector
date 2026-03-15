# AI Voice Fraud Detector

AI-powered system that detects whether an audio recording is **Human speech or AI-generated speech** using **Wav2Vec2 embeddings and a machine learning classifier**.

This project was developed during the **India AI Impact Buildathon**, where our team ranked in the **Top 2% among 40,000+ participants across India**.

---

## 🏆 Achievement

**Top 2% National Finalist – India AI Impact Buildathon**

(![Finalist Certificate](Certificate/HCL%20GUVI%20Certification%20-%20148i0k54718g7C1j1x.png))

---

## Project Overview

This project provides a **FastAPI-based voice authenticity detection API** that classifies audio recordings as **human or AI-generated**.

The system accepts **Base64 encoded audio**, extracts deep speech embeddings using **Wav2Vec2**, and applies a trained machine learning classifier to generate a prediction along with a **confidence score**.

This system can be used to detect:

* AI-generated voice
* Deepfake speech
* Synthetic audio fraud

---

## Key Features

* Accepts **Base64 encoded audio input**
* Classifies **Human vs AI-generated speech**
* Uses **Wav2Vec2 deep audio embeddings**
* Machine learning classifier with **PCA dimensionality reduction**
* **FastAPI REST API**
* **API key authentication**
* **Rate limiting**
* **Prediction caching**

---

## Model Pipeline

1. **Audio preprocessing**

   * Silence trimming
   * Audio normalization

2. **Feature extraction**

   * Wav2Vec2 model extracts deep speech embeddings

3. **Feature processing**

   * Feature scaling
   * PCA dimensionality reduction

4. **Classification**

   * ML classifier predicts **Human vs AI-generated voice**

5. **Output**

   * Classification result
   * Confidence score

---

## My Contributions

* Collected and organized the **training dataset**
* Assisted in **building and training the ML model**
* Implemented the **Base64 audio classification pipeline**
* Worked on **voice detection and evaluation**
* Contributed to the **development and testing of the detection system**

---

## Project Structure

```
project-root/
│
├── app/
│   ├── models/
│   │   ├── scaler.joblib
│   │   ├── pca.joblib
│   │   └── classifier.joblib
│   │
│   ├── feature_extractor.py
│   ├── inference.py
│   ├── main.py
│   ├── rate_limiter.py
│   ├── schemas.py
│   ├── security.py
│   └── utils.py
│
├── Dataset/
│
├── train_model.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

* Python
* FastAPI
* PyTorch
* HuggingFace Transformers
* Wav2Vec2
* Scikit-learn
* LightGBM
* Librosa

---

## API Endpoint

### POST `/api/voice-detection`

Detect whether the audio is **AI-generated or human voice**.

#### Headers

```
x-api-key: your_secret_api_key
Content-Type: application/json
```

#### Request Body

```
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_AUDIO_STRING"
}
```

#### Success Response

```
{
  "status": "success",
  "classification": "HUMAN",
  "confidenceScore": 0.92
}
```

---

## Example cURL Request

```
curl -X POST "http://127.0.0.1:8000/api/voice-detection" \
-H "x-api-key: your_secret_api_key" \
-H "Content-Type: application/json" \
-d '{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_STRING"
}'
```

---

## Notes

* Maximum audio length: **12 seconds**
* Maximum audio size: **5 MB**
* Supported formats: **MP3**

---

## Future Improvements

* Redis-based rate limiting
* GPU batch inference
* Torch-based classifier head
* Multi-language calibration
* Real-time streaming support

---

## Disclaimer

This repository is a **fork of the original team submission for the India AI Impact Buildathon**.
The project was developed collaboratively as part of a team effort.
