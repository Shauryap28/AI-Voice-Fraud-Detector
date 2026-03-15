# AI Voice Fraud Detector

A lightweight **FastAPI-based Voice Authenticity Detection API** that determines whether a voice recording is **human or AI-generated** using **Wav2Vec2 embeddings and a machine learning classifier**.

This project was developed during the **India AI Impact Buildathon**, where our team ranked in the **Top 2% among 40,000+ participants across India**.

---

## 🏆 Achievement

**Top 2% National Finalist – India AI Impact Buildathon**

![Finalist Certificate]((![Finalist Certificate](Certificate/HCL%20GUVI%20Certification%20-%20148i0k54718g7C1j1x.png)))

---

## Project Overview

This system detects **AI-generated voice and deepfake speech** using a machine learning pipeline.

The API accepts **Base64 encoded audio**, extracts speech embeddings using **Wav2Vec2**, and predicts whether the audio is **Human or AI-generated**, returning a **confidence score**.

The system is designed for applications such as:

* Deepfake voice detection
* Voice fraud prevention
* AI-generated speech identification

---

## Features

* Accepts **Base64-encoded audio**
* Detects **AI-generated vs human voice**
* Uses **Wav2Vec2 deep audio embeddings**
* **Scikit-learn classifier with PCA**
* **API key authentication**
* **Rate limiting**
* **Response caching**

---

## Project Structure

```
project-root/
│
├── app/
│   ├── models/                # Saved ML artifacts
│   │   ├── scaler.joblib
│   │   ├── pca.joblib
│   │   └── classifier.joblib
│   │
│   ├── feature_extractor.py   # Wav2Vec2 feature extraction
│   ├── inference.py           # Prediction pipeline
│   ├── main.py                # FastAPI application
│   ├── rate_limiter.py        # API rate limiting
│   ├── schemas.py             # Request/response models
│   ├── security.py            # API key validation
│   └── utils.py               # Audio preprocessing
│
├── Dataset/                   # Training dataset
│
├── train_model.py             # Model training script
├── requirements.txt
├── .env
└── README.md
```

---

## Requirements

* Python **3.9+**
* pip
* virtualenv (recommended)

---

## Setup Instructions

### Clone the repository

```
git clone <repository-url>
cd <repository-folder>
```

---

### Create virtual environment

#### Mac / Linux

```
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```
python -m venv venv
venv\Scripts\activate
```

---

### Install dependencies

```
pip install -r requirements.txt
```

---

### Configure environment variables

Create a `.env` file in the root directory:

```
API_KEY=your_secret_api_key
```

---

## Ensure model files exist

Inside:

```
app/models/
```

You must have:

```
scaler.joblib
pca.joblib
classifier.joblib
```

If these files do not exist, train the model.

---

## Training the Model

### Dataset Structure

```
Dataset/
├── Human/
│   ├── English/
│   ├── Hindi/
│   └── ...
│
└── AI/
    ├── English/
    ├── Hindi/
    └── ...
```

Each language folder contains `.wav` or `.mp3` files.

### Run training

```
python train_model.py
```

After training, the following files will be generated:

```
scaler.joblib
pca.joblib
classifier.joblib
```

Move them to:

```
app/models/
```

---

## Running the API

Start the FastAPI server:

```
uvicorn app.main:app --reload
```

Server will run at:

```
http://127.0.0.1:8000
```

Interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

## API Endpoint

### POST `/api/voice-detection`

Detect whether audio is **AI-generated or human voice**.

### Headers

```
x-api-key: your_secret_api_key
Content-Type: application/json
```

---

### Request Body

```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_AUDIO_STRING"
}
```

---

### Success Response

```json
{
  "status": "success",
  "classification": "HUMAN",
  "confidenceScore": 0.92
}
```

---

### Error Response

```json
{
  "status": "error",
  "message": "Invalid API key"
}
```

---

## Rate Limiting

The API allows:

**15 requests per minute per API key**

If exceeded:

```json
{
  "status": "error",
  "message": "Too many requests. Please try again later."
}
```

---

## Caching

Predictions are cached for **5 minutes**.

This prevents recomputation for identical audio inputs.

---

## Model Pipeline

1. **Audio preprocessing**

   * Silence trimming
   * Normalization

2. **Wav2Vec2 embedding extraction**

3. **Feature scaling**

4. **PCA dimensionality reduction**

5. **Classifier prediction**

6. **Prediction output with confidence score**

---

## Technologies Used

* FastAPI
* Python
* PyTorch
* HuggingFace Transformers
* Wav2Vec2
* Scikit-learn
* LightGBM
* Librosa

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

## My Contributions
Collected and organized the training dataset
Assisted in building and training the ML model
Implemented the Base64 audio classification pipeline
Worked on voice detection and evaluation
Contributed to the development and testing of the detection system
---

## Disclaimer

This repository is a **fork of the original team submission for the India AI Impact Buildathon**.
The project was developed collaboratively as part of a team effort.
