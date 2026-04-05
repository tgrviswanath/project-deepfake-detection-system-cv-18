# Project CV-18 - Deepfake Detection System

Microservice CV system that detects whether a face image is real or AI-generated (deepfake) using a fine-tuned EfficientNet-B0 CNN trained on FaceForensics++.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  FRONTEND  (React - Port 3000)                              │
│  axios POST /api/v1/detect                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP JSON
┌──────────────────────▼──────────────────────────────────────┐
│  BACKEND  (FastAPI - Port 8000)                             │
│  httpx POST /api/v1/cv/detect  →  calls cv-service          │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP JSON
┌──────────────────────▼──────────────────────────────────────┐
│  CV SERVICE  (FastAPI - Port 8001)                          │
│  OpenCV face crop → EfficientNet-B0 → sigmoid output        │
│  Returns { verdict, confidence, face_crop }                 │
└─────────────────────────────────────────────────────────────┘
```

---

## How It Works

```
Image uploaded
    ↓
OpenCV face crop (SSD ResNet-10)
    ↓
EfficientNet-B0 forward pass (ImageNet pretrained, fine-tuned)
    ↓
Sigmoid output → real / fake probability
    ↓
Return: verdict + confidence + face_crop (base64)
```

---

## What's Different from CV-02 (Face Detection)

| | CV-02 Face Detection | CV-18 Deepfake Detection |
|---|---|---|
| Task | Locate faces | Classify real vs fake |
| Model | OpenCV DNN SSD | EfficientNet-B0 (PyTorch) |
| Output | Bounding boxes | Real/Fake verdict + probability |
| Training | Pretrained only | Fine-tuned on FaceForensics++ |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Frontend | React, MUI |
| Backend | FastAPI, httpx |
| CV | PyTorch EfficientNet-B0, OpenCV DNN SSD, Pillow |
| Dataset | FaceForensics++ |
| Deployment | Docker, docker-compose |

---

## Prerequisites

- Python 3.12+
- Node.js — run `nvs use 20.14.0` before starting the frontend

---

## Local Run

### Step 1 — Start CV Service (Terminal 1)

```bash
cd cv-service
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
# EfficientNet-B0 pretrained weights auto-downloaded on first request
```

Verify: http://localhost:8001/health → `{"status":"ok"}`

### Step 2 — Start Backend (Terminal 2)

```bash
cd backend
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Step 3 — Start Frontend (Terminal 3)

```bash
cd frontend
npm install && npm start
```

Opens at: http://localhost:3000

---

## Environment Files

### `backend/.env`

```
APP_NAME=Deepfake Detection API
APP_VERSION=1.0.0
ALLOWED_ORIGINS=["http://localhost:3000"]
CV_SERVICE_URL=http://localhost:8001
```

### `frontend/.env`

```
REACT_APP_API_URL=http://localhost:8000
```

---

## Docker Run

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API docs | http://localhost:8000/docs |
| CV Service docs | http://localhost:8001/docs |

---

## Run Tests

```bash
cd cv-service && venv\Scripts\activate
pytest ../tests/cv-service/ -v

cd backend && venv\Scripts\activate
pytest ../tests/backend/ -v
```

---

## Project Structure

```
project-deepfake-detection-system-cv-18/
├── frontend/                    ← React (Port 3000)
├── backend/                     ← FastAPI (Port 8000)
├── cv-service/                  ← FastAPI CV (Port 8001)
│   └── app/
│       ├── api/routes.py
│       ├── core/face_crop.py    ← OpenCV DNN face crop
│       ├── core/detector.py     ← EfficientNet-B0 inference
│       └── main.py
├── samples/
├── tests/
├── docker/
└── docker-compose.yml
```

---

## API Reference

```
POST /api/v1/detect
Body:     { "image": "<base64>" }
Response: {
  "verdict": "fake",
  "confidence": 91.4,
  "real_probability": 8.6,
  "fake_probability": 91.4,
  "face_crop": "<base64>"
}
```

---

## Dataset

FaceForensics++ — download via their official script from GitHub.
Model weights: EfficientNet-B0 pretrained from torchvision, fine-tuned locally.
