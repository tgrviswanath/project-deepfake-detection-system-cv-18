# Project 18 - Deepfake Detection System (CV)

Detects whether a face image is real or AI-generated (deepfake) using a fine-tuned EfficientNet-B0 CNN trained on FaceForensics++.

## Architecture

```
Frontend :3000  →  Backend :8000  →  CV Service :8001
  React/MUI        FastAPI/httpx      FastAPI/PyTorch EfficientNet
```

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

## What's Different from Earlier Projects

| | Project 02 (Face Detection) | Project 18 (Deepfake Detection) |
|---|---|---|
| Task | Locate faces | Classify real vs fake |
| Model | OpenCV DNN SSD | EfficientNet-B0 (PyTorch) |
| Output | Bounding boxes | Real/Fake verdict + probability |
| Training | Pretrained only | Fine-tuned on FaceForensics++ |

## Local Run

```bash
# Terminal 1 - CV Service
cd cv-service && python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001

# Terminal 2 - Backend
cd backend && python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 3 - Frontend
cd frontend && npm install && npm start
```

- CV Service docs: http://localhost:8001/docs
- Backend docs:   http://localhost:8000/docs
- UI:             http://localhost:3000

## Docker

```bash
docker-compose up --build
```

## Dataset
FaceForensics++ — download via their official script from GitHub.
Model weights: EfficientNet-B0 pretrained from torchvision, fine-tuned locally.
