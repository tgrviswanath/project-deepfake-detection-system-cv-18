# GCP Deployment Guide — Project CV-18 Deepfake Detection System

---

## GCP Services for Deepfake Detection

### 1. Ready-to-Use AI (No Model Needed)

| Service                              | What it does                                                                 | When to use                                        |
|--------------------------------------|------------------------------------------------------------------------------|----------------------------------------------------|
| **Vertex AI**                        | Deploy EfficientNet-B0 deepfake detector as managed endpoint on GPU          | When you need managed inference at scale           |
| **Vertex AI Gemini Vision**          | Gemini Pro Vision for deepfake analysis and manipulation detection via prompt| When you need AI-assisted deepfake analysis        |
| **Cloud Vision API**                 | Detect face quality and manipulation signals as preprocessing                | Preprocessing step for deepfake detection          |

> For deepfake detection, **Vertex AI** with your fine-tuned EfficientNet-B0 is the recommended managed option. Vertex AI Gemini Vision can also analyze images for manipulation signals.

### 2. Host Your Own Model (Keep Current Stack)

| Service                    | What it does                                                        | When to use                                           |
|----------------------------|---------------------------------------------------------------------|-------------------------------------------------------|
| **Cloud Run**              | Run backend + cv-service containers — serverless, scales to zero    | Best match for your current microservice architecture |
| **Artifact Registry**      | Store your Docker images                                            | Used with Cloud Run or GKE                            |

### 3. Frontend Hosting

| Service                    | What it does                                                              |
|----------------------------|---------------------------------------------------------------------------| 
| **Firebase Hosting**       | Host your React frontend — free tier, auto CI/CD from GitHub              |

### 4. Supporting Services

| Service                        | Purpose                                                                   |
|--------------------------------|---------------------------------------------------------------------------|
| **Cloud Storage**              | Store uploaded images and detection results                               |
| **Secret Manager**             | Store API keys and connection strings instead of .env files               |
| **Cloud Monitoring + Logging** | Track detection latency, verdict distributions, request volume            |

---

## Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Firebase Hosting — React Frontend                          │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTPS
┌──────────────────────▼──────────────────────────────────────┐
│  Cloud Run — Backend (FastAPI :8000)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │ Internal HTTPS
        ┌──────────────┴──────────────┐
        │ Option A                    │ Option B
        ▼                             ▼
┌───────────────────┐    ┌────────────────────────────────────┐
│ Cloud Run         │    │ Vertex AI Endpoint                 │
│ CV Service :8001  │    │ EfficientNet-B0 deepfake detector  │
│ PyTorch EfficientNet│  │ Managed inference at scale         │
└───────────────────┘    └────────────────────────────────────┘
```

---

## Prerequisites

```bash
gcloud auth login
gcloud projects create deepfake-cv-project --name="Deepfake Detection"
gcloud config set project deepfake-cv-project
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com aiplatform.googleapis.com \
  storage.googleapis.com cloudbuild.googleapis.com
```

---

## Step 1 — Create Artifact Registry and Push Images

```bash
GCP_REGION=europe-west2
gcloud artifacts repositories create deepfake-repo \
  --repository-format=docker --location=$GCP_REGION
gcloud auth configure-docker $GCP_REGION-docker.pkg.dev
AR=$GCP_REGION-docker.pkg.dev/deepfake-cv-project/deepfake-repo
docker build -f docker/Dockerfile.cv-service -t $AR/cv-service:latest ./cv-service
docker push $AR/cv-service:latest
docker build -f docker/Dockerfile.backend -t $AR/backend:latest ./backend
docker push $AR/backend:latest
```

---

## Step 2 — Upload Model to Cloud Storage

```bash
gsutil mb -l $GCP_REGION gs://deepfake-models-deepfake-cv-project
gsutil cp cv-service/models/deepfake_efficientnet.pth gs://deepfake-models-deepfake-cv-project/models/
```

---

## Step 3 — Deploy to Cloud Run

```bash
gcloud run deploy cv-service \
  --image $AR/cv-service:latest --region $GCP_REGION \
  --port 8001 --no-allow-unauthenticated \
  --min-instances 1 --max-instances 3 --memory 2Gi --cpu 1

CV_URL=$(gcloud run services describe cv-service --region $GCP_REGION --format "value(status.url)")

gcloud run deploy backend \
  --image $AR/backend:latest --region $GCP_REGION \
  --port 8000 --allow-unauthenticated \
  --min-instances 1 --max-instances 5 --memory 1Gi --cpu 1 \
  --set-env-vars CV_SERVICE_URL=$CV_URL
```

---

## Option B — Use Vertex AI Gemini Vision

```python
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import base64, json

vertexai.init(project="deepfake-cv-project", location="europe-west2")
gemini = GenerativeModel("gemini-pro-vision")

def detect_deepfake(image_bytes: bytes) -> dict:
    image_part = Part.from_data(data=image_bytes, mime_type="image/jpeg")
    response = gemini.generate_content([
        image_part,
        "Analyze this face image for deepfake signs. Return JSON: {verdict: 'real'|'fake', confidence: 0-100, signals: ['list of detected signals']}"
    ])
    return json.loads(response.text)
```

---

## Estimated Monthly Cost

| Service                    | Tier                  | Est. Cost          |
|----------------------------|-----------------------|--------------------|
| Cloud Run (backend)        | 1 vCPU / 1 GB         | ~$10–15/month      |
| Cloud Run (cv-service)     | 1 vCPU / 2 GB         | ~$12–18/month      |
| Artifact Registry          | Storage               | ~$1–2/month        |
| Firebase Hosting           | Free tier             | $0                 |
| Cloud Storage (models)     | Standard              | ~$1/month          |
| Vertex AI Gemini Vision    | Pay per token         | ~$5–15/month       |
| **Total (Option A)**       |                       | **~$24–36/month**  |
| **Total (Option B)**       |                       | **~$18–33/month**  |

For exact estimates → https://cloud.google.com/products/calculator

---

## Teardown

```bash
gcloud run services delete backend --region $GCP_REGION --quiet
gcloud run services delete cv-service --region $GCP_REGION --quiet
gcloud artifacts repositories delete deepfake-repo --location=$GCP_REGION --quiet
gsutil rm -r gs://deepfake-models-deepfake-cv-project
gcloud projects delete deepfake-cv-project
```
