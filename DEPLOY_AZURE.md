# Azure Deployment Guide — Project CV-18 Deepfake Detection System

---

## Azure Services for Deepfake Detection

### 1. Ready-to-Use AI (No Model Needed)

| Service                              | What it does                                                                 | When to use                                        |
|--------------------------------------|------------------------------------------------------------------------------|----------------------------------------------------|
| **Azure AI Content Safety**          | Detect manipulated and synthetic content in images                           | When you need managed content authenticity checks  |
| **Azure Machine Learning**           | Deploy EfficientNet-B0 deepfake detector as managed endpoint on GPU          | When you need managed inference at scale           |
| **Azure OpenAI Vision**              | GPT-4V for deepfake analysis and manipulation detection via prompt           | When you need AI-assisted deepfake analysis        |

> For deepfake detection, **Azure Machine Learning** with your fine-tuned EfficientNet-B0 is the recommended managed option. Azure AI Content Safety can detect synthetic/manipulated content.

### 2. Host Your Own Model (Keep Current Stack)

| Service                        | What it does                                                        | When to use                                           |
|--------------------------------|---------------------------------------------------------------------|-------------------------------------------------------|
| **Azure Container Apps**       | Run your 3 Docker containers (frontend, backend, cv-service)        | Best match for your current microservice architecture |
| **Azure Container Registry**   | Store your Docker images                                            | Used with Container Apps or AKS                       |

### 3. Train and Manage Your Model

| Service                        | What it does                                                              | When to use                                           |
|--------------------------------|---------------------------------------------------------------------------|-------------------------------------------------------|
| **Azure Machine Learning**     | Fine-tune EfficientNet-B0 on FaceForensics++, deploy managed endpoints    | Full ML pipeline for deepfake detection               |

### 4. Frontend Hosting

| Service                   | What it does                                                               |
|---------------------------|----------------------------------------------------------------------------|
| **Azure Static Web Apps** | Host your React frontend — free tier available, auto CI/CD from GitHub     |

### 5. Supporting Services

| Service                       | Purpose                                                                  |
|-------------------------------|--------------------------------------------------------------------------|
| **Azure Blob Storage**        | Store uploaded images and detection results                              |
| **Azure Key Vault**           | Store API keys and connection strings instead of .env files              |
| **Azure Monitor + App Insights** | Track detection latency, verdict distributions, request volume       |

---

## Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Azure Static Web Apps — React Frontend                     │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTPS
┌──────────────────────▼──────────────────────────────────────┐
│  Azure Container Apps — Backend (FastAPI :8000)             │
└──────────────────────┬──────────────────────────────────────┘
                       │ Internal
        ┌──────────────┴──────────────┐
        │ Option A                    │ Option B
        ▼                             ▼
┌───────────────────┐    ┌────────────────────────────────────┐
│ Container Apps    │    │ Azure ML Managed Endpoint          │
│ CV Service :8001  │    │ EfficientNet-B0 deepfake detector  │
│ PyTorch EfficientNet│  │ Managed inference at scale         │
└───────────────────┘    └────────────────────────────────────┘
```

---

## Prerequisites

```bash
az login
az group create --name rg-deepfake-detection --location uksouth
az extension add --name containerapp --upgrade
az extension add --name ml --upgrade
```

---

## Step 1 — Create Container Registry and Push Images

```bash
az acr create --resource-group rg-deepfake-detection --name deepfakeacr --sku Basic --admin-enabled true
az acr login --name deepfakeacr
ACR=deepfakeacr.azurecr.io
docker build -f docker/Dockerfile.cv-service -t $ACR/cv-service:latest ./cv-service
docker push $ACR/cv-service:latest
docker build -f docker/Dockerfile.backend -t $ACR/backend:latest ./backend
docker push $ACR/backend:latest
```

---

## Step 2 — Upload Model to Blob Storage

```bash
az storage account create --name deepfakemodels --resource-group rg-deepfake-detection --sku Standard_LRS
az storage container create --name models --account-name deepfakemodels
az storage blob upload --account-name deepfakemodels --container-name models \
  --name deepfake_efficientnet.pth --file cv-service/models/deepfake_efficientnet.pth
```

---

## Step 3 — Deploy Container Apps

```bash
az containerapp env create --name deepfake-env --resource-group rg-deepfake-detection --location uksouth

az containerapp create \
  --name cv-service --resource-group rg-deepfake-detection \
  --environment deepfake-env --image $ACR/cv-service:latest \
  --registry-server $ACR --target-port 8001 --ingress internal \
  --min-replicas 1 --max-replicas 3 --cpu 1 --memory 2.0Gi

az containerapp create \
  --name backend --resource-group rg-deepfake-detection \
  --environment deepfake-env --image $ACR/backend:latest \
  --registry-server $ACR --target-port 8000 --ingress external \
  --min-replicas 1 --max-replicas 5 --cpu 0.5 --memory 1.0Gi \
  --env-vars CV_SERVICE_URL=http://cv-service:8001
```

---

## Option B — Use Azure OpenAI Vision for Deepfake Analysis

```python
from openai import AzureOpenAI
import base64, json

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01"
)

def detect_deepfake(image_bytes: bytes) -> dict:
    image_b64 = base64.b64encode(image_bytes).decode()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": "Analyze this face image for deepfake signs. Return JSON: {verdict: 'real'|'fake', confidence: 0-100, signals: ['list of detected signals']}"}
        ]}]
    )
    return json.loads(response.choices[0].message.content)
```

---

## Estimated Monthly Cost

| Service                  | Tier      | Est. Cost          |
|--------------------------|-----------|--------------------|
| Container Apps (backend) | 0.5 vCPU  | ~$10–15/month      |
| Container Apps (cv-svc)  | 1 vCPU    | ~$15–20/month      |
| Container Registry       | Basic     | ~$5/month          |
| Static Web Apps          | Free      | $0                 |
| Azure OpenAI (GPT-4o)    | Pay per token | ~$5–15/month   |
| **Total (Option A)**     |           | **~$30–40/month**  |
| **Total (Option B)**     |           | **~$20–35/month**  |

For exact estimates → https://calculator.azure.com

---

## Teardown

```bash
az group delete --name rg-deepfake-detection --yes --no-wait
```
