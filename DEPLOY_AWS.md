# AWS Deployment Guide — Project CV-18 Deepfake Detection System

---

## AWS Services for Deepfake Detection

### 1. Ready-to-Use AI (No Model Needed)

| Service                    | What it does                                                                 | When to use                                        |
|----------------------------|------------------------------------------------------------------------------|----------------------------------------------------|
| **Amazon Rekognition**     | Detect face quality, liveness, and manipulation signals                      | Preprocessing step for deepfake detection          |
| **Amazon Bedrock**         | Claude Vision for deepfake analysis and manipulation detection via prompt    | When you need AI-assisted deepfake analysis        |
| **AWS SageMaker**          | Deploy EfficientNet-B0 or custom deepfake detector as managed endpoint       | When you need managed inference at scale           |

> For deepfake detection, **AWS SageMaker** with your fine-tuned EfficientNet-B0 is the recommended managed option. Amazon Rekognition Face Liveness can detect presentation attacks.

### 2. Host Your Own Model (Keep Current Stack)

| Service                    | What it does                                                        | When to use                                           |
|----------------------------|---------------------------------------------------------------------|-------------------------------------------------------|
| **AWS App Runner**         | Run backend container — simplest, no VPC or cluster needed          | Quickest path to production                           |
| **Amazon ECS Fargate**     | Run backend + cv-service containers in a private VPC                | Best match for your current microservice architecture |
| **Amazon ECR**             | Store your Docker images                                            | Used with App Runner, ECS, or EKS                     |

### 3. Train and Manage Your Model

| Service                         | What it does                                                        | When to use                                           |
|---------------------------------|---------------------------------------------------------------------|-------------------------------------------------------|
| **AWS SageMaker**               | Fine-tune EfficientNet-B0 on FaceForensics++, deploy endpoints      | Full ML pipeline for deepfake detection               |
| **Amazon S3**                   | Store FaceForensics++ dataset and trained model weights             | Dataset and model artifact storage                    |

### 4. Frontend Hosting

| Service               | What it does                                                                  |
|-----------------------|-------------------------------------------------------------------------------|
| **Amazon S3**         | Host your React build as a static website                                     |
| **Amazon CloudFront** | CDN in front of S3 — HTTPS, low latency globally                              |

### 5. Supporting Services

| Service                  | Purpose                                                                   |
|--------------------------|---------------------------------------------------------------------------|
| **Amazon S3**            | Store uploaded images and detection results                               |
| **AWS Secrets Manager**  | Store API keys and connection strings instead of .env files               |
| **Amazon CloudWatch**    | Track detection latency, verdict distributions, request volume            |

---

## Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  S3 + CloudFront — React Frontend                           │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTPS
┌──────────────────────▼──────────────────────────────────────┐
│  AWS App Runner / ECS Fargate — Backend (FastAPI :8000)     │
└──────────────────────┬──────────────────────────────────────┘
                       │ Internal
        ┌──────────────┴──────────────┐
        │ Option A                    │ Option B
        ▼                             ▼
┌───────────────────┐    ┌────────────────────────────────────┐
│ ECS Fargate       │    │ SageMaker Endpoint                 │
│ CV Service :8001  │    │ EfficientNet-B0 deepfake detector  │
│ PyTorch EfficientNet│  │ Managed inference at scale         │
└───────────────────┘    └────────────────────────────────────┘
```

---

## Prerequisites

```bash
aws configure
AWS_REGION=eu-west-2
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
```

---

## Step 1 — Create ECR and Push Images

```bash
aws ecr create-repository --repository-name deepfake/cv-service --region $AWS_REGION
aws ecr create-repository --repository-name deepfake/backend --region $AWS_REGION
ECR=$AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR
docker build -f docker/Dockerfile.cv-service -t $ECR/deepfake/cv-service:latest ./cv-service
docker push $ECR/deepfake/cv-service:latest
docker build -f docker/Dockerfile.backend -t $ECR/deepfake/backend:latest ./backend
docker push $ECR/deepfake/backend:latest
```

---

## Step 2 — Upload Model to S3

```bash
aws s3 mb s3://deepfake-models-$AWS_ACCOUNT --region $AWS_REGION
aws s3 cp cv-service/models/deepfake_efficientnet.pth s3://deepfake-models-$AWS_ACCOUNT/models/
```

---

## Step 3 — Deploy with App Runner

```bash
aws apprunner create-service \
  --service-name deepfake-backend \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "'$ECR'/deepfake/backend:latest",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {
        "Port": "8000",
        "RuntimeEnvironmentVariables": {
          "CV_SERVICE_URL": "http://cv-service:8001"
        }
      }
    }
  }' \
  --instance-configuration '{"Cpu": "1 vCPU", "Memory": "2 GB"}' \
  --region $AWS_REGION
```

---

## Option B — Use Amazon Bedrock for Deepfake Analysis

```python
import boto3, json, base64

bedrock = boto3.client("bedrock-runtime", region_name="eu-west-2")

def detect_deepfake(image_bytes: bytes) -> dict:
    image_b64 = base64.b64encode(image_bytes).decode()
    prompt = """Analyze this face image for signs of AI generation or deepfake manipulation.
Look for: unnatural skin texture, inconsistent lighting, blurry edges, artifacts around hair/ears.
Return JSON: {verdict: "real"|"fake", confidence: 0-100, signals: ["list of detected signals"]}"""
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": prompt}
            ]}]
        }),
        contentType="application/json"
    )
    return json.loads(json.loads(response["body"].read())["content"][0]["text"])
```

---

## Estimated Monthly Cost

| Service                    | Tier              | Est. Cost          |
|----------------------------|-------------------|--------------------|
| App Runner (backend)       | 1 vCPU / 2 GB     | ~$20–25/month      |
| App Runner (cv-service)    | 1 vCPU / 2 GB     | ~$20–25/month      |
| ECR + S3 + CloudFront      | Standard          | ~$3–7/month        |
| Amazon Bedrock (Claude)    | Pay per token     | ~$5–15/month       |
| **Total (Option A)**       |                   | **~$43–57/month**  |
| **Total (Option B)**       |                   | **~$28–47/month**  |

For exact estimates → https://calculator.aws

---

## Teardown

```bash
aws ecr delete-repository --repository-name deepfake/backend --force
aws ecr delete-repository --repository-name deepfake/cv-service --force
aws s3 rm s3://deepfake-models-$AWS_ACCOUNT --recursive
aws s3 rb s3://deepfake-models-$AWS_ACCOUNT
```
