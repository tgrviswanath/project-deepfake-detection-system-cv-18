from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "Deepfake Detection CV Service"
    SERVICE_VERSION: str = "1.0.0"
    SERVICE_PORT: int = 8001
    MODEL_PATH: str = "models/deepfake_efficientnet_b0.pth"
    CONFIDENCE_THRESHOLD: float = 0.5
    MAX_IMAGE_SIZE: int = 1280
    # Face detector model files (auto-downloaded)
    PROTOTXT_PATH: str = "models/deploy.prototxt"
    CAFFEMODEL_PATH: str = "models/res10_300x300_ssd_iter_140000.caffemodel"

    class Config:
        env_file = ".env"


settings = Settings()
