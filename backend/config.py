# backend/config.py
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # App
    APP_NAME: str = "DeepGuard API"
    VERSION: str = "2.0.0"
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database (SQLite for dev, swap to PostgreSQL URL for cloud)
    DATABASE_URL: str = "sqlite+aiosqlite:///./deepguard.db"

    # Model paths
    MODEL_DIR: Path = Path("model/weights")
    BASE_MODEL_PATH: Path = Path("model/weights/deepguard_base.pt")
    ADAPTED_MODEL_PATH: Path = Path("model/weights/deepguard_adapted.pt")

    # Detection thresholds
    DANGER_THRESHOLD: float = 0.72
    WARNING_THRESHOLD: float = 0.45
    UNCERTAINTY_BAND: float = 0.15   # auto-prompt feedback when score is in this band around thresholds

    # Feedback & training
    FEEDBACK_BATCH_SIZE: int = 32          # trigger full retrain after this many feedbacks
    MIN_FEEDBACK_FOR_RETRAIN: int = 10     # minimum feedbacks before any retrain
    INSTANT_UPDATE_MOMENTUM: float = 0.85  # EMA momentum for instant feature updates
    MAX_REPLAY_BUFFER: int = 500           # samples kept for replay during retrain

    # Cloud storage (set in env for production)
    S3_BUCKET: str = ""
    S3_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""

    # Redis (for Celery task queue)
    REDIS_URL: str = "redis://localhost:6379/0"

    # Security
    SECRET_KEY: str = "change-me-in-production-use-secrets-manager"
    API_KEY_HEADER: str = "X-DeepGuard-Key"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
