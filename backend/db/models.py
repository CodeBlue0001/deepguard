# backend/db/models.py
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, LargeBinary
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime
import uuid
from config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class Detection(Base):
    """Every detection event logged by the extension."""
    __tablename__ = "detections"

    id              = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id      = Column(String, nullable=False, index=True)  # anonymous user session
    url             = Column(String, nullable=False)
    domain          = Column(String, nullable=False)
    score           = Column(Float, nullable=False)
    state           = Column(String, nullable=False)              # safe / warning / danger
    video_score     = Column(Float, default=0.0)
    audio_score     = Column(Float, default=0.0)
    lipsync_score   = Column(Float, default=0.0)
    frame_hash      = Column(String, nullable=True)               # perceptual hash of sampled frame
    model_version   = Column(String, nullable=False, default="base")
    created_at      = Column(DateTime, default=datetime.utcnow)

class Feedback(Base):
    """User-submitted feedback on a detection."""
    __tablename__ = "feedbacks"

    id              = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    detection_id    = Column(String, nullable=False, index=True)
    session_id      = Column(String, nullable=False)
    user_label      = Column(String, nullable=False)   # "fake" | "real" | "unsure"
    trigger         = Column(String, nullable=False)   # "auto_prompt" | "manual_report"
    model_score     = Column(Float, nullable=False)    # what the model said
    correction      = Column(Boolean, nullable=True)   # True = model was wrong
    frame_data      = Column(LargeBinary, nullable=True)  # compressed frame bytes for replay
    audio_features  = Column(Text, nullable=True)      # JSON-serialized audio feature vector
    used_in_train   = Column(Boolean, default=False)
    created_at      = Column(DateTime, default=datetime.utcnow)

class TrainingRun(Base):
    """Log of every model training / adaptation event."""
    __tablename__ = "training_runs"

    id              = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_type        = Column(String, nullable=False)   # "instant_update" | "full_retrain"
    feedback_count  = Column(Integer, default=0)
    before_accuracy = Column(Float, nullable=True)
    after_accuracy  = Column(Float, nullable=True)
    model_version   = Column(String, nullable=False)
    duration_sec    = Column(Float, nullable=True)
    status          = Column(String, default="pending")  # pending / running / done / failed
    notes           = Column(Text, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)
    finished_at     = Column(DateTime, nullable=True)

class ModelVersion(Base):
    """Registry of every model checkpoint."""
    __tablename__ = "model_versions"

    id              = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    version         = Column(String, nullable=False, unique=True)
    path            = Column(String, nullable=False)
    is_active       = Column(Boolean, default=False)
    accuracy        = Column(Float, nullable=True)
    feedback_trained_on = Column(Integer, default=0)
    created_at      = Column(DateTime, default=datetime.utcnow)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
