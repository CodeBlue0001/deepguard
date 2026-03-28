# backend/api/main.py
"""
DeepGuard FastAPI Backend
Routes:
  POST /detect          — run detection on a video frame
  POST /feedback        — submit user feedback + trigger instant update
  GET  /model/status    — current model version, adapter stats
  GET  /model/history   — training run history
  GET  /stats           — detection stats dashboard
  GET  /health          — health check
"""

import uuid
import json
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update, desc
from pydantic import BaseModel, Field
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from db.models import init_db, get_db, Detection, Feedback, TrainingRun, ModelVersion
from model.deepguard_model import model_manager
from model.preprocessing import (
    decode_frame, process_audio_features,
    process_lipsync_score, compress_frame_for_storage
)
from training.trainer import trainer


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    model_manager.load()
    logger.info("DeepGuard API ready")
    yield
    logger.info("DeepGuard API shutting down")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict to your extension origin in prod
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    session_id:     str   = Field(..., description="Anonymous client session ID")
    url:            str   = Field(..., description="Page URL")
    frame_b64:      str   = Field(..., description="Base64-encoded video frame (DataURL)")
    audio_features: List[float] = Field(default=[], description="128-d audio frequency array")
    lipsync_data:   dict  = Field(default={}, description="Lip-sync correlation data")

class DetectResponse(BaseModel):
    detection_id:    str
    score:           float
    base_score:      float
    state:           str
    is_uncertain:    bool
    should_prompt:   bool          # True = auto-show feedback prompt in extension
    model_version:   str
    adapter_updates: int
    signals:         dict

class FeedbackRequest(BaseModel):
    detection_id:   str
    session_id:     str
    user_label:     str            # "fake" | "real" | "unsure"
    trigger:        str            # "auto_prompt" | "manual_report"
    frame_b64:      Optional[str]  = None   # resend frame for replay buffer
    audio_features: List[float]   = []

class FeedbackResponse(BaseModel):
    feedback_id:      str
    instant_updated:  bool
    retrain_triggered: bool
    adapter_updates:  int
    message:          str


# ─── /detect ──────────────────────────────────────────────────────────────────

@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest, db: AsyncSession = Depends(get_db)):
    try:
        video_tensor = decode_frame(req.frame_b64)
        audio_tensor = process_audio_features(req.audio_features)
        lipsync_score = process_lipsync_score(req.lipsync_data)
    except Exception as e:
        raise HTTPException(400, f"Preprocessing failed: {e}")

    result = model_manager.predict(video_tensor, audio_tensor)

    # Blend lipsync into final score (30% weight)
    blended_score = float(result['score']) * 0.7 + lipsync_score * 0.3
    blended_score = round(min(blended_score, 1.0), 4)
    state = model_manager._score_to_state(blended_score)
    is_uncertain = model_manager._is_uncertain(blended_score)

    # Decide whether to auto-prompt for feedback
    # Prompt when: uncertain score OR danger-but-no-prior-correction on this domain
    should_prompt = is_uncertain or (state == 'danger' and blended_score < 0.88)

    # Persist detection
    domain = _extract_domain(req.url)
    detection_id = str(uuid.uuid4())
    db.add(Detection(
        id=detection_id,
        session_id=req.session_id,
        url=req.url,
        domain=domain,
        score=blended_score,
        state=state,
        video_score=result['base_score'],
        audio_score=0.0,
        lipsync_score=lipsync_score,
        model_version=result['model_version'],
    ))
    await db.commit()

    return DetectResponse(
        detection_id=detection_id,
        score=blended_score,
        base_score=result['base_score'],
        state=state,
        is_uncertain=is_uncertain,
        should_prompt=should_prompt,
        model_version=result['model_version'],
        adapter_updates=result['adapter_feedback_count'],
        signals={
            "visual":  result['base_score'],
            "audio":   0.0,
            "lipsync": lipsync_score,
        }
    )


# ─── /feedback ────────────────────────────────────────────────────────────────

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    req: FeedbackRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    if req.user_label not in ('fake', 'real', 'unsure'):
        raise HTTPException(400, "user_label must be 'fake', 'real', or 'unsure'")

    # Retrieve original detection for context
    det = await db.get(Detection, req.detection_id)
    correction = None
    features   = []

    if det:
        model_said_fake = det.score >= settings.WARNING_THRESHOLD
        user_says_fake  = req.user_label == 'fake'
        correction = model_said_fake != user_says_fake  # True = model was wrong

        # Use stored feature vector from in-memory model prediction
        # (We re-run inference on the resent frame if available)
        if req.frame_b64:
            try:
                v = decode_frame(req.frame_b64)
                a = process_audio_features(req.audio_features)
                pred = model_manager.predict(v, a)
                features = pred['features']
            except Exception as e:
                logger.warning(f"Re-inference failed for feedback: {e}")

    # Compress frame for replay buffer
    frame_bytes = None
    if req.frame_b64:
        frame_bytes = compress_frame_for_storage(req.frame_b64)

    feedback_id = str(uuid.uuid4())
    db.add(Feedback(
        id=feedback_id,
        detection_id=req.detection_id,
        session_id=req.session_id,
        user_label=req.user_label,
        trigger=req.trigger,
        model_score=det.score if det else 0.0,
        correction=correction,
        frame_data=frame_bytes,
        audio_features=json.dumps(req.audio_features) if req.audio_features else None,
    ))
    await db.commit()

    # ── Instant update (synchronous, <10ms) ──
    instant_ok = False
    if req.user_label in ('fake', 'real') and features:
        label = 1 if req.user_label == 'fake' else 0
        instant_ok = trainer.instant_update(features, label, feedback_id)

    # ── Check if full retrain should be triggered ──
    retrain_triggered = False
    total_feedback = await _count_unused_feedback(db)

    if total_feedback >= settings.FEEDBACK_BATCH_SIZE:
        retrain_triggered = True
        run_id = str(uuid.uuid4())
        db.add(TrainingRun(
            id=run_id,
            run_type='full_retrain',
            model_version=model_manager.current_version,
            status='pending'
        ))
        await db.commit()
        # Fire Celery task (non-blocking)
        background_tasks.add_task(_fire_retrain_task, run_id)
        logger.info(f"Full retrain triggered: run_id={run_id}, feedback_count={total_feedback}")

    msg = _build_feedback_message(req.user_label, instant_ok, retrain_triggered, correction)

    return FeedbackResponse(
        feedback_id=feedback_id,
        instant_updated=instant_ok,
        retrain_triggered=retrain_triggered,
        adapter_updates=model_manager.model.adapter.feedback_count,
        message=msg
    )


# ─── /model/status ────────────────────────────────────────────────────────────

@app.get("/model/status")
async def model_status(db: AsyncSession = Depends(get_db)):
    total_detections = (await db.execute(select(func.count(Detection.id)))).scalar()
    total_feedback   = (await db.execute(select(func.count(Feedback.id)))).scalar()
    corrections      = (await db.execute(
        select(func.count(Feedback.id)).where(Feedback.correction == True)
    )).scalar()
    pending_feedback = await _count_unused_feedback(db)

    latest_run = (await db.execute(
        select(TrainingRun).order_by(desc(TrainingRun.created_at)).limit(1)
    )).scalar_one_or_none()

    return {
        "model_version":      model_manager.current_version,
        "adapter_updates":    model_manager.model.adapter.feedback_count,
        "adapter_real_init":  bool(model_manager.model.adapter.real_init),
        "adapter_fake_init":  bool(model_manager.model.adapter.fake_init),
        "total_detections":   total_detections,
        "total_feedback":     total_feedback,
        "model_corrections":  corrections,
        "correction_rate":    round(corrections / total_feedback, 3) if total_feedback else 0,
        "pending_feedback":   pending_feedback,
        "next_retrain_at":    settings.FEEDBACK_BATCH_SIZE - pending_feedback,
        "device":             str(model_manager.device),
        "latest_training_run": {
            "id":     latest_run.id if latest_run else None,
            "status": latest_run.status if latest_run else None,
            "type":   latest_run.run_type if latest_run else None,
            "before": latest_run.before_accuracy if latest_run else None,
            "after":  latest_run.after_accuracy if latest_run else None,
        } if latest_run else None
    }


# ─── /model/history ───────────────────────────────────────────────────────────

@app.get("/model/history")
async def training_history(db: AsyncSession = Depends(get_db)):
    runs = (await db.execute(
        select(TrainingRun).order_by(desc(TrainingRun.created_at)).limit(20)
    )).scalars().all()
    return [
        {
            "id":           r.id,
            "type":         r.run_type,
            "status":       r.status,
            "before_acc":   r.before_accuracy,
            "after_acc":    r.after_accuracy,
            "improvement":  round((r.after_accuracy or 0) - (r.before_accuracy or 0), 4),
            "feedbacks":    r.feedback_count,
            "duration_sec": r.duration_sec,
            "created_at":   r.created_at.isoformat() if r.created_at else None,
        }
        for r in runs
    ]


# ─── /stats ───────────────────────────────────────────────────────────────────

@app.get("/stats")
async def stats(db: AsyncSession = Depends(get_db)):
    by_state = (await db.execute(
        select(Detection.state, func.count(Detection.id))
        .group_by(Detection.state)
    )).all()

    top_domains = (await db.execute(
        select(Detection.domain, func.count(Detection.id).label('n'))
        .where(Detection.state == 'danger')
        .group_by(Detection.domain)
        .order_by(desc('n'))
        .limit(10)
    )).all()

    return {
        "by_state":    {row[0]: row[1] for row in by_state},
        "top_fake_domains": [{"domain": r[0], "detections": r[1]} for r in top_domains],
    }


# ─── /health ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": settings.VERSION,
        "model_loaded": model_manager._loaded
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc
    except Exception:
        return "unknown"

async def _count_unused_feedback(db: AsyncSession) -> int:
    result = await db.execute(
        select(func.count(Feedback.id))
        .where(Feedback.used_in_train == False)
        .where(Feedback.user_label.in_(['fake', 'real']))
    )
    return result.scalar() or 0

def _fire_retrain_task(run_id: str):
    try:
        from training.tasks import run_full_retrain
        run_full_retrain.delay(run_id)
    except Exception as e:
        logger.error(f"Failed to enqueue retrain task: {e}")

def _build_feedback_message(label: str, instant_ok: bool, retrain: bool, correction) -> str:
    parts = []
    if label == 'unsure':
        return "Thanks for your response — we'll keep monitoring this video."
    if instant_ok:
        parts.append("Model adapted instantly to your feedback.")
    if correction:
        parts.append("This was a correction — very valuable for training.")
    if retrain:
        parts.append("Full model retraining triggered in background.")
    return " ".join(parts) if parts else "Feedback recorded. Thank you."


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
