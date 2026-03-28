# backend/training/tasks.py
"""
Celery background tasks for DeepGuard full retraining.
The API triggers these tasks asynchronously so the HTTP response
is never blocked by a GPU training run.
"""

from celery import Celery
from celery.utils.log import get_task_logger
from datetime import datetime
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

app = Celery('deepguard', broker=settings.REDIS_URL, backend=settings.REDIS_URL)
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_expires=3600,
    worker_max_tasks_per_child=10,   # restart worker after 10 tasks to free GPU memory
    task_acks_late=True,
)

logger = get_task_logger(__name__)


@app.task(bind=True, name='tasks.full_retrain', max_retries=2)
def run_full_retrain(self, run_id: str):
    """
    Triggered when feedback buffer hits FEEDBACK_BATCH_SIZE.
    Fetches all unused feedback from DB, runs full fine-tune,
    and updates the TrainingRun record with results.
    """
    from training.trainer import trainer
    from sqlalchemy.orm import Session
    from sqlalchemy import create_engine, select, update
    from db.models import Feedback, TrainingRun, ModelVersion, Base

    # Use sync SQLAlchemy for Celery (not async)
    sync_url = settings.DATABASE_URL.replace('+aiosqlite', '')
    engine   = create_engine(sync_url)

    with Session(engine) as db:
        # Mark run as running
        db.execute(
            update(TrainingRun).where(TrainingRun.id == run_id)
            .values(status='running')
        )
        db.commit()

        # Fetch unused feedback with frame data
        feedbacks = db.execute(
            select(Feedback)
            .where(Feedback.used_in_train == False)
            .where(Feedback.user_label.in_(['fake', 'real']))
            .where(Feedback.frame_data.isnot(None))
            .order_by(Feedback.created_at.desc())
            .limit(settings.MAX_REPLAY_BUFFER)
        ).scalars().all()

        if not feedbacks:
            db.execute(
                update(TrainingRun).where(TrainingRun.id == run_id)
                .values(status='skipped', notes='No usable feedback', finished_at=datetime.utcnow())
            )
            db.commit()
            return {"status": "skipped"}

        samples = [
            {
                'frame_data':    f.frame_data,
                'audio_features': f.audio_features,
                'user_label':    f.user_label
            }
            for f in feedbacks
        ]

        try:
            result = trainer.full_retrain(samples, run_id)

            # Mark feedbacks as used
            feedback_ids = [f.id for f in feedbacks]
            db.execute(
                update(Feedback)
                .where(Feedback.id.in_(feedback_ids))
                .values(used_in_train=True)
            )

            # Register new model version
            new_version = ModelVersion(
                version=result['version'],
                path=str(settings.ADAPTED_MODEL_PATH),
                is_active=True,
                accuracy=result['after_acc'],
                feedback_trained_on=len(feedbacks)
            )
            # Deactivate old versions
            db.execute(update(ModelVersion).values(is_active=False))
            db.add(new_version)

            # Update training run record
            db.execute(
                update(TrainingRun).where(TrainingRun.id == run_id).values(
                    status='done',
                    before_accuracy=result['before_acc'],
                    after_accuracy=result['after_acc'],
                    feedback_count=len(feedbacks),
                    model_version=result['version'],
                    duration_sec=result['duration_sec'],
                    finished_at=datetime.utcnow()
                )
            )
            db.commit()
            logger.info(f"[Task] Retrain complete: {result}")
            return result

        except Exception as e:
            db.execute(
                update(TrainingRun).where(TrainingRun.id == run_id)
                .values(status='failed', notes=str(e), finished_at=datetime.utcnow())
            )
            db.commit()
            logger.error(f"[Task] Retrain failed: {e}")
            raise self.retry(exc=e, countdown=60)
