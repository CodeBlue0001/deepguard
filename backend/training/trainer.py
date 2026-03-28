# backend/training/trainer.py
"""
DeepGuard Hybrid Training Engine

Two modes:
1. Instant update  — called on every feedback, updates OnlineAdapter prototype
                     memory via EMA. No GPU training. Completes in <10ms.

2. Full retrain    — triggered when feedback_count hits FEEDBACK_BATCH_SIZE.
                     Runs as a Celery background task. Fine-tunes the full
                     model (backbone blocks 5-7 + head + fusion) on the
                     accumulated feedback replay buffer, then saves a new
                     checkpoint and swaps it as the active version.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from datetime import datetime
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from model.deepguard_model import model_manager, DeepGuardModel
from model.preprocessing import decode_frame_bytes, process_audio_features


# ─── Replay Buffer Dataset ────────────────────────────────────────────────────

class FeedbackDataset(Dataset):
    """Dataset built from the feedback replay buffer stored in the DB."""

    def __init__(self, samples: list):
        """
        samples: list of dicts with keys:
          frame_data (bytes), audio_features (JSON str), user_label (str)
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Video frame
        frame_tensor = decode_frame_bytes(s['frame_data']) if s.get('frame_data') else \
                       torch.zeros(1, 3, 224, 224)
        frame_tensor = frame_tensor.squeeze(0)  # (3, 224, 224)

        # Audio features
        audio_raw = json.loads(s['audio_features']) if s.get('audio_features') else []
        audio_tensor = process_audio_features(audio_raw).squeeze(0)  # (128,)

        # Label
        label = 1.0 if s['user_label'] == 'fake' else 0.0

        return frame_tensor, audio_tensor, torch.tensor(label, dtype=torch.float32)


# ─── Trainer ──────────────────────────────────────────────────────────────────

class DeepGuardTrainer:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def instant_update(self, features: list, label: int, feedback_id: str):
        """
        Instant prototype update — runs synchronously, no GPU training.
        Called directly from the API on every feedback submission.
        """
        try:
            model_manager.instant_update([features], label)
            logger.info(f"[InstantUpdate] feedback={feedback_id} label={label} "
                        f"total={model_manager.model.adapter.feedback_count}")
            return True
        except Exception as e:
            logger.error(f"[InstantUpdate] Failed: {e}")
            return False

    def full_retrain(self, samples: list, run_id: str) -> dict:
        """
        Full fine-tune on the feedback replay buffer.
        Runs as a background Celery task — do not call synchronously.

        Returns: dict with before/after accuracy and new model version string.
        """
        logger.info(f"[FullRetrain] Starting run {run_id} with {len(samples)} samples")
        start = datetime.utcnow()

        if len(samples) < settings.MIN_FEEDBACK_FOR_RETRAIN:
            return {"status": "skipped", "reason": "insufficient_samples"}

        # ── Build dataset & loader ──
        dataset = FeedbackDataset(samples)
        loader  = DataLoader(
            dataset,
            batch_size=min(8, len(samples)),
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == 'cuda'
        )

        # ── Clone current model for training ──
        train_model = DeepGuardModel().to(self.device)
        train_model.load_state_dict(model_manager.model.state_dict())
        train_model.train()

        # Unfreeze blocks 5+ for fine-tuning
        for name, param in train_model.backbone.named_parameters():
            for part in name.split('.'):
                try:
                    if int(part) >= 5:
                        param.requires_grad = True
                except ValueError:
                    pass

        # ── Optimizer: only fine-tune unfrozen params + head ──
        trainable = [p for p in train_model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable, lr=1e-5, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        criterion = nn.BCEWithLogitsLoss()

        # Evaluate before
        before_acc = self._evaluate(train_model, loader)

        # ── Training loop ──
        best_loss = float('inf')
        epochs = min(20, max(5, len(samples) // 8))

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct    = 0
            total      = 0

            for frames, audio, labels in loader:
                frames = frames.to(self.device)
                audio  = audio.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                adapted, base, _ = train_model(frames, audio)

                # Loss on both adapted and base outputs for stability
                loss = 0.7 * criterion(adapted.unsqueeze(-1), labels.unsqueeze(-1)) + \
                       0.3 * criterion(base.unsqueeze(-1),    labels.unsqueeze(-1))

                loss.backward()
                nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                preds = (adapted.detach() > 0.5).float()
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

            scheduler.step()
            acc = correct / total if total > 0 else 0
            logger.info(f"[FullRetrain] Epoch {epoch+1}/{epochs} "
                        f"loss={epoch_loss/len(loader):.4f} acc={acc:.3f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss

        # Evaluate after
        after_acc = self._evaluate(train_model, loader)

        # ── Swap in retrained model weights ──
        model_manager.model.load_state_dict(train_model.state_dict())

        duration = (datetime.utcnow() - start).total_seconds()
        version  = f"retrained_{run_id[:8]}_{int(datetime.utcnow().timestamp())}"
        model_manager.save(version=version)

        result = {
            "status":       "done",
            "version":      version,
            "before_acc":   round(before_acc, 4),
            "after_acc":    round(after_acc, 4),
            "improvement":  round(after_acc - before_acc, 4),
            "duration_sec": round(duration, 1),
            "samples":      len(samples),
        }
        logger.info(f"[FullRetrain] Complete: {result}")
        return result

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, loader: DataLoader) -> float:
        model.eval()
        correct, total = 0, 0
        for frames, audio, labels in loader:
            frames = frames.to(self.device)
            audio  = audio.to(self.device)
            labels = labels.to(self.device)
            adapted, _, _ = model(frames, audio)
            preds = (adapted > 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        model.train()
        return correct / total if total > 0 else 0.0


trainer = DeepGuardTrainer()
