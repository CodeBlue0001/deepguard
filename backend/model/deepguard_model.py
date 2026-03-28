# backend/model/deepguard_model.py
"""
DeepGuard Model — EfficientNet-B4 backbone + multimodal fusion head
with an online adaptation layer that learns from user feedback in real time.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  Input: video frame (224×224) + audio features (128-d)  │
  └────────────────────┬────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
  EfficientNet-B4            Audio Encoder
  (frozen backbone)          (3-layer MLP)
  → 1792-d features          → 128-d features
         │                           │
         └──────────┬────────────────┘
                    ▼
           Cross-Attention Fusion
           (video queries audio)
                    │
                    ▼
           Deepfake Head (MLP)
           → fake_prob [0,1]
                    │
                    ▼
        ┌───────────┴───────────┐
        │  Online Adapter       │  ← updates from feedback without full retrain
        │  (EMA feature store   │
        │   + prototype memory) │
        └───────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from pathlib import Path
from loguru import logger
from config import settings


# ─── Audio Encoder ────────────────────────────────────────────────────────────

class AudioEncoder(nn.Module):
    def __init__(self, in_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Cross-Attention Fusion ────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """Video features attend over audio features to detect A/V desync."""

    def __init__(self, video_dim: int = 256, audio_dim: int = 128, heads: int = 4):
        super().__init__()
        self.proj_v = nn.Linear(video_dim, 256)
        self.proj_a = nn.Linear(audio_dim, 256)
        self.attn   = nn.MultiheadAttention(embed_dim=256, num_heads=heads, batch_first=True)
        self.norm   = nn.LayerNorm(256)
        self.out    = nn.Linear(256, 256)

    def forward(self, video_feat: torch.Tensor, audio_feat: torch.Tensor) -> torch.Tensor:
        v = self.proj_v(video_feat).unsqueeze(1)   # (B, 1, 256)
        a = self.proj_a(audio_feat).unsqueeze(1)   # (B, 1, 256)
        fused, _ = self.attn(v, a, a)              # video queries audio
        fused = self.norm(fused.squeeze(1) + v.squeeze(1))
        return self.out(fused)


# ─── Online Adapter (Prototype Memory) ────────────────────────────────────────

class OnlineAdapter(nn.Module):
    """
    Maintains two prototype vectors (real_centroid, fake_centroid) in feature space.
    On each feedback event, it updates the relevant centroid via EMA.
    At inference, it computes a similarity-based correction to the base model score.
    This allows immediate adaptation without touching the backbone weights.
    """

    def __init__(self, feat_dim: int = 256, momentum: float = 0.85):
        super().__init__()
        self.momentum    = momentum
        self.feat_dim    = feat_dim
        self.feedback_count = 0

        # Prototype centroids (not trained via backprop — updated via EMA)
        self.register_buffer('real_centroid', torch.zeros(feat_dim))
        self.register_buffer('fake_centroid', torch.zeros(feat_dim))
        self.register_buffer('real_init', torch.tensor(False))
        self.register_buffer('fake_init', torch.tensor(False))

        # Confidence calibration: learned scalar per class
        self.real_confidence = nn.Parameter(torch.tensor(1.0))
        self.fake_confidence = nn.Parameter(torch.tensor(1.0))

    @torch.no_grad()
    def update(self, features: torch.Tensor, label: int):
        """
        label: 0 = real, 1 = fake
        features: (feat_dim,) tensor — the fused feature vector for this sample
        """
        feat = F.normalize(features.float(), dim=0)
        m    = self.momentum

        if label == 1:
            if not self.fake_init:
                self.fake_centroid.copy_(feat)
                self.fake_init.fill_(True)
            else:
                self.fake_centroid.mul_(m).add_(feat * (1 - m))
        else:
            if not self.real_init:
                self.real_centroid.copy_(feat)
                self.real_init.fill_(True)
            else:
                self.real_centroid.mul_(m).add_(feat * (1 - m))

        self.feedback_count += 1

    def forward(self, features: torch.Tensor, base_score: torch.Tensor) -> torch.Tensor:
        """
        Returns an adapted score that blends the base model prediction
        with prototype similarity — stronger weight as more feedback is collected.
        """
        if not (self.real_init and self.fake_init):
            return base_score  # not enough feedback yet — return base score unchanged

        feat   = F.normalize(features.float(), dim=-1)
        r_cent = F.normalize(self.real_centroid.unsqueeze(0), dim=-1)
        f_cent = F.normalize(self.fake_centroid.unsqueeze(0), dim=-1)

        sim_real = torch.sum(feat * r_cent, dim=-1) * self.real_confidence.abs()
        sim_fake = torch.sum(feat * f_cent, dim=-1) * self.fake_confidence.abs()

        # Convert similarities to probability
        prototype_score = torch.sigmoid(sim_fake - sim_real)

        # Blend: grow adapter influence with more feedback (caps at 50% weight)
        alpha = min(0.5, self.feedback_count / 200.0)
        return (1 - alpha) * base_score + alpha * prototype_score


# ─── Full DeepGuard Model ──────────────────────────────────────────────────────

class DeepGuardModel(nn.Module):

    def __init__(self):
        super().__init__()

        # ── Video backbone (EfficientNet-B4, ImageNet pretrained) ──
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=True,
            num_classes=0,          # remove classifier head
            global_pool='avg'
        )
        video_feat_dim = self.backbone.num_features  # 1792

        # Freeze early backbone layers, fine-tune from block 5 onward
        self._freeze_backbone(freeze_up_to_block=4)

        # Project backbone output to 256-d
        self.video_proj = nn.Sequential(
            nn.Linear(video_feat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # ── Audio encoder ──
        self.audio_enc = AudioEncoder(in_dim=128, out_dim=128)

        # ── Cross-attention fusion ──
        self.fusion = CrossAttentionFusion(video_dim=256, audio_dim=128)

        # ── Deepfake classification head ──
        self.head = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # ── Online adapter (no backprop — EMA prototype memory) ──
        self.adapter = OnlineAdapter(feat_dim=256, momentum=settings.INSTANT_UPDATE_MOMENTUM)

    def _freeze_backbone(self, freeze_up_to_block: int = 4):
        frozen = 0
        for name, param in self.backbone.named_parameters():
            block_num = None
            for part in name.split('.'):
                if part.startswith('blocks') or part.isdigit():
                    try:
                        block_num = int(part) if part.isdigit() else None
                    except ValueError:
                        pass
            if block_num is None or block_num <= freeze_up_to_block:
                param.requires_grad = False
                frozen += 1
        logger.info(f"Froze {frozen} backbone parameters up to block {freeze_up_to_block}")

    def extract_features(self, video_frame: torch.Tensor, audio_features: torch.Tensor):
        """Returns (fused_features, raw_logit) — used for adapter updates."""
        vid_raw  = self.backbone(video_frame)               # (B, 1792)
        vid_feat = self.video_proj(vid_raw)                 # (B, 256)
        aud_feat = self.audio_enc(audio_features)           # (B, 128)
        fused    = self.fusion(vid_feat, aud_feat)          # (B, 256)
        combined = torch.cat([fused, aud_feat], dim=-1)     # (B, 384)
        logit    = self.head(combined)                      # (B, 1)
        return fused, logit

    def forward(self, video_frame: torch.Tensor, audio_features: torch.Tensor):
        fused, logit = self.extract_features(video_frame, audio_features)
        base_score   = torch.sigmoid(logit).squeeze(-1)     # (B,)
        adapted      = self.adapter(fused, base_score)      # blend with prototypes
        return adapted, base_score, fused


# ─── Model Manager ────────────────────────────────────────────────────────────

class ModelManager:
    """Singleton that loads, serves, and updates the DeepGuard model."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading DeepGuard model on {self.device}")

        self.model = DeepGuardModel().to(self.device)
        self.model.eval()

        path = settings.ADAPTED_MODEL_PATH if settings.ADAPTED_MODEL_PATH.exists() \
               else settings.BASE_MODEL_PATH

        if path.exists():
            state = torch.load(path, map_location=self.device)
            # Load adapter state separately (it's not part of backbone training)
            model_state = {k: v for k, v in state.items() if not k.startswith('adapter.')}
            adapter_state = {k.replace('adapter.', ''): v
                             for k, v in state.items() if k.startswith('adapter.')}
            self.model.load_state_dict(model_state, strict=False)
            if adapter_state:
                self.model.adapter.load_state_dict(adapter_state, strict=False)
            logger.info(f"Loaded model weights from {path}")
        else:
            logger.warning("No pretrained weights found — using ImageNet init only.")
            settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            self.save(version="base_init")

        self._loaded = True
        self.current_version = self._read_version()

    def _read_version(self) -> str:
        vf = settings.MODEL_DIR / "current_version.txt"
        return vf.read_text().strip() if vf.exists() else "base"

    @torch.no_grad()
    def predict(self, video_tensor: torch.Tensor, audio_tensor: torch.Tensor) -> dict:
        self.model.eval()
        v = video_tensor.to(self.device)
        a = audio_tensor.to(self.device)
        adapted_score, base_score, features = self.model(v, a)

        score = float(adapted_score.squeeze())
        return {
            "score":        round(score, 4),
            "base_score":   round(float(base_score.squeeze()), 4),
            "state":        self._score_to_state(score),
            "is_uncertain": self._is_uncertain(score),
            "features":     features.squeeze().cpu().numpy().tolist(),
            "model_version": self.current_version,
            "adapter_feedback_count": self.model.adapter.feedback_count
        }

    def _score_to_state(self, score: float) -> str:
        if score >= settings.DANGER_THRESHOLD:  return "danger"
        if score >= settings.WARNING_THRESHOLD: return "warning"
        return "safe"

    def _is_uncertain(self, score: float) -> bool:
        """True when prediction is near a decision boundary — should trigger feedback prompt."""
        band = settings.UNCERTAINTY_BAND
        near_warn   = abs(score - settings.WARNING_THRESHOLD) < band
        near_danger = abs(score - settings.DANGER_THRESHOLD)  < band
        return near_warn or near_danger

    @torch.no_grad()
    def instant_update(self, features_list: list, label: int):
        """
        Instantly update the online adapter prototype memory with new feedback.
        Does NOT modify backbone weights — safe to call on every feedback event.
        label: 0 = real, 1 = fake
        """
        for feat_list in features_list:
            feat_tensor = torch.tensor(feat_list, dtype=torch.float32).to(self.device)
            self.model.adapter.update(feat_tensor, label)

        self.save(version=f"adapted_{self.model.adapter.feedback_count}")
        logger.info(f"Instant adapter update: label={label}, total_feedback={self.model.adapter.feedback_count}")

    def save(self, version: str):
        settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = settings.ADAPTED_MODEL_PATH
        torch.save(self.model.state_dict(), path)
        vf = settings.MODEL_DIR / "current_version.txt"
        vf.write_text(version)
        self.current_version = version
        logger.info(f"Saved model version '{version}' to {path}")


model_manager = ModelManager()
