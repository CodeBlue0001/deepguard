# backend/model/preprocessing.py
"""
Frame and audio preprocessing for DeepGuard inference.
Handles base64 frame decoding, face cropping, normalization,
and audio feature extraction from raw frequency data.
"""

import io
import base64
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from loguru import logger

# ─── Image transforms (ImageNet normalization) ────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

video_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def decode_frame(frame_b64: str) -> torch.Tensor:
    """
    Decode a base64-encoded JPEG/PNG frame string to a normalized (1,3,224,224) tensor.
    The extension sends frames as base64 DataURL strings.
    """
    try:
        # Strip DataURL prefix if present
        if ',' in frame_b64:
            frame_b64 = frame_b64.split(',', 1)[1]

        img_bytes = base64.b64decode(frame_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        tensor = video_transform(img).unsqueeze(0)  # (1, 3, 224, 224)
        return tensor
    except Exception as e:
        logger.warning(f"Frame decode failed: {e} — using blank frame")
        return torch.zeros(1, 3, 224, 224)


def decode_frame_bytes(frame_bytes: bytes) -> torch.Tensor:
    """Decode raw bytes (from DB replay buffer) to tensor."""
    try:
        img = Image.open(io.BytesIO(frame_bytes)).convert('RGB')
        return video_transform(img).unsqueeze(0)
    except Exception as e:
        logger.warning(f"Frame bytes decode failed: {e}")
        return torch.zeros(1, 3, 224, 224)


# ─── Audio features ───────────────────────────────────────────────────────────

def process_audio_features(raw_features: list) -> torch.Tensor:
    """
    Normalize and pad/trim raw audio frequency features to 128-d.
    The extension sends a Float32Array of frequency bin values.
    """
    if not raw_features:
        return torch.zeros(1, 128)

    arr = np.array(raw_features, dtype=np.float32)

    # Normalize from dB range [-160, 0] to [0, 1]
    arr = np.clip(arr, -160, 0)
    arr = (arr + 160) / 160.0

    # Resize to exactly 128 dims
    if len(arr) > 128:
        # Downsample by averaging
        indices = np.linspace(0, len(arr) - 1, 128).astype(int)
        arr = arr[indices]
    elif len(arr) < 128:
        arr = np.pad(arr, (0, 128 - len(arr)), mode='constant')

    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, 128)


# ─── Lip-sync score from extension ────────────────────────────────────────────

def process_lipsync_score(lipsync_data: dict) -> float:
    """
    The extension sends raw lip-sync correlation data.
    We convert to a 0-1 score.
    """
    if not lipsync_data:
        return 0.0
    desync_frames = lipsync_data.get('desync_frames', 0)
    total_frames  = lipsync_data.get('total_frames', 1)
    if total_frames == 0:
        return 0.0
    return min(float(desync_frames) / float(total_frames), 1.0)


# ─── Frame compression for DB storage ─────────────────────────────────────────

def compress_frame_for_storage(frame_b64: str, quality: int = 60) -> bytes:
    """Compress frame to JPEG bytes for storage in feedback replay buffer."""
    try:
        if ',' in frame_b64:
            frame_b64 = frame_b64.split(',', 1)[1]
        img_bytes = base64.b64decode(frame_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((112, 112), Image.LANCZOS)  # downsize for storage
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Frame compression failed: {e}")
        return b''
