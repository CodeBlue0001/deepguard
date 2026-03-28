# DeepGuard v2 — Intelligent Deepfake Detection with Live Learning

A cloud-connected browser extension with a custom AI model, human-in-the-loop
feedback, and a hybrid online learning system that gets smarter with every correction.

---

## System architecture

```
Browser (Chrome / Edge)
│
├── content.js          Intercepts videos, sends frames to cloud API
│   ├── Badge overlay   Green / amber / red on every video
│   ├── Report button   Manual "report as fake" on hover
│   └── Feedback prompt Auto-shown when model is uncertain
│
└── background.js       Toolbar icon + detection history

            │  HTTPS
            ▼

Cloud Backend (FastAPI)
│
├── POST /detect        Runs DeepGuard model, returns score + should_prompt flag
├── POST /feedback      Stores feedback → instant adapter update → maybe triggers retrain
├── GET  /model/status  Current version, adapter stats, correction rate
└── GET  /stats         Detection stats, top fake domains

├── DeepGuard Model
│   ├── EfficientNet-B4 backbone (frozen early layers)
│   ├── Audio encoder (Wav2Vec2-style MLP on FFT features)
│   ├── Cross-attention fusion (video queries audio for lip-sync)
│   ├── Deepfake head (MLP → fake probability)
│   └── Online Adapter (prototype memory, EMA, no backprop)
│
├── Celery Worker        Background full retraining task
└── Redis               Task queue + result backend
```

---

## How the hybrid learning works

### Instant update (every feedback, <10ms)
When a user says "this is fake" or "this is real", the system immediately
updates the **OnlineAdapter**'s prototype memory via EMA (Exponential Moving Average).

- Maintains two prototype vectors: `real_centroid` and `fake_centroid`
- At inference, computes cosine similarity to both prototypes
- Blends prototype score with base model score (adapter weight grows with feedback count, caps at 50%)
- Zero backprop — safe to run on every request

### Full retrain (every N feedbacks, background)
When `FEEDBACK_BATCH_SIZE` feedbacks accumulate, a Celery task:
1. Fetches all unused feedback from the replay buffer (up to 500 samples)
2. Fine-tunes EfficientNet blocks 5–7 + fusion head on the feedback data
3. Evaluates before/after accuracy
4. Saves a new checkpoint and activates it
5. Marks feedback as used

The model gets progressively better at the specific types of deepfakes
your users encounter — **domain-specific immunity**.

---

## Deployment

### Option A — Render.com (easiest, free tier available)

1. Push this repo to GitHub
2. Create a new **Web Service** on Render, connect the repo
3. Set build command: `pip install -r backend/requirements.txt`
4. Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Add env vars:
   - `SECRET_KEY` = any random 32-char string
   - `REDIS_URL` = your Render Redis instance URL
6. Copy your service URL (e.g. `https://deepguard.onrender.com`)
7. Paste it into `extension/content.js` as `API_BASE`

### Option B — Docker Compose (AWS EC2 / GCP VM / any VPS)

```bash
git clone <your-repo>
cd deepguard-system

# Set your secret key
echo "SECRET_KEY=$(openssl rand -hex 32)" > .env

# Build and start everything
docker compose up -d

# View logs
docker compose logs -f api
docker compose logs -f celery_worker

# API is now at http://your-server-ip:8000
```

### Option C — Railway / Fly.io
Same as Render — connect GitHub, set env vars, deploy.
Both support Dockerfile-based deploys with Redis add-ons.

---

## Extension setup

1. Open `extension/content.js`
2. Change `API_BASE` to your deployed URL:
   ```js
   const API_BASE = 'https://your-deepguard-api.onrender.com';
   ```
3. Generate icons:
   ```bash
   cd extension && python generate_icons.py
   ```
4. Load in Chrome: `chrome://extensions` → Developer mode → Load unpacked → `extension/`

---

## Training your base model

The system ships with ImageNet-pretrained EfficientNet-B4 weights.
For serious detection accuracy, fine-tune on deepfake datasets first:

```bash
cd backend

# Install deps
pip install -r requirements.txt

# Download FaceForensics++ dataset (requires registration)
# https://github.com/ondyari/FaceForensics

# Run base training (edit paths in train_base.py)
python training/train_base.py \
  --data_dir /path/to/ff++ \
  --epochs 30 \
  --batch_size 16 \
  --output model/weights/deepguard_base.pt
```

### Recommended datasets (all free)

| Dataset | Samples | Notes |
|---------|---------|-------|
| FaceForensics++ | ~5k videos | Best for visual artifacts |
| FakeAVCeleb | ~20k clips | Audio+video, essential for fusion |
| DFDC | ~120k clips | Most variety, requires Kaggle account |
| CelebDF-v2 | ~6k videos | High visual quality fakes |

---

## API reference

### POST /detect
```json
{
  "session_id":     "anonymous-uuid",
  "url":            "https://youtube.com/watch?v=...",
  "frame_b64":      "data:image/jpeg;base64,...",
  "audio_features": [/* 128 floats from Web Audio API */],
  "lipsync_data":   { "desync_frames": 3, "total_frames": 30 }
}
```
Response:
```json
{
  "detection_id":    "uuid",
  "score":           0.84,
  "base_score":      0.79,
  "state":           "danger",
  "is_uncertain":    false,
  "should_prompt":   true,
  "model_version":   "retrained_abc123",
  "adapter_updates": 47,
  "signals": { "visual": 0.79, "audio": 0.12, "lipsync": 0.31 }
}
```

### POST /feedback
```json
{
  "detection_id":   "uuid",
  "session_id":     "anonymous-uuid",
  "user_label":     "fake",
  "trigger":        "auto_prompt",
  "frame_b64":      "data:image/jpeg;base64,...",
  "audio_features": [/* 128 floats */]
}
```

### GET /model/status
Returns adapter stats, correction rate, next retrain ETA.

### GET /stats
Returns detections by state + top domains with fake detections.

---

## Configuration (backend/.env)

```env
SECRET_KEY=your-random-secret
DATABASE_URL=sqlite+aiosqlite:///./deepguard.db
REDIS_URL=redis://redis:6379/0

# Detection thresholds
DANGER_THRESHOLD=0.72
WARNING_THRESHOLD=0.45
UNCERTAINTY_BAND=0.15

# Training
FEEDBACK_BATCH_SIZE=32
MIN_FEEDBACK_FOR_RETRAIN=10
INSTANT_UPDATE_MOMENTUM=0.85
MAX_REPLAY_BUFFER=500
```

---

## Project structure

```
deepguard-system/
├── docker-compose.yml
├── extension/
│   ├── manifest.json
│   ├── content.js          ← main extension logic (cloud-connected v2)
│   ├── background.js
│   ├── popup.html/css/js
│   ├── overlay.css
│   └── icons/
└── backend/
    ├── Dockerfile
    ├── requirements.txt
    ├── config.py
    ├── api/
    │   └── main.py         ← FastAPI routes
    ├── model/
    │   ├── deepguard_model.py   ← EfficientNet + fusion + OnlineAdapter
    │   └── preprocessing.py
    ├── training/
    │   ├── trainer.py      ← instant update + full retrain engine
    │   └── tasks.py        ← Celery task
    └── db/
        └── models.py       ← SQLAlchemy async models
```
