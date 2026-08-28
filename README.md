# Image Quality & Defect Detection

Full-stack app: upload an image, get an overall `ACCEPTABLE` / `DEGRADED` /
`DEFECTIVE` quality label, a 0-100 quality score, a structured list of
detected issues (each with a type, severity, and confidence), a Grad-CAM
heatmap explaining the CNN's prediction, and the interpretable classical-CV
feature values behind the decision.

Covers all six required detection capabilities from the assessment brief:
**blur**, **underexposure**, **overexposure**, **noise**, **corruption**,
and **potential visual defect** — see [Issue taxonomy](#issue-taxonomy)
below for how each maps to code.

## Approach

**Hybrid model** — engineered CV features (Laplacian-variance blur, exposure
histogram stats, wavelet-based noise energy, contrast, saturation,
colorfulness, edge density) fused with a fine-tuned ResNet18 backbone
(ImageNet-pretrained, PyTorch). The CNN classifies distortion type +
regresses a quality score; the engineered features stay directly
interpretable and are reported alongside the CNN's prediction, and are also
used to disambiguate cases the CNN's classes leave coarse (see below).

**Fine-tuning over from-scratch** — a pretrained backbone converges to
useful accuracy in hours on a single 6GB-VRAM GPU; training a CNN from
scratch would not, given the assessment's 48-hour window.

**Backbone: ResNet18, chosen over three follow-up attempts to beat it** —
ResNet34 and ConvNeXt-Tiny were both tried in pursuit of higher accuracy;
ResNet18 won on the metric that mattered (test accuracy), while the
alternatives won on other axes (better score-regression MAE) or didn't
finish (environment instability). Full numbers and reasoning: [Model
selection](EVALUATION.md#model-selection-what-was-tried-and-why-the-baseline-won)
in EVALUATION.md. `train.py` still supports all three via `--backbone`.

**Dataset** — synthetically generated: 2,000 diverse clean photos from the
Oxford-IIIT Pet dataset (auto-downloaded via torchvision), degraded
programmatically across 6 distortion types (blur, Gaussian noise,
salt-and-pepper noise, brightness/exposure shift, JPEG compression, block
corruption) at 5 severity levels each — giving labeled ground truth (type +
severity + derived quality score) for free, 38,000 images total. Splits are
by source image so no photo's variants leak across train/val/test (a model
that memorized one variant of a photo could otherwise get credit for
"generalizing" to another variant of the same photo).

### Issue taxonomy

The CNN's 7 distortion classes and the engineered features' raw flags are
both mapped into a canonical issue-type taxonomy that matches the
assessment's required categories exactly (`backend/quality_decision.py`):

| Required capability | Canonical `issue.type` | Triggered by |
|---|---|---|
| Blur / insufficient sharpness | `blur` | CNN `blur` class, or `laplacian_var` < 50 |
| Underexposure | `underexposure` | CNN `brightness` class (direction resolved via `brightness_mean` < 127), or `brightness_mean` < 60 |
| Overexposure | `overexposure` | CNN `brightness` class (direction resolved via `brightness_mean` >= 127), or `brightness_mean` > 200 |
| Image noise | `noise` | CNN `gaussian_noise`/`salt_pepper` classes, or `noise_energy` > 8 |
| Image corruption / severe degradation | `corruption` | CNN `jpeg`/`block_corrupt` classes |
| Potential visual defect | `defect` | CNN `block_corrupt` class specifically (localized, high-frequency artifacts most resembling a physical/sensor defect) — always forces `quality_label = DEFECTIVE` regardless of score |

`low_contrast` is reported as an additional bonus signal (contrast < 20),
justified per the brief's "additional quality issues may be identified if
technically justified."

The CNN's `brightness` class is direction-agnostic by itself — it only
says exposure is off, not which way. The engineered `brightness_mean`
feature resolves the direction. This is the clearest example of why the
hybrid combination is more informative than either signal alone.

### Quality label thresholds

```
if "defect" in issue_types:      DEFECTIVE            (regardless of score)
elif quality_score >= 75:        ACCEPTABLE
elif quality_score >= 40:        DEGRADED
else:                             DEFECTIVE
```

Thresholds align with the synthetic label scale used to build the dataset
(`quality_score = 100 - (severity-1)*20`, so severity 1 ~ 100, severity 5 ~
20): severity-1 (near-clean) lands solidly in ACCEPTABLE, severities 2-3
land in DEGRADED, severities 4-5 land in DEFECTIVE.

## Project layout

```
backend/
  data_pipeline/
    generate_dataset.py     # builds the labeled synthetic dataset
    features.py              # classical CV feature extraction
  model.py                   # hybrid CNN + feature-fusion architecture
  dataset.py                  # torch Dataset + transforms
  train.py                     # two-stage fine-tuning (frozen -> full)
  evaluate.py                   # precision/recall/F1, confusion matrix, MAE/RMSE
  generalization_check.py        # unseen-domain generalization evidence
  quality_decision.py             # canonical issue taxonomy + ACCEPTABLE/DEGRADED/DEFECTIVE label
  inference.py                     # single-image inference + Grad-CAM
  main.py                           # FastAPI app
  db.py                              # SQLite result storage
frontend/
  index.html                         # upload UI, score/issues display, history
models/
  hybrid_model.pt                     # trained checkpoint (generated)
  eval_report.json                     # test-set evaluation (generated)
  generalization_report.json            # unseen-domain check (generated)
samples/                                 # sample images per quality condition, see samples/README.md
data/                                     # generated dataset (not checked in)
docker-compose.yml
EVALUATION.md                              # full evaluation write-up
```

## Running locally (without Docker)

```bash
python -m venv venv
venv/Scripts/activate        # or source venv/bin/activate on Linux/Mac
pip install -r requirements.txt

# 1. generate the dataset (downloads Oxford-IIIT Pet on first run)
cd backend/data_pipeline && python generate_dataset.py --out ../../data

# 2. train (--backbone resnet18 reproduces the shipped model exactly;
#    train.py also supports resnet34 and convnext_tiny -- see
#    "Model selection" in EVALUATION.md for why resnet18 is what shipped)
cd .. && python train.py --data_root ../data --out ../models/hybrid_model.pt --backbone resnet18

# 3. evaluate
python evaluate.py --data_root ../data --checkpoint ../models/hybrid_model.pt

# 4. (optional) generalization check on an unseen-domain dataset
python generalization_check.py --data_root ../data --checkpoint ../models/hybrid_model.pt

# 5. serve
uvicorn main:app --reload --port 8000
```

Open `frontend/index.html` in a browser (served statically, e.g.
`python -m http.server` from the `frontend/` folder) once the backend is
running on `localhost:8000` — the frontend's `API_BASE` constant assumes
that URL (see note in `frontend/index.html`; a static HTML page has no
build-time env-var injection, so this is a plain constant to edit if you
run the backend elsewhere).

## Database setup

No manual setup needed. `db.py` uses SQLite; the table is created
automatically on backend startup (`init_db()` in `main.py`'s startup
event) at `backend/results.db` by default, or at `$DB_DIR/results.db` if
the `DB_DIR` environment variable is set (this is how Docker Compose points
it at the mounted volume — see below).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_CHECKPOINT_PATH` | `../models/hybrid_model.pt` (relative to `backend/`) | Path to the trained model checkpoint to load |
| `DB_DIR` | `backend/` | Directory the SQLite file is created in |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins for the frontend |
| `MAX_UPLOAD_BYTES` | `20971520` (20MB) | Rejects larger uploads with a 400 |

## API examples

Interactive docs (Swagger UI) are auto-generated by FastAPI at
`http://localhost:8000/docs` once the backend is running.

**Health check:**
```bash
curl http://localhost:8000/health
# {"status":"ok","checkpoint_exists":true,"checkpoint_path":"..."}
```

**Analyze an image:**
```bash
curl -X POST http://localhost:8000/analyze -F "file=@samples/02_blur.jpg"
```
```json
{
  "quality_score": 30.2,
  "quality_label": "DEFECTIVE",
  "issues": [
    {"type": "blur", "severity": "high", "confidence": 0.911, "source": "cnn+engineered_features"}
  ],
  "predicted_distortion": "blur",
  "distortion_confidence": 0.911,
  "class_probabilities": {"none": 0.01, "blur": 0.911, "...": "..."},
  "engineered_features": {
    "laplacian_var": 6.2, "brightness_mean": 118.4, "brightness_std": 52.1,
    "contrast": 52.1, "saturation_mean": 96.3, "noise_energy": 1.8,
    "edge_density": 0.02, "colorfulness": 41.7
  },
  "gradcam_heatmap_base64": "<base64 PNG>",
  "id": 1
}
```

**Retrieve a past result:**
```bash
curl http://localhost:8000/results/1
```

**History:**
```bash
curl http://localhost:8000/history?limit=20
```

Invalid uploads are handled gracefully — a non-image content-type, an
empty file, and undecodable/corrupted bytes all return `400` with a
descriptive `detail` message rather than a server error (verified with
each case; see the "Handling invalid uploads" note in EVALUATION.md).

## Running with Docker

```bash
docker compose up --build
```

Frontend: http://localhost:8080 — Backend: http://localhost:8000

Note: the model must be trained (`models/hybrid_model.pt` present) before
building the backend image, since it's copied in at build time. The SQLite
database is stored in a named Docker volume mounted at `/app/db` inside the
container (`DB_DIR=/app/db`), so results persist across container restarts.

## Cloud deployment (backend on Render, frontend on Vercel)

Cloud deployment is optional per the assessment brief; local Docker Compose
is already sufficient. If deploying anyway, the backend and frontend are
deployed separately, deliberately — the backend (FastAPI + PyTorch +
OpenCV) needs a platform that runs a persistent process; the frontend is a
single static HTML file that fits anywhere. A serverless platform like
Vercel is a poor fit for the *backend* specifically: the PyTorch/OpenCV
dependencies alone exceed typical serverless function size limits, and a
45MB model reloading on every cold start would be slow.

**Backend on Render:**
1. Push this repo to GitHub (already done if you're reading this from there).
2. On [render.com](https://render.com): New → Blueprint → connect this repo. `render.yaml` at the repo root is auto-detected and configures the service (Docker runtime, `backend/Dockerfile`, `/health` as the health check path).
3. Render injects a `PORT` environment variable at runtime; `backend/Dockerfile`'s `CMD` reads it (`--port ${PORT:-8000}`), so no manual port configuration is needed.
4. **Free-tier caveat:** Render's free plan has an ephemeral filesystem (no persistent disk) — the SQLite history resets on every redeploy or restart. Fine for a demo; if you need history to survive restarts, add a persistent disk (`disk:` in `render.yaml`) on a paid plan, or swap SQLite for a managed Postgres (`db.py`'s `create_engine` call is the only place that would need to change).

**Frontend on Vercel:**
1. Deploy the `frontend/` folder as a static site (no build step needed).
2. Update `API_BASE` in `frontend/index.html` to your Render backend's URL (e.g. `https://image-quality-backend.onrender.com`) before deploying — it's a plain JS constant, no env-var injection for a build-less static page.
3. Set `CORS_ORIGINS` on the Render backend to your Vercel domain (via Render's dashboard env vars) instead of leaving it at `*`, once you know the deployed frontend URL.

## Model loading & inference at deployment

The checkpoint (`models/hybrid_model.pt`, ~45MB) is loaded once, lazily, on
the first `/analyze` request (`inference.py: load_model()`), not at process
startup — this keeps `/health` fast and avoids a slow cold start blocking
the whole server. The model and its Grad-CAM wrapper are cached as module
level singletons afterward, so subsequent requests reuse the loaded weights.
Device selection is automatic (`cuda` if available, else `cpu` fallback) —
the same checkpoint runs on either; GPU is only needed for training, not
for serving.

## Evaluation

See [EVALUATION.md](EVALUATION.md) for full precision/recall/F1 per
distortion type, confusion matrix, quality-score MAE/RMSE, and failure-case
analysis on the held-out synthetic test split, plus
`models/generalization_report.json` for the unseen-domain (Oxford
Flowers102) generalization check.
