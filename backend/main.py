import json
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from db import AnalysisResult, SessionLocal, init_db
from inference import analyze_image

DEFAULT_CHECKPOINT = str(Path(__file__).resolve().parent.parent / "models" / "hybrid_model.pt")
CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT_PATH", DEFAULT_CHECKPOINT)
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))  # 20MB

app = FastAPI(
    title="Image Quality & Defect Detection API",
    description="Upload an image to get a quality score, ACCEPTABLE/DEGRADED/DEFECTIVE "
                "label, detected issues (blur, exposure, noise, corruption, defect) with "
                "severity/confidence, an interpretable feature breakdown, and a Grad-CAM "
                "heatmap. See /docs for interactive API documentation.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    """Service health check. checkpoint_exists confirms the trained model is
    present and analyze() will not 503."""
    return {
        "status": "ok",
        "checkpoint_exists": Path(CHECKPOINT_PATH).exists(),
        "checkpoint_path": CHECKPOINT_PATH,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not Path(CHECKPOINT_PATH).exists():
        raise HTTPException(status_code=503, detail="Model checkpoint not found — train the model first")

    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected an image file, got content-type '{file.content_type}'")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=f"File exceeds max size of {MAX_UPLOAD_BYTES} bytes")

    try:
        result = analyze_image(content, CHECKPOINT_PATH)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    db = SessionLocal()
    row = AnalysisResult(
        filename=file.filename or "unnamed",
        quality_score=result["quality_score"],
        quality_label=result["quality_label"],
        predicted_distortion=result["predicted_distortion"],
        issues_json=json.dumps(result["issues"]),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    result["id"] = row.id
    db.close()

    return result


@app.get("/results/{result_id}")
def get_result(result_id: int):
    db = SessionLocal()
    row = db.get(AnalysisResult, result_id)
    db.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return row.to_dict()


@app.get("/history")
def history(limit: int = 50):
    db = SessionLocal()
    rows = db.query(AnalysisResult).order_by(AnalysisResult.id.desc()).limit(limit).all()
    db.close()
    return [r.to_dict() for r in rows]
