"""
main.py

FastAPI app around the phase 2 detector. Local development/testing tool
only -- no auth, no deployment config. Run with:

    uvicorn main:app --reload --port 8000

from this directory (see backend/README.md for full setup). Model loading
and inference logic live in inference_service.py -- this file is just
routes + error translation. Correction persistence lives in
corrections_store.py; the manual retrain trigger and its background
subprocess live in retrain_service.py (dataset_merge.py builds the
combined pilot+corrections dataset it trains against). The review-queue
endpoints read/write data/raw/review_queue/queue.json via queue_store.py
-- that queue is produced offline by batch_prelabel.py, not by this app.
"""

import json
import logging

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import ValidationError

import corrections_store
import inference_service
import queue_store
import retrain_service
from schemas import (
    BoxIn,
    CorrectionCountResponse,
    CorrectResponse,
    HealthResponse,
    PredictResponse,
    QueueNextResponse,
    QueueStatsResponse,
    RetrainStartResponse,
    RetrainStatusResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("car_damage_api")

app = FastAPI(title="Car Damage Detector API")

# Local single-user dev tool: the plain HTML/JS frontend is opened via a
# simple local static server (python -m http.server) on some localhost
# port, so allow any localhost/127.0.0.1 origin rather than hardcoding one.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Always returns 200 -- model_loaded tells you whether /predict will work."""
    return HealthResponse(status="ok", model_loaded=inference_service.is_model_loaded())


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    if not inference_service.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail=inference_service.get_load_error() or "Model is not loaded",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        boxes, width, height = inference_service.run_inference(
            image_bytes, confidence_threshold=confidence_threshold
        )
    except Exception as exc:  # noqa: BLE001 -- turn any decode/inference failure into a clear 400, not a raw 500
        logger.exception("Inference failed")
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}")

    return PredictResponse(boxes=boxes, image_width=width, image_height=height)


@app.post("/correct", response_model=CorrectResponse)
async def correct(
    file: UploadFile = File(..., description="The same image already predicted on"),
    boxes: str = Form(..., description="JSON array of {x_min, y_min, x_max, y_max}"),
):
    """
    Saves a user-corrected set of boxes for `file` as new training data
    (see corrections_store.py). An empty `boxes` array is accepted --
    that's a legitimate "no damage here" negative example.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename")

    try:
        raw_boxes = json.loads(boxes)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"'boxes' is not valid JSON: {exc}")

    if not isinstance(raw_boxes, list):
        raise HTTPException(status_code=400, detail="'boxes' must be a JSON array")

    try:
        validated_boxes = [BoxIn(**b) for b in raw_boxes]
    except (ValidationError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid box in 'boxes': {exc}")

    box_dicts = [
        {
            "x_min": round(b.x_min),
            "y_min": round(b.y_min),
            "x_max": round(b.x_max),
            "y_max": round(b.y_max),
        }
        for b in validated_boxes
    ]

    total = corrections_store.save_correction(image_bytes, file.filename, box_dicts)

    # A correction made through the review queue completes that review in
    # the same call -- no separate "mark reviewed" step for the caller.
    # No-ops (returns False) for a correction on an image never in the queue.
    marked_reviewed = queue_store.mark_reviewed(file.filename)

    return CorrectResponse(saved=True, total_corrections=total, marked_reviewed=marked_reviewed)


@app.get("/corrections/count", response_model=CorrectionCountResponse)
def corrections_count():
    return CorrectionCountResponse(count=corrections_store.get_correction_count())


@app.post("/retrain", response_model=RetrainStartResponse)
def retrain():
    """
    Triggers a retrain against every registered data source plus corrections
    as a background subprocess and returns immediately -- manual only,
    never triggered automatically by /correct.
    """
    try:
        log_path = retrain_service.start_retrain()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return RetrainStartResponse(status="started", log_path=log_path)


@app.get("/retrain/status", response_model=RetrainStatusResponse)
def retrain_status():
    return RetrainStatusResponse(**retrain_service.get_retrain_status())


@app.get("/queue/next", response_model=QueueNextResponse)
def queue_next():
    """
    Returns the next pending review-queue item (least confident first),
    or done=True with a `message` explaining why there's nothing to show
    -- either the queue was never generated (run batch_prelabel.py) or
    every entry has already been reviewed.
    """
    if not queue_store.queue_exists():
        return QueueNextResponse(
            done=True,
            message=(
                "No review queue found. Run "
                "'python batch_prelabel.py --input_dir <folder>' from backend/ first."
            ),
        )

    result = queue_store.get_next_pending()
    if result is None:
        return QueueNextResponse(done=True, message="Review queue is empty -- nothing left to review.")

    filename, entry = result
    return QueueNextResponse(
        done=False,
        filename=filename,
        image_url=f"/queue/image/{filename}",
        boxes=entry["boxes"],
        uncertainty=entry["uncertainty"],
    )


@app.get("/queue/image/{filename}")
def queue_image(filename: str):
    image_path = (queue_store.IMAGES_DIR / filename).resolve()
    images_dir = queue_store.IMAGES_DIR.resolve()
    if not image_path.is_relative_to(images_dir) or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found in review queue")
    return FileResponse(image_path)


@app.get("/queue/stats", response_model=QueueStatsResponse)
def queue_stats():
    return QueueStatsResponse(**queue_store.get_queue_stats())
