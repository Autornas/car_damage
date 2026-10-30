"""
schemas.py

Pydantic request/response models for the car damage detection API. Pure
data shapes -- no inference logic, no model loading.

"""

from typing import List, Optional

from pydantic import BaseModel, model_validator


class BoxOut(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    score: float


class PredictResponse(BaseModel):
    boxes: List[BoxOut]
    image_width: int
    image_height: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class BoxIn(BaseModel):
    """A user-corrected box submitted to POST /correct. No score -- a
    correction is a ground-truth box, not a confidence-weighted guess.
    The current correction UI collects box coordinates only; the dataset
    loader assigns "unknown" when damage_type or part is absent.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @model_validator(mode="after")
    def _check_ordering(self):
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError(
                "Invalid box: x_min must be < x_max and y_min must be < y_max "
                f"(got x_min={self.x_min}, y_min={self.y_min}, "
                f"x_max={self.x_max}, y_max={self.y_max})"
            )
        return self


class CorrectResponse(BaseModel):
    saved: bool
    total_corrections: int
    marked_reviewed: bool = False


class CorrectionCountResponse(BaseModel):
    count: int


class RetrainStartResponse(BaseModel):
    status: str
    log_path: str


class RetrainStatusResponse(BaseModel):
    status: str
    final_loss: Optional[float] = None
    error: Optional[str] = None
    log_path: Optional[str] = None


class QueueNextResponse(BaseModel):
    done: bool = False
    message: Optional[str] = None
    filename: Optional[str] = None
    image_url: Optional[str] = None
    boxes: Optional[List[BoxOut]] = None
    uncertainty: Optional[float] = None


class QueueStatsResponse(BaseModel):
    pending: int
    reviewed: int
    total: int
