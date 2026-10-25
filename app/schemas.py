from pydantic import BaseModel, ConfigDict, Field


class PredictMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=64)
    frame_id: str = Field(min_length=1, max_length=128)
    station_id: str | None = Field(default=None, max_length=128)
    execution_id: str | None = Field(default=None, max_length=128)
    step_id: str | None = Field(default=None, max_length=128)
    confidence: float | None = Field(default=None, ge=0, le=1)


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


class Prediction(BaseModel):
    context: PredictMetadata
    model_version: str
    image_width: int
    image_height: int
    inference_ms: float
    total_ms: float
    detections: list[Detection]
