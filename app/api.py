import hmac
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError

from .predictor import predict
from .schemas import Prediction, PredictMetadata

router = APIRouter()
bearer = HTTPBearer(auto_error=False)
logger = logging.getLogger("sop-detector")


def authorize(request: Request, credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
    if credentials is None or not hmac.compare_digest(credentials.credentials.encode(), request.app.state.settings.access_token.encode()):
        raise HTTPException(401, "invalid_access_token")
    origin = request.headers.get("origin")
    # Swagger's POST requests carry Origin even when served by this service.
    same_origin = str(request.base_url).rstrip("/")
    if origin and origin != same_origin and origin not in request.app.state.settings.allowed_origins:
        raise HTTPException(403, "origin_not_allowed")


@router.get("/health")
def health(request: Request):
    entries = request.app.state.registry.describe()
    enabled = [m for m in entries if m["status"] != "disabled"]
    return {"service": "sop-detector", "status": "ok", "models_ready": bool(enabled) and all(m["status"] == "ready" for m in enabled)}


@router.get("/models", dependencies=[Depends(authorize)])
def models(request: Request):
    return {"device": request.app.state.settings.device, "models": request.app.state.registry.describe()}


@router.post("/predict", response_model=Prediction, dependencies=[Depends(authorize)])
async def prediction(request: Request, file: Annotated[UploadFile, File()], metadata: Annotated[str, Form(max_length=8192)]):
    try:
        try:
            context = PredictMetadata.model_validate_json(metadata)
        except ValidationError:
            raise HTTPException(422, "invalid_prediction_metadata") from None
        if file.content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(415, "only_jpeg_and_png_supported")
        content = await file.read(request.app.state.settings.max_image_bytes + 1)
        if len(content) > request.app.state.settings.max_image_bytes:
            raise HTTPException(413, "image_too_large")
        if not content:
            raise HTTPException(400, "empty_image")
        try:
            return await run_in_threadpool(predict, request.app.state.registry, context, content)
        except HTTPException:
            raise
        except Exception as error:
            logger.error("prediction failed (%s)", type(error).__name__)
            raise HTTPException(503, "inference_failed") from None
    finally:
        await file.close()
