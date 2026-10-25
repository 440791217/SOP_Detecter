from io import BytesIO
from time import perf_counter
import warnings

from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError

from .models import ModelRegistry
from .schemas import Detection, Prediction, PredictMetadata


def predict(registry: ModelRegistry, metadata: PredictMetadata, content: bytes) -> Prediction:
    start = perf_counter()
    if not registry.gate.acquire(blocking=False):
        raise HTTPException(429, "detector_busy")
    try:
        entry = registry.entries.get(metadata.model_id)
        if entry is None:
            raise HTTPException(404, "model_not_found")
        if entry.status != "ready":
            raise HTTPException(503, "model_not_ready")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(content)) as source:
                    if source.format not in {"JPEG", "PNG"}:
                        raise HTTPException(415, "only_jpeg_and_png_supported")
                    if source.width * source.height > registry.settings.max_image_pixels:
                        raise HTTPException(413, "image_dimensions_too_large")
                    source.load()
                    image = ImageOps.exif_transpose(source).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(400, "invalid_image") from None
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise HTTPException(413, "image_dimensions_too_large") from None

        inference_start = perf_counter()
        result = entry.runner.predict(image, imgsz=entry.spec.image_size, device=registry.settings.device,
                                      conf=metadata.confidence if metadata.confidence is not None else entry.spec.confidence,
                                      half=False, verbose=False, save=False, max_det=300)[0]
        # Reading CPU values synchronizes GPU inference before reporting timing.
        boxes = result.boxes
        detections = []
        if boxes is not None:
            for xyxy, confidence, class_id in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
                index = int(class_id)
                detections.append(Detection(class_id=index, class_name=result.names[index], confidence=float(confidence), bbox_xyxy=tuple(xyxy)))
        inference_ms = (perf_counter() - inference_start) * 1000
        return Prediction(context=metadata, model_version=entry.spec.version,
                          image_width=image.width, image_height=image.height,
                          inference_ms=round(inference_ms, 2), total_ms=round((perf_counter() - start) * 1000, 2), detections=detections)
    finally:
        registry.gate.release()
