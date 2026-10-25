"""Opt-in real GPU test: SOP_DETECTOR_GPU_TEST=1 python -m pytest tests/test_gpu.py -s."""
import os
from pathlib import Path
import statistics

import pytest


@pytest.mark.skipif(os.environ.get("SOP_DETECTOR_GPU_TEST") != "1", reason="opt-in GPU/model test")
def test_configured_model_gpu():
    import ultralytics
    from app.config import load_settings
    from app.models import ModelRegistry
    from app.predictor import predict
    from app.schemas import PredictMetadata

    settings = load_settings()
    model_id = next(model.id for model in settings.models if model.enabled)
    registry = ModelRegistry(settings)
    registry.start()
    try:
        assert registry.entries[model_id].status == "ready"
        sample = Path(ultralytics.__file__).parent / "assets/bus.jpg"
        times = []
        for i in range(5):
            result = predict(registry, PredictMetadata(model_id=model_id, frame_id=str(i)), sample.read_bytes())
            assert any(item.class_name in {"person", "bus"} for item in result.detections)
            times.append(result.inference_ms)
        print(f"GPU {model_id} 640, bus.jpg: median={statistics.median(times):.1f}ms, detections={len(result.detections)}")
    finally:
        registry.close()
