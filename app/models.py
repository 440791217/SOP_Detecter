import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

from .config import ROOT, ModelSettings, Settings

logger = logging.getLogger("sop-detector")


def load_yolo(spec: ModelSettings, device: str):
    # A missing local file must never trigger an implicit network download.
    path = (ROOT / spec.weights).resolve()
    if not path.is_file():
        raise FileNotFoundError("weights_missing")
    if path.suffix.lower() != ".pt":
        raise ValueError("only .pt detection models are supported in this version")
    import numpy as np
    import torch
    from ultralytics import YOLO

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("cuda_unavailable")
    model = YOLO(str(path), task="detect")
    if model.task != "detect":
        raise ValueError("only object detection models are supported")
    model.to(device)
    # Initialize the predictor and kernels once, not on the first live request.
    model.predict(np.zeros((spec.image_size, spec.image_size, 3), dtype=np.uint8),
                  imgsz=spec.image_size, device=device, half=False, verbose=False, save=False)
    return model


@dataclass
class ModelEntry:
    spec: ModelSettings
    runner: Any = None
    status: str = "not_loaded"
    error: str | None = None


class ModelRegistry:
    def __init__(self, settings: Settings, loader: Callable = load_yolo):
        self.settings = settings
        self.loader = loader
        self.entries = {m.id: ModelEntry(m, status="not_loaded" if m.enabled else "disabled") for m in settings.models}
        self.gate = Lock()  # One inference across all models; no GPU request queue.
        self._started = False

    def start(self):
        if self._started:
            return
        self._started = True
        for entry in self.entries.values():
            if not entry.spec.enabled:
                continue
            try:
                entry.runner = self.loader(entry.spec, self.settings.device)
                entry.status = "ready"
                logger.info("model %s loaded, version=%s", entry.spec.id, entry.spec.version)
            except Exception as error:
                entry.status = "unavailable"
                entry.error = "weights_missing" if isinstance(error, FileNotFoundError) else "load_failed"
                # Do not log raw exceptions: they may contain local paths or input data.
                logger.error("model %s unavailable (%s, %s)", entry.spec.id, entry.error, type(error).__name__)

    def describe(self):
        result = []
        for entry in self.entries.values():
            names = getattr(entry.runner, "names", {}) if entry.runner is not None else {}
            if isinstance(names, list):
                names = dict(enumerate(names))
            result.append({"id": entry.spec.id, "version": entry.spec.version, "status": entry.status,
                           "error": entry.error, "classes": [{"id": int(k), "name": v} for k, v in names.items()]})
        return result

    def close(self):
        # The ASGI server drains live requests before lifespan shutdown.
        with self.gate:
            for entry in self.entries.values():
                entry.runner = None
                if entry.status == "ready":
                    entry.status = "stopped"
            import sys
            if "torch" in sys.modules:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
