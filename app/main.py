from contextlib import asynccontextmanager
from datetime import datetime
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .api import router
from .config import Settings, load_settings
from .models import ModelRegistry


class RequestSizeLimit:
    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            consumed += len(message.get("body", b""))
            if consumed > self.limit:
                raise HTTPException(413, "request_too_large")
            return message

        await self.app(scope, limited_receive, send)


def create_app(settings: Settings | None = None, registry: ModelRegistry | None = None):
    settings = settings or load_settings()
    registry = registry or ModelRegistry(settings)

    @asynccontextmanager
    async def lifespan(app):
        registry.start()
        try:
            yield
        finally:
            registry.close()

    app = FastAPI(title="SOP Detecter", version="0.1.0", lifespan=lifespan)
    app.state.settings, app.state.registry = settings, registry
    app.add_middleware(RequestSizeLimit, limit=settings.max_image_bytes + 1024 * 1024)
    app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins,
                       allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])
    app.include_router(router)
    return app


def main(config_file: str | Path | None = None):
    settings = load_settings(config_file)
    directory = Path(settings.log_directory)
    directory.mkdir(parents=True, exist_ok=True)
    filename = directory / f"sop-detector-runtime-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(filename, encoding="utf-8"), logging.StreamHandler()])
    # Keep third-party settings/cache separate from model weights and source files.
    cache = Path(__file__).resolve().parents[1] / ".cache/ultralytics"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cache))
    import uvicorn
    # Never use reload/workers here: each worker would duplicate resident models.
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1, access_log=False, log_config=None)


if __name__ == "__main__":
    main()
