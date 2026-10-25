from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from threading import Event

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.config import ModelSettings, Settings
from app.main import create_app
from app.models import ModelRegistry


class Values:
    def __init__(self, values): self.values = values
    def cpu(self): return self
    def tolist(self): return self.values


class FakeModel:
    names = {0: "part"}

    def __init__(self):
        self.entered, self.release = Event(), Event()
        self.block = False

    def predict(self, image, **kwargs):
        self.entered.set()
        if self.block:
            assert self.release.wait(5)
        result = type("Result", (), {"names": self.names, "boxes": type("Boxes", (), {
            "xyxy": Values([[1, 2, 30, 40]]), "conf": Values([0.9]), "cls": Values([0])})()})()
        return [result]


@pytest.fixture
def harness():
    settings = Settings(device="cpu", models=[ModelSettings(id="parts", version="test-v1", weights="weights/test.pt")])
    model = FakeModel()
    loads = []
    def loader(spec, device):
        loads.append(spec.id)
        return model
    registry = ModelRegistry(settings, loader)
    with TestClient(create_app(settings, registry)) as client:
        yield client, registry, model, loads


def image_bytes():
    output = BytesIO()
    Image.new("RGB", (80, 60)).save(output, format="JPEG")
    return output.getvalue()


def send(client, image=None, **changes):
    metadata = {"model_id": "parts", "frame_id": "frame-1", "station_id": "7", "execution_id": "9", "step_id": "3"}
    metadata.update(changes)
    return client.post("/predict", headers={"Authorization": "Bearer 88888888"},
                       files={"file": ("frame.jpg", image if image is not None else image_bytes(), "image/jpeg")},
                       data={"metadata": json.dumps(metadata)})


def test_model_loads_once_and_response_keeps_context(harness):
    client, registry, _, loads = harness
    registry.start()
    for _ in range(2):
        response = send(client)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["context"]["frame_id"] == "frame-1"
        assert data["context"]["step_id"] == "3"
        assert data["model_version"] == "test-v1"
        assert (data["image_width"], data["image_height"]) == (80, 60)
        assert data["detections"][0]["bbox_xyxy"] == [1, 2, 30, 40]
    assert loads == ["parts"]
    assert client.get("/health").json()["models_ready"] is True


def test_missing_model_never_returns_a_fake_success():
    settings = Settings(device="cpu", models=[ModelSettings(id="missing", version="1", weights="weights/missing.pt")])
    registry = ModelRegistry(settings)
    with TestClient(create_app(settings, registry)) as client:
        assert client.get("/health").json()["models_ready"] is False
        assert send(client, model_id="missing").status_code == 503
        models = client.get("/models", headers={"Authorization": "Bearer 88888888"}).json()["models"]
        assert models[0]["error"] == "weights_missing"


def test_authentication_and_cors(harness):
    client, *_ = harness
    assert client.get("/models").status_code == 401
    headers = {"Authorization": "Bearer 88888888", "Origin": "http://untrusted.example"}
    assert client.get("/models", headers=headers).status_code == 403
    headers["Origin"] = "http://localhost:9527"
    response = client.get("/models", headers=headers)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == headers["Origin"]


def test_invalid_image_metadata_and_unknown_model(harness):
    client, *_ = harness
    assert send(client, image=b"not an image").status_code == 400
    assert send(client, confidence=2).status_code == 422
    assert send(client, model_id="not-found").status_code == 404


def test_swagger_same_origin_prediction(harness):
    client, *_ = harness
    for address in ("http://localhost:18092", "http://127.0.0.1:18092"):
        response = client.post(address + "/predict",
            headers={"Authorization": "Bearer 88888888", "Origin": address},
            files={"file": ("frame.jpg", image_bytes(), "image/jpeg")},
            data={"metadata": json.dumps({"model_id": "parts", "frame_id": "docs-test"})})
        assert response.status_code == 200, response.text


def test_busy_requests_are_rejected_without_queueing(harness):
    client, _, model, _ = harness
    model.block = True
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(send, client)
        assert model.entered.wait(5)
        try:
            response = send(client, frame_id="frame-2")
            assert response.status_code == 429, response.text
        finally:
            model.release.set()
        assert first.result().status_code == 200
    assert send(client).status_code == 200


def test_upload_limits_and_lock_recovery(harness):
    client, registry, *_ = harness
    registry.settings.max_image_bytes = 1024
    assert send(client, image=b"x" * 1025).status_code == 413
    assert send(client).status_code == 200
    assert send(client, image=b"").status_code == 400


def test_streamed_request_body_limit(harness):
    client, registry, *_ = harness
    limit = registry.settings.max_image_bytes + 1024 * 1024
    response = send(client, image=b"x" * (limit + 1))
    assert response.status_code == 413, response.text


def test_decoded_pixel_limit(harness):
    client, registry, *_ = harness
    registry.settings.max_image_pixels = 4096
    assert send(client).status_code == 413


def test_shutdown_releases_model():
    settings = Settings(device="cpu", models=[ModelSettings(id="parts", version="1", weights="unused.pt")])
    registry = ModelRegistry(settings, lambda *_: FakeModel())
    with TestClient(create_app(settings, registry)):
        assert registry.entries["parts"].runner is not None
    assert registry.entries["parts"].runner is None


@pytest.mark.parametrize("change", [
    {"allowed_origins": ["*"]}, {"device": "cuda:5"}, {"port": 0}, {"log_directory": "logs"},
    {"models": [{"id": "same", "version": "1", "weights": "a.pt"}] * 2},
])
def test_invalid_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        Settings(**change)
