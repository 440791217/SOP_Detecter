import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[1]


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    version: str = Field(min_length=1, max_length=64)
    weights: str
    enabled: bool = True
    image_size: int = Field(default=640, ge=64, le=1280, multiple_of=32)
    confidence: float = Field(default=0.25, ge=0, le=1)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = Field(default=18092, ge=1, le=65535)
    access_token: str = Field(default="88888888", min_length=8, max_length=256)
    allowed_origins: list[str] = ["http://localhost:9527", "http://127.0.0.1:9527"]
    log_directory: str = "D:/projects/logs"
    device: str = "cuda:0"
    max_image_bytes: int = Field(default=5 * 1024 * 1024, ge=1024, le=20 * 1024 * 1024)
    max_image_pixels: int = Field(default=16_000_000, ge=4096, le=40_000_000)
    models: list[ModelSettings] = Field(default_factory=list, max_length=4)

    @field_validator("allowed_origins")
    @classmethod
    def origins(cls, values):
        from urllib.parse import urlsplit
        for value in values:
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment or "*" in value:
                raise ValueError("allowed_origins must contain exact HTTP origins")
        return values

    @field_validator("device")
    @classmethod
    def device_name(cls, value):
        if value not in {"cpu", "cuda:0"}:
            raise ValueError("device must be cpu or cuda:0")
        return value

    @model_validator(mode="after")
    def unique_models(self):
        if len({m.id for m in self.models}) != len(self.models):
            raise ValueError("model IDs must be unique")
        if not Path(self.log_directory).is_absolute():
            raise ValueError("log_directory must be absolute")
        return self


def load_settings(path: str | Path | None = None) -> Settings:
    path = Path(path) if path else ROOT / "configs/settings.json"
    return Settings.model_validate(json.loads(path.read_text(encoding="utf-8-sig")))
