"""Explicitly download the pinned official development model; never called by the API."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt"
SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"


def main():
    target = ROOT / "weights/yolov8m.pt"
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
        print("Verified model already present:", target)
        return
    temporary = target.with_suffix(".pt.part")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(URL, timeout=60) as response, temporary.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 60 * 1024 * 1024:
                    raise RuntimeError("unexpected model size")
                output.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError("model checksum mismatch")
        temporary.replace(target)
        print("Official development model downloaded and verified:", target)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
