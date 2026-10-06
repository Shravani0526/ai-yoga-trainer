"""MoveNet model registry: where the TFLite files live and how to obtain them.

The deployed app is self-contained once the ``.tflite`` file is committed to
``ai/models/`` (recommended, ~5 MB for Lightning). If the file is missing, the app
downloads it on first start from the URLs below (override with ``MOVENET_MODEL_URL``).
"""

from __future__ import annotations

import io
import os
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

MODELS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelSpec:
    variant: str
    filename: str
    input_size: int
    urls: tuple
    description: str


MODEL_SPECS = {
    "lightning": ModelSpec(
        variant="lightning",
        filename="movenet_singlepose_lightning_float16.tflite",
        input_size=192,
        urls=(
            "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite",
            "https://www.kaggle.com/api/v1/models/google/movenet/tfLite/singlepose-lightning-tflite-float16/4/download",
        ),
        description="Fastest variant (192x192 input) - recommended for webcam and cloud CPUs.",
    ),
    "thunder": ModelSpec(
        variant="thunder",
        filename="movenet_singlepose_thunder_float16.tflite",
        input_size=256,
        urls=(
            "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite",
            "https://www.kaggle.com/api/v1/models/google/movenet/tfLite/singlepose-thunder-tflite-float16/4/download",
        ),
        description="More accurate but slower (256x256 input).",
    ),
}


def get_spec(variant: str) -> ModelSpec:
    try:
        return MODEL_SPECS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown MoveNet variant '{variant}'. Choose from {list(MODEL_SPECS)}") from exc


def model_path(variant: str = "lightning") -> Path:
    """Path where the model for ``variant`` is (or will be) stored."""
    override = os.environ.get("MOVENET_MODEL_PATH")
    if override:
        return Path(override)
    return MODELS_DIR / get_spec(variant).filename


def _extract_tflite(payload: bytes) -> Optional[bytes]:
    """Return raw .tflite bytes from a download that may be raw, .tar.gz or .zip."""
    # TFLite flatbuffers carry the file identifier "TFL3" at byte offset 4.
    if len(payload) > 8 and payload[4:8] == b"TFL3":
        return payload
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".tflite"):
                    handle = tar.extractfile(member)
                    if handle:
                        return handle.read()
    except (tarfile.TarError, OSError):
        pass
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            for name in zf.namelist():
                if name.endswith(".tflite"):
                    return zf.read(name)
    except zipfile.BadZipFile:
        pass
    return None


def download_model(variant: str = "lightning", timeout: int = 90) -> Path:
    """Download the model for ``variant`` and store it in ``ai/models``."""
    spec = get_spec(variant)
    target = model_path(variant)
    target.parent.mkdir(parents=True, exist_ok=True)

    urls: List[str] = []
    custom = os.environ.get("MOVENET_MODEL_URL")
    if custom:
        urls.append(custom)
    urls.extend(spec.urls)

    errors = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ai-yoga-trainer/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
            data = _extract_tflite(payload)
            if data:
                tmp = target.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(target)
                return target
            errors.append(f"{url}: downloaded file is not a TFLite model")
        except Exception as exc:  # noqa: BLE001 - we report every failure to the user
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "Could not download the MoveNet model automatically.\n"
        + "\n".join(errors)
        + f"\nDownload the '{variant}' float16 TFLite model manually and save it as:\n  {target}"
    )


def ensure_model(variant: str = "lightning") -> Path:
    """Return the local model path, downloading the file first if it does not exist."""
    path = model_path(variant)
    if path.exists() and path.stat().st_size > 100_000:
        return path
    return download_model(variant)
