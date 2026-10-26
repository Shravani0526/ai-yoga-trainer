"""Browser-webcam support for Streamlit (WebRTC) - used by app.py.

``streamlit-webrtc`` streams the visitor's own webcam to the server, where every frame goes
through the same ``YogaPipeline`` that the Colab notebook uses. The processor class is kept
here (not in app.py) because Streamlit re-executes app.py on every interaction.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Optional

try:  # optional dependency: the app falls back to video upload / photo when missing
    import av
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer  # noqa: F401

    WEBRTC_AVAILABLE = True
except Exception:  # noqa: BLE001
    av = None
    VideoProcessorBase = object  # type: ignore[misc,assignment]
    WebRtcMode = None  # type: ignore[assignment]
    webrtc_streamer = None  # type: ignore[assignment]
    WEBRTC_AVAILABLE = False

from ai.models.registry import ensure_model
from ai.pipeline import FrameResult, YogaPipeline
from ai.pose.movenet import MoveNetDetector
from utils.drawing import annotate_frame, ascii_text
import cv2


class YogaProcessor(VideoProcessorBase):  # type: ignore[misc]
    """Runs the full yoga pipeline on every webcam frame (executed in a worker thread)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._params: Dict[str, Any] = {}
        self._params_key = ""
        self._built_key = ""
        self._epoch = 0
        self._built_epoch = -1
        self._t0 = time.monotonic()
        self.pipeline: Optional[YogaPipeline] = None
        self.latest: Optional[FrameResult] = None
        self.error: Optional[str] = None
        self._detector: Optional[MoveNetDetector] = None
        self._detector_variant = ""

    # ----------------------------------------------------------------- called from the UI thread
    def set_params(self, params: Dict[str, Any]) -> None:
        """Update configuration; the pipeline is rebuilt only when something really changed."""
        key = json.dumps(params, sort_keys=True, default=str)
        with self._lock:
            self._params = dict(params)
            self._params_key = key

    def restart(self) -> None:
        """Discard the running session / calibration data and start a fresh one."""
        with self._lock:
            self._epoch += 1

    def get_latest(self) -> Optional[FrameResult]:
        return self.latest

    # ----------------------------------------------------------------- worker thread
    def _ensure_pipeline(self) -> Optional[YogaPipeline]:
        with self._lock:
            params, key, epoch = dict(self._params), self._params_key, self._epoch
        if not params:
            return None
        if self.pipeline is not None and key == self._built_key and epoch == self._built_epoch:
            return self.pipeline
        variant = params.get("model_variant", "lightning")
        if self._detector is None or self._detector_variant != variant:
            self._detector = MoveNetDetector(variant, model_file=str(ensure_model(variant)))
            self._detector_variant = variant
        pipe = YogaPipeline(self._detector, params["pose"], params.get("variant", "auto"),
                            params.get("level", "Beginner"), params.get("calibrations"),
                            source="webcam")
        if params.get("mode") == "calibrate":
            pipe.start_calibration()
        else:
            pipe.start_session("webcam")
        self.pipeline, self._built_key, self._built_epoch = pipe, key, epoch
        self._t0 = time.monotonic()
        return pipe

    def recv(self, frame):  # type: ignore[override]
        img = frame.to_ndarray(format="bgr24")
        try:
            pipe = self._ensure_pipeline()
            if pipe is None:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            result = pipe.process_frame(img, time.monotonic() - self._t0)
            self.latest = result
            out = annotate_frame(img, result, mirror=bool(self._params.get("mirror", True)))
        except Exception as exc:  # noqa: BLE001 - never kill the stream; show the problem instead
            self.error = f"{type(exc).__name__}: {exc}"
            out = img.copy()
            cv2.putText(out, ascii_text("Processing error - see the page for details"), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 235), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(out, format="bgr24")
