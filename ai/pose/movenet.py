"""MoveNet single-pose detector (TensorFlow Lite).

``MoveNetDetector.detect(frame_bgr)`` returns a (17, 3) array of ``(x, y, score)`` in
*pixel* coordinates of the input frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ai.models.registry import ensure_model, get_spec
from ai.pose.keypoints import NUM_KEYPOINTS


def _make_interpreter(model_file: str, num_threads: int):
    """Create a TFLite interpreter from tflite_runtime or full TensorFlow."""
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        return Interpreter(model_path=model_file, num_threads=num_threads)
    except ImportError:
        import tensorflow as tf  # imported lazily: heavy

        return tf.lite.Interpreter(model_path=model_file, num_threads=num_threads)


class MoveNetDetector:
    """Wrapper around the MoveNet SinglePose Lightning / Thunder TFLite models."""

    def __init__(self, variant: str = "lightning", model_file: Optional[str] = None,
                 num_threads: int = 2):
        self.spec = get_spec(variant)
        self.variant = variant
        self.input_size = self.spec.input_size
        path = Path(model_file) if model_file else ensure_model(variant)
        self._interp = _make_interpreter(str(path), num_threads)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        # Some exported models have a fixed input size different from our default.
        shape = self._in["shape"]
        if len(shape) == 4 and int(shape[1]) == int(shape[2]) and int(shape[1]) > 0:
            self.input_size = int(shape[1])

    # ------------------------------------------------------------------ preprocessing
    def _preprocess(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        size = self.input_size
        scale = size / float(max(h, w))
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        dtype = self._in["dtype"]
        tensor = canvas.astype(dtype)  # uint8 models take raw bytes, float models take 0..255
        return np.expand_dims(tensor, 0), scale, pad_x, pad_y

    # ------------------------------------------------------------------ inference
    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run MoveNet and return keypoints as pixel coordinates (17, 3) = (x, y, score)."""
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros((NUM_KEYPOINTS, 3), dtype=float)
        h, w = frame_bgr.shape[:2]
        tensor, scale, pad_x, pad_y = self._preprocess(frame_bgr)
        self._interp.set_tensor(self._in["index"], tensor)
        self._interp.invoke()
        out = np.asarray(self._interp.get_tensor(self._out["index"]), dtype=float)
        out = out.reshape(NUM_KEYPOINTS, 3)  # (y, x, score) normalised to the padded input

        size = float(self.input_size)
        xs = (out[:, 1] * size - pad_x) / scale
        ys = (out[:, 0] * size - pad_y) / scale
        kps = np.stack([np.clip(xs, 0, w - 1), np.clip(ys, 0, h - 1), out[:, 2]], axis=1)
        return kps
