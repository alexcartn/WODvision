"""Ecriture video H.264 via PyAV (lisible dans un navigateur, sans ffmpeg externe)."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np


class H264Writer:
    def __init__(self, path: str | Path, width: int, height: int, fps: float, crf: int = 23):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(self._path), mode="w")
        # Dimensions paires requises par yuv420p.
        self.width, self.height = width - width % 2, height - height % 2
        codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
        self._stream = self._container.add_stream(codec, rate=max(1, round(fps)))
        self._stream.width, self._stream.height = self.width, self.height
        self._stream.pix_fmt = "yuv420p"
        if codec == "libx264":
            self._stream.options = {"crf": str(crf), "preset": "veryfast"}

    def write(self, image_bgr: np.ndarray) -> None:
        image = image_bgr[: self.height, : self.width]
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()

    def __enter__(self) -> H264Writer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
