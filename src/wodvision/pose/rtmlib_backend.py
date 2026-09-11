"""Backend de pose base sur rtmlib (RTMPose, COCO-17), Apache 2.0.

Reutilise le registre de modeles de `rtmlib.Body.MODE` (urls + tailles
d'entree par mode leger / equilibre / performance) plutot que de dupliquer
les URLs, pour rester aligne avec les versions de rtmlib installees.

Les poids ONNX sont telecharges automatiquement au premier usage vers
`~/.cache/rtmlib` (necessite un acces reseau une seule fois).
"""

from __future__ import annotations

import logging

import numpy as np
from rtmlib import RTMPose
from rtmlib.tools.solution.body import Body

from wodvision.pose.base import PoseEstimator
from wodvision.types import Keypoints

logger = logging.getLogger(__name__)


def select_device() -> str:
    """Choisit 'cuda' si onnxruntime expose un provider CUDA, sinon 'cpu'."""
    import onnxruntime as ort

    if "CUDAExecutionProvider" in ort.get_available_providers():
        return "cuda"
    return "cpu"


class RTMLibPoseEstimator(PoseEstimator):
    """Pose 2D COCO-17 via RTMPose (rtmlib), sur une bbox fournie par l'appelant."""

    def __init__(
        self,
        mode: str = "balanced",
        device: str | None = None,
        backend: str = "onnxruntime",
    ) -> None:
        if mode not in Body.MODE:
            raise ValueError(f"mode inconnu: {mode!r}, attendu parmi {list(Body.MODE)}")

        resolved_device = device or select_device()
        pose_url = Body.MODE[mode]["pose"]
        pose_input_size = Body.MODE[mode]["pose_input_size"]

        logger.info("rtmlib pose backend : mode=%s device=%s", mode, resolved_device)
        self._model = RTMPose(
            pose_url,
            model_input_size=pose_input_size,
            to_openpose=False,
            backend=backend,
            device=resolved_device,
        )

    def infer(self, image: np.ndarray, bbox: tuple[float, float, float, float]) -> Keypoints:
        keypoints, scores = self._model(image, bboxes=[list(bbox)])
        return Keypoints(xy=keypoints[0], score=scores[0])
