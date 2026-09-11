"""Interface commune aux backends de pose 2D."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from wodvision.types import Keypoints


class PoseEstimator(ABC):
    """Estime la pose 2D (COCO-17) d'une personne dans une bbox donnee."""

    @abstractmethod
    def infer(self, image: np.ndarray, bbox: tuple[float, float, float, float]) -> Keypoints:
        """Infere les keypoints pour la personne delimitee par `bbox`.

        Args:
            image: image BGR complete (pas recadree).
            bbox: (x1, y1, x2, y2) en pixels, format xyxy.

        Returns:
            Keypoints au format COCO-17 (xy en pixels dans le repere de `image`).
        """
        raise NotImplementedError
