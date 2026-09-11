"""Detection de personnes, basee sur le detecteur YOLOX fourni par rtmlib.

rtmlib (Apache 2.0) embarque son propre detecteur ; on le reutilise plutot
que d'ajouter une dependance de detection separee, conformement au choix par
defaut de la stack (section 5 de CLAUDE.md).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from rtmlib import YOLOX
from rtmlib.tools.solution.body import Body

from wodvision.pose.rtmlib_backend import select_device

logger = logging.getLogger(__name__)

# rtmlib.YOLOX ne renvoie pas les scores de confiance en det_mode='human'
# (seules les boites survivent, deja filtrees par son score_thr interne,
# 0.7 par defaut) : dupliquer sa logique NMS interne pour recuperer le score
# exact serait fragile face aux futures versions de rtmlib. On assigne donc
# une confiance uniforme aux detections retenues — simplification documentee,
# pas un bug — suffisante pour ByteTrack et l'AthleteSelector.
ASSUMED_CONFIDENCE = 0.9


@dataclass(slots=True)
class PersonDetection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float


class PersonDetector:
    """Detecte les personnes dans une frame (classe COCO 'person' uniquement)."""

    def __init__(
        self,
        mode: str = "balanced",
        device: str | None = None,
        backend: str = "onnxruntime",
    ) -> None:
        if mode not in Body.MODE:
            raise ValueError(f"mode inconnu: {mode!r}, attendu parmi {list(Body.MODE)}")

        resolved_device = device or select_device()
        det_url = Body.MODE[mode]["det"]
        det_input_size = Body.MODE[mode]["det_input_size"]

        logger.info("rtmlib person detector : mode=%s device=%s", mode, resolved_device)
        self._model = YOLOX(
            det_url,
            model_input_size=det_input_size,
            backend=backend,
            device=resolved_device,
        )

    def detect(self, image: np.ndarray) -> list[PersonDetection]:
        boxes = self._model(image)
        detections = []
        for x1, y1, x2, y2 in boxes:
            bbox = (float(x1), float(y1), float(x2), float(y2))
            detections.append(PersonDetection(bbox_xyxy=bbox, confidence=ASSUMED_CONFIDENCE))
        return detections
