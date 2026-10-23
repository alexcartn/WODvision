"""Video synthetique : fond texture + rameur filaire, avec mouvement de camera.

Les modeles (detecteur, pose) sont remplaces par des doubles qui renvoient la
verite terrain : on teste la plomberie (rotation, stabilisation, cache,
analyse, rendu) sans poids de modele ni video reelle.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tests.synthetic.rowing import RowerParams, generate_rower
from wodvision.io.video_writer import H264Writer
from wodvision.types import Keypoints

W, H = 640, 480
EDGES = [(5, 7), (7, 9), (5, 11), (11, 13), (13, 15)]


class FakePose:
    def __init__(self, xy: np.ndarray, score: np.ndarray):
        self._xy, self._score, self._i = xy, score, 0

    def infer(self, image, bbox):
        i = min(self._i, len(self._xy) - 1)
        self._i += 1
        return Keypoints(xy=self._xy[i].copy(), score=self._score[i].copy())


class FakeDetector:
    """Detecte a l'indice de frame courant (donne par le faux modele de pose)."""

    def __init__(self, bboxes: list[tuple[float, float, float, float]], pose: FakePose):
        self._bboxes, self._pose = bboxes, pose
        self.calls = 0

    def detect(self, image):
        from wodvision.detection.person import PersonDetection

        self.calls += 1
        b = self._bboxes[min(self._pose._i, len(self._bboxes) - 1)]
        return [PersonDetection(bbox_xyxy=b, confidence=0.9)]


def camera_matrix(t: float, pan_px_s: float, roll_deg_s: float) -> np.ndarray:
    m = cv2.getRotationMatrix2D((W / 2, H / 2), roll_deg_s * t, 1.0)
    m[:, 2] += [pan_px_s * t, 0.5 * pan_px_s * t]
    return m


def make_scene(path: Path, params: RowerParams, pan_px_s: float = 0.0, roll_deg_s: float = 0.0):
    """Ecrit la video et renvoie (FakeDetector, FakePose, verite terrain keypoints image)."""
    t, xy_world, score, _ = generate_rower(params)
    rng = np.random.default_rng(3)
    bg = cv2.GaussianBlur(rng.integers(0, 255, (H, W), dtype=np.uint8), (0, 0), 2.0)
    bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

    xy_img = np.zeros_like(xy_world)
    bboxes = []
    with H264Writer(path, W, H, params.fps, crf=18) as writer:
        for i, ti in enumerate(t):
            m = camera_matrix(ti, pan_px_s, roll_deg_s)
            frame = cv2.warpAffine(bg, m, (W, H), borderMode=cv2.BORDER_REFLECT)
            pts = xy_world[i] @ m[:, :2].T + m[:, 2]
            xy_img[i] = pts
            for a, b in EDGES:
                cv2.line(frame, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]),
                         (0, 0, 255), 6)
            used = pts[[5, 7, 9, 11, 13, 15]]
            x1, y1 = used.min(axis=0) - 30
            x2, y2 = used.max(axis=0) + 30
            bboxes.append((float(x1), float(y1), float(x2), float(y2)))
            writer.write(frame)
    pose = FakePose(xy_img, score)
    return FakeDetector(bboxes, pose), pose, xy_img
