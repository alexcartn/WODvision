import cv2
import numpy as np
import pytest

from wodvision.stabilization.camera_motion import (
    CameraMotionEstimator,
    apply_transform,
    motion_amplitude_px,
)

W, H = 640, 480


def make_world() -> np.ndarray:
    rng = np.random.default_rng(1)
    world = rng.integers(0, 255, (H * 2, W * 2), dtype=np.uint8)
    world = cv2.GaussianBlur(world, (0, 0), 2.0)
    return cv2.cvtColor(world, cv2.COLOR_GRAY2BGR)


def camera_view(world, tx, ty, rot_deg):
    """Vue de la camera : translation + rotation autour du centre de la vue."""
    center = (W / 2 + W / 2, H / 2 + H / 2)
    m = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
    m[:, 2] += [-W / 2 + tx, -H / 2 + ty]
    return cv2.warpAffine(world, m, (W, H)), m


def test_static_camera_is_identity():
    world = make_world()
    est = CameraMotionEstimator()
    for _ in range(5):
        step = est.update(camera_view(world, 0, 0, 0)[0], None)
    assert np.allclose(step.to_ref, np.eye(3), atol=0.5)
    assert motion_amplitude_px(np.array([step.to_ref]), W, H) < 1.0


def test_recovers_pan_and_roll():
    world = make_world()
    est = CameraMotionEstimator()
    # Trajectoire de camera : pan progressif + roulis progressif.
    poses = [(i * 3.0, i * 1.5, i * 0.8) for i in range(12)]
    steps = [est.update(camera_view(world, *p)[0], None) for p in poses]
    # Un point fixe du monde doit retomber au meme endroit apres stabilisation.
    _, m0 = camera_view(world, *poses[0])
    pt_world = np.array([[500.0, 480.0]])
    p_img0 = pt_world @ m0[:, :2].T + m0[:, 2]
    for pose, step in zip(poses, steps, strict=True):
        _, m = camera_view(world, *pose)
        p_img = pt_world @ m[:, :2].T + m[:, 2]
        stabilized = apply_transform(step.to_ref, p_img)
        assert np.linalg.norm(stabilized - p_img0) < 3.0


def test_ignores_moving_athlete_region():
    world = make_world()
    est = CameraMotionEstimator()
    for i in range(6):
        img, _ = camera_view(world, 0, 0, 0)
        x = 200 + i * 15
        img[150:400, x : x + 120] = 255 - img[150:400, x : x + 120]  # "athlete" qui bouge
        step = est.update(img, (x, 150, x + 120, 400))
    assert np.allclose(step.to_ref, np.eye(3), atol=1.0)


def test_unreliable_when_no_texture():
    est = CameraMotionEstimator()
    blank = np.full((H, W, 3), 128, dtype=np.uint8)
    est.update(blank, None)
    step = est.update(blank, None)
    assert not step.reliable
    assert step.to_ref == pytest.approx(np.eye(3))
