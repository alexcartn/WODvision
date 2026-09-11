import numpy as np

from wodvision.analysis.angles import (
    L_ANKLE,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    angle_from_vertical_deg,
    elbow_angle_deg,
    hip_angle_deg,
    joint_angle_deg,
    knee_angle_deg,
    pick_visible_side,
    shin_angle_deg,
    trunk_lean_deg,
)


def test_joint_angle_right_angle() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 0.0])
    c = np.array([0.0, 1.0])
    assert abs(joint_angle_deg(a, b, c) - 90.0) < 1e-6


def test_joint_angle_straight_line() -> None:
    a = np.array([-1.0, 0.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    assert abs(joint_angle_deg(a, b, c) - 180.0) < 1e-6


def test_knee_angle_straight_leg() -> None:
    keypoints = np.zeros((17, 2))
    keypoints[L_HIP] = [0.0, 0.0]
    keypoints[L_KNEE] = [0.0, 1.0]
    keypoints[L_ANKLE] = [0.0, 2.0]
    assert abs(knee_angle_deg(keypoints, side="left") - 180.0) < 1e-6


def test_knee_angle_bent_leg() -> None:
    keypoints = np.zeros((17, 2))
    keypoints[R_HIP] = [0.0, 0.0]
    keypoints[R_KNEE] = [0.0, 1.0]
    keypoints[R_ANKLE] = [1.0, 1.0]
    assert abs(knee_angle_deg(keypoints, side="right") - 90.0) < 1e-6


def test_hip_and_elbow_angle_use_correct_keypoints() -> None:
    keypoints = np.zeros((17, 2))
    keypoints[L_SHOULDER] = [0.0, 0.0]
    keypoints[L_HIP] = [0.0, 1.0]
    keypoints[L_KNEE] = [1.0, 1.0]
    assert abs(hip_angle_deg(keypoints, side="left") - 90.0) < 1e-6

    keypoints[R_SHOULDER] = [0.0, 0.0]
    keypoints[R_ELBOW] = [0.0, 1.0]
    keypoints[R_WRIST] = [0.0, 2.0]
    assert abs(elbow_angle_deg(keypoints, side="right") - 180.0) < 1e-6


def test_angle_from_vertical_pure_vertical_segment() -> None:
    assert abs(angle_from_vertical_deg((0.0, 0.0), (0.0, 5.0)) - 0.0) < 1e-6


def test_angle_from_vertical_pure_horizontal_segment() -> None:
    assert abs(angle_from_vertical_deg((0.0, 0.0), (5.0, 0.0)) - 90.0) < 1e-6


def test_angle_from_vertical_45_degrees() -> None:
    assert abs(angle_from_vertical_deg((0.0, 0.0), (3.0, 3.0)) - 45.0) < 1e-6


def test_trunk_lean_and_shin_angle_use_correct_keypoints() -> None:
    keypoints = np.zeros((17, 2))
    keypoints[L_SHOULDER] = [0.0, 0.0]
    keypoints[L_HIP] = [2.0, 4.0]  # incline vers l'avant
    assert trunk_lean_deg(keypoints, side="left") > 0.0

    keypoints[R_KNEE] = [0.0, 0.0]
    keypoints[R_ANKLE] = [0.0, 3.0]  # tibia parfaitement vertical
    assert abs(shin_angle_deg(keypoints, side="right")) < 1e-6


def test_pick_visible_side_prefers_higher_mean_score() -> None:
    scores = np.zeros((10, 17))
    scores[:, [0, 1, 2, 3, 4]] = 0.9  # face keypoints, non pertinents ici
    scores[:, list(range(5, 17))] = 0.1
    # Rendre le cote droit plus confiant que le gauche.
    from wodvision.analysis.angles import RIGHT_LIMB

    scores[:, list(RIGHT_LIMB)] = 0.8
    assert pick_visible_side(scores) == "right"
