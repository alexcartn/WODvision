import numpy as np

from tests.synthetic.generators import generate_squat_trajectory
from wodvision.signal.resampling import resample_uniform


def test_resample_clean_signal_matches_truth() -> None:
    traj = generate_squat_trajectory(n_reps=3, fps=60.0, seed=1)
    result = resample_uniform(
        traj.t_obs_s, traj.y_obs_m, traj.valid_obs, target_fps=60.0, max_gap_frames=5
    )
    assert not result.missing.any()
    y_ref = np.interp(result.t_s, traj.t_true_s, traj.y_true_m)
    assert np.max(np.abs(result.y - y_ref)) < 1e-3


def test_resample_handles_vfr() -> None:
    traj = generate_squat_trajectory(n_reps=3, fps=60.0, vfr_jitter_s=0.01, seed=2)
    result = resample_uniform(
        traj.t_obs_s, traj.y_obs_m, traj.valid_obs, target_fps=60.0, max_gap_frames=5
    )
    y_ref = np.interp(result.t_s, traj.t_true_s, traj.y_true_m)
    assert np.max(np.abs(result.y - y_ref)) < 0.05


def test_resample_flags_long_gaps_as_missing() -> None:
    traj = generate_squat_trajectory(n_reps=3, fps=60.0, dropout_fraction=0.0, seed=3)
    # Force un long trou explicite : frames 100..130 invalides (30 frames a 60fps
    # = 0.5s, largement au-dessus de max_gap_frames=5).
    valid = traj.valid_obs.copy()
    valid[100:130] = False

    result = resample_uniform(traj.t_obs_s, traj.y_obs_m, valid, target_fps=60.0, max_gap_frames=5)
    # Bornes valides encadrant le trou : index 99 (valide) et 130 (valide).
    t_gap_start = traj.t_obs_s[99]
    t_gap_end = traj.t_obs_s[130]
    in_gap = (result.t_s > t_gap_start) & (result.t_s < t_gap_end)
    assert in_gap.sum() > 0
    assert result.missing[in_gap].all()
    assert not result.missing[~in_gap].any()


def test_resample_small_dropout_not_flagged_missing() -> None:
    traj = generate_squat_trajectory(n_reps=3, fps=60.0, seed=4)
    valid = traj.valid_obs.copy()
    valid[100] = False  # une seule frame manquante : trou court

    result = resample_uniform(traj.t_obs_s, traj.y_obs_m, valid, target_fps=60.0, max_gap_frames=5)
    assert not result.missing.any()
