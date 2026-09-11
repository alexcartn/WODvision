import numpy as np

from tests.synthetic.generators import generate_squat_trajectory
from wodvision.analysis.reps import segment_reps
from wodvision.signal.filtering import butterworth_lowpass
from wodvision.signal.resampling import resample_uniform

PLATE_DIAMETER_M = 0.45
PROMINENCE_FRACTION = 0.5
MIN_REP_DISTANCE_S = 0.5


def _resample_and_filter(traj, fps: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    resampled = resample_uniform(
        traj.t_obs_s, traj.y_obs_m, traj.valid_obs, target_fps=fps, max_gap_frames=5
    )
    y_filtered = butterworth_lowpass(resampled.y, fps=fps, cutoff_hz=6.0, order=4)
    return resampled.t_s, y_filtered


def test_segment_reps_counts_clean_signal() -> None:
    traj = generate_squat_trajectory(n_reps=5, fps=60.0, seed=10)
    t_s, y = _resample_and_filter(traj)

    windows = segment_reps(t_s, y, PLATE_DIAMETER_M, PROMINENCE_FRACTION, MIN_REP_DISTANCE_S)

    assert len(windows) == 5
    for window, t_bottom_true in zip(windows, traj.bottom_times_s, strict=True):
        assert abs(window.t_bottom - t_bottom_true) < 0.1
        assert window.t_start < window.t_bottom < window.t_end


def test_segment_reps_robust_to_noise() -> None:
    traj = generate_squat_trajectory(n_reps=5, fps=60.0, noise_std_m=0.01, seed=11)
    t_s, y = _resample_and_filter(traj)

    windows = segment_reps(t_s, y, PLATE_DIAMETER_M, PROMINENCE_FRACTION, MIN_REP_DISTANCE_S)

    assert len(windows) == 5
    for window, t_bottom_true in zip(windows, traj.bottom_times_s, strict=True):
        assert abs(window.t_bottom - t_bottom_true) < 0.15


def test_segment_reps_rejects_truncated_first_rep() -> None:
    fps = 60.0
    traj = generate_squat_trajectory(
        n_reps=4, fps=fps, lead_in_s=1.0, duration_per_rep_s=2.0, seed=12
    )
    t_s, y = _resample_and_filter(traj, fps=fps)

    # Coupe le debut du signal en plein milieu de la 1ere phase excentrique
    # (25% dans la descente) : aucune phase STANDING complete n'est visible
    # avant cette premiere rep, qui doit donc etre rejetee.
    t_cut = 1.0 + 0.25 * 2.0
    cut_idx = int(np.searchsorted(t_s, t_cut))
    t_s_cut, y_cut = t_s[cut_idx:], y[cut_idx:]

    windows = segment_reps(
        t_s_cut, y_cut, PLATE_DIAMETER_M, PROMINENCE_FRACTION, MIN_REP_DISTANCE_S
    )

    assert len(windows) == 3
    remaining_bottoms = traj.bottom_times_s[1:]
    for window, t_bottom_true in zip(windows, remaining_bottoms, strict=True):
        assert abs(window.t_bottom - t_bottom_true) < 0.15


def test_segment_reps_no_peaks_returns_empty() -> None:
    t_s = np.linspace(0, 5, 300)
    y = np.zeros_like(t_s)
    windows = segment_reps(t_s, y, PLATE_DIAMETER_M, PROMINENCE_FRACTION, MIN_REP_DISTANCE_S)
    assert windows == []
