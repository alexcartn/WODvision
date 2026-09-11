import numpy as np

from tests.synthetic.generators import generate_squat_trajectory
from wodvision.signal.filtering import butterworth_lowpass
from wodvision.signal.resampling import resample_uniform


def test_filter_reduces_noise_and_preserves_amplitude() -> None:
    fps = 60.0
    traj = generate_squat_trajectory(n_reps=3, fps=fps, noise_std_m=0.02, depth_m=0.5, seed=5)
    resampled = resample_uniform(
        traj.t_obs_s, traj.y_obs_m, traj.valid_obs, target_fps=fps, max_gap_frames=5
    )
    y_filtered = butterworth_lowpass(resampled.y, fps=fps, cutoff_hz=6.0, order=4)

    noise_residual = np.std(resampled.y - np.interp(resampled.t_s, traj.t_true_s, traj.y_true_m))
    filtered_residual = np.std(
        y_filtered - np.interp(resampled.t_s, traj.t_true_s, traj.y_true_m)
    )
    assert filtered_residual < noise_residual

    # L'amplitude du squat (profondeur) doit rester globalement preservee.
    assert abs(y_filtered.min() - (-0.5)) < 0.05


def test_filter_rejects_cutoff_above_nyquist() -> None:
    y = np.sin(np.linspace(0, 10, 200))
    try:
        butterworth_lowpass(y, fps=10.0, cutoff_hz=6.0, order=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for cutoff >= Nyquist")
