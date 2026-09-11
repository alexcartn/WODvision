import numpy as np

from wodvision.signal.kinematics import acceleration, velocity


def test_velocity_of_linear_ramp_is_constant() -> None:
    t = np.linspace(0, 2, 200)
    y = 0.5 * t  # vitesse vraie = 0.5 m/s
    v = velocity(y, t)
    assert np.allclose(v, 0.5, atol=1e-3)


def test_acceleration_of_constant_velocity_is_zero() -> None:
    t = np.linspace(0, 2, 200)
    y = 0.5 * t
    v = velocity(y, t)
    a = acceleration(v, t)
    assert np.allclose(a, 0.0, atol=1e-2)


def test_velocity_sign_matches_direction() -> None:
    t = np.linspace(0, 1, 100)
    y = -0.3 * t  # la barre descend : vitesse negative
    v = velocity(y, t)
    assert np.all(v < 0)
