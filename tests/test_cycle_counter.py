import numpy as np

from wodvision.analysis.cycle_counter import count_cycles, find_bottoms_between


def test_count_cycles_on_clean_sine() -> None:
    t = np.linspace(0, 10, 600)
    y = np.sin(2 * np.pi * 0.5 * t)  # 5 cycles sur 10s (une periode = 2s)
    peaks = count_cycles(t, y, prominence_fraction=0.3, min_distance_s=1.0)
    assert len(peaks) == 5


def test_count_cycles_respects_min_distance() -> None:
    t = np.linspace(0, 10, 600)
    y = np.sin(2 * np.pi * 0.5 * t)
    peaks_far = count_cycles(t, y, min_distance_s=1.9)
    peaks_too_far = count_cycles(t, y, min_distance_s=3.0)
    assert len(peaks_far) == 5
    assert len(peaks_too_far) < 5


def test_count_cycles_flat_signal_returns_empty() -> None:
    t = np.linspace(0, 5, 100)
    y = np.zeros_like(t)
    assert count_cycles(t, y) == []


def test_count_cycles_ignores_small_noise_below_prominence() -> None:
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 600)
    y = np.sin(2 * np.pi * 0.5 * t) + rng.normal(0, 0.02, size=t.shape)
    peaks = count_cycles(t, y, prominence_fraction=0.3, min_distance_s=1.0)
    assert len(peaks) == 5


def test_find_bottoms_between_on_clean_sine() -> None:
    t = np.linspace(0, 10, 600)
    y = np.sin(2 * np.pi * 0.5 * t)  # periode 2s, sommets a t=0.5,2.5,4.5,6.5,8.5
    tops = count_cycles(t, y, prominence_fraction=0.3, min_distance_s=1.0)
    bottoms = find_bottoms_between(t, y, tops)
    assert len(bottoms) == len(tops) - 1
    for top_before, bottom, top_after in zip(tops[:-1], bottoms, tops[1:], strict=True):
        assert top_before < bottom < top_after
        # Le minimum d'un sinus est a mi-chemin entre deux maxima consecutifs.
        assert abs(bottom - (top_before + top_after) / 2) < 0.1


def test_find_bottoms_between_needs_at_least_two_tops() -> None:
    t = np.linspace(0, 5, 100)
    y = np.sin(t)
    assert find_bottoms_between(t, y, []) == []
    assert find_bottoms_between(t, y, [1.0]) == []
