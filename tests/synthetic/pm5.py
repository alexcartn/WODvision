"""PM5 synthetique : encodeurs d'octets BLE (format de la spec) et sessions coherentes
avec un rameur video synthetique (decalage d'horloge, decalage de coups, gigue, biais)."""

from __future__ import annotations

import numpy as np

from wodvision.pm5 import spec


def _le(value: int, size: int) -> bytes:
    return int(round(value)).to_bytes(size, "little")


def enc_stroke_data(elapsed_s, distance_m, drive_length_m, drive_time_s, recovery_time_s,
                    stroke_distance_m, peak_lbf, avg_lbf, work_j, stroke_count) -> bytes:
    return (_le(elapsed_s / 0.01, 3) + _le(distance_m / 0.1, 3) + _le(drive_length_m / 0.01, 1)
            + _le(drive_time_s / 0.01, 1) + _le(recovery_time_s / 0.01, 2)
            + _le(stroke_distance_m / 0.01, 2) + _le(peak_lbf / 0.1, 2) + _le(avg_lbf / 0.1, 2)
            + _le(work_j / 0.1, 2) + _le(stroke_count, 2))


def enc_additional_stroke_data(elapsed_s, power_w, calories_hr, stroke_count) -> bytes:
    return (_le(elapsed_s / 0.01, 3) + _le(power_w, 2) + _le(calories_hr, 2)
            + _le(stroke_count, 2) + bytes(9))


def enc_general_status(elapsed_s, stroke_state, drag_factor, distance_m=0.0) -> bytes:
    return (_le(elapsed_s / 0.01, 3) + _le(distance_m / 0.1, 3) + bytes([0, 0, 1, 1, stroke_state])
            + bytes(3) + bytes(3) + bytes([0, drag_factor]))


def enc_additional_status_1(elapsed_s, rate_spm, hr=255) -> bytes:
    return (_le(elapsed_s / 0.01, 3) + _le(3500, 2) + bytes([rate_spm, hr]) + _le(12000, 2)
            + _le(12000, 2) + _le(0, 2) + bytes(3) + bytes([0]))


def enc_force_curve(curve: list[int], words_per_chunk: int = 9) -> list[bytes]:
    chunks = [curve[i : i + words_per_chunk] for i in range(0, len(curve), words_per_chunk)]
    out = []
    for seq, words in enumerate(chunks):
        header = bytes([(len(chunks) << 4) | len(words), seq])
        out.append(header + b"".join(_le(w, 2) for w in words))
    return out


def bell_curve(peak_pos: float, n: int = 40, amplitude: float = 600.0, humps: int = 1) -> list[int]:
    """Courbe de force : gaussienne dont le pic est a peak_pos ; humps=2 ajoute un creux."""
    x = np.linspace(0, 1, n)
    y = np.exp(-0.5 * ((x - peak_pos) / 0.22) ** 2)
    if humps == 2:
        y = 0.8 * np.exp(-0.5 * ((x - 0.25) / 0.09) ** 2) + np.exp(-0.5 * ((x - 0.7) / 0.09) ** 2)
    return [int(v) for v in amplitude * y / y.max()]


def make_events(
    report,
    lag: int = 3,
    offset_s: float = 12.3,
    phase: str = "finish",
    drive_bias_s: float = 0.04,
    jitter_s: float = 0.02,
    curve=None,
    with_status: bool = True,
    t0_host: float = 1_700_000_000.0,
    seed: int = 1,
    true_torso_m: float = 0.5,
):
    """Journal d'evenements PM5 coherent avec `report` (strokes video + handle_travel)."""
    rng = np.random.default_rng(seed)
    ev = report.stroke_events
    n = len(ev)
    med_period = float(np.median([s.t_next_catch - s.t_catch for s in ev]))
    # Coups "virtuels" pour les indices video -lag..n-1 (extrapolation avant le premier).
    virt = []
    for i in range(-lag, n):
        if i >= 0:
            virt.append((ev[i].t_catch, ev[i].t_finish, ev[i].t_next_catch,
                         report.strokes[i].handle_travel))
        else:
            shift = i * med_period
            virt.append((ev[0].t_catch + shift, ev[0].t_finish + shift,
                         ev[0].t_next_catch + shift, report.strokes[0].handle_travel))
    events: list[tuple[float, int, bytes]] = []

    def pm_time(t_video: float) -> float:
        return t_video + offset_s

    for j, (t_c, t_f, t_n, travel) in enumerate(virt):
        drive = (t_f - t_c) - drive_bias_s + rng.normal(0, jitter_s)
        period = t_n - t_c
        recovery = period - drive
        tau = pm_time(t_f if phase == "finish" else t_n) + rng.normal(0, 0.01)
        t_host = t0_host + tau + 0.05
        peak = 420.0 + 10 * rng.standard_normal()
        events.append((t_host, spec.STROKE_DATA, enc_stroke_data(
            tau, 10.0 * j, travel * true_torso_m, drive, recovery, 9.0, peak, peak * 0.55,
            250.0, j + 1)))
        events.append((t_host + 0.01, spec.ADDITIONAL_STROKE_DATA,
                       enc_additional_stroke_data(tau, 180, 900, j + 1)))
        if curve is not None:
            shape = curve(j) if callable(curve) else curve
            for k, chunk in enumerate(enc_force_curve(shape)):
                events.append((t_host + 0.02 + 0.005 * k, spec.FORCE_CURVE, chunk))
        if with_status:
            t_catch_pm = pm_time(t_c) + 0.05
            t_fin_pm = pm_time(t_f) - 0.02
            for t_s in np.arange(max(0.0, pm_time(t_c) - 0.2), pm_time(t_n), 0.1):
                state = (spec.StrokeState.DRIVING if t_catch_pm <= t_s < t_fin_pm
                         else spec.StrokeState.RECOVERY)
                events.append((t0_host + t_s + 0.05, spec.GENERAL_STATUS,
                               enc_general_status(t_s, state, 115)))
    events.sort(key=lambda e: e[0])
    return events
