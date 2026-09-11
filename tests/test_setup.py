"""Verifie que le toolchain (packaging, imports) fonctionne."""

import wodvision
from wodvision.config import Config, load_config


def test_version() -> None:
    assert wodvision.__version__


def test_default_config() -> None:
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.signal.butterworth_cutoff_hz == 6.0
    assert cfg.calibration.plate_diameter_mm == 450.0
