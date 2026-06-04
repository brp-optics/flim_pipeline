"""Unit tests for src/config.py + config/default.yaml."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_config         # noqa: E402
from src.fitting import (                  # noqa: E402
    DEFAULT_MIN_PHOTONS_BY_NCOMP,
    DEFAULT_TAU_BOUNDS,
    DEFAULT_TAU_MEAN_BOUNDS,
    DEFAULT_CHI2_LO,
    DEFAULT_CHI2_HI,
    DEFAULT_PREFERRED_N_COMP,
)
from src.metrics import DEFAULT_TAUMEAN_CFG  # noqa: E402


def test_load_config_keys_present():
    cfg = load_config()
    for key in ("min_photons_by_ncomp", "chi2_lo", "chi2_hi",
                "tau_bounds", "tau_mean_bounds", "taumean_cfg",
                "ratio_cfg", "preferred_n_comp",
                "rep_rate_hz", "nadh_filters"):
        assert key in cfg, f"missing key: {key}"


def test_min_photons_normalization():
    cfg = load_config()
    mph = cfg["min_photons_by_ncomp"]
    # None is the fallback, mapped from 'default' in YAML
    assert None in mph
    assert 2 in mph
    assert isinstance(list(mph.keys())[0], (int, type(None)))


def test_yaml_matches_library_defaults():
    """The YAML mirrors src/fitting.py defaults.  If they drift, fail loudly."""
    cfg = load_config()
    assert cfg["min_photons_by_ncomp"] == DEFAULT_MIN_PHOTONS_BY_NCOMP
    assert cfg["chi2_lo"] == DEFAULT_CHI2_LO
    assert cfg["chi2_hi"] == DEFAULT_CHI2_HI
    assert cfg["tau_bounds"] == DEFAULT_TAU_BOUNDS
    assert cfg["tau_mean_bounds"] == DEFAULT_TAU_MEAN_BOUNDS
    assert cfg["taumean_cfg"] == DEFAULT_TAUMEAN_CFG
    assert cfg["preferred_n_comp"] == DEFAULT_PREFERRED_N_COMP


def test_tau_bounds_are_tuples():
    cfg = load_config()
    for ft, params in cfg["tau_bounds"].items():
        for p, rng in params.items():
            assert isinstance(rng, tuple), f"{ft}/{p} should be tuple"
            assert len(rng) == 2


def test_taumean_cfg_is_nested_tuples():
    cfg = load_config()
    for ft, pairs in cfg["taumean_cfg"].items():
        assert isinstance(pairs, tuple)
        for pair in pairs:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
