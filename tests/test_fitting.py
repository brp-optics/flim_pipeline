"""Unit tests for src/fitting.py and the compute_tau_mean part of src/metrics.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.fitting import (    # noqa: E402
    _parse_bin_radius,
    _prefer_shift_zero,
    best_fit_key,
    compute_fit_mask,
    DEFAULT_MIN_PHOTONS_BY_NCOMP,
)
from src.metrics import compute_tau_mean   # noqa: E402


# ---------------------------------------------------------------------------
# _parse_bin_radius
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,expected", [
    ("fitet-sz-c2-b10", 10),
    ("fitet-sf-c2-b5",  5),
    ("fitet-sz-b1",     1),
    ("session::fitet-sz-c2-b9::stem", 9),
    ("nothing_here",    0),     # no -b -> 0
])
def test_parse_bin_radius(key, expected):
    assert _parse_bin_radius(key) == expected


# ---------------------------------------------------------------------------
# _prefer_shift_zero
# ---------------------------------------------------------------------------

def test_prefer_shift_zero_picks_sz_over_sf():
    df = pd.DataFrame({"fit_set_key": [
        "S::fitet-sf-c2-b5::stem",
        "S::fitet-sz-c2-b5::stem",
        "S::fitet-sf-c2-b10::stem",
    ]})
    assert _prefer_shift_zero(df) == "S::fitet-sz-c2-b5::stem"


def test_prefer_shift_zero_fallback_when_no_sz():
    df = pd.DataFrame({"fit_set_key": [
        "S::fitet-sf-c2-b5::stem",
        "S::fitet-sf-c2-b10::stem",
    ]})
    assert _prefer_shift_zero(df) == "S::fitet-sf-c2-b5::stem"


# ---------------------------------------------------------------------------
# best_fit_key
# ---------------------------------------------------------------------------

def test_best_fit_key_picks_highest_bin_sz_at_preferred_ncomp():
    df = pd.DataFrame({
        "sdt_filename": ["A.sdt"] * 4,
        "fit_set_key": [
            "S::fitet-sf-c2-b10::A",   # wrong shift
            "S::fitet-sz-c2-b5::A",    # ok, b=5
            "S::fitet-sz-c2-b10::A",   # ok, b=10 <- expected
            "S::fitet-sz-c3-b10::A",   # wrong n_comp
        ],
        "n_components": [2, 2, 2, 3],
    })
    out = best_fit_key("A.sdt", df, fixation_type="form")
    assert out == "S::fitet-sz-c2-b10::A"


def test_best_fit_key_returns_none_when_no_sz_at_preferred_ncomp():
    df = pd.DataFrame({
        "sdt_filename": ["A.sdt", "A.sdt"],
        "fit_set_key": ["S::fitet-sf-c2-b10::A", "S::fitet-sz-c3-b10::A"],
        "n_components": [2, 3],
    })
    # form prefers n_comp=2 by default; the only n=2 fit has sf, not sz.
    assert best_fit_key("A.sdt", df, fixation_type="form") is None


def test_best_fit_key_returns_none_when_filename_absent():
    df = pd.DataFrame({
        "sdt_filename": ["B.sdt"], "fit_set_key": ["S::fitet-sz-c2-b9::B"],
        "n_components": [2],
    })
    assert best_fit_key("MISSING.sdt", df, fixation_type="form") is None


# ---------------------------------------------------------------------------
# compute_tau_mean (sanity)
# ---------------------------------------------------------------------------

def test_compute_tau_mean_form_2comp():
    a1 = np.array([[0.7, 0.5]]); tau1 = np.array([[400.0, 400.0]])
    a2 = np.array([[0.3, 0.5]]); tau2 = np.array([[2500.0, 2500.0]])
    fd = {"a1": a1, "tau1": tau1, "a2": a2, "tau2": tau2}
    tm = compute_tau_mean(fd, "form")
    np.testing.assert_allclose(tm, [[0.7*400 + 0.3*2500, 0.5*400 + 0.5*2500]])


def test_compute_tau_mean_missing_returns_none():
    assert compute_tau_mean({"a1": np.zeros((2, 2))}, "form") is None


# ---------------------------------------------------------------------------
# compute_fit_mask
# ---------------------------------------------------------------------------

def _good_form_fd(shape=(8, 8), photons=5000.0):
    # All pixels meet every criterion.
    return {
        "photons": np.full(shape, photons),
        "chi2":    np.full(shape, 1.0),
        "a1":      np.full(shape, 0.5),
        "a2":      np.full(shape, 0.5),
        "tau1":    np.full(shape, 400.0),
        "tau2":    np.full(shape, 2500.0),
    }


def test_compute_fit_mask_all_pass():
    fd = _good_form_fd()
    mask, brk = compute_fit_mask(fd, fixation_type="form", b_val=1, n_components=2)
    assert mask.all()
    assert brk["pct_final"] == 100.0


def test_compute_fit_mask_photon_threshold_excludes():
    # photons too low after binning -> all pixels rejected.
    fd = _good_form_fd(photons=10.0)
    mask, brk = compute_fit_mask(fd, fixation_type="form", b_val=1, n_components=2)
    assert not mask.any()
    assert brk["n_photon_ok"] == 0


def test_compute_fit_mask_chi2_rejected():
    fd = _good_form_fd()
    fd["chi2"][:] = 5.0    # outside [0.8, 2.0]
    mask, _ = compute_fit_mask(fd, fixation_type="form", b_val=1, n_components=2)
    assert not mask.any()


def test_compute_fit_mask_tau_bounds_rejected():
    fd = _good_form_fd()
    fd["tau2"][:] = 200.0    # below 800 ps lower bound
    mask, _ = compute_fit_mask(fd, fixation_type="form", b_val=1, n_components=2)
    assert not mask.any()


def test_compute_fit_mask_live_taumean_floor():
    # live has tau_mean >= 250 ps requirement
    a1 = np.full((4, 4), 1.0); a2 = np.full((4, 4), 0.0)
    fd = {
        "photons": np.full((4, 4), 5000.0),
        "chi2":    np.full((4, 4), 1.0),
        "a1":      a1, "a2": a2,
        "tau1":    np.full((4, 4), 100.0),    # tau_mean = 100 < 250 -> reject
        "tau2":    np.full((4, 4), 2500.0),
    }
    mask, _ = compute_fit_mask(fd, fixation_type="live", b_val=1, n_components=2)
    assert not mask.any()


def test_compute_fit_mask_returns_default_when_shape_unknown():
    mask, brk = compute_fit_mask({}, fixation_type="form", b_val=1, n_components=2)
    assert mask.shape == (1, 1)
    assert brk == {}
