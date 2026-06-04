"""Unit tests for src/phasor.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.phasor import (    # noqa: E402
    otsu_threshold,
    intensity_mask,
    smooth_phasor,
    compute_phasor_raw,
    apply_phasor_cal,
    draw_semicircle,
    DEFAULT_REP_RATE_HZ,
)


# ---------------------------------------------------------------------------
# otsu_threshold
# ---------------------------------------------------------------------------

def test_otsu_threshold_separates_bimodal():
    # Build a strongly-bimodal image and verify the threshold lies between
    # the two means (Otsu does not always land in the dead centre; just that
    # it cleanly separates the two clusters).
    img = np.concatenate([
        np.full(1000,  5.0),
        np.full(1000, 95.0),
    ]).reshape((40, 50))
    thr = otsu_threshold(img)
    assert 5.0 < thr < 95.0
    mask = img >= thr
    # Half of pixels above, half below
    assert mask.sum() == 1000


def test_otsu_threshold_zero_on_empty_or_constant():
    assert otsu_threshold(np.zeros((4, 4))) == 0.0
    assert otsu_threshold(np.array([[1.0]])) == 0.0


# ---------------------------------------------------------------------------
# intensity_mask
# ---------------------------------------------------------------------------

def test_intensity_mask_keeps_bright_pixels():
    img = np.zeros((20, 20))
    img[5:15, 5:15] = 100.0   # bright square
    mask, thr = intensity_mask(img, blur_sigma=0.5)
    # The bright square should be in the mask
    assert mask[10, 10]
    # Corners should not be
    assert not mask[0, 0]


def test_intensity_mask_fallback_when_otsu_degenerate():
    # Noisy near-uniform image with a small bright tail.  Otsu typically
    # threshold-bias high here and keeps a tiny fraction; the fallback then
    # kicks in to keep the top fallback_frac.
    rng = np.random.default_rng(0)
    img = rng.normal(50.0, 1.0, size=(40, 40))
    img[0:2, 0:2] = 1000.0   # a few extreme outliers to wreck Otsu
    mask, _ = intensity_mask(img, fallback_frac=0.5, blur_sigma=1.0)
    frac = mask.sum() / mask.size
    # Either Otsu kept a normal fraction OR fallback gave us ~50%.  In any
    # case the result should not be all-True/all-False degenerate.
    assert 0.05 < frac < 0.95


# ---------------------------------------------------------------------------
# smooth_phasor
# ---------------------------------------------------------------------------

def test_smooth_phasor_nan_where_no_photons():
    G = np.full((10, 10), 0.5)
    S = np.full((10, 10), 0.3)
    ph = np.zeros((10, 10))
    ph[5:8, 5:8] = 100.0
    G_sm, S_sm = smooth_phasor(G, S, ph, sigma=1.0)
    # Pixels far from the photon patch -> NaN
    assert np.isnan(G_sm[0, 0])
    # Pixels near the photon patch -> defined
    assert np.isfinite(G_sm[6, 6])


# ---------------------------------------------------------------------------
# compute_phasor_raw + apply_phasor_cal
# ---------------------------------------------------------------------------

def _single_exp_decay(tau_ns, n_bins=256, t_window_ns=12.5, photons=10000.0):
    """Build a (1, 1, n_bins) decay for a single-exponential lifetime."""
    t = np.linspace(0, t_window_ns, n_bins, endpoint=False)
    profile = np.exp(-t / tau_ns)
    profile = profile / profile.sum() * photons
    return profile.reshape(1, 1, n_bins), t


def test_compute_phasor_raw_single_exp_on_semicircle():
    # A single-exponential decay should sit on the universal semicircle.
    # i.e. G^2 - G + S^2 = 0  =>  (G - 0.5)^2 + S^2 = 0.25
    decay, t = _single_exp_decay(tau_ns=2.0)
    G, S, ph = compute_phasor_raw(decay, t, rep_rate_hz=DEFAULT_REP_RATE_HZ)
    g, s = float(G[0, 0]), float(S[0, 0])
    radial = (g - 0.5) ** 2 + s ** 2
    assert abs(radial - 0.25) < 0.02
    assert ph[0, 0] == pytest.approx(10000.0, rel=1e-3)


def test_apply_phasor_cal_identity_when_no_correction():
    G = np.array([[0.5]]); S = np.array([[0.3]])
    Gc, Sc = apply_phasor_cal(G, S, phase_corr=0.0, mod_corr=1.0)
    np.testing.assert_allclose(Gc, G)
    np.testing.assert_allclose(Sc, S)


def test_apply_phasor_cal_rotation_90deg():
    G = np.array([[1.0]]); S = np.array([[0.0]])
    Gc, Sc = apply_phasor_cal(G, S, phase_corr=np.pi / 2, mod_corr=1.0)
    # 90deg rotation: (1, 0) -> (0, 1)
    np.testing.assert_allclose(Gc, [[0.0]], atol=1e-10)
    np.testing.assert_allclose(Sc, [[1.0]], atol=1e-10)


# ---------------------------------------------------------------------------
# draw_semicircle (smoke test)
# ---------------------------------------------------------------------------

def test_draw_semicircle_runs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    draw_semicircle(ax)
    plt.close(fig)
