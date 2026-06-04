"""Per-pixel phasor calculation, calibration, smoothing, and visualisation.

Migrated from notebooks/20260515_KPC_phasor.py (Phase D).  Behaviour is
unchanged; module-level constants (REP_RATE_HZ, blur sigmas) are exposed
as defaults so notebooks can override.

Functions:
  otsu_threshold     - threshold a 2-D photon image
  intensity_mask     - Otsu-with-fallback boolean mask for a photon image
  smooth_phasor      - photon-weighted Gaussian smoothing of G, S
  compute_phasor_raw - cos/sin tensor-dot from a TCSPC decay
  apply_phasor_cal   - rotate + scale raw phasor by chroma calibration
  draw_semicircle    - universal FLIM semicircle with lifetime tick marks
  get_time_ns        - bin-centre times for an SdtFile (renamed from _get_time_ns)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Defaults (mirror Phase D values)
# ---------------------------------------------------------------------------

DEFAULT_REP_RATE_HZ      = 80e6                       # laser rep rate, Hz
DEFAULT_OMEGA            = 2.0 * np.pi * DEFAULT_REP_RATE_HZ  # rad/s
DEFAULT_MASK_FALLBACK_FRAC = 0.50
DEFAULT_MASK_BLUR_SIGMA    = 2.0
DEFAULT_PHASOR_BLUR_SIGMA  = 2.0
DEFAULT_LIFETIME_TICKS_NS  = (0.3, 0.5, 1.0, 2.0, 4.0)


# ---------------------------------------------------------------------------
# Photon mask
# ---------------------------------------------------------------------------

def otsu_threshold(img: np.ndarray) -> float:
    """Otsu threshold of a 2-D photon-count image."""
    flat = img[np.isfinite(img) & (img > 0)].ravel()
    if flat.size < 2:
        return 0.0
    counts, edges = np.histogram(
        flat, bins=256, range=(float(flat.min()), float(flat.max())),
    )
    total = float(counts.sum())
    if total == 0:
        return 0.0
    bin_mid = 0.5 * (edges[:-1] + edges[1:])
    mu_tot  = float((counts * bin_mid).sum()) / total
    best_var, best_thr = -1.0, float(edges[0])
    w0 = sum0 = 0.0
    for i in range(len(counts)):
        w0   += counts[i]
        sum0 += counts[i] * bin_mid[i]
        if w0 == 0 or w0 == total:
            continue
        w1    = total - w0
        mu0   = sum0 / w0
        mu1   = (mu_tot * total - sum0) / w1
        var_b = (w0 / total) * (w1 / total) * (mu0 - mu1) ** 2
        if var_b > best_var:
            best_var = var_b
            best_thr = float(edges[i + 1])
    return best_thr


def intensity_mask(
    photon_img: np.ndarray,
    fallback_frac: float = DEFAULT_MASK_FALLBACK_FRAC,
    blur_sigma: float    = DEFAULT_MASK_BLUR_SIGMA,
):
    """Otsu mask with a top-percentile fallback when Otsu is degenerate.

    Returns (mask, threshold).  Applies a Gaussian blur (sigma=blur_sigma)
    before Otsu so the threshold follows cell-level structure rather than
    per-pixel shot noise.  Falls back to keeping the top fallback_frac
    fraction of pixels if Otsu keeps <5% or >95%.
    """
    blurred = gaussian_filter(photon_img.astype(float), sigma=blur_sigma)
    thresh  = otsu_threshold(blurred)
    mask    = blurred >= thresh
    frac    = mask.sum() / float(mask.size)
    if frac < 0.05 or frac > 0.95:
        lo = float(np.nanpercentile(
            blurred[blurred > 0], 100.0 * (1.0 - fallback_frac)
        ))
        thresh = lo
        mask   = blurred >= lo
    return mask, thresh


def smooth_phasor(
    G_cal: np.ndarray,
    S_cal: np.ndarray,
    photons: np.ndarray,
    sigma: float = DEFAULT_PHASOR_BLUR_SIGMA,
) -> tuple:
    """Photon-weighted Gaussian smoothing of G and S in spatial domain.

    Filters G*photons and photons separately before dividing -- equivalent to
    averaging the underlying TCSPC decays over a local neighbourhood.
    Returns (G_sm, S_sm) with NaN where photons == 0 after smoothing.
    """
    ph    = np.nan_to_num(photons, nan=0.0)
    G_num = np.nan_to_num(G_cal * photons, nan=0.0)
    S_num = np.nan_to_num(S_cal * photons, nan=0.0)
    ph_sm = gaussian_filter(ph,    sigma=sigma)
    G_sm  = np.where(ph_sm > 0,
                     gaussian_filter(G_num, sigma=sigma) / ph_sm, np.nan)
    S_sm  = np.where(ph_sm > 0,
                     gaussian_filter(S_num, sigma=sigma) / ph_sm, np.nan)
    return G_sm, S_sm


# ---------------------------------------------------------------------------
# Phasor computation
# ---------------------------------------------------------------------------

def get_time_ns(sdt_obj, n_bins: int) -> np.ndarray:
    """Return bin-centre times in nanoseconds for a loaded SdtFile."""
    t = sdt_obj.times[0].astype(float) * 1e9
    if t.size == n_bins + 1:
        return 0.5 * (t[:-1] + t[1:])
    if t.size == n_bins:
        return t
    dt = 12.5 / n_bins
    return np.arange(n_bins) * dt + 0.5 * dt


def compute_phasor_raw(
    decay: np.ndarray,
    time_ns: np.ndarray,
    rep_rate_hz: float = DEFAULT_REP_RATE_HZ,
) -> tuple:
    """Compute raw (uncalibrated) G, S, and photon count per pixel.

    Args:
        decay:       (ny, nx, n_timebins) float
        time_ns:     (n_timebins,) bin-centre times in ns
        rep_rate_hz: laser repetition rate (default 80 MHz)

    Returns (G, S, photons) each shape (ny, nx).  G, S are NaN where
    photons == 0.
    """
    photons  = decay.sum(axis=2)
    safe_n   = np.where(photons > 0, photons, 1.0)
    omega_ns = 2.0 * np.pi * rep_rate_hz * 1e-9
    cos_t    = np.cos(omega_ns * time_ns)
    sin_t    = np.sin(omega_ns * time_ns)
    G = np.tensordot(decay, cos_t, axes=[[2], [0]]) / safe_n
    S = np.tensordot(decay, sin_t, axes=[[2], [0]]) / safe_n
    G = np.where(photons > 0, G, np.nan)
    S = np.where(photons > 0, S, np.nan)
    return G, S, photons


def apply_phasor_cal(
    G_raw: np.ndarray,
    S_raw: np.ndarray,
    phase_corr: float,
    mod_corr:   float,
) -> tuple:
    """Rotate (by phase_corr) and scale (by mod_corr) raw phasor arrays."""
    c = np.cos(phase_corr)
    s = np.sin(phase_corr)
    G_cal = mod_corr * (G_raw * c - S_raw * s)
    S_cal = mod_corr * (G_raw * s + S_raw * c)
    return G_cal, S_cal


# ---------------------------------------------------------------------------
# Universal semicircle plot
# ---------------------------------------------------------------------------

def draw_semicircle(
    ax,
    rep_rate_hz: float = DEFAULT_REP_RATE_HZ,
    lifetime_ticks_ns: tuple = DEFAULT_LIFETIME_TICKS_NS,
    lw: float = 1.2,
    alpha: float = 0.45,
    color: str = "gray",
    set_axes: bool = True,
):
    """Draw the universal FLIM semicircle with lifetime tick marks.

    Single-exponential lifetimes lie on the semicircle from (0,0) to (1,0)
    through (0.5, 0.5).  G = 1/(1 + (omega*tau)^2), S = omega*tau / (1 + ...).

    Args:
        ax:              matplotlib Axes.
        rep_rate_hz:     laser rep rate, defines omega.
        lifetime_ticks_ns: tau values at which to plot tick markers.
        lw / alpha / color: semicircle line styling.
        set_axes:        if True, set xlim/ylim/aspect/grid/xlabel/ylabel.
    """
    theta = np.linspace(0.0, np.pi, 300)
    ax.plot(0.5 + 0.5 * np.cos(theta), 0.5 * np.sin(theta),
            color=color, lw=lw, alpha=alpha, zorder=2)
    omega_ns = 2.0 * np.pi * rep_rate_hz * 1e-9
    for tau in lifetime_ticks_ns:
        denom = 1.0 + (omega_ns * tau) ** 2
        gx    = 1.0 / denom
        sx    = (omega_ns * tau) / denom
        ax.plot(gx, sx, "o", color=color, ms=3, alpha=alpha, zorder=3)
        ax.annotate(f"{tau:g} ns", (gx, sx), fontsize=7, color=color,
                    xytext=(4, 4), textcoords="offset points",
                    alpha=min(1.0, alpha + 0.2))
    if set_axes:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 0.6)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("G")
        ax.set_ylabel("S")
