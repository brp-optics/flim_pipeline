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
    """Otsu threshold of a 2-D photon-count image.

    Computes Otsu's method (maximize between-class variance) on a 256-bin
    histogram of finite, positive pixel values.  Returns 0.0 for degenerate
    images (fewer than 2 positive pixels, or zero total weight).

    Parameters
    ----------
    img : np.ndarray
        2-D float array of photon counts.  Non-finite and non-positive values
        are excluded before computing the histogram.

    Returns
    -------
    float
        Threshold value.  Pixels >= threshold are considered foreground
        (cell area).

    Assumptions
    -----------
    img should have a bimodal distribution (cells vs background) for Otsu
    to work well.  Falls back gracefully to 0.0 on degenerate input.

    Examples
    --------
    >>> thresh = otsu_threshold(photons)
    >>> mask = photons >= thresh

    Dependencies
    ------------
    numpy
    """
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
    """Boolean intensity mask with Otsu threshold and top-percentile fallback.

    Applies a Gaussian blur before Otsu so the threshold follows cell-level
    structure rather than per-pixel shot noise.  Falls back to keeping the
    top fallback_frac fraction of pixels when Otsu is degenerate (keeps
    <5% or >95% of pixels).

    Parameters
    ----------
    photon_img : np.ndarray
        2-D float array of photon counts (raw, not blurred).
    fallback_frac : float, optional
        Fraction of pixels to retain in the fallback mode.  Default 0.5
        (top 50%).
    blur_sigma : float, optional
        Gaussian blur sigma (pixels) applied before thresholding.
        Default 2.0.

    Returns
    -------
    tuple
        (mask, threshold) where mask is a bool 2-D ndarray (True = foreground)
        and threshold is the float photon level used as the cutoff.

    Assumptions
    -----------
    photon_img has non-negative values.  Pixels <= 0 do not affect the
    threshold calculation.

    Examples
    --------
    >>> mask, thresh = intensity_mask(photons)
    >>> print(f'{mask.mean()*100:.1f}% of pixels accepted')

    Dependencies
    ------------
    numpy, scipy.ndimage.gaussian_filter, otsu_threshold
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

    Filters G*photons and photons separately before dividing, which is
    equivalent to averaging the underlying TCSPC decays over a local
    neighbourhood.  This preserves the physical meaning of the phasor
    (it is the phasor of the spatially-averaged decay, not the average
    of per-pixel phasors).

    Parameters
    ----------
    G_cal : np.ndarray
        2-D calibrated G (real part of phasor), shape (ny, nx).
    S_cal : np.ndarray
        2-D calibrated S (imaginary part of phasor), shape (ny, nx).
    photons : np.ndarray
        2-D photon count per pixel, same shape as G_cal.
    sigma : float, optional
        Gaussian blur radius in pixels.  Default 2.0.

    Returns
    -------
    tuple
        (G_sm, S_sm), each 2-D ndarray.  NaN where the smoothed photon
        count is zero (i.e. near the image edges or in empty regions).

    Assumptions
    -----------
    G_cal, S_cal, and photons must have the same shape.  NaN pixels in
    G_cal or S_cal are treated as zero contribution (nan_to_num).

    Examples
    --------
    >>> G_sm, S_sm = smooth_phasor(G_cal, S_cal, photons, sigma=2.0)

    Dependencies
    ------------
    numpy, scipy.ndimage.gaussian_filter
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
    """Return bin-centre times in nanoseconds for a loaded SdtFile.

    Handles three cases: times array has n_bins+1 edges (compute midpoints),
    n_bins centres (use as-is), or any other length (fall back to a uniform
    grid spanning 12.5 ns, the standard TCSPC window at 80 MHz repetition).

    Parameters
    ----------
    sdt_obj : sdtfile.SdtFile
        Loaded .sdt file object (from sdtfile.SdtFile(path)).
    n_bins : int
        Expected number of time bins (e.g. 256).  Must match the decay
        array's last axis length.

    Returns
    -------
    np.ndarray
        1-D array of shape (n_bins,) giving bin-centre times in nanoseconds.

    Assumptions
    -----------
    sdt_obj.times[0] exists and can be cast to float.  The 12.5 ns fallback
    assumes a 80 MHz laser (12.5 ns period); adjust if rep rate differs.

    Examples
    --------
    >>> t = get_time_ns(sdt, decay.shape[2])
    >>> G, S, ph = compute_phasor_raw(decay, t)

    Dependencies
    ------------
    numpy, sdtfile (via sdt_obj)
    """
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

    G and S are the real and imaginary parts of the first Fourier component
    of the TCSPC decay at the fundamental frequency (laser rep rate).  They
    are normalized by photon count so each pixel lies in [0, 1] on the
    phasor plot.  Apply apply_phasor_cal() to correct for the IRF.

    Parameters
    ----------
    decay : np.ndarray
        3-D float array, shape (ny, nx, n_timebins).  Each [y, x, :] is the
        TCSPC histogram for that pixel.
    time_ns : np.ndarray
        1-D array of bin-centre times in nanoseconds, shape (n_timebins,).
        Obtain from get_time_ns().
    rep_rate_hz : float, optional
        Laser repetition rate in Hz.  Default 80e6 (80 MHz).

    Returns
    -------
    tuple
        (G, S, photons), each shape (ny, nx).  G and S are NaN where the
        photon count is zero.  photons is the sum over the time axis.

    Assumptions
    -----------
    decay values are non-negative (raw photon counts).  time_ns length must
    match decay.shape[2].

    Examples
    --------
    >>> G, S, photons = compute_phasor_raw(decay, get_time_ns(sdt, decay.shape[2]))
    >>> G_cal, S_cal = apply_phasor_cal(G, S, phase_corr, mod_corr)

    Dependencies
    ------------
    numpy
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
    """Rotate (by phase_corr) and scale (by mod_corr) raw phasor arrays.

    Calibration corrects for the instrument response function (IRF) phase
    shift and modulation depth.  The calibration parameters (phase_corr,
    mod_corr) are derived in Phase B from a known-lifetime reference
    (chroma slide) and stored in sdt_metadata_cal.csv.

    Parameters
    ----------
    G_raw : np.ndarray
        2-D raw G (real phasor component) from compute_phasor_raw().
    S_raw : np.ndarray
        2-D raw S (imaginary phasor component).
    phase_corr : float
        Phase correction in radians (from 'phasor_cal_phase_rad' column).
    mod_corr : float
        Modulation correction factor (from 'phasor_cal_mod' column).
        Values > 1 increase G and S (instrument had lower modulation than
        the reference).

    Returns
    -------
    tuple
        (G_cal, S_cal), each 2-D ndarray of the same shape as the inputs.

    Examples
    --------
    >>> G_cal, S_cal = apply_phasor_cal(
    ...     G, S, row['phasor_cal_phase_rad'], row['phasor_cal_mod'])

    Dependencies
    ------------
    numpy
    """
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
    """Draw the universal FLIM semicircle with lifetime tick marks on a phasor plot.

    Single-exponential lifetimes lie on the semicircle from (0,0) to (1,0)
    passing through (0.5, 0.5).  Each point is calculated from:
      G = 1 / (1 + (omega*tau)^2),  S = omega*tau / (1 + (omega*tau)^2)
    Tick markers are drawn at user-specified lifetime values with annotations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw.  Should already have calibrated G, S scatter
        plotted for context.
    rep_rate_hz : float, optional
        Laser rep rate defining omega = 2*pi*rep_rate.  Default 80 MHz.
    lifetime_ticks_ns : tuple of float, optional
        Tau values (nanoseconds) at which to draw tick markers and labels.
        Default (0.3, 0.5, 1.0, 2.0, 4.0) ns.
    lw : float, optional
        Line width for the semicircle arc.  Default 1.2.
    alpha : float, optional
        Opacity for both the arc and tick markers.  Default 0.45.
    color : str, optional
        Color for the arc and tick markers.  Default 'gray'.
    set_axes : bool, optional
        If True, set xlim=(-0.05, 1.05), ylim=(-0.05, 0.6), aspect='equal',
        grid, and axis labels 'G'/'S'.  Default True.

    Side Effects
    ------------
    Modifies ax in place (adds Line2D, scatter artists, and text annotations).

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.scatter(G_cal[mask], S_cal[mask], s=1, alpha=0.3)
    >>> draw_semicircle(ax)

    Dependencies
    ------------
    numpy, matplotlib (via ax)
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
