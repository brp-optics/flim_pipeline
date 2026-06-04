"""Fit selection and per-pixel quality masking.

Previously defined inline in notebooks/20260517_KPC_fit.py.  All config
constants (MIN_PHOTONS_BY_NCOMP, CHI2_LO, CHI2_HI, TAU_BOUNDS,
TAU_MEAN_BOUNDS) live here as DEFAULT_* and match the Phase E values
in use as of 2026-06-01.  Notebooks may pass overrides.

Functions:
  _parse_bin_radius   - extract bin radius from a fit_set_key string
  _prefer_shift_zero  - among candidate rows, prefer the shift-zero folder
  best_fit_key        - highest-bin sz fit at the preferred n_components
  compute_fit_mask    - 5-criterion boolean mask + per-criterion breakdown
"""

from __future__ import annotations

import re

import numpy as np
from scipy.ndimage import uniform_filter

from .metrics import compute_tau_mean  # for tau_mean criterion


# ---------------------------------------------------------------------------
# Default config (mirrors Phase E values)
# ---------------------------------------------------------------------------

DEFAULT_MIN_PHOTONS_BY_NCOMP = {
    1:     500,
    2:   1_000,
    3:   8_000,
    None: 1_000,
}
DEFAULT_CHI2_LO = 0.8
DEFAULT_CHI2_HI = 2.0
DEFAULT_TAU_BOUNDS = {
    "glu": {
        "tau1": (0.0,    200.0),
        "tau2": (50.0,  1200.0),
        "tau3": (800.0, 6000.0),
    },
    "form": {
        "tau1": (50.0,  1500.0),
        "tau2": (800.0, 6000.0),
    },
    "live": {
        "tau1": (50.0,  1500.0),
        "tau2": (800.0, 6000.0),
    },
}
DEFAULT_TAU_MEAN_BOUNDS = {
    "glu":  None,
    "form": None,
    "live": (250.0, None),
}
DEFAULT_PREFERRED_N_COMP = {"glu": 3, "form": 2, "live": 2}


# ---------------------------------------------------------------------------
# fit_set_key helpers
# ---------------------------------------------------------------------------

def _parse_bin_radius(key: str) -> int:
    """Parse bin radius from a fit_set_key, e.g. 'fitet-sz-c2-b10' -> 10.
    Returns 0 if no '-b<digits>' token is found.
    """
    m = re.search(r"[-_]b(\d+)", str(key), re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _prefer_shift_zero(candidates) -> str:
    """Among candidate rows (DataFrame with 'fit_set_key' column), return the
    fit_set_key of the shift-zero ('-sz-') folder if one exists; otherwise
    return the first row.
    """
    sz_rows = candidates[
        candidates["fit_set_key"].str.contains("-sz-", case=False, na=False)
    ]
    chosen = sz_rows if not sz_rows.empty else candidates
    return str(chosen.iloc[0]["fit_set_key"])


def best_fit_key(
    filename: str,
    fit_map_df,
    fixation_type: str | None = None,
    preferred_n_comp: dict | None = None,
) -> str | None:
    """Return the highest-bin fit_set_key matching the hard constraints:
       n_components == preferred_n_comp[fixation_type]  AND  shift-zero folder.

    Returns None if no fit satisfies both constraints.

    Args:
        filename:         the .sdt filename to find a fit for.
        fit_map_df:       DataFrame with columns 'sdt_filename', 'fit_set_key',
                          'n_components'.
        fixation_type:    'glu' | 'form' | 'live' (selects preferred n_comp).
        preferred_n_comp: override for DEFAULT_PREFERRED_N_COMP.
    """
    pref = preferred_n_comp if preferred_n_comp is not None else DEFAULT_PREFERRED_N_COMP

    rows = fit_map_df[fit_map_df["sdt_filename"] == filename]
    if rows.empty:
        return None
    preferred = pref.get(str(fixation_type)) if fixation_type else None
    if preferred is not None and "n_components" in rows.columns:
        rows = rows[rows["n_components"] == preferred]
        if rows.empty:
            return None
    rows = rows[rows["fit_set_key"].str.contains("-sz-", case=False, na=False)]
    if rows.empty:
        return None
    rows = rows.assign(_bin=rows["fit_set_key"].apply(_parse_bin_radius))
    return str(rows.sort_values("_bin", ascending=False).iloc[0]["fit_set_key"])


# ---------------------------------------------------------------------------
# Quality mask
# ---------------------------------------------------------------------------

def compute_fit_mask(
    fd: dict,
    fixation_type: str,
    b_val: int,
    sdt_photons: np.ndarray | None = None,
    n_components: int | None = None,
    *,
    min_photons_by_ncomp: dict | None = None,
    chi2_lo: float = DEFAULT_CHI2_LO,
    chi2_hi: float = DEFAULT_CHI2_HI,
    tau_bounds: dict | None = None,
    tau_mean_bounds: dict | None = None,
    taumean_cfg: dict | None = None,
) -> tuple:
    """Combine 5 quality criteria into one boolean mask.

    Criteria (must all pass):
      1. Box-kernel smoothed photon count >= min_photons_by_ncomp[n_components]
      2. chi^2 in [chi2_lo, chi2_hi]
      3. All amplitude components >= 0 and their sum > 0
      4. Tau components within tau_bounds[fixation_type]
      5. amplitude-weighted tau_mean within tau_mean_bounds[fixation_type]
         (if set; otherwise skipped)

    Returns (mask_final, breakdown_dict).
    """
    mph_cfg = (min_photons_by_ncomp
               if min_photons_by_ncomp is not None else DEFAULT_MIN_PHOTONS_BY_NCOMP)
    tb_cfg  = tau_bounds      if tau_bounds      is not None else DEFAULT_TAU_BOUNDS
    tmb_cfg = tau_mean_bounds if tau_mean_bounds is not None else DEFAULT_TAU_MEAN_BOUNDS

    shape = next(
        (v.shape for v in fd.values() if isinstance(v, np.ndarray) and v.ndim == 2),
        None,
    )
    if shape is None:
        return np.zeros((1, 1), dtype=bool), {}

    n_total = shape[0] * shape[1]

    # 1. Photon count (binned)
    min_ph     = mph_cfg.get(n_components, mph_cfg[None])
    photon_src = fd.get("photons") if "photons" in fd else sdt_photons
    if photon_src is not None and photon_src.shape == shape:
        kernel_area = (2 * b_val + 1) ** 2
        smoothed    = uniform_filter(photon_src.astype(float), size=2 * b_val + 1)
        mask_photon = smoothed * kernel_area >= min_ph
    else:
        mask_photon = np.ones(shape, dtype=bool)

    # 2. chi^2
    chi2 = fd.get("chi2")
    if chi2 is not None and chi2.shape == shape:
        mask_chi2 = np.isfinite(chi2) & (chi2 >= chi2_lo) & (chi2 <= chi2_hi)
    else:
        mask_chi2 = np.ones(shape, dtype=bool)

    # 3. amplitude positivity + non-zero sum
    amp_names = [p for p in ("a1", "a2", "a3") if p in fd and fd[p].shape == shape]
    if amp_names:
        mask_amp = np.ones(shape, dtype=bool)
        for p in amp_names:
            mask_amp &= fd[p] >= 0.0
        mask_amp &= sum(fd[p] for p in amp_names) > 0.0
    else:
        mask_amp = np.ones(shape, dtype=bool)

    # 4. tau bounds
    tau_cfg  = tb_cfg.get(fixation_type, {})
    mask_tau = np.ones(shape, dtype=bool)
    for param, (lo, hi) in tau_cfg.items():
        if param in fd and fd[param].shape == shape:
            arr = fd[param]
            mask_tau &= np.isfinite(arr) & (arr >= lo) & (arr <= hi)

    # 5. amplitude-weighted tau_mean bounds
    taumean_window = tmb_cfg.get(fixation_type)
    if taumean_window is not None:
        tau_m = compute_tau_mean(fd, fixation_type, taumean_cfg=taumean_cfg)
        if tau_m is not None and tau_m.shape == shape:
            lo_tm, hi_tm = taumean_window
            mask_taumean = np.isfinite(tau_m)
            if lo_tm is not None:
                mask_taumean &= tau_m >= lo_tm
            if hi_tm is not None:
                mask_taumean &= tau_m <= hi_tm
        else:
            mask_taumean = np.ones(shape, dtype=bool)
    else:
        mask_taumean = np.ones(shape, dtype=bool)

    mask_final = mask_photon & mask_chi2 & mask_amp & mask_tau & mask_taumean

    breakdown = {
        "n_total":      n_total,
        "n_photon_ok":  int(mask_photon.sum()),
        "n_chi2_ok":    int(mask_chi2.sum()),
        "n_amp_ok":     int(mask_amp.sum()),
        "n_tau_ok":     int(mask_tau.sum()),
        "n_taumean_ok": int(mask_taumean.sum()),
        "n_final":      int(mask_final.sum()),
        "pct_final":    round(100.0 * mask_final.sum() / n_total, 1),
    }
    return mask_final, breakdown
