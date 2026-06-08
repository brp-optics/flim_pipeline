"""I/O helpers for SPCImage .asc exports and Phase E .npy masks.

These functions previously lived inline in notebooks/20260517_KPC_fit.py
(Phase E) and were duplicated in quickpeek3.py and view_reference_panel.py.
Behaviour is unchanged from the Phase E originals; data_dirs is now passed
explicitly instead of being read from a global.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def _infer_param_name(stem: str) -> str:
    """Map an .asc file stem to a parameter name (a1, tau1, chi2, ...).

    Matches common SPCImage export suffixes via a priority-ordered list of
    regex patterns.  Falls back to the trailing token after the last
    underscore when no pattern matches.

    Parameters
    ----------
    stem : str
        .asc filename without extension, e.g. 'myfile_0000-a1' or
        'myfile_0000_chi'.

    Returns
    -------
    str
        Canonical parameter name: 'a1', 'a2', 'a3', 'tau1', 'tau2',
        'tau3', 'tau_mean', 'chi2', 'photons', 'scatter', 'shift',
        'G', or 'S'.  Returns the last underscore-delimited token if no
        pattern matches.

    Dependencies
    ------------
    re
    """
    s = stem.lower()
    for pattern, name in [
        (r"[-_]a1[_%]?$",                             "a1"),
        (r"[-_]a2[_%]?$",                             "a2"),
        (r"[-_]a3[_%]?$",                             "a3"),
        (r"[-_]t1$|[-_]tau1$",                        "tau1"),
        (r"[-_]t2$|[-_]tau2$",                        "tau2"),
        (r"[-_]t3$|[-_]tau3$",                        "tau3"),
        (r"[-_]tm$|[-_]tau_?mean$|[-_]mean$",         "tau_mean"),
        (r"[-_]chi$|[-_]chisq?$",                     "chi2"),
        (r"[-_]photons?$|[-_]int(ensity)?$|[-_]cnt$", "photons"),
        (r"[-_]scatter$|[-_]sc$",                     "scatter"),
        (r"[-_]shift$",                               "shift"),
        (r"[-_]g$",                                   "G"),
        (r"[-_]s$",                                   "S"),
    ]:
        if re.search(pattern, s):
            return name
    parts = s.rsplit("_", 1)
    return parts[-1] if len(parts) > 1 else s


def _strip_param_suffix(stem: str) -> str:
    """Remove the parameter suffix from an .asc file stem to recover the base stem.

    The base stem is the .sdt filename without extension and without the
    trailing parameter token added by SPCImage on export.

    Parameters
    ----------
    stem : str
        .asc filename without extension, e.g. 'myfile_0000-a1' or
        'myfile_0000_tau2'.

    Returns
    -------
    str
        Base stem, e.g. 'myfile_0000'.

    Dependencies
    ------------
    re
    """
    return re.sub(
        r"[-_]?(a[123]|t[123]|tau[123]|chi|chisq?|photons?|intensity|cnt|"
        r"tm|tau_?mean|scatter|sc|shift|[gs])[_%]?$",
        "",
        stem,
        flags=re.IGNORECASE,
    ).strip("_- ")


# ---------------------------------------------------------------------------
# Fit folder navigation
# ---------------------------------------------------------------------------

def parse_fit_folder(folder_name: str) -> dict:
    """Parse BH fit-folder or export-folder metadata from the folder name.

    Fit folder examples:
      'fitet-sz-b2'              -> b=2, shift=zero, n_components=None
      'fitet-sf-b5-c3'           -> b=5, shift=free, n_components=3
      'fitet-sz-b2-1component'   -> b=2, shift=zero, n_components=1

    Export folder examples (is_export=True):
      'exet-fitet-sz-b2'                -> asc exports, b=2, shift=zero
      'exet600-1400-0-500-fitet-sz-b2'  -> tif exports, color LUT 600-1400 ps,
                                           intensity LUT 0-500, b=2

    Parameters
    ----------
    folder_name : str
        Name of the SPCImage fit or export folder (not a full path).

    Returns
    -------
    dict
        Keys and meaning:
          'b_val'        -- int, spatial binning radius (kernel = 2*b+1)
          'shift_free'   -- bool, True if shift was fitted freely ('-sf-')
          'kernel_size'  -- int, total kernel edge length = 2*b+1
          'n_components' -- int or None, number of exponential components
          'is_export'    -- bool, True if this is an export folder (exet*)
          'export_format'-- 'asc', 'tif', or None
          'color_lo/hi'  -- float or None, tif colormap range (ps)
          'intensity_lo/hi' -- float or None, tif intensity range

    Assumptions
    -----------
    Folder names follow the BH SPCImage naming convention.  Unknown tokens
    are silently ignored; defaults are b_val=1, shift_free=False.

    Examples
    --------
    >>> parse_fit_folder('fitet-sz-b2-c2')
    {'b_val': 2, 'shift_free': False, 'kernel_size': 5, 'n_components': 2, ...}

    Dependencies
    ------------
    re
    """
    result = {
        "b_val": 1, "shift_free": False, "kernel_size": 3,
        "n_components": None, "is_export": False, "export_format": None,
        "color_lo": None, "color_hi": None,
        "intensity_lo": None, "intensity_hi": None,
    }

    name = folder_name

    # Detect export folder: exet{N}-{N}-{N}-{N}-... (tif) or exet-... (asc)
    m_tif = re.match(r"^exet(\d+)-(\d+)-(\d+)-(\d+)[-_](.+)$", name, re.IGNORECASE)
    m_asc = re.match(r"^exet[-_](.+)$", name, re.IGNORECASE)
    if m_tif:
        result["is_export"]     = True
        result["export_format"] = "tif"
        result["color_lo"]      = float(m_tif.group(1))
        result["color_hi"]      = float(m_tif.group(2))
        result["intensity_lo"]  = float(m_tif.group(3))
        result["intensity_hi"]  = float(m_tif.group(4))
        name = m_tif.group(5)    # remainder: 'fitet-sz-b2'
    elif m_asc:
        result["is_export"]     = True
        result["export_format"] = "asc"
        name = m_asc.group(1)    # remainder: 'fitet-sz-b2'

    m_b = re.search(r"(?:^|[-_])b(\d+)", name, re.IGNORECASE)
    if m_b:
        b = int(m_b.group(1))
        result["b_val"]       = b
        result["kernel_size"] = 2 * b + 1

    m_s = re.search(r"(?:^|[-_])s([zf])(?=[-_]|$)", name, re.IGNORECASE)
    if m_s:
        result["shift_free"] = m_s.group(1).lower() == "f"

    m_c = re.search(r"(?:^|[-_])c(\d+)(?:omponents?)?(?=[-_]|$)", name, re.IGNORECASE)
    if not m_c:
        m_c = re.search(r"(?:^|[-_])(\d+)components?(?=[-_]|$)", name, re.IGNORECASE)
    if m_c:
        result["n_components"] = int(m_c.group(1))

    return result


def fit_dir_from_key(fit_set_key: str, data_dirs: Iterable[Path]) -> tuple:
    """Return (fit_dir, session_root, rel_dir, base_stem) for a fit_set_key.

    fit_set_key encodes a 3-part address: which session, which subdirectory
    within that session, and which .sdt base stem.  This function resolves
    the session_root against data_dirs to produce an absolute fit_dir.

    Parameters
    ----------
    fit_set_key : str
        '<session_root>::<rel_dir>::<base_stem>', e.g.
        '20260429_KPC_fixed_dishes_on_SLIM::fitet-sz-b2::3_KPCWT0430form20min_..._0000'.
        The session_root is matched by directory name (not full path).
    data_dirs : iterable of Path
        Session root directories to search (from get_data_dirs()).

    Returns
    -------
    tuple
        (fit_dir, session_root, rel_dir, base_stem) where fit_dir is
        Path or None if the session root is not found in data_dirs.

    Assumptions
    -----------
    fit_set_key must contain exactly two '::' separators.  Session roots
    are matched by their final path component (d.name == session_root).

    Examples
    --------
    >>> fit_dir, _, _, stem = fit_dir_from_key(row['fit_set_key'], data_dirs)
    >>> if fit_dir is not None:
    ...     fd = load_asc_fit_set(fit_dir, stem)

    Dependencies
    ------------
    pathlib
    """
    session_root, rel_dir, base_stem = fit_set_key.split("::", 2)
    session_dir = next((d for d in data_dirs if d.name == session_root), None)
    fit_dir = (session_dir / Path(rel_dir)) if session_dir else None
    return fit_dir, session_root, rel_dir, base_stem


# ---------------------------------------------------------------------------
# Array loading
# ---------------------------------------------------------------------------

def load_asc_fit_set(fit_dir: Path, base_stem: str) -> dict:
    """Load every .asc parameter map for one base stem from a fit folder.

    Globs all .asc files in fit_dir, filters to those whose stripped stem
    matches base_stem, infers the parameter name, and loads each as a 2-D
    ndarray.  Summary files (*_statistic*.asc) are skipped.

    Parameters
    ----------
    fit_dir : Path
        Directory containing SPCImage .asc exports (from fit_dir_from_key).
    base_stem : str
        .sdt base stem to match, e.g. '3_KPCWT0430form20min_..._0000'.
        Matching is case-insensitive.

    Returns
    -------
    dict
        {param_name: 2-D ndarray} where param_name is one of 'a1', 'a2',
        'tau1', 'tau2', 'chi2', 'photons', etc.  Files that fail to load
        or are not 2-D are silently skipped.

    Side Effects
    ------------
    Reads .asc files from disk.

    Assumptions
    -----------
    .asc files are space-delimited grids of float values, optionally with a
    one-line header (skiprows=1 is tried on ValueError).  Only 2-D non-empty
    arrays are returned.

    Examples
    --------
    >>> fd = load_asc_fit_set(fit_dir, '3_KPCWT0430form20min_..._0000')
    >>> fd.keys()  # {'a1', 'a2', 'tau1', 'tau2', 'chi2', 'photons'}

    Dependencies
    ------------
    numpy, pathlib, re, _infer_param_name, _strip_param_suffix
    """
    fd: dict = {}
    for fp in sorted(fit_dir.glob("*.asc")):
        if re.search(r"_statistic", fp.stem, re.IGNORECASE):
            continue
        if _strip_param_suffix(fp.stem).lower() != base_stem.lower():
            continue
        param = _infer_param_name(fp.stem)
        try:
            data = np.loadtxt(str(fp), dtype=float)
        except ValueError:
            try:
                data = np.loadtxt(str(fp), dtype=float, skiprows=1)
            except Exception:
                continue
        except Exception:
            continue
        if data.ndim == 2 and data.size > 0:
            fd[param] = data
    return fd


def load_saved_mask(
    fit_set_key: str,
    data_dirs: Iterable[Path],
) -> np.ndarray | None:
    """Load the Phase E quality mask (.npy) for a fit_set_key, or None if missing.

    Resolves the fit_set_key to a directory, then looks for a file named
    '{base_stem}_fit_mask.npy'.  Returns None rather than raising if the
    file is absent or the session root is not found in data_dirs.

    Parameters
    ----------
    fit_set_key : str
        '<session_root>::<rel_dir>::<base_stem>' (same format as the
        'fit_set_key' column in fit_analysis_summary.csv).
    data_dirs : iterable of Path
        Session root directories (from get_data_dirs()).

    Returns
    -------
    np.ndarray or None
        Boolean 2-D array (True = pixel passed quality criteria) or None
        if the .npy file does not exist or the session root is not found.

    Side Effects
    ------------
    Reads a .npy file from disk.

    Examples
    --------
    >>> mask = load_saved_mask(row['fit_set_key'], data_dirs)
    >>> if mask is not None:
    ...     pct_ok = mask.mean() * 100

    Dependencies
    ------------
    numpy, pathlib, fit_dir_from_key
    """
    fit_dir, _, _, base_stem = fit_dir_from_key(fit_set_key, data_dirs)
    if fit_dir is None:
        return None
    p = fit_dir / f"{base_stem}_fit_mask.npy"
    return np.load(str(p)) if p.exists() else None


def load_image_bundle(mask_path) -> tuple[dict, str]:
    """Load the standard image bundle for one .sdt file given its mask path.

    Picks SPCImage .asc files alongside the mask using exact filename suffixes.
    This is a convenience wrapper for the common case; for flexible (partial)
    loading use load_asc_fit_set() instead.

    Expected files (all must exist):
      {stem}_photons.asc           -- raw photon count image
      {stem}_color coded value.asc -- SPCImage's amplitude-weighted tau_mean
      {stem}_a1.asc, _a2.asc      -- amplitude components
      {stem}_chi.asc               -- chi-squared goodness-of-fit image

    Parameters
    ----------
    mask_path : path-like
        Path to the Phase E quality mask file ('{stem}_fit_mask.npy').
        The stem and folder are inferred from this path.

    Returns
    -------
    tuple
        (arrs, base_stem) where arrs is a dict with keys:
          'photons'  -- 2-D ndarray, raw photon count per pixel
          'tau_mean' -- 2-D ndarray, amplitude-weighted lifetime (ps) from fit
          'a1', 'a2' -- 2-D ndarray, amplitude components
          'chi2'     -- 2-D ndarray, chi-squared values
          'mask'     -- 2-D bool ndarray (True = quality-accepted pixel)
          'a1/a2'    -- 2-D ndarray, ratio (NaN where a2 == 0)
        base_stem is the .sdt filename without extension or parameter suffix.

    Raises
    ------
    FileNotFoundError or OSError
        If any of the required .asc files is missing.

    Side Effects
    ------------
    Reads .asc and .npy files from disk.

    Examples
    --------
    >>> arrs, stem = load_image_bundle(row['fit_mask_path'])
    >>> plt.imshow(np.where(arrs['mask'], arrs['tau_mean'], np.nan))

    Dependencies
    ------------
    numpy, pathlib
    """
    p      = Path(mask_path)
    folder = p.parent
    stem   = p.name.replace("_fit_mask.npy", "")
    arrs = {
        "photons":  np.loadtxt(folder / f"{stem}_photons.asc"),
        "tau_mean": np.loadtxt(folder / f"{stem}_color coded value.asc"),
        "a1":       np.loadtxt(folder / f"{stem}_a1.asc"),
        "a2":       np.loadtxt(folder / f"{stem}_a2.asc"),
        "chi2":     np.loadtxt(folder / f"{stem}_chi.asc"),
        "mask":     np.load(p),
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        arrs["a1/a2"] = np.where(arrs["a2"] > 0, arrs["a1"] / arrs["a2"],
                                  np.nan)
    return arrs, stem
