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

    Falls back to the trailing token after the last underscore.
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
    """Remove the parameter suffix from an .asc file stem to recover the
    base stem of the original .sdt file.
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

    Returns dict with keys:
      b_val, shift_free, kernel_size, n_components,
      is_export, export_format, color_lo/hi, intensity_lo/hi.
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

    fit_dir is None if the session directory is not found in data_dirs.

    Args:
        fit_set_key: '<session_root>::<rel_dir>::<base_stem>'
        data_dirs:   list of session root Paths to search.
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

    Returns {param_name: 2D ndarray}.  Files whose stripped stem does not
    match base_stem are skipped, as are *_statistic*.asc summary files.
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
    """Load the Phase E _fit_mask.npy for a fit_set_key, or None if missing.

    Args:
        fit_set_key: '<session_root>::<rel_dir>::<base_stem>'
        data_dirs:   list of session root Paths to search.
    """
    fit_dir, _, _, base_stem = fit_dir_from_key(fit_set_key, data_dirs)
    if fit_dir is None:
        return None
    p = fit_dir / f"{base_stem}_fit_mask.npy"
    return np.load(str(p)) if p.exists() else None


def load_image_bundle(mask_path) -> tuple[dict, str]:
    """Load the standard image bundle for one .sdt file given its mask path.

    Picks SPCImage .asc files alongside the mask by their exact suffixes:
      photons   = {stem}_photons.asc
      tau_mean  = {stem}_color coded value.asc       (SPCImage's own tau_mean)
      a1, a2    = {stem}_a1.asc, {stem}_a2.asc
      chi2      = {stem}_chi.asc
      mask      = the .npy file at mask_path

    Returns (arrs, base_stem).  arrs keys: photons, tau_mean, a1, a2, chi2,
    mask, and a derived 'a1/a2' = a1/a2 (NaN where a2==0).

    Raises if any .asc file is missing; if you'd rather have flexible matching
    use load_asc_fit_set() instead.
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
