"""Batch processing for Phase E.  Wraps the per-fit-set processing loop.

Migrated from notebooks/20260517_KPC_fit.py Step 15.  Behaviour is
identical to the inline loop: same cache logic, same skip/print messages,
same low-retention threshold, same per-row dict shape in the returned
mask_summary DataFrame.

Public:
  process_all_fit_sets - the full loop over every sample fit_set_key
  process_one_fit_set  - one fit set (mostly for unit testing)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.io import (
    fit_dir_from_key,
    load_asc_fit_set,
    parse_fit_folder,
)
from src.fitting import (
    compute_fit_mask,
    DEFAULT_MIN_PHOTONS_BY_NCOMP,
    DEFAULT_CHI2_LO,
    DEFAULT_CHI2_HI,
    DEFAULT_TAU_BOUNDS,
    DEFAULT_TAU_MEAN_BOUNDS,
)


LOW_RETENTION_PCT = 1.0   # threshold below which a fit is flagged


def _load_sdt_photons_fallback(sdt_path) -> np.ndarray | None:
    """Sum a TCSPC decay along the time axis to get a 2-D photon image."""
    try:
        from sdtfile import SdtFile
        sdt_obj = SdtFile(str(sdt_path))
        return sdt_obj.data[0].astype(float).sum(axis=2)
    except Exception:
        return None


def process_one_fit_set(
    fit_set_key: str,
    fixation_type: str | None,
    sdt_path,
    data_dirs: Iterable[Path],
    *,
    min_photons_by_ncomp: dict | None = None,
    chi2_lo: float = DEFAULT_CHI2_LO,
    chi2_hi: float = DEFAULT_CHI2_HI,
    tau_bounds: dict | None = None,
    tau_mean_bounds: dict | None = None,
    taumean_cfg: dict | None = None,
) -> dict | None:
    """Load + compute + save a mask for one fit_set_key.

    Returns the summary row dict (with keys matching the inline loop), or
    None if the fit dir or .asc files are missing.
    """
    fit_dir, session_root, rel_dir, base_stem = fit_dir_from_key(
        fit_set_key, data_dirs)
    if fit_dir is None or not fit_dir.exists():
        print(f"  [SKIP] fit dir not found: {fit_set_key[:60]}")
        return None

    fd = load_asc_fit_set(fit_dir, base_stem)
    if not fd:
        print(f"  [SKIP] no .asc files loaded: {base_stem}")
        return None

    folder_meta = parse_fit_folder(Path(rel_dir).name)
    b_val       = folder_meta["b_val"]
    shift_free  = folder_meta["shift_free"]
    n_comp      = folder_meta.get("n_components")

    sdt_photons = None
    if "photons" not in fd and sdt_path is not None:
        sdt_photons = _load_sdt_photons_fallback(sdt_path)

    mask, breakdown = compute_fit_mask(
        fd, fixation_type, b_val, sdt_photons, n_comp,
        min_photons_by_ncomp=min_photons_by_ncomp,
        chi2_lo=chi2_lo, chi2_hi=chi2_hi,
        tau_bounds=tau_bounds, tau_mean_bounds=tau_mean_bounds,
        taumean_cfg=taumean_cfg,
    )
    mask_npy = fit_dir / f"{base_stem}_fit_mask.npy"
    # Defensive write: catch OSError (file lock / partial-write / permission)
    # and retry once after removing any half-written stub.  If still failing
    # log loudly so the user can diagnose, but keep the row in the summary so
    # downstream stats stay complete.
    try:
        np.save(str(mask_npy), mask)
    except OSError as e:
        print(f"  [WARN] np.save failed for {base_stem}: {e!r}")
        try:
            if mask_npy.exists():
                mask_npy.unlink()
            np.save(str(mask_npy), mask)
            print(f"  [WARN]   retry succeeded after removing stub.")
        except OSError as e2:
            print(f"  [ERROR] np.save retry failed: {e2!r}.  Mask not "
                  f"persisted; summary row still emitted.")

    row = {
        "fit_set_key":   fit_set_key,
        "session_root":  session_root,
        "fit_folder":    rel_dir,
        "base_stem":     base_stem,
        "b_val":         b_val,
        "shift_free":    shift_free,
        "fixation_type": fixation_type,
        **breakdown,
    }
    return row


def process_all_fit_sets(
    fit_map_df: pd.DataFrame,
    sdt_df: pd.DataFrame,
    data_dirs: Iterable[Path],
    *,
    force_recompute: bool = True,
    cache_csv_path: Path | None = None,
    progress_every: int = 20,
    # ---- mask config (passed through to compute_fit_mask) ----
    min_photons_by_ncomp: dict | None = None,
    chi2_lo: float = DEFAULT_CHI2_LO,
    chi2_hi: float = DEFAULT_CHI2_HI,
    tau_bounds: dict | None = None,
    tau_mean_bounds: dict | None = None,
    taumean_cfg: dict | None = None,
) -> tuple[pd.DataFrame, list]:
    """Run Phase E Step 15: load .asc + mask for every sample fit_set_key.

    Args:
        fit_map_df:     output of Phase B; cols 'sdt_filename', 'fit_set_key'.
        sdt_df:         output of Phase A/C; needs 'filename', 'file_type',
                        'fixation_type', 'filepath' columns.
        data_dirs:      session-root Paths for fit_dir resolution.
        force_recompute: if False, skip fit sets whose .npy mask exists AND
                        whose row is in the cache CSV.
        cache_csv_path: path to fit_mask_summary.csv (used only when
                        force_recompute is False).
        progress_every: print a progress line every N computed fit sets.

        ...mask config kwargs are forwarded to compute_fit_mask().

    Returns:
        (mask_summary_df, low_retention_log)
        low_retention_log is a list of (sdt_filename, fit_folder, b_val, pct).
    """
    sample_filenames = set(sdt_df.loc[sdt_df["file_type"] == "sample",
                                       "filename"])
    sample_key_rows  = fit_map_df[fit_map_df["sdt_filename"].isin(sample_filenames)]
    sample_keys      = sorted(set(sample_key_rows["fit_set_key"]))
    print(f"Unique fit sets to process: {len(sample_keys)}")

    cache_df = pd.DataFrame()
    if not force_recompute and cache_csv_path and Path(cache_csv_path).exists():
        cache_df = pd.read_csv(cache_csv_path).set_index("fit_set_key")
        print(f"  Cache: {len(cache_df)} existing entries in "
              f"{Path(cache_csv_path).name}")

    fn_to_meta = sdt_df.set_index("filename")[
        ["filepath", "fixation_type"]
    ].to_dict("index")
    key_to_fn  = (sample_key_rows.groupby("fit_set_key")["sdt_filename"]
                                  .first().to_dict())

    rows: list[dict] = []
    low_ret: list[tuple] = []
    n_cached = n_computed = 0

    for fit_set_key in sample_keys:
        fit_dir, session_root, rel_dir, base_stem = fit_dir_from_key(
            fit_set_key, data_dirs)
        if fit_dir is None or not fit_dir.exists():
            print(f"  [SKIP] fit dir not found: {fit_set_key[:60]}")
            continue

        mask_npy = fit_dir / f"{base_stem}_fit_mask.npy"

        # Cache hit
        if (not force_recompute and mask_npy.exists()
                and fit_set_key in cache_df.index):
            row_dict = cache_df.loc[fit_set_key]
            if isinstance(row_dict, pd.DataFrame):
                row_dict = row_dict.iloc[0]
            row_dict = row_dict.to_dict()
            row_dict["fit_set_key"] = fit_set_key
            rows.append(row_dict)
            n_cached += 1
            continue

        # Cache miss: compute
        sdt_filename  = key_to_fn.get(fit_set_key)
        sdt_meta      = fn_to_meta.get(sdt_filename, {}) if sdt_filename else {}
        fixation_type = sdt_meta.get("fixation_type")
        sdt_path      = sdt_meta.get("filepath")

        row = process_one_fit_set(
            fit_set_key, fixation_type, sdt_path, data_dirs,
            min_photons_by_ncomp=min_photons_by_ncomp,
            chi2_lo=chi2_lo, chi2_hi=chi2_hi,
            tau_bounds=tau_bounds, tau_mean_bounds=tau_mean_bounds,
            taumean_cfg=taumean_cfg,
        )
        if row is None:
            continue

        row["sdt_filename"] = sdt_filename   # add late so it sits between cols
        # reorder to match the inline loop's dict insertion order
        ordered = {
            "fit_set_key":   row["fit_set_key"],
            "sdt_filename":  sdt_filename,
            "session_root":  row["session_root"],
            "fit_folder":    row["fit_folder"],
            "base_stem":     row["base_stem"],
            "b_val":         row["b_val"],
            "shift_free":    row["shift_free"],
            "fixation_type": row["fixation_type"],
        }
        for k, v in row.items():
            if k not in ordered:
                ordered[k] = v
        rows.append(ordered)

        pct = ordered.get("pct_final", 100.0)
        if pct < LOW_RETENTION_PCT:
            low_ret.append((sdt_filename or ordered["base_stem"],
                            ordered["fit_folder"], ordered["b_val"], pct))

        n_computed += 1
        if n_computed % progress_every == 0:
            n_tot = ordered.get("n_total", 1) or 1
            def _pct(k):
                return round(100.0 * ordered.get(k, 0) / n_tot)
            print(f"  computed {n_computed}  {ordered['base_stem'][:32]}"
                  f"  ph={_pct('n_photon_ok')}%"
                  f"  chi2={_pct('n_chi2_ok')}%"
                  f"  amp={_pct('n_amp_ok')}%"
                  f"  tau={_pct('n_tau_ok')}%"
                  f"  taum={_pct('n_taumean_ok')}%"
                  f"  kept={ordered.get('pct_final', '?')}%")

    mask_summary_df = pd.DataFrame(rows)
    print(f"\nDone: {len(mask_summary_df)} fit sets  "
          f"({n_cached} cached, {n_computed} newly computed)")
    return mask_summary_df, low_ret
