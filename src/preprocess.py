"""Pre-analysis DataFrame helpers: condition labelling, subset filtering,
analysis-df assembly.

Functions:
  label_of          - filename stem -> 'CELLTYPE condition' string
  pockels_class     - bin Pockels values into 'Low' / 'High'
  build_analysis_df - merge fit_analysis_summary + sdt_metadata + annotations
  filter_subset     - apply standard matched-condition filters to a df
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Condition labelling
# ---------------------------------------------------------------------------

def label_of(stem: str) -> str:
    """Filename stem -> 'CELLTYPE condition' string (e.g. 'BKO form 20min').

    Parses cell type (KPCWT or BKO) and fixation condition (live, form 20min,
    form 10min, form, glu) from the filename stem by case-insensitive substring
    matching.

    Parameters
    ----------
    stem : str
        Filename stem (with or without extension), e.g.
        '3_KPCWT0430form20min_740nm_3024mW_u1_poc0p2_20x0p75NA_457s50_g70_z1_256pix_basicFLIM_120f_0000'.

    Returns
    -------
    str
        Label of the form 'CELLTYPE condition', e.g. 'KPCWT form 20min',
        'BKO live', 'KPCWT glu'.  Returns 'KPCWT ?' or 'BKO ?' if the
        condition cannot be determined.

    Assumptions
    -----------
    'BKO' (case-insensitive) -> BKO; anything else -> KPCWT.
    Condition tokens are searched in priority order: LIVE > FORM20MIN >
    FORM10MIN > FORM > GLU.

    Examples
    --------
    >>> label_of('3_KPCWT0430form20min_740nm...')
    'KPCWT form 20min'
    >>> label_of('5_BKO0501live_740nm...')
    'BKO live'

    Dependencies
    ------------
    None (stdlib only)
    """
    s = str(stem).upper()
    cell = "BKO" if "BKO" in s else "KPCWT"
    if "LIVE" in s:
        cond = "live"
    elif "FORM20MIN" in s:
        cond = "form 20min"
    elif "FORM10MIN" in s:
        cond = "form 10min"
    elif "FORM" in s:
        cond = "form"
    elif "GLU" in s:
        cond = "glu"
    else:
        cond = "?"
    return f"{cell} {cond}"


# ---------------------------------------------------------------------------
# Pockels classification
# ---------------------------------------------------------------------------

def pockels_class(values, threshold: float = 0.3):
    """Bin Pockels cell voltage into 'Low' (<= threshold) / 'High' (> threshold).

    Used to distinguish the two laser-power regimes in the dataset: low Pockels
    (<=0.25, higher photon count) and high Pockels (>=0.45, lower photon count).
    The 'PC' column in df_analysis is derived from this function via
    build_analysis_df().

    Parameters
    ----------
    values : scalar, list, np.ndarray, or pd.Series
        Pockels cell voltage reading(s), typically in [0, 1].
    threshold : float, optional
        Boundary between 'Low' and 'High'.  Default 0.3 sits between the
        two natural clusters (<=0.25 and >=0.45) in the KPC dataset.

    Returns
    -------
    str, list, or pd.Series
        Same container type as the input.  Each element is 'Low', 'High',
        or float('nan') (for NaN inputs).

    Assumptions
    -----------
    NaN inputs (pd.isna returns True) are passed through as float('nan').
    numpy arrays and plain lists are returned as lists, not arrays.

    Examples
    --------
    >>> pockels_class(0.25)           # -> 'Low'
    >>> pockels_class(0.45)           # -> 'High'
    >>> pockels_class(df['pockels'])  # -> pd.Series of 'Low'/'High'

    Dependencies
    ------------
    numpy, pandas
    """
    if isinstance(values, pd.Series):
        return values.apply(lambda v: float("nan") if pd.isna(v)
                            else ("High" if v > threshold else "Low"))
    if np.isscalar(values):
        if pd.isna(values):
            return float("nan")
        return "High" if values > threshold else "Low"
    return ["High" if v > threshold else "Low" for v in values]


# ---------------------------------------------------------------------------
# DataFrame assembly
# ---------------------------------------------------------------------------

def build_analysis_df(
    fit_analysis_csv: str | Path,
    sdt_metadata_csv: str | Path,
    *,
    annotation_csv: str | Path | None = None,
    filepath_map_csv: str | Path | None = None,
    pockels_threshold: float = 0.3,
    derive_date: bool = True,
    derive_pc: bool = True,
    extra_sdt_cols: tuple = ("pockels", "treatment_duration", "frame_index",
                              "phasor_cal_phase_rad", "phasor_cal_mod",
                              "power_mW"),
) -> pd.DataFrame:
    """Build the canonical analysis DataFrame used by quickpeek3 and Phase F.

    Steps:
      1. Read fit_analysis_summary.csv (one row per .sdt file that survived
         Phase E quality filtering).
      2. Merge any of `extra_sdt_cols` not already present from
         sdt_metadata_cal.csv.
      3. Optionally merge position_annotation.csv on filename (adds
         'annotation' column; missing filenames get '(unannotated)').
      4. Optionally merge filepath_map.csv (raw .sdt filepath).
      5. Optionally derive `date` (from session_root, format YYYYMMDD) and
         `PC` ('Low'/'High' from pockels with `pockels_threshold`).

    Parameters
    ----------
    fit_analysis_csv : str or Path
        Path to fit_analysis_summary.csv (Phase E output).  One row per
        accepted .sdt image.  Must contain a 'filename' column.
    sdt_metadata_csv : str or Path
        Path to sdt_metadata_cal.csv (Phase A/C output).  Joined on
        'filename'.
    annotation_csv : str or Path, optional
        Path to position_annotation.csv (adds 'annotation' column, e.g.
        'colony_deep', 'colony_edge').
    filepath_map_csv : str or Path, optional
        Path to filepath_map.csv (adds 'filepath' with the absolute path to
        the raw .sdt file; machine-specific).
    pockels_threshold : float, optional
        Cutoff for 'Low'/'High' PC binning.  Default 0.3.
    derive_date : bool, optional
        If True, parse 'date' (datetime) from the YYYYMMDD prefix of
        session_root.
    derive_pc : bool, optional
        If True, add 'PC' column ('Low'/'High') from pockels column.
    extra_sdt_cols : tuple of str, optional
        Columns to pull from sdt_metadata if not already in fit_df.
        Default: pockels, treatment_duration, frame_index,
        phasor_cal_phase_rad, phasor_cal_mod, power_mW.

    Returns
    -------
    pd.DataFrame
        Merged analysis DataFrame.  One row per accepted .sdt image.
        Key columns: filename, session_root, fixation_type, cell_type,
        em_filter_nm, n_components, tau_mean_median_ps, amp_ratio_median,
        fit_set_key, fit_mask_path, n_px_final, pct_final, plus any
        annotation/date/PC columns derived above.

    Side Effects
    ------------
    Reads CSV files from disk.  annotation_csv and filepath_map_csv are
    silently skipped if the file does not exist.

    Assumptions
    -----------
    All CSVs use 'filename' as the join key.  session_root must start with
    a YYYYMMDD date prefix for derive_date to succeed.

    Examples
    --------
    >>> df_analysis = build_analysis_df(
    ...     'results/fit_analysis_summary.csv',
    ...     'results/sdt_metadata_cal.csv',
    ...     annotation_csv='results/position_annotation.csv',
    ... )

    Dependencies
    ------------
    pandas, pathlib, pockels_class
    """
    sdt = pd.read_csv(sdt_metadata_csv)
    df  = pd.read_csv(fit_analysis_csv)

    # Bring in metadata columns that aren't already in fit_df.
    cols_to_merge = [c for c in extra_sdt_cols
                     if c not in df.columns and c in sdt.columns]
    if cols_to_merge:
        df = df.merge(sdt[["filename"] + cols_to_merge],
                      on="filename", how="left")

    if annotation_csv is not None:
        annot_path = Path(annotation_csv)
        if annot_path.exists():
            annot = pd.read_csv(annot_path)
            df = df.merge(annot[["filename", "annotation"]],
                          on="filename", how="left")
            df["annotation"] = df["annotation"].fillna("(unannotated)")

    if filepath_map_csv is not None:
        fp_path = Path(filepath_map_csv)
        if fp_path.exists():
            fp_map = pd.read_csv(fp_path)
            df = df.merge(fp_map[["filename", "filepath"]],
                          on="filename", how="left")

    if derive_date and "session_root" in df.columns:
        df["date"] = pd.to_datetime(
            df.session_root.str.split("_").str[0], format="%Y%m%d",
            errors="coerce",
        )

    if derive_pc and "pockels" in df.columns:
        df["PC"] = pockels_class(df["pockels"], threshold=pockels_threshold)

    return df


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_subset(df: pd.DataFrame, **conditions) -> pd.DataFrame:
    """Filter a DataFrame to rows matching every condition in kwargs.

    All conditions are combined with logical AND.  NaN values in any
    column never satisfy a condition (fillna(False) is applied internally).

    Each kwarg value is interpreted as:
      - scalar:                 df[col] == value
      - list/tuple/set/Series:  df[col].isin(value)
      - callable:               df[col].map(callable)

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame (not modified; a copy is returned).
    **conditions
        Keyword arguments where each key is a column name and the value
        is a scalar, collection, or callable as described above.

    Returns
    -------
    pd.DataFrame
        Filtered copy of df containing only matching rows.

    Raises
    ------
    KeyError
        If a condition key is not a column in df.

    Assumptions
    -----------
    To explicitly drop NaN rows in a column, pass col=pd.notna (callable
    form); otherwise NaN rows are simply excluded by fillna(False).

    Examples
    --------
    >>> sub = filter_subset(df, fixation_type='form',
    ...                         em_filter_nm=[457, 535],
    ...                         PC='Low',
    ...                         annotation='colony_edge')
    >>> sub = filter_subset(df, pockels=pd.notna)  # drop rows where pockels is NaN

    Dependencies
    ------------
    pandas, numpy
    """
    mask = pd.Series(True, index=df.index)
    for col, value in conditions.items():
        if col not in df.columns:
            raise KeyError(f"filter_subset: column {col!r} not in DataFrame")
        s = df[col]
        if callable(value):
            cond = s.map(value)
        elif isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
            cond = s.isin(list(value))
        else:
            cond = (s == value)
        mask &= cond.fillna(False)
    return df.loc[mask].copy()
