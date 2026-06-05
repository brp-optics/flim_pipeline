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
    """Filename stem -> 'CELLTYPE condition' string (e.g. 'BKO form 20min')."""
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
    """Bin Pockels values into 'Low' (<= threshold) / 'High' (> threshold).

    Accepts a scalar, list, pandas Series, or numpy array.  Returns the same
    container type.  NaN inputs -> NaN.
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
      3. Optionally merge position_annotation.csv on filename.
      4. Optionally merge filepath_map.csv (raw .sdt filepath).
      5. Optionally derive `date` (from session_root, format YYYYMMDD) and
         `PC` ('Low'/'High' from pockels with `pockels_threshold`).

    Args:
        fit_analysis_csv:  path to fit_analysis_summary.csv (Phase E output).
        sdt_metadata_csv:  path to sdt_metadata_cal.csv (Phase A/C output).
        annotation_csv:    optional path to position_annotation.csv.
        filepath_map_csv:  optional path to filepath_map.csv (for raw .sdt
                           paths).
        pockels_threshold: cutoff for 'Low'/'High' PC binning.
        derive_date:       add a 'date' column (datetime).
        derive_pc:         add a 'PC' column ('Low'/'High').
        extra_sdt_cols:    columns to pull from sdt_metadata if not in fit_df.

    Returns the merged DataFrame.
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

    Each kwarg is one of:
      - scalar value:           df[col] == value
      - list/tuple/set/Series:  df[col].isin(value)
      - callable(series)->bool: df[col].map(callable)

    Special handling:
      - if a key isn't in df.columns, raises KeyError with a helpful message.
      - to drop NaN rows in a column, pass `col=pd.notna` or use df.dropna().

    Examples:
        filter_subset(df, fixation_type='form',
                          em_filter_nm=[457, 475],
                          PC='Low',
                          annotation='colony_deep',
                          treatment_duration='10min')
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
