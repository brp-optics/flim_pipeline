"""Unit tests for src/preprocess.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocess import (    # noqa: E402
    label_of,
    pockels_class,
    filter_subset,
    build_analysis_df,
)


# ---------------------------------------------------------------------------
# label_of
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stem,expected", [
    ("3_KPCWTform10min_xxx_0000",     "KPCWT form 10min"),
    ("4_BKOCO2form10min_xxx_0000",    "BKO form 10min"),
    ("3_KPCWTform20min_xxx_0000",     "KPCWT form 20min"),
    ("4_BKOCO2form20min_xxx_0000",    "BKO form 20min"),
    ("3_KPCWTlive_xxx_0000",          "KPCWT live"),
    ("3_BKOCO2live_xxx_0000",         "BKO live"),
    ("15_KPC_live_posn3_xxx_0002",    "KPCWT live"),
    ("9_BKO_glu_xxx",                 "BKO glu"),
    ("unknown_thing",                 "KPCWT ?"),
])
def test_label_of(stem, expected):
    assert label_of(stem) == expected


# ---------------------------------------------------------------------------
# pockels_class
# ---------------------------------------------------------------------------

def test_pockels_class_scalar():
    assert pockels_class(0.1) == "Low"
    assert pockels_class(0.3) == "Low"
    assert pockels_class(0.6) == "High"


def test_pockels_class_nan():
    assert np.isnan(pockels_class(float("nan")))


def test_pockels_class_series():
    s = pd.Series([0.1, 0.5, 0.6, float("nan")])
    out = pockels_class(s)
    assert list(out[:3]) == ["Low", "High", "High"]
    assert pd.isna(out.iloc[3])


def test_pockels_class_custom_threshold():
    assert pockels_class(0.4, threshold=0.5) == "Low"
    assert pockels_class(0.4, threshold=0.3) == "High"


# ---------------------------------------------------------------------------
# filter_subset
# ---------------------------------------------------------------------------

def _make_df():
    return pd.DataFrame({
        "fixation_type":      ["live", "form", "form", "live"],
        "em_filter_nm":       [457,    457,    535,    475],
        "PC":                 ["Low",  "Low",  "Low",  "High"],
        "annotation":         ["colony_deep", "colony_deep",
                               "colony_edge", "colony_deep"],
        "treatment_duration": [None, "10min", "20min", None],
        "metric":             [1.0, 2.0, 3.0, 4.0],
    })


def test_filter_subset_single_scalar():
    df = _make_df()
    sub = filter_subset(df, fixation_type="form")
    assert list(sub.metric) == [2.0, 3.0]


def test_filter_subset_list_isin():
    df = _make_df()
    sub = filter_subset(df, em_filter_nm=[457, 475])
    assert list(sub.metric) == [1.0, 2.0, 4.0]


def test_filter_subset_multiple_conditions():
    df = _make_df()
    sub = filter_subset(df, fixation_type="live", PC="Low")
    assert list(sub.metric) == [1.0]


def test_filter_subset_unknown_column():
    df = _make_df()
    with pytest.raises(KeyError):
        filter_subset(df, not_a_column="x")


def test_filter_subset_callable():
    df = _make_df()
    sub = filter_subset(df, metric=lambda v: v > 2.0)
    assert list(sub.metric) == [3.0, 4.0]


# ---------------------------------------------------------------------------
# build_analysis_df (filesystem-backed)
# ---------------------------------------------------------------------------

def test_build_analysis_df_against_real_results():
    """Against the real results/* files, build the DataFrame and verify
    enough surface-level invariants.  This exercises the merge logic end-to-end.
    """
    pr = _PROJECT_ROOT
    df = build_analysis_df(
        fit_analysis_csv = pr / "results" / "fit_analysis_summary.csv",
        sdt_metadata_csv = pr / "results" / "sdt_metadata_cal.csv",
        annotation_csv   = pr / "results" / "position_annotation.csv",
    )
    assert len(df) > 0
    assert "pockels" in df.columns
    assert "date" in df.columns
    assert "PC" in df.columns
    assert "annotation" in df.columns
    assert df["PC"].dropna().isin(["Low", "High"]).all()


def test_build_analysis_df_minimal(tmp_path):
    """build_analysis_df works without optional annotation/filepath args."""
    fit_csv = tmp_path / "fit.csv"
    sdt_csv = tmp_path / "sdt.csv"
    pd.DataFrame({
        "filename":      ["a.sdt", "b.sdt"],
        "session_root":  ["20260101_x", "20260102_x"],
        "fixation_type": ["form", "live"],
    }).to_csv(fit_csv, index=False)
    pd.DataFrame({
        "filename":   ["a.sdt", "b.sdt"],
        "pockels":    [0.1, 0.5],
        "treatment_duration": ["10min", None],
    }).to_csv(sdt_csv, index=False)

    df = build_analysis_df(fit_csv, sdt_csv)
    assert "pockels" in df.columns
    assert "PC" in df.columns
    assert "date" in df.columns
    assert list(df.PC) == ["Low", "High"]
