"""Unit tests for pipeline/batch.py and pipeline/export.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.batch  import process_all_fit_sets, process_one_fit_set  # noqa: E402
from pipeline.export import (    # noqa: E402
    write_per_folder_quality_summaries,
    write_low_retention_log,
    print_retention_stats,
)


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------

def _build_fixture(tmp_path: Path):
    """One session with one fit folder + one .sdt + a 'form' file_type."""
    session = tmp_path / "SESSION_X"
    folder  = session / "fitet-sz-c2-b1"
    folder.mkdir(parents=True)
    stem    = "img_0001"
    # photons high enough to pass MIN_PHOTONS for n_comp=2 (= 1000 after binning)
    np.savetxt(folder / f"{stem}_photons.asc",         np.full((4, 4), 1000.0))
    np.savetxt(folder / f"{stem}_chi.asc",             np.full((4, 4), 1.0))
    np.savetxt(folder / f"{stem}_a1.asc",              np.full((4, 4), 0.5))
    np.savetxt(folder / f"{stem}_a2.asc",              np.full((4, 4), 0.5))
    np.savetxt(folder / f"{stem}_t1.asc",              np.full((4, 4), 400.0))
    np.savetxt(folder / f"{stem}_t2.asc",              np.full((4, 4), 2500.0))

    fit_map = pd.DataFrame({
        "sdt_filename": [f"{stem}.sdt"],
        "fit_set_key":  [f"SESSION_X::fitet-sz-c2-b1::{stem}"],
    })
    sdt = pd.DataFrame({
        "filename":      [f"{stem}.sdt"],
        "file_type":     ["sample"],
        "fixation_type": ["form"],
        "filepath":      [str(tmp_path / "nonexistent.sdt")],
    })
    return session, folder, stem, fit_map, sdt


# ---------------------------------------------------------------------------
# process_one_fit_set
# ---------------------------------------------------------------------------

def test_process_one_fit_set_writes_npy_and_returns_row(tmp_path):
    session, folder, stem, fit_map, sdt = _build_fixture(tmp_path)
    row = process_one_fit_set(
        f"SESSION_X::fitet-sz-c2-b1::{stem}",
        fixation_type="form",
        sdt_path=None,
        data_dirs=[session],
    )
    assert row is not None
    assert row["base_stem"] == stem
    assert row["fit_folder"] == "fitet-sz-c2-b1"
    assert row["b_val"] == 1
    assert row["fixation_type"] == "form"
    assert row["n_total"] == 16
    assert row["pct_final"] == 100.0
    assert (folder / f"{stem}_fit_mask.npy").exists()


def test_process_one_fit_set_missing_dir_returns_none(tmp_path):
    out = process_one_fit_set(
        "MISSING::fitet-sz-c2-b1::stem",
        fixation_type="form",
        sdt_path=None,
        data_dirs=[tmp_path],
    )
    assert out is None


# ---------------------------------------------------------------------------
# process_all_fit_sets
# ---------------------------------------------------------------------------

def test_process_all_fit_sets_returns_dataframe(tmp_path):
    session, folder, stem, fit_map, sdt = _build_fixture(tmp_path)
    df, low_ret = process_all_fit_sets(fit_map, sdt, [session],
                                        force_recompute=True)
    assert len(df) == 1
    expected_cols = {"fit_set_key", "sdt_filename", "session_root", "fit_folder",
                     "base_stem", "b_val", "shift_free", "fixation_type",
                     "n_total", "n_final", "pct_final"}
    assert expected_cols.issubset(df.columns)
    assert low_ret == []
    assert (folder / f"{stem}_fit_mask.npy").exists()


def test_process_all_fit_sets_low_retention_recorded(tmp_path):
    session, folder, stem, fit_map, sdt = _build_fixture(tmp_path)
    # Drop photons to trigger 0% retention
    (folder / f"{stem}_photons.asc").unlink()
    np.savetxt(folder / f"{stem}_photons.asc", np.zeros((4, 4)))
    df, low_ret = process_all_fit_sets(fit_map, sdt, [session],
                                        force_recompute=True)
    assert len(df) == 1
    assert df.iloc[0]["pct_final"] == 0.0
    assert len(low_ret) == 1
    assert low_ret[0][0] == f"{stem}.sdt"


# ---------------------------------------------------------------------------
# export functions
# ---------------------------------------------------------------------------

def test_write_per_folder_quality_summaries(tmp_path):
    session, folder, stem, fit_map, sdt = _build_fixture(tmp_path)
    df = pd.DataFrame([{
        "fit_set_key":  f"SESSION_X::fitet-sz-c2-b1::{stem}",
        "session_root": "SESSION_X",
        "fit_folder":   "fitet-sz-c2-b1",
        "n_final":      10,
        "pct_final":    62.5,
    }])
    n = write_per_folder_quality_summaries(df, [session])
    assert n == 1
    out = folder / "fit_quality_summary.csv"
    assert out.exists()
    written = pd.read_csv(out)
    # drop_cols removes the join keys
    assert "fit_set_key" not in written.columns
    assert "n_final" in written.columns


def test_write_low_retention_log_writes_records(tmp_path):
    p = tmp_path / "warnings.txt"
    n = write_low_retention_log(
        [("a.sdt", "fitet-sz-c2-b1", 1, 0.5),
         ("b.sdt", "fitet-sz-c2-b1", 1, 0.2)],
        p, verbose=False,
    )
    assert n == 2
    text = p.read_text(encoding="utf-8")
    assert "a.sdt" in text and "b.sdt" in text
    assert text.startswith("#")   # header line


def test_write_low_retention_log_removes_stale(tmp_path):
    p = tmp_path / "warnings.txt"
    p.write_text("stale\n", encoding="utf-8")
    n = write_low_retention_log([], p, verbose=False)
    assert n == 0
    assert not p.exists()


def test_print_retention_stats_empty_is_noop(capsys):
    print_retention_stats(pd.DataFrame())
    captured = capsys.readouterr()
    assert captured.out == ""
