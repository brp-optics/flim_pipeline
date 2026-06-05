"""Unit tests for src/io.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# make src/ importable regardless of cwd
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.io import (    # noqa: E402
    _infer_param_name,
    _strip_param_suffix,
    parse_fit_folder,
    fit_dir_from_key,
    load_asc_fit_set,
    load_saved_mask,
    load_image_bundle,
)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stem,expected", [
    ("13_KPC_live_xxx_a1",       "a1"),
    ("13_KPC_live_xxx_a2",       "a2"),
    ("13_KPC_live_xxx_a3",       "a3"),
    ("13_KPC_live_xxx_t1",       "tau1"),
    ("13_KPC_live_xxx_t2",       "tau2"),
    ("13_KPC_live_xxx_t3",       "tau3"),
    ("13_KPC_live_xxx_tau1",     "tau1"),
    ("13_KPC_live_xxx_chi",      "chi2"),
    ("13_KPC_live_xxx_photons",  "photons"),
    ("13_KPC_live_xxx_shift",    "shift"),
])
def test_infer_param_name(stem, expected):
    assert _infer_param_name(stem) == expected


def test_infer_param_name_fallback():
    assert _infer_param_name("foo_bar_qux") == "qux"


@pytest.mark.parametrize("stem,expected", [
    ("13_KPC_live_basicFLIM_120f_0002_a1",      "13_KPC_live_basicFLIM_120f_0002"),
    ("13_KPC_live_basicFLIM_120f_0002_tau1",    "13_KPC_live_basicFLIM_120f_0002"),
    ("13_KPC_live_basicFLIM_120f_0002_chi",     "13_KPC_live_basicFLIM_120f_0002"),
    ("13_KPC_live_basicFLIM_120f_0002_photons", "13_KPC_live_basicFLIM_120f_0002"),
])
def test_strip_param_suffix(stem, expected):
    assert _strip_param_suffix(stem) == expected


# ---------------------------------------------------------------------------
# Fit folder parsing
# ---------------------------------------------------------------------------

def test_parse_fit_folder_basic_b_shift():
    r = parse_fit_folder("fitet-sz-b5")
    assert r["b_val"] == 5
    assert r["kernel_size"] == 11
    assert r["shift_free"] is False
    assert r["n_components"] is None
    assert r["is_export"] is False


def test_parse_fit_folder_shift_free_with_ncomp():
    r = parse_fit_folder("fitet-sf-b10-c3")
    assert r["b_val"] == 10
    assert r["shift_free"] is True
    assert r["n_components"] == 3


def test_parse_fit_folder_word_form_ncomp():
    r = parse_fit_folder("fitet-sz-b2-1component")
    assert r["b_val"] == 2
    assert r["n_components"] == 1


def test_parse_fit_folder_export_tif():
    r = parse_fit_folder("exet600-1400-0-500-fitet-sz-b2")
    assert r["is_export"] is True
    assert r["export_format"] == "tif"
    assert r["color_lo"] == 600.0
    assert r["color_hi"] == 1400.0
    assert r["intensity_lo"] == 0.0
    assert r["intensity_hi"] == 500.0
    assert r["b_val"] == 2


def test_parse_fit_folder_export_asc():
    r = parse_fit_folder("exet-fitet-sz-b3")
    assert r["is_export"] is True
    assert r["export_format"] == "asc"
    assert r["b_val"] == 3


# ---------------------------------------------------------------------------
# Fit-set-key navigation
# ---------------------------------------------------------------------------

def test_fit_dir_from_key_resolves_session(tmp_path):
    a = tmp_path / "A"
    sx = tmp_path / "SESSION_X"
    b = tmp_path / "B"
    for p in (a, sx, b):
        p.mkdir()
    key = "SESSION_X::fitet-sz-b9::stem_0001"
    fit_dir, session, rel, stem = fit_dir_from_key(key, [a, sx, b])
    assert session == "SESSION_X"
    assert rel == "fitet-sz-b9"
    assert stem == "stem_0001"
    assert fit_dir == sx / "fitet-sz-b9"


def test_fit_dir_from_key_missing_session(tmp_path):
    fit_dir, *_ = fit_dir_from_key("UNKNOWN::x::y", [tmp_path / "A"])
    assert fit_dir is None


# ---------------------------------------------------------------------------
# ASC + mask loaders
# ---------------------------------------------------------------------------

def test_load_asc_fit_set_picks_matching_stem(tmp_path):
    stem_a = "img_0001"
    stem_b = "other_0001"
    np.savetxt(tmp_path / f"{stem_a}_a1.asc", np.ones((4, 4)))
    np.savetxt(tmp_path / f"{stem_a}_a2.asc", np.full((4, 4), 2.0))
    np.savetxt(tmp_path / f"{stem_b}_a1.asc", np.zeros((4, 4)))
    np.savetxt(tmp_path / f"{stem_a}_statistic.asc", np.full((4, 4), 9.0))

    fd = load_asc_fit_set(tmp_path, stem_a)
    assert set(fd.keys()) == {"a1", "a2"}
    np.testing.assert_array_equal(fd["a1"], np.ones((4, 4)))
    np.testing.assert_array_equal(fd["a2"], np.full((4, 4), 2.0))


def test_load_saved_mask_roundtrip(tmp_path):
    session = tmp_path / "SESSION_X"
    folder = session / "fitet-sz-b9"
    folder.mkdir(parents=True)
    mask = np.array([[True, False], [False, True]])
    np.save(folder / "stem_A_fit_mask.npy", mask)

    out = load_saved_mask("SESSION_X::fitet-sz-b9::stem_A", [session])
    np.testing.assert_array_equal(out, mask)


def test_load_saved_mask_missing_returns_none(tmp_path):
    session = tmp_path / "SESSION_X"
    (session / "fitet-sz-b9").mkdir(parents=True)
    out = load_saved_mask("SESSION_X::fitet-sz-b9::stem_NO", [session])
    assert out is None


# ---------------------------------------------------------------------------
# load_image_bundle
# ---------------------------------------------------------------------------

def test_load_image_bundle_roundtrip(tmp_path):
    stem = "img_0001"
    np.savetxt(tmp_path / f"{stem}_photons.asc",            np.full((4, 4), 1000.0))
    np.savetxt(tmp_path / f"{stem}_color coded value.asc", np.full((4, 4), 1500.0))
    np.savetxt(tmp_path / f"{stem}_a1.asc",                 np.full((4, 4), 0.6))
    np.savetxt(tmp_path / f"{stem}_a2.asc",                 np.full((4, 4), 0.4))
    np.savetxt(tmp_path / f"{stem}_chi.asc",                np.full((4, 4), 1.0))
    np.save   (tmp_path / f"{stem}_fit_mask.npy", np.ones((4, 4), dtype=bool))

    arrs, base_stem = load_image_bundle(tmp_path / f"{stem}_fit_mask.npy")
    assert base_stem == stem
    assert set(arrs.keys()) == {"photons", "tau_mean", "a1", "a2",
                                  "chi2", "mask", "a1/a2"}
    np.testing.assert_allclose(arrs["a1/a2"], np.full((4, 4), 1.5))
