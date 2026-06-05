"""Unit tests for utils/helpers.py."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.helpers import (    # noqa: E402
    set2_palette,
    violin_panel,
    violin_grid,
    show_image_row,
    DEFAULT_IMAGE_PANELS,
)


def test_set2_palette_keys_match_input():
    cts = ["KPCWT", "BKO"]
    pal = set2_palette(cts)
    assert set(pal.keys()) == set(cts)


def test_set2_palette_distinct_colors():
    cts = ["A", "B", "C"]
    pal = set2_palette(cts)
    arrs = [tuple(pal[c]) for c in cts]
    assert len(set(arrs)) == 3


def test_violin_panel_returns_cts_and_groups():
    df = pd.DataFrame({
        "cell_type": ["KPCWT"] * 5 + ["BKO"] * 5,
        "metric":    list(range(5)) + list(range(10, 15)),
    })
    fig, ax = plt.subplots()
    cts, groups = violin_panel(
        ax, df, "metric",
        ct_order=["KPCWT", "BKO"],
        ct_colors=set2_palette(["KPCWT", "BKO"]),
    )
    plt.close(fig)
    assert cts == ["KPCWT", "BKO"]
    assert len(groups) == 2
    assert len(groups[0]) == 5
    assert len(groups[1]) == 5


def test_violin_panel_skips_underpowered_groups():
    df = pd.DataFrame({
        "cell_type": ["KPCWT"] * 4 + ["BKO"] * 1,    # BKO has < min_n=2
        "metric":    list(range(5)),
    })
    fig, ax = plt.subplots()
    cts, groups = violin_panel(
        ax, df, "metric",
        ct_order=["KPCWT", "BKO"],
        ct_colors=set2_palette(["KPCWT", "BKO"]),
    )
    plt.close(fig)
    assert cts == ["KPCWT"]


def test_violin_panel_no_data_text():
    df = pd.DataFrame({"cell_type": [], "metric": []})
    fig, ax = plt.subplots()
    cts, groups = violin_panel(
        ax, df, "metric",
        ct_order=["KPCWT", "BKO"],
        ct_colors=set2_palette(["KPCWT", "BKO"]),
    )
    plt.close(fig)
    assert cts == []
    assert groups == []


def test_violin_grid_2d_layout():
    rng = np.random.default_rng(0)
    fixation = ["form", "live"] * 80
    channel  = [457, 535] * 80
    cells    = ["KPCWT"] * 80 + ["BKO"] * 80
    df = pd.DataFrame({
        "fixation_type": rng.permutation(fixation),
        "em_filter_nm":  rng.permutation(channel),
        "cell_type":     rng.permutation(cells),
        "metric":        rng.normal(0, 1, 160),
    })
    fig, axes = violin_grid(
        df, "metric",
        row_by="fixation_type", col_by="em_filter_nm",
        group_order=["KPCWT", "BKO"],
        suptitle="test",
    )
    assert axes.shape == (2, 2)
    plt.close(fig)


def test_violin_grid_single_dim():
    df = pd.DataFrame({
        "session": ["s1"] * 6 + ["s2"] * 6,
        "cell_type": (["KPCWT"] * 3 + ["BKO"] * 3) * 2,
        "metric":  [1.0, 2, 3, 10, 11, 12, 4, 5, 6, 7, 8, 9],
    })
    fig, axes = violin_grid(
        df, "metric",
        col_by="session",
        group_order=["KPCWT", "BKO"],
    )
    assert axes.shape == (1, 2)
    plt.close(fig)


def test_violin_grid_empty_panel_does_not_crash():
    df = pd.DataFrame({
        "fixation_type": ["form"] * 4,
        "em_filter_nm":  [457] * 4,
        "cell_type":     ["KPCWT"] * 2 + ["BKO"] * 2,
        "metric":        [1.0, 2, 10, 11],
    })
    # row='live' has no rows -> panel shows "no data"
    fig, axes = violin_grid(
        df, "metric",
        row_by="fixation_type", col_by="em_filter_nm",
        row_order=["form", "live"],
        group_order=["KPCWT", "BKO"],
    )
    assert axes.shape == (2, 1)
    plt.close(fig)


def test_show_image_row_renders():
    shape = (8, 8)
    arrs = {
        "photons":  np.full(shape, 100.0),
        "tau_mean": np.full(shape, 1500.0),
        "a1":       np.full(shape, 0.6),
        "a2":       np.full(shape, 0.4),
        "a1/a2":    np.full(shape, 1.5),
        "chi2":     np.full(shape, 1.0),
        "mask":     np.ones(shape, dtype=bool),
    }
    fig, axes = plt.subplots(1, len(DEFAULT_IMAGE_PANELS) + 1)
    show_image_row(axes, arrs)
    plt.close(fig)


def test_show_image_row_grays_out_rejected():
    shape = (8, 8)
    mask = np.zeros(shape, dtype=bool)
    mask[2:6, 2:6] = True
    arrs = {
        "photons":  np.full(shape, 100.0),
        "tau_mean": np.full(shape, 1500.0),
        "a1":       np.full(shape, 0.6),
        "a2":       np.full(shape, 0.4),
        "a1/a2":    np.full(shape, 1.5),
        "chi2":     np.full(shape, 1.0),
        "mask":     mask,
    }
    fig, axes = plt.subplots(1, len(DEFAULT_IMAGE_PANELS) + 1)
    show_image_row(axes, arrs)
    plt.close(fig)
