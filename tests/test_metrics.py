"""Unit tests for src/metrics.py group statistics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.metrics import (    # noqa: E402
    benjamini_hochberg,
    rank_biserial_r,
    effect_stats,
    perm_test,
)


# ---------------------------------------------------------------------------
# rank_biserial_r
# ---------------------------------------------------------------------------

def test_rank_biserial_r_sign_positive_when_x_larger():
    x = np.array([10.0, 11.0, 12.0])
    y = np.array([1.0, 2.0, 3.0])
    r = rank_biserial_r(x, y)
    assert r > 0.99   # essentially +1: all x ranks > all y ranks


def test_rank_biserial_r_sign_negative_when_x_smaller():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 11.0, 12.0])
    r = rank_biserial_r(x, y)
    assert r < -0.99


def test_rank_biserial_r_zero_when_identical():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    assert abs(rank_biserial_r(x, y)) < 1e-9


def test_rank_biserial_r_nan_on_empty():
    assert np.isnan(rank_biserial_r(np.array([]), np.array([1.0])))


# ---------------------------------------------------------------------------
# benjamini_hochberg (q-values, matches Phase F implementation)
# ---------------------------------------------------------------------------

def test_benjamini_hochberg_returns_qvalues_at_least_p():
    p = [0.01, 0.02, 0.03, 0.04]
    q = benjamini_hochberg(p)
    # Each q-value should be >= corresponding p-value
    for pi, qi in zip(p, q):
        assert qi >= pi - 1e-12


def test_benjamini_hochberg_clamps_to_one():
    q = benjamini_hochberg([0.9, 0.95, 0.99])
    assert q.max() <= 1.0


def test_benjamini_hochberg_monotonic_in_sorted_p():
    p = np.array([0.001, 0.01, 0.05, 0.5, 0.9])
    q = benjamini_hochberg(p)
    # When p is sorted, q is also sorted (non-decreasing)
    assert np.all(np.diff(q) >= -1e-12)


def test_benjamini_hochberg_empty_returns_empty():
    out = benjamini_hochberg([])
    assert out.size == 0


# ---------------------------------------------------------------------------
# effect_stats
# ---------------------------------------------------------------------------

def test_effect_stats_basic_positive_difference():
    df = pd.DataFrame({
        "cell_type": ["KPCWT"] * 5 + ["BKO"] * 5,
        "metric":    [10.0, 11, 12, 13, 14] + [1.0, 2, 3, 4, 5],
    })
    e = effect_stats(df, "metric", group_a="KPCWT", group_b="BKO", n_boot=200)
    assert e is not None
    assert e["diff"] > 0
    assert e["r"] > 0
    assert e["p"] < 0.05
    assert e["n_a"] == 5 and e["n_b"] == 5


def test_effect_stats_returns_none_when_underpowered():
    df = pd.DataFrame({"cell_type": ["KPCWT"], "metric": [1.0]})
    assert effect_stats(df, "metric", group_a="KPCWT", group_b="BKO") is None


def test_effect_stats_ci_brackets_point_estimate():
    df = pd.DataFrame({
        "cell_type": ["KPCWT"] * 10 + ["BKO"] * 10,
        "metric":    np.concatenate([np.full(10, 5.0), np.full(10, 1.0)]),
    })
    e = effect_stats(df, "metric", group_a="KPCWT", group_b="BKO", n_boot=200)
    # With no variation within groups, CI degenerates to the point estimate
    assert e["ci_lo"] <= e["diff"] <= e["ci_hi"]


# ---------------------------------------------------------------------------
# perm_test
# ---------------------------------------------------------------------------

def test_perm_test_significant_with_clear_separation():
    df = pd.DataFrame({
        "cell_type": ["A"] * 8 + ["B"] * 8,
        "metric":    list(np.arange(100.0, 108.0)) + list(np.arange(0.0, 8.0)),
    })
    res = perm_test(df, "metric", group_a="A", group_b="B",
                    n_perm=500, seed=0)
    assert res is not None
    obs, null, p_emp = res
    assert obs > 0
    assert p_emp < 0.05


def test_perm_test_underpowered_returns_none():
    df = pd.DataFrame({"cell_type": ["A"], "metric": [1.0]})
    assert perm_test(df, "metric", group_a="A", group_b="B") is None
