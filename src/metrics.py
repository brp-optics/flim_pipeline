"""Per-image and group-level metrics for FLIM data.

Provides:
  - compute_tau_mean: amplitude-weighted mean lifetime per pixel
  - rank_biserial_r: effect size for Mann-Whitney U
  - benjamini_hochberg: FDR adjustment
  - effect_stats: bootstrap CI + Mann-Whitney for two-group comparison
  - perm_test: permutation null distribution for two-group comparison

Defaults (DEFAULT_TAUMEAN_CFG) match the Phase E pipeline:
  glu  -> a2*tau2 + a3*tau3 / (a2+a3)
  form -> a1*tau1 + a2*tau2 / (a1+a2)
  live -> same as form
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Default TAUMEAN config (matches Phase E defaults)
# ---------------------------------------------------------------------------

DEFAULT_TAUMEAN_CFG: dict = {
    "glu":  (("a2", "tau2"), ("a3", "tau3")),
    "form": (("a1", "tau1"), ("a2", "tau2")),
    "live": (("a1", "tau1"), ("a2", "tau2")),
}


# ---------------------------------------------------------------------------
# Amplitude-weighted mean lifetime
# ---------------------------------------------------------------------------

def compute_tau_mean(
    fd: dict,
    fixation_type: str,
    taumean_cfg: dict | None = None,
) -> np.ndarray | None:
    """Compute amplitude-weighted tau_mean (ps) from fit arrays.

    Returns a 2-D array in the same units as the tau exports, or None if any
    required parameter is absent.

    Args:
        fd:            {param_name: ndarray} as returned by load_asc_fit_set.
        fixation_type: 'glu' | 'form' | 'live'.
        taumean_cfg:   override the default (a, tau) component pairs.
    """
    cfg = taumean_cfg if taumean_cfg is not None else DEFAULT_TAUMEAN_CFG
    components = cfg.get(fixation_type)
    if components is None:
        return None
    if any(a not in fd or t not in fd for a, t in components):
        return None
    a_arrs   = [fd[a] for a, _ in components]
    tau_arrs = [fd[t] for _, t in components]
    a_sum    = sum(a_arrs)
    with np.errstate(invalid="ignore", divide="ignore"):
        tau_m = np.where(
            a_sum > 0,
            sum(a * t for a, t in zip(a_arrs, tau_arrs)) / a_sum,
            np.nan,
        )
    return tau_m


# ---------------------------------------------------------------------------
# Group statistics
# ---------------------------------------------------------------------------

def rank_biserial_r(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation: r = 2U/(n1*n2) - 1.

    Positive r => x tends to be LARGER than y (matches the sign convention used
    in Phase F).  Returns NaN if either group is empty.
    """
    from scipy.stats import mannwhitneyu
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    U = mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2.0 * U / (n1 * n2) - 1.0)


def benjamini_hochberg(pvals) -> np.ndarray:
    """Return BH-corrected q-values for a sequence of p-values.

    Matches the Phase F implementation: monotonicity enforced from largest to
    smallest p (the standard Benjamini-Hochberg step-up procedure).  To get
    "null rejected at alpha", check `q <= alpha` element-wise on the result.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    rank  = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    q = np.minimum(1.0, p * n / rank)
    # Enforce monotonicity from largest to smallest p
    for i in range(n - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i + 1]])
    return q


def effect_stats(
    sub,
    metric: str,
    group_col: str = "cell_type",
    group_a: str = "KPCWT",
    group_b: str = "BKO",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict | None:
    """Median difference + 95% bootstrap CI + Mann-Whitney U for two groups.

    Args:
        sub:        DataFrame with `group_col` and `metric` columns.
        metric:     numeric column to compare.
        group_col:  column distinguishing the two groups.
        group_a/b:  the two group labels.  diff = median(a) - median(b).
        n_boot:     bootstrap samples.
        seed:       RNG seed.

    Returns None if either group has < 2 finite observations.
    """
    from scipy.stats import mannwhitneyu
    rng = np.random.default_rng(seed)
    a = sub.loc[sub[group_col] == group_a, metric].dropna().to_numpy()
    b = sub.loc[sub[group_col] == group_b, metric].dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return None
    obs = float(np.median(a) - np.median(b))
    boot = np.array([
        np.median(rng.choice(a, len(a), replace=True)) -
        np.median(rng.choice(b, len(b), replace=True))
        for _ in range(n_boot)
    ])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    r    = 2 * u / (len(a) * len(b)) - 1
    return dict(diff=obs,
                ci_lo=float(ci_lo), ci_hi=float(ci_hi),
                p=float(p), r=float(r),
                n_a=int(len(a)), n_b=int(len(b)))


def perm_test(
    sub,
    metric: str,
    group_col: str = "cell_type",
    group_a: str = "KPCWT",
    group_b: str = "BKO",
    n_perm: int = 2000,
    seed: int = 0,
):
    """Two-sample permutation test on the median difference.

    Returns (obs_diff, null_distribution, p_empirical) or None if either group
    has < 2 finite observations.
    """
    rng = np.random.default_rng(seed)
    a = sub.loc[sub[group_col] == group_a, metric].dropna().to_numpy()
    b = sub.loc[sub[group_col] == group_b, metric].dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return None
    obs    = float(np.median(a) - np.median(b))
    pooled = np.concatenate([a, b])
    n_a    = len(a)
    null   = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(pooled)
        null[i] = np.median(pooled[:n_a]) - np.median(pooled[n_a:])
    p_emp = float((np.abs(null) >= abs(obs)).mean())
    return obs, null, p_emp
