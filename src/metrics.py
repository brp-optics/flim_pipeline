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
    """Compute amplitude-weighted tau_mean from fit arrays.

    tau_mean = sum(a_i * tau_i) / sum(a_i) over the component pairs
    specified by taumean_cfg for the given fixation_type.

    Default component pairs (DEFAULT_TAUMEAN_CFG):
      glu:  (a2, tau2) + (a3, tau3)  -- artifact a1/tau1 excluded
      form: (a1, tau1) + (a2, tau2)
      live: (a1, tau1) + (a2, tau2)

    Parameters
    ----------
    fd : dict
        {param_name: 2-D ndarray} as returned by load_asc_fit_set().
        Must contain all amplitude and tau arrays named in taumean_cfg.
    fixation_type : str
        'glu', 'form', or 'live'.  Selects the component pairs to use.
    taumean_cfg : dict, optional
        Override for DEFAULT_TAUMEAN_CFG.  Must map fixation_type to a
        sequence of ('a_name', 'tau_name') pairs.

    Returns
    -------
    np.ndarray or None
        2-D array in the same units as the tau exports (typically ps).
        NaN where the amplitude sum is zero.  Returns None if fixation_type
        is not in taumean_cfg or any required parameter is absent from fd.

    Assumptions
    -----------
    tau export units are ps (as output by SPCImage).  NaN pixels in a or
    tau arrays propagate to NaN in the result via numpy arithmetic.

    Examples
    --------
    >>> tau_m = compute_tau_mean(fd, 'form')
    >>> median_ps = np.nanmedian(tau_m[mask])

    Dependencies
    ------------
    numpy
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
    """Rank-biserial correlation (effect size for Mann-Whitney U): r = 2U/(n1*n2) - 1.

    Ranges from -1 (all y > x) to +1 (all x > y).  Positive r means x tends
    to be larger than y, matching the sign convention used in Phase F (KPCWT
    as group_a, BKO as group_b, so positive r => KPCWT > BKO).

    Parameters
    ----------
    x : np.ndarray
        1-D array of finite values for group A (e.g. KPCWT).
    y : np.ndarray
        1-D array of finite values for group B (e.g. BKO).

    Returns
    -------
    float
        Rank-biserial r in [-1, 1].  Returns float('nan') if either array
        is empty.

    Assumptions
    -----------
    x and y should not contain NaN (use dropna() before calling).

    Examples
    --------
    >>> a = df.loc[df['cell_type'] == 'KPCWT', 'tau_mean_median_ps'].dropna().values
    >>> b = df.loc[df['cell_type'] == 'BKO',   'tau_mean_median_ps'].dropna().values
    >>> r = rank_biserial_r(a, b)  # positive => KPCWT has longer lifetime

    Dependencies
    ------------
    numpy, scipy.stats.mannwhitneyu
    """
    from scipy.stats import mannwhitneyu
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    U = mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2.0 * U / (n1 * n2) - 1.0)


def benjamini_hochberg(pvals) -> np.ndarray:
    """Return Benjamini-Hochberg FDR-corrected q-values for a sequence of p-values.

    Implements the standard step-up procedure: rank p-values, compute
    q = p * n / rank, then enforce monotonicity from largest to smallest p
    so that q[i] <= q[j] whenever p[i] <= p[j].  To identify discoveries,
    check q <= alpha element-wise on the returned array.

    Parameters
    ----------
    pvals : sequence of float
        Raw p-values from multiple tests (order does not matter; the
        returned array is in the same order as the input).

    Returns
    -------
    np.ndarray
        BH q-values, same length as pvals.  All values in [0, 1].

    Assumptions
    -----------
    pvals should be in [0, 1].  An empty input returns an empty array.

    Examples
    --------
    >>> q = benjamini_hochberg(p_list)
    >>> significant = q <= 0.05

    Dependencies
    ------------
    numpy
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

    Computes the observed median difference (group_a minus group_b) and its
    bootstrap 95% CI, then runs a two-sided Mann-Whitney U test and computes
    rank-biserial r as the effect size.

    Parameters
    ----------
    sub : pd.DataFrame
        DataFrame slice containing both groups.  Rows with NaN in `metric`
        are dropped before analysis.
    metric : str
        Numeric column name to compare (e.g. 'tau_mean_median_ps').
    group_col : str, optional
        Column that identifies each group.  Default 'cell_type'.
    group_a : str, optional
        Label of group A (positive diff means A > B).  Default 'KPCWT'.
    group_b : str, optional
        Label of group B.  Default 'BKO'.
    n_boot : int, optional
        Number of bootstrap replicates for the CI.  Default 2000.
    seed : int, optional
        RNG seed for reproducibility.  Default 0.

    Returns
    -------
    dict or None
        Keys: 'diff' (float, median_a - median_b), 'ci_lo', 'ci_hi'
        (95% bootstrap CI bounds), 'p' (Mann-Whitney two-sided p-value),
        'r' (rank-biserial effect size in [-1,1]), 'n_a', 'n_b' (sample
        sizes).  Returns None if either group has < 2 finite observations.

    Examples
    --------
    >>> stats = effect_stats(df, 'tau_mean_median_ps')
    >>> print(f"diff={stats['diff']:.1f} ps, p={stats['p']:.4f}, r={stats['r']:.2f}")

    Dependencies
    ------------
    numpy, scipy.stats.mannwhitneyu
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

    Generates a null distribution by repeatedly shuffling the pooled values
    of both groups and computing the median difference.  The empirical p-value
    is the fraction of permutations where the absolute null difference is at
    least as large as the observed absolute difference.

    Parameters
    ----------
    sub : pd.DataFrame
        DataFrame containing both groups.  NaN values in `metric` are dropped.
    metric : str
        Numeric column to compare.
    group_col : str, optional
        Group-identifying column.  Default 'cell_type'.
    group_a : str, optional
        Label for group A.  Default 'KPCWT'.
    group_b : str, optional
        Label for group B.  Default 'BKO'.
    n_perm : int, optional
        Number of permutations.  Default 2000.
    seed : int, optional
        RNG seed.  Default 0.

    Returns
    -------
    tuple or None
        (obs_diff, null, p_empirical) where obs_diff is the observed median
        difference, null is a float array of shape (n_perm,) (the null
        distribution), and p_empirical is the two-sided empirical p-value.
        Returns None if either group has < 2 finite observations.

    Examples
    --------
    >>> obs, null, p = perm_test(df, 'amp_ratio_median', n_perm=5000)
    >>> plt.hist(null, bins=40)
    >>> plt.axvline(obs, color='red')

    Dependencies
    ------------
    numpy
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
