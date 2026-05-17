# %% [markdown]
# # Phase F: Group comparisons
#
# Reads per-file summary CSVs from Phases D and E; no raw data reloading.
#
# Steps:
# 19. Load and merge Phase D/E summaries; report coverage
# 20. Photobleaching check: metrics vs acquisition_time within each session
# 21. Violin plots: tau_mean and amplitude ratio grouped by cell_type
# 22. Pairwise statistical tests: Mann-Whitney U with Benjamini-Hochberg FDR
# 23. Phasor vs fit cross-validation: tau_phi vs tau_mean scatter
# 24. Group summary table -> results/group_summary.csv

# %%
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# -- Configuration ----------------------------------------------------------
# Cell types and fixation types to include, in display order
CELL_TYPE_ORDER = ["KPCWT", "BKO", "DKO"]
FIXATION_ORDER  = ["glu", "form", "live"]

# Minimum images per group to include in statistical tests
MIN_N = 3

# FDR threshold for significance marking
FDR_ALPHA = 0.05

# Metrics from Phase E (fit) and Phase D (phasor) to analyse
FIT_METRICS    = ["tau_mean_median_ps", "amp_ratio_median"]
PHASOR_METRICS = ["tau_phi_ns", "tau_mod_ns", "G_cal_wmean", "S_cal_wmean"]

METRIC_LABELS = {
    "tau_mean_median_ps": "tau_mean median (ps)",
    "amp_ratio_median":   "amplitude ratio (median)",
    "tau_phi_ns":         "tau_phi (ns)",
    "tau_mod_ns":         "tau_mod (ns)",
    "G_cal_wmean":        "G_cal (photon-weighted mean)",
    "S_cal_wmean":        "S_cal (photon-weighted mean)",
}

# %% [markdown]
# ## Step 19: Load and merge summaries

# %%
results_dir = Path("../results")

sdt_df = pd.read_csv(results_dir / "sdt_metadata_cal.csv")
sdt_df["acquisition_time"] = pd.to_datetime(sdt_df["acquisition_time"])

fit_df    = pd.read_csv(results_dir / "fit_analysis_summary.csv")
phasor_df = pd.read_csv(results_dir / "sdt_phasor_summary.csv")

# Join phasor columns onto fit summary (outer join to keep files with only one)
phasor_cols = ["filename", "tau_phi_ns", "tau_mod_ns",
               "G_cal_wmean", "S_cal_wmean", "M_wmean",
               "n_px_masked"]
df = fit_df.merge(
    phasor_df[phasor_cols].rename(columns={"n_px_masked": "n_px_phasor"}),
    on="filename", how="outer",
)

# Bring in acquisition_time and session_root from sdt_df if absent
meta_cols = [c for c in ["acquisition_time", "session_root"]
             if c not in df.columns]
if meta_cols:
    df = df.merge(sdt_df[["filename"] + meta_cols], on="filename", how="left")
df["acquisition_time"] = pd.to_datetime(df["acquisition_time"])

# Merge position annotations
annot_path = results_dir / "position_annotation.csv"
if annot_path.exists():
    annot_df = pd.read_csv(annot_path)
    df = df.merge(
        annot_df[["filename", "annotation"]].rename(
            columns={"annotation": "position_annotation"}
        ),
        on="filename", how="left",
    )
    df["position_annotation"] = df["position_annotation"].fillna("(unannotated)")
else:
    df["position_annotation"] = "(unannotated)"

print(f"Combined dataframe: {len(df)} rows")
print("\nCoverage by fixation_type x cell_type:")
print(df.groupby(["fixation_type", "cell_type"], dropna=False).size().to_string())

print("\nMetric availability:")
all_metrics = [m for m in FIT_METRICS + PHASOR_METRICS if m in df.columns]
for m in all_metrics:
    print(f"  {m:30s}: {df[m].notna().sum()}/{len(df)}")

# Convenience subsets
fix_present  = [f for f in FIXATION_ORDER  if f in df["fixation_type"].dropna().unique()]
ct_present   = [c for c in CELL_TYPE_ORDER if c in df["cell_type"].dropna().unique()]

# %% [markdown]
# ## Step 20: Photobleaching check
#
# Plot each metric vs acquisition_time, grouped by session.
# A monotonic decrease within a session suggests photobleaching.
# Files from the same session are connected by a line.

# %%
CHECK_METRICS = [m for m in ["tau_mean_median_ps", "tau_phi_ns", "G_cal_wmean"]
                 if m in df.columns and df[m].notna().any()]

if not CHECK_METRICS:
    print("No metrics available for photobleaching check -- run Phases D and E first.")
else:
    for fix_type in fix_present:
        sub = df[df["fixation_type"] == fix_type].copy()
        if sub.empty:
            continue

        sessions  = sorted(sub["session_root"].dropna().unique())
        cmap      = plt.cm.tab10(np.linspace(0, 0.9, max(len(sessions), 1)))
        clr       = {s: cmap[i] for i, s in enumerate(sessions)}

        n_panels = len(CHECK_METRICS)
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4),
                                 squeeze=False)

        for ax, metric in zip(axes[0], CHECK_METRICS):
            for sess in sessions:
                pts = (sub[sub["session_root"] == sess]
                       .dropna(subset=["acquisition_time", metric])
                       .sort_values("acquisition_time"))
                if pts.empty:
                    continue
                ax.plot(pts["acquisition_time"], pts[metric], "o-",
                        color=clr[sess], ms=4, lw=1, alpha=0.75, label=sess)
            ax.set_xlabel("acquisition time")
            ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            ax.set_title(metric)
            ax.tick_params(axis="x", labelrotation=30)

        handles = [mpatches.Patch(color=clr[s], label=s) for s in sessions]
        if handles:
            axes[0][-1].legend(handles=handles, fontsize=7,
                               bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.suptitle(f"Photobleaching check -- {fix_type}", fontsize=10)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## Step 21: Violin plots by cell_type x fixation_type
#
# Each violin shows the distribution of per-image medians.
# One figure per metric; one panel per fixation_type.
# Individual data points overlaid as a strip plot.

# %%
palette  = plt.cm.Set2(np.linspace(0, 0.8, max(len(ct_present), 1)))
ct_color = {ct: palette[i] for i, ct in enumerate(ct_present)}

for metric in all_metrics:
    label = METRIC_LABELS.get(metric, metric)
    fig, axes = plt.subplots(1, len(fix_present),
                             figsize=(4 * len(fix_present), 5),
                             squeeze=False)

    for ax, fix_type in zip(axes[0], fix_present):
        sub = (df[(df["fixation_type"] == fix_type) &
                  df["cell_type"].isin(ct_present)]
               .dropna(subset=[metric]))

        cts_here  = [ct for ct in ct_present if (sub["cell_type"] == ct).any()]
        groups    = [sub[sub["cell_type"] == ct][metric].values for ct in cts_here]
        colors    = [ct_color[ct] for ct in cts_here]

        if not groups:
            ax.set_title(f"{fix_type}\n(no data)")
            continue

        parts = ax.violinplot(groups, positions=range(len(groups)),
                              showmedians=True, showextrema=True)
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.65)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color("k")
                parts[key].set_linewidth(1.2)

        rng = np.random.default_rng(seed=0)
        for j, (grp, col) in enumerate(zip(groups, colors)):
            jitter = rng.uniform(-0.07, 0.07, len(grp))
            ax.scatter(j + jitter, grp, s=14, color=col, alpha=0.6, zorder=3)

        ax.set_xticks(range(len(cts_here)))
        ax.set_xticklabels(cts_here, fontsize=9)
        ax.set_ylabel(label)
        n_str = "  ".join(f"{ct}:{(sub['cell_type']==ct).sum()}"
                          for ct in cts_here)
        ax.set_title(f"{fix_type}\nn={n_str}", fontsize=8)

    fig.suptitle(label, fontsize=11)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Step 22: Pairwise statistical tests
#
# For each (fixation_type, metric): Mann-Whitney U on all pairs of cell_types
# with >= MIN_N images. Benjamini-Hochberg FDR applied across all pairs
# within each (fixation_type, metric) block.
#
# Effect size: rank-biserial r = 1 - 2U/(n1*n2).
# Interpretation: |r| < 0.3 small, 0.3-0.5 medium, > 0.5 large.
# Sign: positive r means group_a tends to be larger than group_b.

# %%
def benjamini_hochberg(pvals: list) -> np.ndarray:
    """Return BH-corrected q-values for a list of p-values."""
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


def rank_biserial_r(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation as effect size for Mann-Whitney U."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    U = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(1.0 - 2.0 * U / (n1 * n2))


stat_rows = []

for fix_type in fix_present:
    sub  = df[df["fixation_type"] == fix_type]
    cts  = [ct for ct in ct_present
            if len(sub[sub["cell_type"] == ct].dropna(subset=["tau_mean_median_ps"]
                                                       if "tau_mean_median_ps" in df.columns
                                                       else [])) >= MIN_N
            or len(sub[sub["cell_type"] == ct]) >= MIN_N]
    if len(cts) < 2:
        continue
    pairs = list(itertools.combinations(cts, 2))

    for metric in all_metrics:
        p_raw     = []
        pair_data = []
        for ct_a, ct_b in pairs:
            x = sub[sub["cell_type"] == ct_a][metric].dropna().values
            y = sub[sub["cell_type"] == ct_b][metric].dropna().values
            if len(x) < MIN_N or len(y) < MIN_N:
                continue
            U_res = stats.mannwhitneyu(x, y, alternative="two-sided")
            p_raw.append(U_res.pvalue)
            pair_data.append({
                "fixation_type": fix_type,
                "metric":        metric,
                "group_a":       ct_a,
                "group_b":       ct_b,
                "n_a":           len(x),
                "n_b":           len(y),
                "median_a":      float(np.median(x)),
                "median_b":      float(np.median(y)),
                "U":             float(U_res.statistic),
                "p_raw":         float(U_res.pvalue),
                "r":             rank_biserial_r(x, y),
            })

        if not pair_data:
            continue
        q_vals = benjamini_hochberg(p_raw)
        for row, q in zip(pair_data, q_vals):
            row["q_bh"] = float(q)
            row["sig"]  = bool(q < FDR_ALPHA)
            stat_rows.append(row)

stat_df = pd.DataFrame(stat_rows)
print(f"Statistical tests: {len(stat_df)} comparisons")

if not stat_df.empty:
    sig = stat_df[stat_df["sig"]]
    print(f"Significant after BH-FDR (q < {FDR_ALPHA}): {len(sig)}")
    if not sig.empty:
        print(sig[["fixation_type", "metric", "group_a", "group_b",
                   "median_a", "median_b", "r", "q_bh"]]
              .round(4).to_string(index=False))

stat_df.to_csv(results_dir / "stat_tests.csv", index=False)
print(f"\nSaved: {results_dir / 'stat_tests.csv'}")

# %%
# -- Effect size heatmap (r) with significance markers --------------------
if not stat_df.empty:
    n_met = len(all_metrics)
    n_fix = len(fix_present)
    fig, axes = plt.subplots(n_met, n_fix,
                             figsize=(3.2 * n_fix, 3.0 * n_met),
                             squeeze=False)

    for r_idx, metric in enumerate(all_metrics):
        for c_idx, fix_type in enumerate(fix_present):
            ax  = axes[r_idx][c_idx]
            sub_stat = stat_df[
                (stat_df["metric"] == metric) &
                (stat_df["fixation_type"] == fix_type)
            ]
            cts = [ct for ct in ct_present
                   if len(df[(df["fixation_type"] == fix_type) &
                             (df["cell_type"] == ct)]) >= MIN_N]
            n   = len(cts)
            if n < 2:
                ax.set_visible(False)
                continue

            ct_idx = {ct: i for i, ct in enumerate(cts)}
            mat_r  = np.full((n, n), np.nan)
            mat_q  = np.full((n, n), np.nan)
            for _, row in sub_stat.iterrows():
                i = ct_idx.get(row["group_a"])
                j = ct_idx.get(row["group_b"])
                if i is None or j is None:
                    continue
                mat_r[i, j] =  row["r"]
                mat_r[j, i] = -row["r"]
                mat_q[i, j] = mat_q[j, i] = row["q_bh"]

            im = ax.imshow(mat_r, cmap="RdBu_r", vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="r")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(cts, fontsize=7, rotation=45, ha="right")
            ax.set_yticklabels(cts, fontsize=7)
            for i2 in range(n):
                for j2 in range(n):
                    if np.isfinite(mat_q[i2, j2]) and mat_q[i2, j2] < FDR_ALPHA:
                        ax.text(j2, i2, "*", ha="center", va="center",
                                fontsize=14, color="k", fontweight="bold")
            ax.set_title(f"{fix_type}\n{METRIC_LABELS.get(metric, metric)}",
                         fontsize=7)

    plt.suptitle("Effect size r (rank-biserial)   * = BH-FDR q < "
                 + str(FDR_ALPHA), fontsize=10)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Step 23: Phasor vs fit cross-validation
#
# Scatter plot of phasor-derived lifetime (tau_phi, tau_mod) against
# amplitude-weighted tau_mean from fitting. Both are converted to ns.
# High Spearman r -> both methods track the same signal.
# A systematic offset indicates calibration bias in one method.

# %%
xval_pairs = [
    ("tau_phi_ns", "tau_mean_median_ps"),
    ("tau_mod_ns", "tau_mean_median_ps"),
]

palette_fix = plt.cm.Set1(np.linspace(0, 0.8, max(len(fix_present), 1)))
fix_color   = {ft: palette_fix[i] for i, ft in enumerate(fix_present)}

for y_col, x_col in xval_pairs:
    if y_col not in df.columns or x_col not in df.columns:
        continue
    sub = df.dropna(subset=[x_col, y_col]).copy()
    if sub.empty:
        print(f"No data for {y_col} vs {x_col}")
        continue

    # Convert fit metric to ns if exported in ps
    x_vals  = sub[x_col] / 1000.0 if x_col.endswith("_ps") else sub[x_col]
    x_label = x_col.replace("_ps", "_ns")

    fig, ax = plt.subplots(figsize=(6, 5))
    for ft in fix_present:
        m = sub["fixation_type"] == ft
        if not m.any():
            continue
        ax.scatter(x_vals[m], sub.loc[m, y_col],
                   color=fix_color[ft], alpha=0.65, s=18, label=ft)

    # Unity line over the data range
    lo = float(min(x_vals.min(), sub[y_col].min()))
    hi = float(max(x_vals.max(), sub[y_col].max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4, label="y=x")

    r_sp, p_sp = stats.spearmanr(x_vals, sub[y_col])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_label}\n"
                 f"Spearman r={r_sp:.3f}  p={p_sp:.2g}  n={len(sub)}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Step 24: Group summary table
#
# Mean +/- SEM across images, stratified by fixation_type x cell_type.

# %%
summary_rows = []
for fix_type in fix_present:
    for ct in ct_present:
        sub = df[(df["fixation_type"] == fix_type) & (df["cell_type"] == ct)]
        if sub.empty:
            continue
        rec = {"fixation_type": fix_type, "cell_type": ct, "n_files": len(sub)}
        for metric in all_metrics:
            vals = sub[metric].dropna()
            rec[f"{metric}_n"]    = len(vals)
            rec[f"{metric}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            rec[f"{metric}_sem"]  = float(vals.sem())  if len(vals) > 1 else np.nan
        summary_rows.append(rec)

summary_df = pd.DataFrame(summary_rows)

# Print a readable view: mean (SEM) per metric
if not summary_df.empty:
    print("Group means (SEM in parentheses):\n")
    for metric in all_metrics:
        mn_col  = f"{metric}_mean"
        sem_col = f"{metric}_sem"
        n_col   = f"{metric}_n"
        if mn_col not in summary_df.columns:
            continue
        print(f"  {METRIC_LABELS.get(metric, metric)}")
        for _, row in summary_df.iterrows():
            mn  = row.get(mn_col)
            sem = row.get(sem_col)
            n   = int(row.get(n_col, 0))
            val = f"{mn:.3g} ({sem:.2g})" if pd.notna(mn) and pd.notna(sem) else "n/a"
            print(f"    {row['fixation_type']:6s}  {row['cell_type']:8s}  "
                  f"n={n:3d}   {val}")
        print()

summary_df.to_csv(results_dir / "group_summary.csv", index=False)
print(f"Saved: {results_dir / 'group_summary.csv'}")

# %% [markdown]
# ## Summary
#
# Phase F complete.
#
# Step 20 -- Photobleaching: if tau_mean or G_cal drift systematically within
#   a session, consider splitting the session or excluding late-acquired images.
#
# Step 21 -- Violins: each point is one field of view (one image).
#   Spread reflects true biological variability + measurement noise.
#
# Step 22 -- Statistics: Mann-Whitney U (non-parametric, no normality assumption).
#   BH-FDR applied per (fixation_type, metric) block.
#   Effect size r: |r| < 0.3 small, 0.3-0.5 medium, > 0.5 large.
#   Note: treating each image as an independent replicate is conservative if
#   multiple images came from the same dish. Mixed-effects models would be
#   needed to account for dish-level clustering.
#
# Step 23 -- Cross-validation: phasor tau_phi should correlate with fit tau_mean.
#   Decorrelation suggests a systematic artifact in one method (e.g., IRF
#   mismatch, poor chi^2 masking, or calibration error).
#
# Saved:
#   results/stat_tests.csv      -- all pairwise Mann-Whitney tests
#   results/group_summary.csv   -- group means +/- SEM
