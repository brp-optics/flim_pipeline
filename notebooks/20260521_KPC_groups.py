# %% [markdown]
# # Phase F: Group comparisons
#
# Reads per-file summary CSVs from Phases D and E; no raw data reloading.
#
# -----------------------------------------------------------------------
# Unit of analysis
# -----------------------------------------------------------------------
# The unit is one .sdt image file.  Each row in fit_analysis_summary.csv
# represents one acquired field of view.  The fit metric values reported
# (tau_mean_median_ps, amp_ratio_median) are the MEDIAN of quality-masked
# pixels within that image -- not individual pixels.
#
# Consequence: statistical tests compare image-level medians, which is
# appropriate for detecting differences between cell populations but
# underestimates within-image heterogeneity.
#
# -----------------------------------------------------------------------
# Like-for-like filtering (CRITICAL)
# -----------------------------------------------------------------------
# Mixing files fitted with different SPCImage models (2-comp vs 3-comp)
# or with different IRF-shift modes (shift=0 vs shift!=0) produces
# bimodal distributions that can wash out or reverse apparent effects.
#
# ANALYSIS_N_COMP and ANALYSIS_SHIFT_GROUP (config below) restrict the
# analysis to a homogeneous subset:
#   ANALYSIS_N_COMP     -- preferred number of fit components per
#                          fixation_type.  Files with a different n_comp
#                          are excluded from df_analysis.
#   ANALYSIS_SHIFT_GROUP -- "shift=0" | "shift!=0" | None.
#                          Derived from fit_qc_summary.csv (Phase C).
#
# The unfiltered df is retained for diagnostic steps (Steps 19b, 20, 23).
# ALL group comparisons (Steps 21, 22, 24-27) use df_analysis.
#
# -----------------------------------------------------------------------
# Amplitude ratio definition (WARNING: raw ratio, not fraction)
# -----------------------------------------------------------------------
# amp_ratio is the RAW ratio of two SPCImage amplitude components:
#   glu  : a2 / a3  (free NADH amplitude / bound NADH amplitude)
#   form : a1 / a2  (free / bound)
#   live : a1 / a2
# Values are NOT bounded by [0, 1] -- for glu, typical a2/a3 ~ 1-5.
# To get the normalised free-NADH fraction use amp_num / (amp_num + amp_den)
# computed from the per-pixel maps in Phase E.
# Violin plots and summary tables display the raw ratio.
#
# -----------------------------------------------------------------------
# tau_mean definition
# -----------------------------------------------------------------------
# tau_mean is the amplitude-weighted mean lifetime (ps) from Phase E.
#   glu  : (a2*tau2 + a3*tau3) / (a2+a3)  -- artifact a1/tau1 excluded
#   form : (a1*tau1 + a2*tau2) / (a1+a2)
#   live : same as form
# For glu files fitted with a 2-comp fallback, tau_mean is NaN because
# the a3/tau3 components are absent.  Such files contribute NaN to
# tau_mean_median_ps and are silently absent from tau_mean comparisons.
#
# -----------------------------------------------------------------------
# Statistical test
# -----------------------------------------------------------------------
# Step 22 uses a two-sided Mann-Whitney U test on per-image medians.
# All pairwise cell_type combinations per (fixation_type, metric) block
# are tested together; p-values are corrected with Benjamini-Hochberg FDR.
# Effect size: rank-biserial r = 1 - 2U/(n1*n2).
#   |r| < 0.3 small, 0.3-0.5 medium, > 0.5 large.
#   Positive r: group_a tends to be larger than group_b.
# Only cell_type groups with >= MIN_N images are included.
#
# -----------------------------------------------------------------------
# IRF drift and calibration sensitivity
# -----------------------------------------------------------------------
# The IRF (Instrument Response Function) encodes the timing jitter of the
# TCSPC detector.  If the IRF peak shifts between sessions or during a
# session, fitted lifetimes are systematically wrong by an amount that
# scales with the shift.
#
# Phase D (phasor): uses a per-session IRF calibration file measured each
#   session.  Between-session IRF drift is therefore corrected.  Within-
#   session drift is not corrected and broadens the phasor cloud.
#
# Phase E (reconvolution fitting from SPCImage): this pipeline imports
#   already-fitted parameters -- it does NOT re-fit or re-calibrate.
#   Sensitivity to IRF drift depends entirely on which IRF file was used
#   inside SPCImage:
#     - If SPCImage used a SINGLE IRF file for all sessions, any session
#       where the IRF had drifted will have systematically shifted lifetimes.
#     - If SPCImage used PER-SESSION IRF files, between-session drift is
#       corrected at the fitting stage.
#   This pipeline cannot determine which was done.  Confirm with the
#   SPCImage project files before interpreting inter-session differences.
#
# The SPCImage "shift" parameter (shift=0 vs shift!=0) is an in-fitting
#   correction for IRF timing offset.  shift=0 files absorbed any timing
#   offset into the lifetime values.  shift!=0 files corrected for it but
#   introduce a correlated free parameter that can cause bimodal tau
#   distributions (two local minima for shift vs tau).
#
# -----------------------------------------------------------------------
# Inter-session confounding (CRITICAL WARNING)
# -----------------------------------------------------------------------
# Step 19b checks for bimodality by imaging session.  If tau_mean_median_ps
# separates strongly by session_root, the comparison may be confounded:
#
#   Fixed samples (glu, form): fixation time, fixation protocol, and
#     fixative concentration can vary between sessions.  Different fixation
#     times alter the fraction of bound vs free NADH independently of
#     genotype.  If different genotypes were predominantly imaged in
#     different sessions, session is a confound for genotype.
#
#   Live samples: if only one cell type was imaged per session, session
#     and cell_type are completely confounded and no statistical comparison
#     is possible without a session-corrected model (e.g., mixed effects).
#
# BEFORE interpreting any group differences:
#   1. Check which sessions each cell type appears in.
#   2. If there is no within-session replication across genotypes, the
#      group comparison is not interpretable.
#   3. If session bimodality is present, consult the experimental log to
#      determine whether fixation conditions differed.
#
# -----------------------------------------------------------------------
# Known limitations and cautions
# -----------------------------------------------------------------------
# 1. Pseudo-replication: pixels within one image are spatially correlated
#    (SPCImage bins over (2b+1)^2 neighborhoods).  Using per-image medians
#    avoids this.  But multiple images from the same dish are not fully
#    independent -- no dish-level random effect is modeled.
#
# 2. Asymmetric tau_mean filter in Phase E: live images apply an extra
#    tau_mean >= 250 ps criterion that glu and form do not.  This may
#    affect the live distribution shape relative to fixed conditions.
#
# 3. Photobleaching: Step 20 checks for metric drift within a session.
#    If drift is detected, acquisition order should be treated as a
#    covariate or early/late acquisitions excluded before final analysis.
#
# 4. n_components column in fit_analysis_summary.csv: parsed from the fit
#    folder name string (e.g. "fitet-sf-3component-b5" -> 3).  If the
#    folder uses an older naming scheme ("c2", "c3") the regex may fail
#    and n_components will be None.  Check the distribution printout in
#    Step 19 before interpreting ANALYSIS_N_COMP filtering.
#
# 5. ANALYSIS_SHIFT_GROUP is now a per-fixation-type dict.  If all files
#    of a given fixation_type were fitted with a free shift in SPCImage
#    (shift!=0), a "shift=0" filter silently excludes all of them.  A
#    WARNING is printed in Step 19 when this happens.
#
# -----------------------------------------------------------------------
# Steps
# -----------------------------------------------------------------------
# 19.  Load and merge Phase D/E summaries; report coverage
# 19b. Bimodality diagnosis (unfiltered): session, fit model, calibration
# 20.  Photobleaching check: metrics vs acquisition_time within session
# 21.  Violin plots: tau_mean and amplitude ratio grouped by cell_type
# 22.  Pairwise statistical tests: Mann-Whitney U with BH-FDR
# 23.  Phasor vs fit cross-validation: tau_phi vs tau_mean scatter
# 24.  Group summary table -> results/group_summary.csv
# 25.  Per-channel comparison (em_filter_nm facets)
# 26-27. Additional diagnostic plots

# %%
from pathlib import Path
import itertools
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# -- Configuration ----------------------------------------------------------
# Cell types and fixation types to include, in display order
CELL_TYPE_ORDER = ["KPCWT", "BKO", "DKO"]
FIXATION_ORDER  = ["glu", "form", "live"]

# Figure output
FIGURES_DIR  = Path("../results/figures")
SAVE_FIGURES = True   # set False to skip saving


def _savefig(fig, name: str) -> None:
    if SAVE_FIGURES:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")

# Minimum images per group to include in statistical tests
MIN_N = 3

# FDR threshold for significance marking
FDR_ALPHA = 0.05

# Metrics from Phase E (fit) and Phase D (phasor) to analyse
FIT_METRICS    = ["tau_mean_median_ps", "amp_ratio_median"]
PHASOR_METRICS = ["tau_phi_ns", "tau_mod_ns", "G_cal_wmean", "S_cal_wmean"]

METRIC_LABELS = {
    "tau_mean_median_ps": "tau_mean median (ps)",
    # glu: a2/(a2+a3) free/bound NADH ratio (artifact component excluded)
    # form/live: a1/(a1+a2) free/bound NADH ratio
    "amp_ratio_median":   "amp ratio (median)",
    "tau_phi_ns":         "tau_phi (ns)",
    "tau_mod_ns":         "tau_mod (ns)",
    "G_cal_wmean":        "G_cal (photon-weighted mean)",
    "S_cal_wmean":        "S_cal (photon-weighted mean)",
}

# Restrict group comparisons to a single fit model so every violin contains
# only like-for-like measurements.  Mixing n_comp or shift groups produces
# bimodal distributions that wash out any biological signal.
#
# ANALYSIS_N_COMP: preferred n_components per fixation_type.
#   Files where best_fit_key() resolved to a different n_comp are excluded.
#   Set a value to None to include all n_comp variants for that fixation_type.
ANALYSIS_N_COMP = {"glu": 3, "form": 2, "live": 2}

# ANALYSIS_SHIFT_GROUP: per-fixation-type dict.
#   Value: "shift=0" | "shift!=0" | None (no filter for that type).
#   Derived from fit_qc_summary.csv (Phase C shift_nonzero_px column).
#   If fit_qc_summary.csv is absent, all files are treated as "shift=0".
#
#   IMPORTANT: if all files of a given fixation_type were fitted with a
#   free IRF shift in SPCImage (shift!=0), setting that type to "shift=0"
#   will produce an empty analysis group.  Check the shift breakdown printed
#   in Step 19 before changing these values.
#
#   Typical reasons all live files end up as shift!=0:
#     - SPCImage was configured with shift=free for the live acquisition set
#     - The live IRF timing offset was consistently non-zero across sessions
#   In that case set live to "shift!=0" to retain those files, but note that
#   the shift parameter correlates with the fitted lifetimes (see header).
ANALYSIS_SHIFT_GROUP = {
    "glu":  "shift=0",
    "form": "shift=0",
    "live": "shift=0",  # best_fit_key now prefers sz folders; check WARNING in Step 19
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

# Session x cell_type cross-tab: printed here as an early confounding check.
# If each cell type appears ONLY in its own session(s) the comparison is
# confounded -- any group difference could be a session/batch effect.
print("\nSession x cell_type (files per cell): check for session/genotype confounding")
if "session_root" in df.columns and "cell_type" in df.columns:
    _xtab = (df.groupby(["fixation_type", "session_root", "cell_type"], dropna=False)
               .size()
               .unstack("cell_type", fill_value=0))
    print(_xtab.to_string())
    _ct_cols = [c for c in CELL_TYPE_ORDER if c in _xtab.columns]
    for _ft in FIXATION_ORDER:
        _sub_xt = _xtab[_xtab.index.get_level_values("fixation_type") == _ft][_ct_cols]
        if _sub_xt.empty:
            continue
        # A session is "confounded" if only one cell type has > 0 files in that session
        _confounded = [idx for idx in _sub_xt.index
                       if (_sub_xt.loc[idx] > 0).sum() < 2]
        if _confounded:
            print(f"  CAUTION [{_ft}]: {len(_confounded)} session(s) have only "
                  f"one cell type -- session and genotype are partially confounded.")

print("\nMetric availability:")
all_metrics = [m for m in FIT_METRICS + PHASOR_METRICS if m in df.columns]
for m in all_metrics:
    print(f"  {m:30s}: {df[m].notna().sum()}/{len(df)}")

# %%
# -- Parse n_components and shift_group from fit metadata -------------------
#
# n_components: Phase E (Step 18) now writes an explicit n_components column
#   to fit_analysis_summary.csv.  Use it directly when present; fall back to
#   regex parsing of fit_set_key for older CSV files that lack the column.
#   Regex rules:
#     "...fitet-sf-3component-b5..." -> 3
#     "...fitet-sf-c2-b3..."        -> 2
#
# shift_group: merged from fit_qc_summary.csv (Phase C output).
#   shift_nonzero_px > 0 means SPCImage used a free IRF shift.
#   If fit_qc_summary.csv is absent, every file is labelled "shift=0".

def _parse_n_comp_from_key(key):
    if not isinstance(key, str):
        return np.nan
    m = re.search(r"[-_](\d+)component", key, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"[-_]c(\d+)(?=[-_b]|$)", key, re.IGNORECASE)
    return float(m.group(1)) if m else np.nan

# Prefer the explicit column; fall back to key-string parsing
if "n_components" not in df.columns or df["n_components"].isna().all():
    df["n_components"] = df["fit_set_key"].apply(_parse_n_comp_from_key)
    print("n_components: parsed from fit_set_key string (no explicit column found)")
else:
    # Fill any gaps with key-string parsing
    _missing = df["n_components"].isna()
    if _missing.any():
        df.loc[_missing, "n_components"] = (
            df.loc[_missing, "fit_set_key"].apply(_parse_n_comp_from_key)
        )
    print("n_components: using explicit column from fit_analysis_summary.csv")

qc_path = results_dir / "fit_qc_summary.csv"
if qc_path.exists():
    _qc = pd.read_csv(qc_path)[["filename", "shift_nonzero_px"]].drop_duplicates("filename")
    df  = df.merge(_qc, on="filename", how="left")
    df["shift_group"] = df["shift_nonzero_px"].apply(
        lambda x: "shift!=0" if (pd.notna(x) and float(x) > 0) else "shift=0"
    )
    print(f"\nShift groups merged from fit_qc_summary.csv:")
    print(df.groupby(["fixation_type", "shift_group"], dropna=False).size().to_string())
else:
    df["shift_group"] = "shift=0"
    print("\nfit_qc_summary.csv not found; all files treated as shift=0")

print("\nn_components distribution:")
print(df.groupby(["fixation_type", "n_components"], dropna=False).size().to_string())

# %%
# -- Filter to like-for-like analysis subset --------------------------------
#
# Drop files that don't match ANALYSIS_N_COMP or ANALYSIS_SHIFT_GROUP so that
# violins and statistical tests compare only homogeneous fit populations.
# Stored as df_analysis; the raw df is kept for diagnostics (Steps 19b, 20, 23).

df_analysis = df.copy()

if ANALYSIS_N_COMP:
    for fix_type, preferred_nc in ANALYSIS_N_COMP.items():
        if preferred_nc is None:
            continue
        mask_ok = (
            (df_analysis["fixation_type"] != fix_type) |
            df_analysis["n_components"].isna() |
            (df_analysis["n_components"] == preferred_nc)
        )
        n_drop = (~mask_ok).sum()
        if n_drop > 0:
            print(f"  n_comp filter: dropped {n_drop} {fix_type} rows"
                  f" not matching {preferred_nc}-comp")
        df_analysis = df_analysis[mask_ok]

if isinstance(ANALYSIS_SHIFT_GROUP, dict):
    # Per-fixation-type shift filter
    for fix_type, sg_val in ANALYSIS_SHIFT_GROUP.items():
        if sg_val is None:
            continue
        mask_shift = (
            (df_analysis["fixation_type"] != fix_type) |
            df_analysis["shift_group"].isna() |
            (df_analysis["shift_group"] == sg_val)
        )
        n_drop = (~mask_shift).sum()
        if n_drop > 0:
            print(f"  shift filter [{fix_type}]: dropped {n_drop} rows"
                  f" not in '{sg_val}'")
        df_analysis = df_analysis[mask_shift]
elif ANALYSIS_SHIFT_GROUP is not None:
    # Global shift filter (legacy scalar)
    mask_shift = (
        df_analysis["shift_group"].isna() |
        (df_analysis["shift_group"] == ANALYSIS_SHIFT_GROUP)
    )
    n_drop = (~mask_shift).sum()
    if n_drop > 0:
        print(f"  shift filter: dropped {n_drop} rows not in '{ANALYSIS_SHIFT_GROUP}'")
    df_analysis = df_analysis[mask_shift]

print(f"\nAnalysis dataset: {len(df_analysis)} rows  "
      f"(from {len(df)} total; "
      f"n_comp={ANALYSIS_N_COMP}, shift={ANALYSIS_SHIFT_GROUP})")
print(df_analysis.groupby(["fixation_type", "cell_type"], dropna=False).size().to_string())

# Warn loudly if any fixation_type is completely absent after filtering
for _ft in FIXATION_ORDER:
    _n_ft = (df_analysis["fixation_type"] == _ft).sum()
    _n_before = (df["fixation_type"] == _ft).sum()
    if _n_before > 0 and _n_ft == 0:
        print(f"\n  WARNING: {_ft} has 0 rows in df_analysis (was {_n_before} before filtering).")
        print(f"    Check ANALYSIS_N_COMP['{_ft}'] and ANALYSIS_SHIFT_GROUP['{_ft}'].")
        _ft_sg = df[df["fixation_type"] == _ft]["shift_group"].value_counts()
        _ft_nc = df[df["fixation_type"] == _ft]["n_components"].value_counts()
        print(f"    shift_group distribution:\n{_ft_sg.to_string()}")
        print(f"    n_components distribution:\n{_ft_nc.to_string()}")

# Convenience subsets derived from the filtered dataset
fix_present = [f for f in FIXATION_ORDER  if f in df_analysis["fixation_type"].dropna().unique()]
ct_present  = [c for c in CELL_TYPE_ORDER if c in df_analysis["cell_type"].dropna().unique()]

# %%
# -- Step 19c: Numeric group summary ----------------------------------------
#
# Median, mean, SD, and n for every metric broken down by
# (fixation_type, em_filter_nm, cell_type).  Use this table to cross-check
# violin plots and compare against prior runs.  Printed before any figures
# so it is visible even if a later cell crashes.

print("=" * 72)
print("NUMERIC GROUP SUMMARY  (df_analysis -- like-for-like filtered)")
print("=" * 72)

_grp_cols_19c = [c for c in ["fixation_type", "em_filter_nm", "cell_type"]
                 if c in df_analysis.columns]

for _m in all_metrics:
    _lbl = METRIC_LABELS.get(_m, _m)
    _sub19 = df_analysis[_grp_cols_19c + [_m]].dropna(subset=[_m])
    if _sub19.empty:
        continue
    _agg19 = (_sub19.groupby(_grp_cols_19c, dropna=False)[_m]
              .agg(n="count", median="median", mean="mean", std="std")
              .round(1))
    print(f"\n{_lbl}  [{_m}]")
    print(_agg19.to_string())

    # Direction: which cell_type has the higher median per (fix, channel)?
    print("  Direction (descending median):")
    _chans19 = sorted(df_analysis["em_filter_nm"].dropna().unique())
    for _ft19 in fix_present:
        for _ch19 in _chans19:
            _meds = {}
            for _ct19 in ct_present:
                _v19 = df_analysis[
                    (df_analysis["fixation_type"] == _ft19) &
                    (df_analysis["em_filter_nm"]  == _ch19) &
                    (df_analysis["cell_type"]     == _ct19)
                ][_m].dropna()
                if len(_v19) >= 2:
                    _meds[_ct19] = float(_v19.median())
            if len(_meds) >= 2:
                _sorted19 = sorted(_meds.items(), key=lambda kv: -kv[1])
                _dir19 = " > ".join(f"{k}({v:.1f})" for k, v in _sorted19)
                print(f"    {_ft19} / {_ch19} nm: {_dir19}")

print("\n" + "=" * 72)

# %%
# -- Step 19d: Per-session breakdown (session x channel x cell_type) ---------
#
# Median per (fixation_type, em_filter_nm, session_root, cell_type).
# Uses df_analysis which is already filtered to shift=0 and preferred n_comp,
# so session-size imbalances between cell_types cannot skew the aggregate.
# If a session contains only one cell_type the comparison is confounded --
# any group difference in that fixation_type could be a batch effect.

print("=" * 72)
print("PER-SESSION BREAKDOWN  (df_analysis: shift=0, preferred n_comp)")
print("=" * 72)

_sd_cols = [c for c in ["fixation_type", "em_filter_nm", "session_root", "cell_type"]
            if c in df_analysis.columns]

for _m in all_metrics:
    _lbl = METRIC_LABELS.get(_m, _m)
    _sub_sd = df_analysis[_sd_cols + [_m]].dropna(subset=[_m])
    if _sub_sd.empty:
        continue
    _agg_sd = (_sub_sd.groupby(_sd_cols, dropna=False)[_m]
               .agg(n="count", median="median")
               .round(1))
    print(f"\n{_lbl}  [{_m}]")
    print(_agg_sd.to_string())

# Confounding summary (independent of metric)
if "session_root" in df_analysis.columns:
    print("\nSession/cell_type overlap per fixation_type:")
    for _ft_s in fix_present:
        _sub_ft = df_analysis[df_analysis["fixation_type"] == _ft_s]
        _sess_ct = (_sub_ft.groupby(["session_root", "cell_type"], dropna=False)
                    .size().unstack("cell_type", fill_value=0))
        _ct_cols_s = [c for c in ct_present if c in _sess_ct.columns]
        if len(_ct_cols_s) < 2:
            print(f"  {_ft_s}: only one cell_type present -- cannot assess overlap")
            continue
        _n_both  = (_sess_ct[_ct_cols_s] > 0).all(axis=1).sum()
        _n_total = len(_sess_ct)
        if _n_both == 0:
            print(f"  {_ft_s}: FULLY CONFOUNDED -- 0/{_n_total} sessions"
                  f" contain both cell_types")
        elif _n_both < _n_total:
            print(f"  {_ft_s}: PARTIAL -- {_n_both}/{_n_total} sessions"
                  f" contain both cell_types")
        else:
            print(f"  {_ft_s}: OK -- all {_n_total} sessions contain both cell_types")

print("\n" + "=" * 72)

# %%
# -- Step 19b: Bimodality diagnosis -----------------------------------------
#
# Before group comparison, test the three hypotheses for bimodal distributions:
#   (A) Imaging day    -- color by session_root
#   (B) Fit model mix  -- color by n_components or shift_group
#   (C) Calibration    -- compare phasor tau_phi with fit tau_mean; if both
#       are bimodal and modes map to sessions, suspect calibration drift.
#
# Uses the UNFILTERED df so session/model mixing is visible before filtering.

_diag_metric = "tau_mean_median_ps"
if _diag_metric in df.columns and df[_diag_metric].notna().any():
    _sessions  = sorted(df["session_root"].dropna().unique())
    _n_comps   = sorted(df["n_components"].dropna().unique())
    _sgroups   = sorted(df["shift_group"].dropna().unique())
    _s_cmap    = plt.cm.tab10(np.linspace(0, 0.9, max(len(_sessions), 1)))
    _nc_cmap   = plt.cm.Set1(np.linspace(0, 0.8, max(len(_n_comps), 1)))
    _sg_colors = {"shift=0": "steelblue", "shift!=0": "darkorange"}

    for _ft in [f for f in FIXATION_ORDER if f in df["fixation_type"].dropna().unique()]:
        _sub = df[(df["fixation_type"] == _ft) & df[_diag_metric].notna()]
        if _sub.empty:
            continue
        _vals_all = _sub[_diag_metric].values
        _lo = float(np.percentile(_vals_all, 1))
        _hi = float(np.percentile(_vals_all, 99))
        _bins = np.linspace(_lo, _hi, 60)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # (A) By imaging session
        for i, sess in enumerate(_sessions):
            v = _sub[_sub["session_root"] == sess][_diag_metric].values
            if v.size > 0:
                axes[0].hist(v, bins=_bins, alpha=0.55, density=True,
                             color=_s_cmap[i], label=sess[:20])
        axes[0].set_title(f"(A) By imaging session\n{_ft}")
        axes[0].set_xlabel(_diag_metric)
        axes[0].legend(fontsize=6, loc="upper right")

        # (B) By fit model (n_comp and shift_group)
        for nc in _n_comps:
            v = _sub[_sub["n_components"] == nc][_diag_metric].values
            if v.size > 0:
                nc_idx = list(_n_comps).index(nc)
                axes[1].hist(v, bins=_bins, alpha=0.55, density=True,
                             color=_nc_cmap[nc_idx], label=f"{int(nc)}-comp")
        for sg in _sgroups:
            v = _sub[_sub["shift_group"] == sg][_diag_metric].values
            if v.size > 0:
                axes[1].hist(v, bins=_bins, alpha=0.35, density=True,
                             color=_sg_colors.get(sg, "gray"),
                             linestyle="--", histtype="step",
                             linewidth=1.5, label=sg)
        axes[1].set_title(f"(B) By fit model\n{_ft}")
        axes[1].set_xlabel(_diag_metric)
        axes[1].legend(fontsize=7, loc="upper right")

        # (C) Phasor vs fit: if phasor tau_phi is also bimodal,
        #     and modes map to sessions, it is a calibration issue.
        if "tau_phi_ns" in _sub.columns and _sub["tau_phi_ns"].notna().any():
            _phi_vals = _sub["tau_phi_ns"].dropna().values * 1000  # ns -> ps equivalent
            _phi_lo   = float(np.percentile(_phi_vals, 1))
            _phi_hi   = float(np.percentile(_phi_vals, 99))
            for i, sess in enumerate(_sessions):
                v = _sub[_sub["session_root"] == sess]["tau_phi_ns"].dropna().values * 1000
                if v.size > 0:
                    axes[2].hist(v, bins=np.linspace(_phi_lo, _phi_hi, 60),
                                 alpha=0.55, density=True,
                                 color=_s_cmap[i], label=sess[:20])
            axes[2].set_title(f"(C) Phasor tau_phi (session colored)\n{_ft}")
            axes[2].set_xlabel("tau_phi (ps equiv.)")
            axes[2].legend(fontsize=6, loc="upper right")
        else:
            axes[2].text(0.5, 0.5, "tau_phi not available\n(run Phase D first)",
                         ha="center", va="center", transform=axes[2].transAxes,
                         fontsize=10, color="gray")
            axes[2].set_title(f"(C) Phasor tau_phi\n{_ft}")

        fig.suptitle(
            f"Bimodality diagnosis -- {_ft}  "
            f"(unfiltered n={len(_sub)})\n"
            "Mode aligns with (A)=imaging day, (B)=fit model, (C)=calibration",
            fontsize=9,
        )
        plt.tight_layout()
        _savefig(fig, f"fig_bimodality_diag_{_ft}")
        plt.show()
else:
    print("tau_mean_median_ps not available -- run Phase E first.")

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
        _savefig(fig, f"fig_batch_check_{fix_type}")
        plt.show()

# %% [markdown]
# ## Step 21: Violin plots by cell_type x fixation_type x em_filter_nm
#
# Each violin shows the distribution of per-image medians.
# Layout: rows = fixation_type, columns = emission channel (457 / 535 nm).
# Individual data points overlaid as a strip plot.
# Both channel panels share the same y-axis so amplitudes are directly comparable.

# %%
palette  = plt.cm.Set2(np.linspace(0, 0.8, max(len(ct_present), 1)))
ct_color = {ct: palette[i] for i, ct in enumerate(ct_present)}

_channels_21 = sorted(df_analysis["em_filter_nm"].dropna().unique())
_n_rows      = len(fix_present)
_n_cols      = max(len(_channels_21), 1)

for metric in all_metrics:
    label = METRIC_LABELS.get(metric, metric)
    fig, axes = plt.subplots(_n_rows, _n_cols,
                             figsize=(4 * _n_cols, 4 * _n_rows),
                             squeeze=False,
                             sharey=True)

    for ri, fix_type in enumerate(fix_present):
        for ci, ch in enumerate(_channels_21):
            ax = axes[ri][ci]
            sub = (df_analysis[(df_analysis["fixation_type"] == fix_type) &
                               (df_analysis["em_filter_nm"]  == ch) &
                               df_analysis["cell_type"].isin(ct_present)]
                   .dropna(subset=[metric]))

            # Require >= 2 non-NaN values per group: violinplot KDE fails with n < 2.
            cts_here = [ct for ct in ct_present
                        if (sub["cell_type"] == ct).sum() >= 2]
            groups   = [sub[sub["cell_type"] == ct][metric].values for ct in cts_here]
            colors   = [ct_color[ct] for ct in cts_here]

            if not groups:
                ax.set_title(f"{fix_type} / {ch} nm\n(no data)", fontsize=8)
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
            if ci == 0:
                ax.set_ylabel(label)
            n_str = "  ".join(f"{ct}:{(sub['cell_type']==ct).sum()}"
                              for ct in cts_here)
            ax.set_title(f"{fix_type} / {ch} nm\nn={n_str}", fontsize=8)

    fig.suptitle(label, fontsize=11)
    plt.tight_layout()
    _savefig(fig, f"fig_violin_{metric}")
    plt.show()

# %%
# -- Step 21b: Direction check (text) ----------------------------------------
# Prints the median per group for every violin so the direction of any
# difference is unambiguous regardless of color-blind or scaling issues.

print("=" * 72)
print("STEP 21 DIRECTION CHECK  (median per group, analysis dataset)")
print("=" * 72)
_chans_21b = sorted(df_analysis["em_filter_nm"].dropna().unique())
for _m21 in all_metrics:
    _lbl21 = METRIC_LABELS.get(_m21, _m21)
    _any21  = False
    _lines21 = []
    for _ft21 in fix_present:
        for _ch21 in _chans_21b:
            _meds21 = {}
            for _ct21 in ct_present:
                _v21 = df_analysis[
                    (df_analysis["fixation_type"] == _ft21) &
                    (df_analysis["em_filter_nm"]  == _ch21) &
                    (df_analysis["cell_type"]     == _ct21)
                ][_m21].dropna()
                if len(_v21) >= 2:
                    _meds21[_ct21] = float(_v21.median())
                    _any21 = True
            if _meds21:
                _s21 = sorted(_meds21.items(), key=lambda kv: -kv[1])
                _dir21 = " > ".join(f"{k}({v:.1f})" for k, v in _s21)
                _lines21.append(f"  {_ft21:6s} / {_ch21} nm : {_dir21}")
    if _any21:
        print(f"\n{_lbl21}  [{_m21}]")
        for _ln in _lines21:
            print(_ln)
print("=" * 72)

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
    sub  = df_analysis[df_analysis["fixation_type"] == fix_type]
    # Include a cell type if it has >= MIN_N total rows in df_analysis for this
    # fixation_type.  Per-metric NaN filtering is handled inside the metric loop
    # by the `if len(x) < MIN_N or len(y) < MIN_N: continue` guard.
    cts  = [ct for ct in ct_present
            if len(sub[sub["cell_type"] == ct]) >= MIN_N]
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
                   if len(df_analysis[(df_analysis["fixation_type"] == fix_type) &
                                      (df_analysis["cell_type"] == ct)]) >= MIN_N]
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
    _savefig(fig, "fig_effect_size_heatmap")
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
    _savefig(fig, f"fig_xval_{y_col}_vs_{x_col}")
    plt.show()

# %% [markdown]
# ## Step 24: Group summary table
#
# Mean +/- SEM across images, stratified by fixation_type x cell_type.

# %%
summary_rows = []
for fix_type in fix_present:
    for ct in ct_present:
        sub = df_analysis[(df_analysis["fixation_type"] == fix_type) &
                          (df_analysis["cell_type"] == ct)]
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
# ## Step 25: Per-channel comparison
#
# Violin plots of amplitude ratio and tau_mean faceted by emission channel
# (em_filter_nm). Shows whether BKO vs KPCWT differences are consistent
# across 457 nm (NADH) and 535 nm (FAD / scatter) channels.

# %%
CHANNEL_METRICS = [m for m in ["amp_ratio_median", "tau_mean_median_ps"]
                   if m in df.columns]

channels = sorted(df_analysis["em_filter_nm"].dropna().unique())

if len(channels) < 2:
    print("Only one emission channel in data -- skipping per-channel comparison.")
elif not CHANNEL_METRICS:
    print("No fit metrics available -- run Phase E first.")
else:
    for metric in CHANNEL_METRICS:
        label   = METRIC_LABELS.get(metric, metric)
        n_fix   = len(fix_present)
        n_ct    = len(ct_present)
        n_ch    = len(channels)
        # positions: channel 0 at 0..n_ct-1, channel 1 at n_ct+1..2*n_ct,
        # with a gap of 2 between groups; dashed line at x = n_ct - 0.5 + 1
        gap     = 1.5
        ch_offsets = [i * (n_ct + gap) for i in range(n_ch)]

        fig, axes = plt.subplots(1, n_fix, figsize=(max(5, 2.2 * n_ct * n_ch) * n_fix, 5),
                                 squeeze=False)

        for ax, fix_type in zip(axes[0], fix_present):
            all_vals_by_ch = {}   # ch -> pooled values for channel-level significance

            for ci, ch in enumerate(channels):
                sub = (df_analysis[(df_analysis["fixation_type"] == fix_type) &
                                   (df_analysis["em_filter_nm"]  == ch) &
                                   df_analysis["cell_type"].isin(ct_present)]
                                  .dropna(subset=[metric]))
                cts_here = [ct for ct in ct_present
                            if (sub["cell_type"] == ct).any()]
                all_vals_by_ch[ch] = sub[metric].values

                rng = np.random.default_rng(seed=ci)
                for j, ct in enumerate(cts_here):
                    grp  = sub[sub["cell_type"] == ct][metric].values
                    pos  = ch_offsets[ci] + j
                    col  = ct_color[ct]
                    if len(grp) >= 2:
                        parts = ax.violinplot([grp], positions=[pos],
                                              showmedians=True, showextrema=True)
                        for pc in parts["bodies"]:
                            pc.set_facecolor(col); pc.set_alpha(0.65)
                        for key2 in ("cmedians", "cbars", "cmins", "cmaxes"):
                            if key2 in parts:
                                parts[key2].set_color("k")
                    elif len(grp) == 1:
                        ax.plot(pos, grp[0], "o", color=col, ms=8, zorder=4)
                    if len(grp) > 0:
                        ax.scatter(pos + rng.uniform(-0.07, 0.07, len(grp)),
                                   grp, s=12, color=col, alpha=0.7, zorder=5)

            # Dashed separator between channel groups
            sep_x = ch_offsets[0] + n_ct - 0.5 + gap * 0.5
            ylo, yhi = ax.get_ylim()
            ax.axvline(sep_x, color="0.4", lw=1.2, ls="--", zorder=1)

            # Channel significance: Mann-Whitney between the two channel pools
            vals_ch = [all_vals_by_ch.get(ch, np.array([])) for ch in channels]
            if len(vals_ch) == 2 and len(vals_ch[0]) >= 3 and len(vals_ch[1]) >= 3:
                mw = stats.mannwhitneyu(vals_ch[0], vals_ch[1], alternative="two-sided")
                if mw.pvalue < FDR_ALPHA:
                    mid_y = ax.get_ylim()[1] * 0.97
                    ax.text(sep_x, mid_y, "*",
                            ha="center", va="top", fontsize=16, fontweight="bold",
                            color="k")

            # X-tick labels: cell-type labels under each violin, channel label above cluster
            tick_pos   = []
            tick_label = []
            for ci, ch in enumerate(channels):
                for j, ct in enumerate(ct_present):
                    tick_pos.append(ch_offsets[ci] + j)
                    tick_label.append(ct)
                mid = ch_offsets[ci] + (n_ct - 1) / 2.0
                ax.text(mid, ax.get_ylim()[0],
                        f"{int(ch)} nm", ha="center", va="top",
                        fontsize=8, color="0.35",
                        transform=ax.get_xaxis_transform())

            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_label, fontsize=8, rotation=30, ha="right")
            ax.set_xlim(ch_offsets[0] - 0.8,
                        ch_offsets[-1] + n_ct - 1 + 0.8)
            ax.set_ylabel(label)
            ax.set_title(fix_type, fontsize=9)

        # Legend: cell type colours
        handles_leg = [mpatches.Patch(color=ct_color[ct], label=ct)
                       for ct in ct_present]
        axes[0][-1].legend(handles=handles_leg, fontsize=8,
                           bbox_to_anchor=(1.02, 1), loc="upper left")

        fig.suptitle(f"{label}  by channel  (* = MW p < {FDR_ALPHA})", fontsize=11)
        plt.tight_layout()
        _savefig(fig, f"fig_channel_{metric}")
        plt.show()

# %% [markdown]
# ## Step 26: Within-image heterogeneity
#
# Coefficient of variation (CV = std/mean) of tau_mean and amplitude ratio
# per image. High CV indicates spatially heterogeneous metabolism within a
# field of view. Compare BKO vs KPCWT to test whether KO affects metabolic
# uniformity, not just mean level.

# %%
CV_METRICS = {
    "tau_mean_cv":  "tau_mean CV (std/mean)",
    "amp_ratio_cv": "amplitude ratio CV",
}
cv_cols = [c for c in CV_METRICS if c in df.columns]

if not cv_cols:
    print("CV columns not found -- re-run Phase E Step 18 to compute them.")
else:
    for cv_col in cv_cols:
        label = CV_METRICS[cv_col]
        fig, axes = plt.subplots(1, len(fix_present),
                                 figsize=(4 * len(fix_present), 5),
                                 squeeze=False)
        for ax, fix_type in zip(axes[0], fix_present):
            sub = (df_analysis[(df_analysis["fixation_type"] == fix_type) &
                               df_analysis["cell_type"].isin(ct_present)]
                              .dropna(subset=[cv_col]))
            cts_here = [ct for ct in ct_present
                        if (sub["cell_type"] == ct).any()]
            groups   = [sub[sub["cell_type"] == ct][cv_col].values
                        for ct in cts_here]
            colors   = [ct_color[ct] for ct in cts_here]

            if groups:
                parts = ax.violinplot(groups,
                                      positions=range(len(groups)),
                                      showmedians=True, showextrema=True)
                for pc, c in zip(parts["bodies"], colors):
                    pc.set_facecolor(c); pc.set_alpha(0.65)
                for key2 in ("cmedians", "cbars", "cmins", "cmaxes"):
                    if key2 in parts:
                        parts[key2].set_color("k")
                rng = np.random.default_rng(seed=0)
                for j, (grp, c) in enumerate(zip(groups, colors)):
                    ax.scatter(j + rng.uniform(-0.07, 0.07, len(grp)),
                               grp, s=14, color=c, alpha=0.6, zorder=3)
                ax.set_xticks(range(len(cts_here)))
                ax.set_xticklabels(cts_here, fontsize=9)

            n_str = "  ".join(f"{ct}:{(sub['cell_type']==ct).sum()}"
                              for ct in cts_here)
            ax.set_title(f"{fix_type}\nn={n_str}", fontsize=8)
            ax.set_ylabel(label)

        fig.suptitle(label, fontsize=11)
        plt.tight_layout()
        _savefig(fig, f"fig_heterogeneity_{cv_col}")
        plt.show()

# %% [markdown]
# ## Step 27: Sample size table
#
# N images per (fixation_type x cell_type x channel). Used for the methods
# slide and to flag underpowered groups before interpreting statistics.

# %%
size_tbl = (df_analysis[df_analysis["cell_type"].isin(ct_present)]
            .groupby(["fixation_type", "cell_type", "em_filter_nm"],
                     dropna=False)
            .size()
            .reset_index(name="n_images"))

# Pivot for readability: rows = fixation x channel, cols = cell_type
if not size_tbl.empty:
    pivot = size_tbl.pivot_table(
        index=["fixation_type", "em_filter_nm"],
        columns="cell_type",
        values="n_images",
        aggfunc="sum",
        fill_value=0,
    )
    # Enforce column order
    col_order = [c for c in CELL_TYPE_ORDER if c in pivot.columns]
    pivot = pivot[col_order]
    print("Sample sizes (N images):")
    print(pivot.to_string())

    # Visual table as a matplotlib figure
    row_labels = [f"{ft} / {int(ch) if not pd.isna(ch) else 'NA'} nm"
                  for ft, ch in pivot.index]
    cell_text  = pivot.values.tolist()
    col_labels = list(pivot.columns)

    fig, ax = plt.subplots(figsize=(max(4, 1.6 * len(col_labels)),
                                    max(2, 0.45 * len(row_labels) + 1.2)))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.6)
    ax.set_title("Sample sizes (N images per group)", fontsize=12, pad=10)
    plt.tight_layout()
    _savefig(fig, "fig_sample_sizes")
    plt.show()

    size_tbl.to_csv(results_dir / "sample_sizes.csv", index=False)
    print(f"Saved: {results_dir / 'sample_sizes.csv'}")

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
