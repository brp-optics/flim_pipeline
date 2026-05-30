# %% [markdown]
# # Reference Image Panel Finder
#
# Produces a DataFrame of candidate images for the 8-image inter-session
# reference panel.  Prints the filename list at every filter step where
# fewer than 100 rows remain, so you can inspect what is being dropped.

# %%
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("../results")

# Eligibility thresholds -- set a value to None to skip that filter
MAX_POCKELS  = 0.25
N_COMP       = 2
EM_FILTER    = 457   # set None to include all emission filters
OBJECTIVE    = "20x0.75NA"
USB_ATTEN    = 1
SHIFT_GROUP  = "shift=0"

# Form treatment duration filter -- set None to include all durations
FORM_DURATION = None   # was "10min"; relaxed to include 20min

# Sessions / fixation-type combinations to exclude
EXCLUDE_SESSION_FT = {
    "20260501_KPC_fixed_dishes_on_SLIM": ["live"],
}

# Number of top candidates to show per group at the end
TOP_N = 5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _show_files(df, label):
    n = len(df)
    print(f"\n[filter] {label}: {n} rows")
    if 0 < n < 100:
        cols = [c for c in ["fixation_type", "cell_type", "session_root",
                             "session_date", "frame_index", "em_filter_nm",
                             "pockels", "treatment_duration", "filename"]
                if c in df.columns]
        short = df[cols].copy()
        short["filename"] = short["filename"].str[-55:]
        print(short.to_string(index=True))

def _extract_bin(key):
    if pd.isna(key):
        return np.nan
    m = re.search(r"-b(\d+)", str(key))
    return int(m.group(1)) if m else np.nan

def _shift_from_key(k):
    if pd.isna(k):
        return np.nan
    return "shift=0" if re.search(r"-(sf|sz)-", str(k)) else "shift!=0"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

sdt_df = pd.read_csv(RESULTS_DIR / "sdt_metadata_cal.csv")
fit_df = pd.read_csv(RESULTS_DIR / "fit_analysis_summary.csv")

sdt_df["acquisition_time"] = pd.to_datetime(sdt_df["acquisition_time"], errors="coerce")

if "session_date" not in sdt_df.columns:
    sdt_df["session_date"] = sdt_df["session_root"].str.extract(r"(\d{8})", expand=False)

# ---------------------------------------------------------------------------
# Merge metadata into fit_df
# ---------------------------------------------------------------------------

NEED_COLS = ["pockels", "frame_index", "z_position", "objective", "usb_atten",
             "file_type", "session_date", "treatment_duration", "session_root"]
merge_cols = [c for c in NEED_COLS if c not in fit_df.columns]
if merge_cols:
    fit_df = fit_df.merge(sdt_df[["filename"] + merge_cols], on="filename", how="left")

if "session_root_x" in fit_df.columns:
    fit_df["session_root"] = fit_df["session_root_x"].fillna(fit_df["session_root_y"])
    fit_df.drop(columns=["session_root_x", "session_root_y"], inplace=True)

if "shift_group" not in fit_df.columns:
    if "shift_group" in sdt_df.columns:
        fit_df = fit_df.merge(sdt_df[["filename", "shift_group"]], on="filename", how="left")
    else:
        fit_df["shift_group"] = fit_df["fit_set_key"].map(_shift_from_key)

fit_df["_bin"] = fit_df["fit_set_key"].map(_extract_bin)
# fit_analysis_summary.csv already has one row per file: Phase E's best_fit_key()
# selects the preferred fit (highest n_comp, shift-zero, per-session bin override).
# No further bin filtering is needed here; _bin is kept as a display column.

# ---------------------------------------------------------------------------
# Sequential filters with filename printout
# ---------------------------------------------------------------------------

df = fit_df.copy()
_show_files(df, "start")

# 1. Biological samples only
if "file_type" in df.columns:
    df = df[df["file_type"] == "sample"]
_show_files(df, "after file_type=='sample'")

# 2. Emission filter
if EM_FILTER is not None:
    df = df[df["em_filter_nm"] == EM_FILTER]
    _show_files(df, f"after em_filter_nm=={EM_FILTER}")

# 3. Pockels
if MAX_POCKELS is not None and "pockels" in df.columns:
    df = df[df["pockels"] <= MAX_POCKELS]
    _show_files(df, f"after pockels<={MAX_POCKELS}")

# 4. n_components
df = df[df["n_components"] == N_COMP]
_show_files(df, f"after n_components=={N_COMP}")

# 5. Bin: no filter -- Phase E already selected the best fit per file.
#    _bin is shown in display columns so you can see which bin was used.
_show_files(df, "after n_components (bin not filtered -- Phase E already chose best)")

# 6. Shift group
if SHIFT_GROUP and "shift_group" in df.columns:
    df = df[df["shift_group"] == SHIFT_GROUP]
    _show_files(df, f"after shift_group=='{SHIFT_GROUP}'")

# 7. Objective
if OBJECTIVE and "objective" in df.columns:
    df = df[df["objective"] == OBJECTIVE]
    _show_files(df, f"after objective=='{OBJECTIVE}'")

# 8. USB attenuation
if USB_ATTEN is not None and "usb_atten" in df.columns:
    df = df[df["usb_atten"] == USB_ATTEN]
    _show_files(df, f"after usb_atten=={USB_ATTEN}")

# 9. frame_index: not filtered globally -- shown as display column
if "frame_index" in df.columns:
    print(f"\n[info] frame_index range: {sorted(df['frame_index'].unique())} (not filtered)")

# 10. Session exclusions
if "session_root" in df.columns:
    for sess, fts in EXCLUDE_SESSION_FT.items():
        mask = df["session_root"].str.startswith(sess, na=False)
        if fts is None:
            df = df[~mask]
        else:
            df = df[~(mask & df["fixation_type"].isin(fts))]
_show_files(df, "after session exclusions")

# 11. Form duration (optional)
if FORM_DURATION and "treatment_duration" in df.columns:
    _form = df["fixation_type"] == "form"
    _ok   = df["treatment_duration"].str.strip() == FORM_DURATION
    df = df[~_form | _ok]
    _show_files(df, f"after form duration=='{FORM_DURATION}'")

# 12. Keep only live and form
df = df[df["fixation_type"].isin(["live", "form"])]
_show_files(df, "after keep live+form only")

df = df.reset_index(drop=True)
print(f"\nEligible rows: {len(df)}")
if not df.empty:
    print(df.groupby(["fixation_type", "cell_type", "session_date"],
                     dropna=False).size().rename("n").to_string())

# ---------------------------------------------------------------------------
# Output DataFrame
# ---------------------------------------------------------------------------

OUT_COLS = [c for c in [
    "fixation_type", "cell_type", "session_root", "session_date",
    "filename", "fit_set_key",
    "em_filter_nm", "pockels", "objective", "usb_atten",
    "frame_index", "z_position", "n_components", "_bin", "shift_group",
    "treatment_duration", "amp_ratio_median", "tau_mean_median_ps",
    "n_px_final", "pct_final",
] if c in df.columns]

candidates = df[OUT_COLS].copy()

# ---------------------------------------------------------------------------
# Rank within each (fixation_type, cell_type) group
# ---------------------------------------------------------------------------

if not candidates.empty:
    RANK_METRICS = ["amp_ratio_median", "tau_mean_median_ps"]
    grp_med = (
        candidates.groupby(["fixation_type", "cell_type"])[RANK_METRICS]
        .median()
        .rename(columns={m: f"{m}_gmed" for m in RANK_METRICS})
    )
    candidates = candidates.merge(grp_med, on=["fixation_type", "cell_type"], how="left")

    _amp_sd = candidates["amp_ratio_median"].std(ddof=1) or 1.0
    _tau_sd = candidates["tau_mean_median_ps"].std(ddof=1) or 1.0
    candidates["_dist"] = np.sqrt(
        ((candidates["amp_ratio_median"] - candidates["amp_ratio_median_gmed"]) / _amp_sd) ** 2 +
        ((candidates["tau_mean_median_ps"] - candidates["tau_mean_median_ps_gmed"]) / _tau_sd) ** 2
    )
    candidates = candidates.sort_values(["fixation_type", "cell_type", "_dist"]).reset_index(drop=True)

    GROUPS = [("live","KPCWT"),("live","BKO"),("form","KPCWT"),("form","BKO")]
    print("\n" + "=" * 70)
    print(f"TOP {TOP_N} PER GROUP (closest to group median)")
    print("=" * 70)
    for ft, ct in GROUPS:
        sub = candidates[(candidates["fixation_type"]==ft) & (candidates["cell_type"]==ct)].head(TOP_N)
        print(f"\n--- {ft.upper()}  {ct}  (n={len(candidates[(candidates['fixation_type']==ft)&(candidates['cell_type']==ct)])}) ---")
        if sub.empty:
            print("  (none)")
        else:
            _s = sub.copy()
            _s["filename"] = _s["filename"].str[-55:]
            with pd.option_context("display.max_colwidth", 57, "display.float_format", "{:.4f}".format,
                                   "display.max_columns", 25, "display.width", 200):
                print(_s.to_string(index=True))

    out_path = RESULTS_DIR / "reference_panel_candidates.csv"
    candidates.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
