# %% [markdown]
# # Phase C: Data integrity
#
# Steps:
# 7. Fit parameter inventory -- which params exported per file, shift=0 check
# 8. Orientation verification -- SDT intensity vs SPCImage photon map
# 9. Chi-squared quality -- distributions and spatial maps
# 10. Fit parameter distributions -- tau and amplitude by fixation type
# 11. Amplitude ratios -- a2/(a2+a3) for glu, a1/(a1+a2) for form/live

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sdtfile import SdtFile
from scipy.ndimage import uniform_filter as _uniform_filter

# -- Configuration ----------------------------------------------------------
WIN_DATA_DIRS = [
    Path(r"E:\18_RK_Circadian\data\raw\20260429_KPC_fixed_dishes_on_SLIM"),
    Path(r"E:\18_RK_Circadian\data\raw\20260501_KPC_fixed_dishes_on_SLIM"),
    Path(r"E:\18_RK_Circadian\data\raw\20260509_KPC_fixed_dishes_on_SLIM"),
    Path(r"E:\18_RK_Circadian\data\raw\20260508_KPC_live_on_SLIM"),
    Path(r"E:\18_RK_Circadian\data\raw\20260517_KPC_live_on_SLIM"),
    Path(r"E:\18_RK_Circadian\data\raw\20260520_KPC_fixed_dishes_SLIM"),
    Path(r"E:\18_RK_Circadian\data\raw\20260521_KPC_live_SLIM"),
    Path(r"E:\18_RK_Circadian\data\raw\20260522_KPC_fixed_dishes_SLIM"),
]
LIN_DATA_DIRS = [
    Path("/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260429_KPC_fixed_dishes_on_SLIM"),
    Path("/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260501_KPC_fixed_dishes_on_SLIM"),
    Path("/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260509_KPC_fixed_dishes_on_SLIM"),
    Path("/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260508_KPC_live_on_SLIM"),
    Path("/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260517_KPC_live_on_SLIM"),
    Path("/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260520_KPC_fixed_dishes_SLIM"),
    Path("/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260521_KPC_live_SLIM"),
    Path("/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260522_KPC_fixed_dishes_SLIM"),
]
CURRENT_OS = "Win"
data_dirs = WIN_DATA_DIRS if CURRENT_OS == "Win" else LIN_DATA_DIRS

CHI2_LO = 0.8    # flag pixels below this
CHI2_HI = 2.0    # flag pixels above this

# Orientation of SPCImage .asc arrays relative to the SDT pixel array.
# OpenScan flips vertically vs BH. Confirmed in Step 8; update if wrong.
# Options: None | "flipud" | "fliplr" | "transpose" | "flipud+transpose"
ORIENTATION_TRANSFORM = None    # confirmed in Step 8: SDT and SPCImage raw are already aligned

# Preferred number of fit components per fixation type (for selecting fit model
# when multiple exports are present in fit_map.csv):
#   glu:  3-component (ultra-short artifact a1 + two NADH components)
#         If only 2-comp fits exist, best_fit_key() falls back automatically.
#   form/live: 2-component (standard free/bound NADH)
PREFERRED_N_COMP = {"glu": 3, "form": 2, "live": 2}

# Per-session bin-radius override.  See Phase E for rationale.
PREFERRED_BIN_OVERRIDE = {
    "20260509_KPC_fixed_dishes_on_SLIM": 10,
}

# Amplitude ratio definition per fixation type:
#   glu:       a2/(a2+a3)  -- skip ultra-short artifact component a1 (~16 ps)
#   form/live: a1/(a1+a2)  -- standard free/total NADH ratio
RATIO_CFG = {
    "glu":  ("a2", "a3"),
    "form": ("a1", "a2"),
    "live": ("a1", "a2"),
}

# Number of parallel workers for loading .asc files.
# Each worker opens files concurrently; set to 1 to disable threading.
LOAD_WORKERS = 8

# Tau1 realism threshold (ps).  In a 2-component shift=0 fit of glu, tau1 values
# below this are likely absorbing the ultra-short fixation artifact rather than
# representing free NADH.  Used in Step 7 to flag unrealistic pixels.
TAU1_REALISM_PS = 200.0

# Minimum binned-photon count for the tau1 realism pixel gate.
# Pixels below this are excluded before computing the tau1 fraction,
# removing background pixels where the fit is poorly constrained.
# Match these values to Phase E MIN_PHOTONS_BY_NCOMP thresholds.
MIN_PHOTONS_BY_NCOMP = {2: 200, 3: 100}
MIN_PHOTONS_DEFAULT  = 200   # used when n_components is unknown

# %%
# -- Load Phase B outputs ---------------------------------------------------
results_dir = Path("../results")

sdt_df = pd.read_csv(results_dir / "sdt_metadata_cal.csv")
fp_map = pd.read_csv(results_dir / "filepath_map.csv")
# fp_map may have duplicate filename rows (multiple drives / data dirs scanned).
# Keep first occurrence per filename to avoid fan-out in the merge.
fp_map = fp_map.drop_duplicates(subset="filename", keep="first")
sdt_df = sdt_df.merge(fp_map[["filename", "filepath"]], on="filename", how="left")
# Drop duplicates: first by full row (catches exact copies), then by filename
# (catches near-duplicates where rows differ only in a metadata column).
n_before = len(sdt_df)
sdt_df = sdt_df.drop_duplicates()
sdt_df = sdt_df.drop_duplicates(subset="filename", keep="first")
if len(sdt_df) < n_before:
    print(f"Dropped {n_before - len(sdt_df)} duplicate rows after merge.")
sdt_df["filepath"]         = sdt_df["filepath"].map(Path)
sdt_df["acquisition_time"] = pd.to_datetime(sdt_df["acquisition_time"])
sdt_df["has_fit_export"]   = sdt_df["has_fit_export"].fillna(False).astype(bool)

fit_map_df = pd.read_csv(results_dir / "fit_map.csv")

print(f"Loaded {len(sdt_df)} SDT files")
print(sdt_df["file_type"].value_counts().to_string())
print(f"\nLoaded fit_map: {len(fit_map_df)} fit-set/SDT pairs")
if "n_components" in fit_map_df.columns:
    print(fit_map_df["n_components"].value_counts().sort_index().to_string())

# %%
# -- Rebuild fit_data from .asc files --------------------------------------

def _infer_param_name(stem: str) -> str:
    """Map .asc filename stem suffix to canonical parameter name."""
    s = stem.lower()
    for pattern, name in [
        (r"[-_]a1[_%]?$",                              "a1"),
        (r"[-_]a2[_%]?$",                              "a2"),
        (r"[-_]a3[_%]?$",                              "a3"),
        (r"[-_]t1$|[-_]tau1$",                         "tau1"),
        (r"[-_]t2$|[-_]tau2$",                         "tau2"),
        (r"[-_]t3$|[-_]tau3$",                         "tau3"),
        (r"[-_]tm$|[-_]tau_?mean$|[-_]mean$",          "tau_mean"),
        (r"[-_]chi$|[-_]chisq?$",                      "chi2"),
        (r"[-_]photons?$|[-_]int(ensity)?$|[-_]cnt$",  "photons"),
        (r"[-_]scatter$|[-_]sc$",                      "scatter"),
        (r"[-_]shift$",                                 "shift"),
        (r"[-_]g$",                                     "G"),
        (r"[-_]s$",                                     "S"),
    ]:
        if re.search(pattern, s):
            return name
    parts = s.rsplit("_", 1)
    return parts[-1] if len(parts) > 1 else s


def _parse_bin_param(folder: str) -> int:
    """Extract SPCImage bin parameter b from fit folder name.
    e.g. 'fitet-sf-c2-b3' -> 3.  Returns 0 if not found (no smoothing)."""
    m = re.search(r"-b(\d+)", str(folder), re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _strip_param_suffix(stem: str) -> str:
    """Remove trailing parameter token to recover the base image name."""
    return re.sub(
        r"[-_]?(a[123]|t[123]|tau[123]|chi|chisq?|photons?|intensity|cnt|"
        r"tm|tau_?mean|scatter|sc|shift|[gs])[_%]?$",
        "",
        stem,
        flags=re.IGNORECASE,
    ).strip("_- ")


def _load_asc(filepath: Path) -> dict | None:
    """Load a single SPCImage .asc grid. Returns None on failure."""
    param = _infer_param_name(filepath.stem)
    try:
        data = np.loadtxt(str(filepath), dtype=float)
    except ValueError:
        try:
            data = np.loadtxt(str(filepath), dtype=float, skiprows=1)
        except Exception:
            return None
    except Exception:
        return None
    if data.ndim != 2 or data.size == 0:
        return None
    return {"param": param, "data": data}


# Targeted parallel load.
# 1. Collect every unique fit_set_key from fit_map_df (no dependency on
#    sample_fits or best_fit_key, both defined later).
# 2. Each key encodes the exact directory and base_stem, so we can use
#    fit_dir.glob("*.asc") instead of rglob over the whole tree.
# 3. ThreadPoolExecutor parallelises the text-file I/O across LOAD_WORKERS.
#
# Key format: "{session_root}::{rel_dir}::{base_stem}"

from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _asc

_all_keys: dict = {}  # fit_set_key -> (session_root, rel_dir, base_stem)
for _, _row in fit_map_df.iterrows():
    _fkey = str(_row["fit_set_key"])
    if _fkey in _all_keys:
        continue
    try:
        _sr, _rd, _bs = _fkey.split("::", 2)
        _all_keys[_fkey] = (_sr, _rd, _bs)
    except ValueError:
        pass


def _load_one_fit_set(args):
    """Load .asc files for one fit set. Designed to run in a thread pool."""
    fkey, sr, rd, bs = args
    _sd = next((d for d in data_dirs if d.name == sr), None)
    if _sd is None:
        return fkey, None, "no_session"
    fit_dir = _sd / Path(rd)
    if not fit_dir.exists():
        return fkey, None, "missing"
    fd = {"_session_root": sr, "_folder": rd, "_base_stem": bs}
    for fp in sorted(fit_dir.glob("*.asc")):
        if re.search(r"_statistic", fp.stem, re.IGNORECASE):
            continue
        if _strip_param_suffix(fp.stem).lower() != bs.lower():
            continue
        res = _load_asc(fp)
        if res:
            fd[res["param"]] = res["data"]
            fd[f"_path_{res['param']}"] = fp
    return fkey, fd, "ok"


_load_args = [(fkey, sr, rd, bs) for fkey, (sr, rd, bs) in _all_keys.items()]
print(f"\nLoading {len(_load_args)} fit sets using {LOAD_WORKERS} workers...")

fit_data: dict = {}
_n_ok = _n_missing = 0
with _TPE(max_workers=LOAD_WORKERS) as _pool:
    _futures = {_pool.submit(_load_one_fit_set, a): a[0] for a in _load_args}
    _done = 0
    for _fut in _asc(_futures):
        _fkey, _fd, _status = _fut.result()
        _done += 1
        if _fd is not None:
            fit_data[_fkey] = _fd
            _n_ok += 1
        else:
            _n_missing += 1
        if _done % 100 == 0 or _done == len(_load_args):
            print(f"  {_done}/{len(_load_args)} done...", end="\r", flush=True)

print(f"\nLoaded {_n_ok} fit sets  ({_n_missing} dirs missing/not found)")

# %%
# Diagnostic: what param names actually loaded?
all_found = sorted(set(k for fd in fit_data.values()
                       for k in fd if not k.startswith("_")))
print("Param names found across all fit_data entries:", all_found)

print("\nSample fit sets (first 3):")
for _sample_key in list(fit_data.keys())[:3]:
    _plist = [k for k in fit_data[_sample_key] if not k.startswith("_")]
    print(f"  {_sample_key}")
    print(f"    params: {_plist}")

# Helper: select best fit_set_key for a file using fit_map_df + PREFERRED_N_COMP
def _parse_bin_radius(key: str) -> int:
    """Parse bin radius from fit_set_key string, e.g. 'fitet-sz-c2-b10' -> 10.
    Returns 0 if no bin token found."""
    m = re.search(r"[-_]b(\d+)", str(key), re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _prefer_shift_zero(candidates) -> str:
    """Among candidate rows, return the fit_set_key of the shift-zero folder.

    Prefers rows whose key contains '-sz-' (SPCImage shift=zero) over '-sf-'
    (shift=free).  Falls back to the first row if no '-sz-' match exists.

    This matters when a file has BOTH a shift-zero and a shift-free fit of the
    same n_components: best_fit_key must consistently pick one so that
    shift_nonzero_px is computed on the correct folder.
    """
    sz_rows = candidates[candidates["fit_set_key"].str.contains("-sz-", case=False, na=False)]
    chosen  = sz_rows if not sz_rows.empty else candidates
    return str(chosen.iloc[0]["fit_set_key"])


def best_fit_key(filename: str, fixation_type: str = None) -> str | None:
    """Return the preferred fit_set_key for a file.

    Selection priority:
      0. If the session has a PREFERRED_BIN_OVERRIDE entry, restrict candidates
         to that bin radius before any other selection.  Falls through if the
         preferred bin has no matching fit set for this file.
      1. Match PREFERRED_N_COMP[fixation_type]; among ties prefer shift-zero folder.
      2. Else: highest n_components; among ties prefer shift-zero folder.
      3. Else: first row.
    """
    rows = fit_map_df[fit_map_df["sdt_filename"] == filename]
    if rows.empty:
        return None
    # Apply per-session bin override (explicit re-export at higher bin)
    if PREFERRED_BIN_OVERRIDE:
        _sess = str(rows.iloc[0]["fit_set_key"]).split("::")[0]
        if _sess in PREFERRED_BIN_OVERRIDE:
            _bin_rows = rows[rows["fit_set_key"].apply(_parse_bin_radius)
                            == PREFERRED_BIN_OVERRIDE[_sess]]
            if not _bin_rows.empty:
                rows = _bin_rows
    preferred = PREFERRED_N_COMP.get(str(fixation_type)) if fixation_type else None
    if preferred is not None and "n_components" in rows.columns:
        exact = rows[rows["n_components"] == preferred]
        if not exact.empty:
            return _prefer_shift_zero(exact)
    if "n_components" in rows.columns:
        best_nc = rows["n_components"].max()
        top     = rows[rows["n_components"] == best_nc]
        return _prefer_shift_zero(top)
    return _prefer_shift_zero(rows)


# Convenience: apply orientation correction to a fit array
def apply_orientation(arr: np.ndarray, transform) -> np.ndarray:
    if transform is None:
        return arr
    if transform == "flipud":
        return np.flipud(arr)
    if transform == "fliplr":
        return np.fliplr(arr)
    if transform == "transpose":
        return arr.T
    if transform == "flipud+transpose":
        return np.flipud(arr).T
    raise ValueError(f"Unknown transform: {transform!r}")


# %% [markdown]
# ## Step 7: Fit parameter inventory
#
# Confirm which parameters are available per fixation type.
# SPCImage exports: a1/tau1 (all), a2/tau2 (most), a3/tau3 (glu),
# chi, photons, G, S, scatter. Shift was fixed at zero -- verify below.

# %%
ALL_PARAMS = ["a1", "tau1", "a2", "tau2", "a3", "tau3",
              "chi2", "photons", "G", "S", "scatter", "shift", "tau_mean"]

sample_fits = sdt_df[
    (sdt_df["file_type"] == "sample") & sdt_df["has_fit_export"]
].copy()
print(f"Sample files with fit exports: {len(sample_fits)}\n")

inv_rows = []
for _, row in sample_fits.iterrows():
    key = best_fit_key(row["filename"], row.get("fixation_type"))
    if key is None or key not in fit_data:
        continue
    fd = fit_data[key]
    fm_match = fit_map_df[
        (fit_map_df["sdt_filename"] == row["filename"]) &
        (fit_map_df["fit_set_key"]  == key)
    ]
    n_comp = int(fm_match.iloc[0]["n_components"]) if not fm_match.empty else None
    inv_rows.append({
        "filename":      row["filename"],
        "fixation_type": row.get("fixation_type"),
        "n_components":  n_comp,
        **{p: (p in fd) for p in ALL_PARAMS},
    })

inv_df = pd.DataFrame(inv_rows)

print("Parameter availability by fixation type:")
for fix, grp in inv_df.groupby("fixation_type", dropna=False):
    n = len(grp)
    print(f"\n  {fix} ({n} files):")
    if "n_components" in grp.columns:
        nc = grp["n_components"].value_counts().sort_index()
        print(f"    fit models (n_comp): {dict(nc)}")
    for p in ALL_PARAMS:
        if p in grp.columns and grp[p].any():
            print(f"    {p:12s}: {int(grp[p].sum())}/{n}")

# %%
# Shift check: shift was fixed to 0 in SPCImage -- verify
shift_issues = []
for _, row in sample_fits.iterrows():
    key = best_fit_key(row["filename"], row.get("fixation_type"))
    if key is None or key not in fit_data:
        continue
    fd = fit_data[key]
    if "shift" not in fd:
        continue
    arr = fd["shift"]
    nz = int(np.count_nonzero(np.nan_to_num(arr)))
    if nz > 0:
        shift_issues.append({
            "filename":   row["filename"],
            "nonzero_px": nz,
            "max_abs":    float(np.nanmax(np.abs(arr))),
        })

if shift_issues:
    print(f"\nWARNING: {len(shift_issues)} files have non-zero shift values:")
    for s in shift_issues:
        print(f"  {s['filename']}: {s['nonzero_px']} px, max={s['max_abs']:.4f}")
else:
    print("\nShift OK: zero (or not exported) for all sample files.")

# Track which filenames have free (nonzero) shift -- used to split Steps 10-11.
shift_nonzero_files = set(s["filename"] for s in shift_issues)
print(f"\nShift groups: shift=0: {len(sample_fits) - len(shift_nonzero_files)}  "
      f"shift!=0: {len(shift_nonzero_files)}")

# %%
# Save per-file shift statistics to results/shift_metadata.csv.
# Columns: filename, fit_set_key, has_shift_export, shift_nonzero_px,
#          shift_max_abs_ps, shift_mean_ps.
# Downstream notebooks can merge this on filename to filter or split by shift mode.
shift_meta_rows = []
for _, row in sample_fits.iterrows():
    fn  = row["filename"]
    key = best_fit_key(fn, row.get("fixation_type"))
    if key is None or key not in fit_data:
        shift_meta_rows.append({
            "filename": fn, "fit_set_key": key,
            "has_shift_export": False,
            "shift_nonzero_px": 0, "shift_max_abs_ps": np.nan,
            "shift_mean_ps": np.nan, "shift_std_ps": np.nan,
        })
        continue
    fd = fit_data[key]
    has_shift = "shift" in fd
    if has_shift:
        arr = fd["shift"]
        valid = np.isfinite(arr)
        nz    = int(np.count_nonzero(np.nan_to_num(arr)))
        mx    = float(np.nanmax(np.abs(arr))) if valid.any() else np.nan
        mn    = float(np.nanmean(arr))        if valid.any() else np.nan
        sd    = float(np.nanstd(arr))         if valid.any() else np.nan
    else:
        nz, mx, mn, sd = 0, np.nan, np.nan, np.nan
    shift_meta_rows.append({
        "filename":          fn,
        "fit_set_key":       key,
        "has_shift_export":  has_shift,
        "shift_nonzero_px":  nz,
        "shift_max_abs_ps":  mx,
        "shift_mean_ps":     mn,
        "shift_std_ps":      sd,
    })

shift_meta_df = pd.DataFrame(shift_meta_rows)
# shift_meta_df is merged into fit_qc_summary.csv at the save step below.
if not shift_meta_df.empty:
    n_free = int((shift_meta_df["shift_nonzero_px"] > 0).sum())
    print(f"  Files with nonzero shift: {n_free} / {len(shift_meta_df)}")
    if n_free:
        print(shift_meta_df[shift_meta_df["shift_nonzero_px"] > 0][
            ["filename", "shift_nonzero_px", "shift_max_abs_ps", "shift_mean_ps"]
        ].to_string())

# %%
# Tau1 realism check.
# Fraction of valid pixels where tau1 < TAU1_REALISM_PS.
# Elevated fractions (especially in glu shift=0 2-comp fits) indicate the short
# component is absorbing the fixation artifact rather than free NADH.
# Results are merged into fit_qc_summary.csv and plotted in Step 10.
tau1_realism_rows = []
_tau1_n_no_photons = 0

for _, row in sample_fits.iterrows():
    fn  = row["filename"]
    fix = row.get("fixation_type")
    key = best_fit_key(fn, fix)
    rec = {
        "filename":            fn,
        "fixation_type":       fix,
        "tau1_n_valid":        0,
        "tau1_frac_lt_thresh": np.nan,
        "tau1_median_ps":      np.nan,
        "tau1_min_photons":    np.nan,
    }
    if key and key in fit_data:
        fd = fit_data[key]
        # Only compute when both tau1 and photons are exported.
        if "tau1" not in fd or "photons" not in fd:
            if "tau1" in fd:
                _tau1_n_no_photons += 1
            tau1_realism_rows.append(rec)
            continue

        # Look up n_components to choose the photon threshold.
        _fm = fit_map_df[
            (fit_map_df["sdt_filename"] == fn) &
            (fit_map_df["fit_set_key"]  == key)
        ]
        n_comp = (int(_fm.iloc[0]["n_components"])
                  if not _fm.empty and "n_components" in _fm.columns
                  else None)
        min_ph = MIN_PHOTONS_BY_NCOMP.get(n_comp, MIN_PHOTONS_DEFAULT)
        rec["tau1_min_photons"] = float(min_ph)

        # Apply the same spatial binning SPCImage used (kernel 2b+1 square).
        b      = _parse_bin_param(fd.get("_folder", ""))
        ph_raw = fd["photons"].astype(float)
        ph_bin = (_uniform_filter(ph_raw, size=(2 * b + 1)) if b > 0 else ph_raw)

        tau1_arr = fd["tau1"]
        valid = (np.isfinite(tau1_arr) & (tau1_arr > 0) &
                 np.isfinite(ph_bin)   & (ph_bin >= min_ph))
        n = int(valid.sum())
        if n > 0:
            vals = tau1_arr[valid]
            rec["tau1_n_valid"]        = n
            rec["tau1_frac_lt_thresh"] = float((vals < TAU1_REALISM_PS).mean())
            rec["tau1_median_ps"]      = float(np.median(vals))
    tau1_realism_rows.append(rec)

tau1_df = pd.DataFrame(tau1_realism_rows)
tau1_df["shift_group"] = tau1_df["filename"].apply(
    lambda fn: "shift!=0" if fn in shift_nonzero_files else "shift=0"
)

_n_gated = int((tau1_df["tau1_n_valid"] > 0).sum())
print(f"\nTau1 realism check  (threshold = {TAU1_REALISM_PS} ps, binned photon gate):")
print(f"  Files computed (photons available): {_n_gated} / {len(tau1_df)}")
if _tau1_n_no_photons:
    print(f"  Files skipped (tau1 present but no photons export): {_tau1_n_no_photons}")
_sub = tau1_df[tau1_df["tau1_n_valid"] > 0]
if not _sub.empty:
    print(_sub.groupby(["fixation_type", "shift_group"])[
        ["tau1_frac_lt_thresh", "tau1_median_ps", "tau1_min_photons"]
    ].mean().round(3).to_string())
high = tau1_df[tau1_df["tau1_frac_lt_thresh"] > 0.3]
if not high.empty:
    print(f"\n  {len(high)} file(s) with >30% pixels below threshold:")
    print(high[["filename", "fixation_type", "shift_group",
                "tau1_frac_lt_thresh", "tau1_median_ps",
                "tau1_min_photons"]].to_string())

# %% [markdown]
# ## Step 8: Orientation verification
#
# OpenScan is reported to flip the image vertically relative to BH native orientation.
# Compare the SDT summed-intensity map against the SPCImage photon count map
# under several candidate transforms. The active transform is highlighted.
# Update ORIENTATION_TRANSFORM at the top of the notebook if needed.

# %%
_cand = sample_fits[
    sample_fits["fit_params_available"].str.contains("photons", na=False)
]
if _cand.empty:
    _cand = sample_fits
orient_row = _cand.iloc[0]
print(f"Orientation check: {orient_row['filename']}")

sdt_obj       = SdtFile(str(orient_row["filepath"]))
sdt_intensity = sdt_obj.data[0].astype(float).sum(axis=2)

key = best_fit_key(orient_row["filename"], orient_row.get("fixation_type"))
fd  = fit_data.get(key, {}) if key else {}
ref_map = next((fd[p] for p in ("photons", "a1", "a2", "a3") if p in fd), None)

if ref_map is None:
    print("No usable fit export found for orientation check.")
else:
    candidates = {
        "SPCImage (raw)": ref_map,
        "flipud":         np.flipud(ref_map),
        "fliplr":         np.fliplr(ref_map),
        "transpose":      ref_map.T,
    }
    fig, axes = plt.subplots(1, len(candidates) + 1,
                             figsize=(4.5 * (len(candidates) + 1), 4))

    sdt_vmax = float(np.percentile(sdt_intensity[sdt_intensity > 0], 99))
    axes[0].imshow(sdt_intensity, cmap="inferno", vmin=0, vmax=sdt_vmax)
    axes[0].set_title("SDT intensity")
    axes[0].axis("off")

    for ax, (label, img) in zip(axes[1:], candidates.items()):
        pos_pix = img[img > 0]
        vmax = float(np.percentile(pos_pix, 99)) if pos_pix.size > 0 else 1.0
        ax.imshow(img, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
        if label == ORIENTATION_TRANSFORM:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("lime")
                spine.set_linewidth(3)

    plt.suptitle(orient_row["filename"][:70], fontsize=8)
    plt.tight_layout()
    plt.show()

    print(f"SDT shape: {sdt_intensity.shape}   fit export shape: {ref_map.shape}")
    print(f"Active transform: {ORIENTATION_TRANSFORM!r}  (green border)")
    if sdt_intensity.shape != ref_map.shape:
        print("WARNING: shape mismatch -- check scan dimensions in Phase A")

# %% [markdown]
# ## Step 9: Chi-squared quality check
#
# Reduced chi^2 should peak near 1.0 for well-fitted pixels.
# Values << 1 suggest over-fitting or too few photons.
# Values >> 2 indicate poor fit quality and should be masked in Phase E.

# %%
chi2_records = []
for _, row in sample_fits.iterrows():
    key = best_fit_key(row["filename"], row.get("fixation_type"))
    if key is None or key not in fit_data:
        continue
    fd = fit_data[key]
    if "chi2" not in fd:
        continue
    arr  = apply_orientation(fd["chi2"], ORIENTATION_TRANSFORM)
    valid = np.isfinite(arr) & (arr > 0)
    chi2_records.append({
        "filename":      row["filename"],
        "fixation_type": row.get("fixation_type"),
        "chi2_mean":     float(arr[valid].mean()) if valid.any() else np.nan,
        "chi2_median":   float(np.median(arr[valid])) if valid.any() else np.nan,
        "frac_hi":       float((arr[valid] > CHI2_HI).mean()) if valid.any() else np.nan,
        "_arr":          arr,
    })

chi2_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                         for r in chi2_records])

print(f"Chi^2 from {len(chi2_df)} files (thresholds: lo={CHI2_LO}, hi={CHI2_HI}):")
if not chi2_df.empty:
    print(chi2_df.groupby("fixation_type")[["chi2_mean", "chi2_median", "frac_hi"]]
          .mean().round(3).to_string())

# %%
# Chi^2 distributions by fixation type
fix_types = sorted(chi2_df["fixation_type"].dropna().unique())
if fix_types:
    fig, axes = plt.subplots(1, len(fix_types), figsize=(5 * len(fix_types), 4),
                             squeeze=False)
    for ax, ft in zip(axes[0], fix_types):
        recs = [r for r in chi2_records if r["fixation_type"] == ft]
        vals = np.concatenate([r["_arr"][np.isfinite(r["_arr"]) & (r["_arr"] > 0)]
                                for r in recs])
        ax.hist(vals, bins=100, range=(0, 5), density=True, color="steelblue", alpha=0.8)
        ax.axvline(1.0, color="red", lw=1.2, linestyle="--", label="chi^2=1")
        ax.axvspan(0, CHI2_LO, alpha=0.1, color="orange", label="flag zone")
        ax.axvspan(CHI2_HI, 5, alpha=0.1, color="orange")
        ax.set_xlabel("chi^2")
        ax.set_ylabel("density")
        ax.set_title(f"{ft}  (n={len(recs)} files)")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.show()

# %%
# Spatial chi^2 map paired with photon intensity -- one file per fixation type
for ft in fix_types:
    recs = [r for r in chi2_records if r["fixation_type"] == ft]
    if not recs:
        continue
    rec = recs[0]

    # Load matching intensity (SDT sum or photons export)
    row_match = sample_fits[sample_fits["filename"] == rec["filename"]]
    intensity = None
    if not row_match.empty:
        _key = best_fit_key(row_match.iloc[0]["filename"],
                            row_match.iloc[0].get("fixation_type"))
        _fd  = fit_data.get(_key, {}) if _key else {}
        intensity = next((apply_orientation(_fd[p], ORIENTATION_TRANSFORM)
                          for p in ("photons", "a1", "a2", "a3") if p in _fd), None)
        if intensity is None:
            try:
                sdt_obj = SdtFile(str(row_match.iloc[0]["filepath"]))
                intensity = sdt_obj.data[0].astype(float).sum(axis=2)
            except Exception:
                pass

    n_cols = 2 if intensity is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    if intensity is not None:
        vmax_i = float(np.percentile(intensity[intensity > 0], 99)) if (intensity > 0).any() else 1
        axes[0].imshow(intensity, cmap="inferno", vmin=0, vmax=vmax_i)
        axes[0].set_title("intensity (photons)")
        axes[0].axis("off")

    im = axes[-1].imshow(rec["_arr"], cmap="RdYlGn_r", vmin=0.5, vmax=3.0)
    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label="chi^2")
    axes[-1].set_title("chi^2")
    axes[-1].axis("off")

    plt.suptitle(f"{ft}: {rec['filename'][:60]}", fontsize=8)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Step 10: Fit parameter distributions
#
# tau values are in whatever units SPCImage exports (typically ps).
# Amplitude values are normalized (sum = 1) by SPCImage.
#
# Component assignment (SPCImage sorts by lifetime, tau1 < tau2 < tau3):
#   glu:       component 1 is the ~16 ps artifact; components 2 and 3 are NADH
#   form/live: component 1 = shorter NADH, component 2 = longer NADH

# %%
# Step 10: one figure per (fixation_type, shift_group, n_components) combination.
# Splitting by n_comp prevents bimodal/trimodal distributions caused by mixing
# 2-comp and 3-comp fits (tau1 role differs: free NADH vs artifact absorber).
_shift_sort = {"shift=0": 0, "shift!=0": 1}

for fix_type in ["glu", "form", "live"]:
    all_rows = sample_fits[sample_fits["fixation_type"] == fix_type]
    if all_rows.empty:
        continue

    # Index files by (shift_label, n_comp) so each combination gets its own figure.
    # Store (row, fit_set_key) pairs to avoid calling best_fit_key() twice.
    _groups: dict = {}  # (shift_label, n_comp) -> list of (row, key)
    for _, row in all_rows.iterrows():
        fn  = row["filename"]
        fix = row.get("fixation_type")
        key = best_fit_key(fn, fix)
        if key is None or key not in fit_data:
            continue
        _sl = "shift!=0" if fn in shift_nonzero_files else "shift=0"
        _fm = fit_map_df[
            (fit_map_df["sdt_filename"] == fn) &
            (fit_map_df["fit_set_key"]  == key)
        ]
        _nc = (int(_fm.iloc[0]["n_components"])
               if not _fm.empty and "n_components" in _fm.columns
               else None)
        _gkey = (_sl, _nc)
        _groups.setdefault(_gkey, []).append((row, key))

    # Iterate in order: shift=0 before shift!=0, lower n_comp first.
    for (_sl, _nc) in sorted(_groups, key=lambda x: (_shift_sort.get(x[0], 9), x[1] or 0)):
        _entries = _groups[(_sl, _nc)]
        _nc_str  = f"{_nc}-comp" if _nc else "?-comp"

        # For 2-comp glu fits tau3/a3 are absent; trim params_show accordingly.
        if fix_type == "glu" and _nc != 2:
            params_show = ["tau1", "tau2", "tau3", "a1", "a2", "a3", "chi2"]
        elif fix_type == "glu":
            params_show = ["tau1", "tau2", "a1", "a2", "chi2"]
        else:
            params_show = ["tau1", "tau2", "a1", "a2", "chi2"]

        fig, axes = plt.subplots(1, len(params_show),
                                 figsize=(3 * len(params_show), 3), squeeze=False)

        for ax, param in zip(axes[0], params_show):
            arrays = []
            for _row, _key in _entries:
                fd = fit_data[_key]
                if param not in fd:
                    continue
                arr = apply_orientation(fd[param], ORIENTATION_TRANSFORM).ravel()
                arrays.append(arr[np.isfinite(arr)])

            if not arrays:
                ax.set_title(f"{param}\n(no data)")
                continue

            vals = np.concatenate(arrays)
            if param.startswith("a"):
                lo, hi = 0.0, 1.0
            elif param == "chi2":
                lo, hi = 0.0, 5.0
            else:
                lo = float(np.percentile(vals[vals > 0], 1)) if (vals > 0).any() else 0
                hi = float(np.percentile(vals, 99))
                if hi <= lo:
                    hi = lo + 1.0   # constant array (e.g. all-zero shift map)

            ax.hist(vals, bins=80, range=(lo, hi), density=True,
                    color="steelblue", alpha=0.8)
            ax.set_xlabel(param)
            ax.set_title(f"{param}\nmedian={np.nanmedian(vals):.4g}")

        fig.suptitle(
            f"Fit parameters -- {fix_type}  ({len(_entries)} files)"
            f"  [{_sl}, {_nc_str}]",
            fontsize=10,
        )
        plt.tight_layout()
        plt.show()

# %%
# Tau1 realism summary plot: fraction of pixels below TAU1_REALISM_PS,
# shown per file as a scatter, grouped by fixation_type and shift_group.
# High fractions in glu shift=0 indicate the short component is absorbing
# the fixation artifact.  Expect lower fractions with 3-comp or free-shift fits.
if not tau1_df.empty and tau1_df["tau1_frac_lt_thresh"].notna().any():
    _fix_types = sorted(tau1_df["fixation_type"].dropna().unique())
    _sg_colors = {"shift=0": "steelblue", "shift!=0": "darkorange"}

    fig, axes = plt.subplots(1, len(_fix_types),
                             figsize=(4 * len(_fix_types), 4), squeeze=False)
    for ax, ft in zip(axes[0], _fix_types):
        sub = tau1_df[tau1_df["fixation_type"] == ft].dropna(
            subset=["tau1_frac_lt_thresh"]
        )
        for sg, color in _sg_colors.items():
            s = sub[sub["shift_group"] == sg]
            if s.empty:
                continue
            ax.scatter(
                [sg] * len(s), s["tau1_frac_lt_thresh"],
                color=color, alpha=0.7, s=30, label=sg,
            )
            ax.axhline(s["tau1_frac_lt_thresh"].mean(), color=color,
                       lw=1.5, linestyle="--", alpha=0.8)
        ax.axhline(0.3, color="red", lw=1, linestyle=":", alpha=0.6,
                   label="30% flag level")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"frac tau1 < {TAU1_REALISM_PS:.0f} ps")
        ax.set_title(ft)
        ax.legend(fontsize=7)
    fig.suptitle("Tau1 realism check by fixation type and shift group",
                 fontsize=10)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Step 11: Amplitude ratios
#
# Per-pixel ratio map and distribution across all files of each fixation type.
# Ratio convention set by RATIO_CFG at the top:
#   glu:       a2/(a2+a3)  -- fraction of non-artifact signal in shorter component
#   form/live: a1/(a1+a2)  -- fraction of total NADH signal in shorter component
#
# Results are shown separately for shift=0 and shift!=0 fits so the effect of
# allowing a free IRF shift can be assessed.

# %%
for fix_type, (num_p, den_p) in RATIO_CFG.items():
    all_rows = sample_fits[sample_fits["fixation_type"] == fix_type]
    if all_rows.empty:
        continue

    _r0  = all_rows[~all_rows["filename"].isin(shift_nonzero_files)]
    _rnz = all_rows[ all_rows["filename"].isin(shift_nonzero_files)]
    shift_groups_r = []
    if not _r0.empty:
        shift_groups_r.append(("shift=0",  _r0))
    if not _rnz.empty:
        shift_groups_r.append(("shift!=0", _rnz))

    label = f"{num_p}/({num_p}+{den_p})"

    for shift_label, rows in shift_groups_r:
        ratio_maps = []
        for _, row in rows.iterrows():
            key = best_fit_key(row["filename"], row.get("fixation_type"))
            if key is None or key not in fit_data:
                continue
            fd = fit_data[key]
            if num_p not in fd or den_p not in fd:
                continue
            a_n = apply_orientation(fd[num_p], ORIENTATION_TRANSFORM)
            a_d = apply_orientation(fd[den_p], ORIENTATION_TRANSFORM)
            denom = a_n + a_d
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(denom > 0, a_n / denom, np.nan)
            ratio_maps.append(ratio)

        if not ratio_maps:
            print(f"  {fix_type} [{shift_label}]: no data for {num_p}/({num_p}+{den_p})")
            continue

        flat = np.concatenate([m.ravel() for m in ratio_maps])
        flat = flat[np.isfinite(flat)]

        fig, (ax_map, ax_hist) = plt.subplots(1, 2, figsize=(10, 4))

        im = ax_map.imshow(ratio_maps[0], cmap="RdBu_r", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax_map, label=label)
        ax_map.set_title(f"{fix_type} [{shift_label}]: ratio map (first file)")
        ax_map.axis("off")

        ax_hist.hist(flat, bins=80, range=(0, 1), density=True,
                     color="steelblue", alpha=0.8)
        ax_hist.set_xlabel(label)
        ax_hist.set_ylabel("density")
        ax_hist.set_title(
            f"{fix_type} [{shift_label}]: n={len(ratio_maps)} files\n"
            f"mean={flat.mean():.3f}  median={np.median(flat):.3f}  "
            f"std={flat.std():.3f}"
        )
        plt.tight_layout()
        plt.show()

# %%
# Save all Phase C per-file quality stats to fit_qc_summary.csv.
# Merges chi2 stats, shift stats, and tau1 realism stats on filename.
_qc = chi2_df.copy()

_shift_cols = ["filename", "fit_set_key", "has_shift_export",
               "shift_nonzero_px", "shift_max_abs_ps",
               "shift_mean_ps", "shift_std_ps"]
if "shift_meta_df" in dir() and not shift_meta_df.empty:
    _qc = _qc.merge(
        shift_meta_df[[c for c in _shift_cols if c in shift_meta_df.columns]],
        on="filename", how="left",
    )

_tau1_cols = ["filename", "tau1_n_valid", "tau1_frac_lt_thresh",
              "tau1_median_ps", "tau1_min_photons"]
if "tau1_df" in dir() and not tau1_df.empty:
    _qc = _qc.merge(
        tau1_df[[c for c in _tau1_cols if c in tau1_df.columns]],
        on="filename", how="left",
    )

_qc_path = results_dir / "fit_qc_summary.csv"
_qc.to_csv(_qc_path, index=False)
print(f"Saved fit_qc_summary: {len(_qc)} rows, {len(_qc.columns)} columns -> {_qc_path}")
print(f"  Columns: {_qc.columns.tolist()}")

# %% [markdown]
# ## Step 12: Position annotation template
#
# Each field of view needs a morphology label before group analysis.
# Labels: single_cell | colony_deep | colony_edge | colony_island | no_cells | other
#
# Matching sets (same session + frame_index): transmitted, FLIM 457s50, FLIM 535s50.
# The annotation is at the field-of-view level (one label per position),
# so matching files share the same frame_index within a session+sample.
#
# Workflow:
#   1. Run this cell -> writes results/position_annotation.csv (blank annotations)
#   2. Open in Excel / LibreOffice, fill in the "annotation" column
#   3. Downstream notebooks read and merge by filename

# %%
ANNOTATION_OPTIONS = ["single_cell", "colony_deep", "colony_edge",
                      "colony_island", "no_cells", "other"]

annot_cols = ["filename", "session_root", "session_date", "fixation_type",
              "cell_type", "sample_index", "frame_index", "em_filter_nm",
              "acquisition_time"]
annot_cols_present = [c for c in annot_cols if c in sdt_df.columns]

annot_df = (
    sdt_df[sdt_df["file_type"] == "sample"][annot_cols_present]
    .sort_values(["session_root", "sample_index", "frame_index", "em_filter_nm"],
                 na_position="last")
    .copy()
    .reset_index(drop=True)
)
annot_df["annotation"] = ""    # user fills: one of ANNOTATION_OPTIONS
annot_df["notes"]      = ""    # free text

annot_path = results_dir / "position_annotation.csv"

if annot_path.exists():
    # Merge with existing file: preserve filled annotations, rebuild metadata columns.
    # Uses filename as the join key -- safe because sdt_df is already deduplicated above.
    existing = pd.read_csv(annot_path).drop_duplicates(subset="filename", keep="first")
    saved_cols = ["filename", "annotation", "notes"]
    saved = existing[[c for c in saved_cols if c in existing.columns]]
    merged = annot_df.merge(saved, on="filename", how="left", suffixes=("", "_saved"))
    # Prefer saved values over blanks
    for col in ("annotation", "notes"):
        saved_col = col + "_saved"
        if saved_col in merged.columns:
            merged[col] = merged[saved_col].where(
                merged[saved_col].notna() & (merged[saved_col] != ""),
                merged[col]
            )
            merged.drop(columns=[saved_col], inplace=True)
    merged.to_csv(annot_path, index=False)
    n_filled = int((merged["annotation"].notna() & (merged["annotation"] != "")).sum())
    print(f"Updated {annot_path}: {n_filled}/{len(merged)} annotated")
else:
    annot_df.to_csv(annot_path, index=False)
    print(f"Wrote annotation template: {annot_path}  ({len(annot_df)} rows)")

print(f"\nValid annotation values: {ANNOTATION_OPTIONS}")
print("Fill the 'annotation' column in Excel, then re-run downstream notebooks.")
print("\nFirst few rows:")
print(annot_df[["filename", "fixation_type", "frame_index",
                "em_filter_nm", "annotation"]].head(10).to_string())

# %% [markdown]
# ### Step 12b: Side-by-side intensity browser
#
# Loop through matching positions (same session + sample_index + frame_index + pockels)
# and display transmitted-light (TD), 457/50 nm, and 535/50 nm intensity images side
# by side. Pockels is included in the group key so that files at different power
# settings (e.g. poc0p25 vs poc0p6) each appear as a separate position row.
#
# Set SHOW_ONLY_UNANNOTATED = True (default) to skip positions that already have
# an annotation in position_annotation.csv.
#
# Tip: run this once, keep the figure window open, then fill position_annotation.csv.

# %%
SHOW_ONLY_UNANNOTATED = True   # False to show all positions (including annotated)

def _channel_label(em_nm) -> str:
    """Return TD / 457 / 535 / or numeric label from em_filter_nm."""
    if pd.isna(em_nm):
        return "unknown"
    s = str(em_nm).upper()
    if "TD" in s or "TRANS" in s:
        return "TD"
    try:
        v = float(em_nm)
        if 450 <= v <= 465:
            return "457"
        if 528 <= v <= 545:
            return "535"
        return f"{int(v)}nm"
    except (ValueError, TypeError):
        return s[:8]


sample_sdt = sdt_df[sdt_df["file_type"] == "sample"].copy()
sample_sdt["_ch"] = sample_sdt["em_filter_nm"].apply(_channel_label)

# Include pockels in the grouping key so different power settings are shown
# separately (each warrants its own annotation entry).
_group_cols = [c for c in ["session_root", "sample_index", "frame_index", "pockels"]
               if c in sample_sdt.columns]
grouped = sample_sdt.groupby(_group_cols, dropna=False)
print(f"Positions found: {len(grouped)}\n")

# Pre-load existing annotations so we can skip already-annotated positions.
_existing_annot: dict = {}
if SHOW_ONLY_UNANNOTATED and annot_path.exists():
    _ea = pd.read_csv(annot_path)
    _existing_annot = dict(zip(
        _ea["filename"].astype(str),
        _ea["annotation"].fillna("").astype(str),
    ))
    print(f"Loaded {len(_existing_annot)} existing annotations; "
          "will skip fully-annotated positions.")

_CH_ORDER = ["TD", "457", "535"]
_n_shown = _n_skipped_annot = 0

for pos_key, grp in grouped:
    # If SHOW_ONLY_UNANNOTATED, skip groups where every file already has a
    # non-empty annotation.
    if SHOW_ONLY_UNANNOTATED and _existing_annot:
        group_fns = grp["filename"].astype(str).tolist()
        all_done  = all(_existing_annot.get(fn, "") != "" for fn in group_fns)
        if all_done:
            _n_skipped_annot += 1
            continue

    ch_map = {}
    for _, row in grp.iterrows():
        ch = row["_ch"]
        if ch not in ch_map:        # keep first occurrence per channel
            ch_map[ch] = row

    present = [c for c in _CH_ORDER if c in ch_map]
    present += [c for c in ch_map if c not in _CH_ORDER]
    if not present:
        continue

    pos_str = " / ".join(
        str(k) for k in (pos_key if isinstance(pos_key, tuple) else (pos_key,))
    )

    fig, axes = plt.subplots(1, len(present),
                             figsize=(4.5 * len(present), 4), squeeze=False)
    axes = axes[0]
    fig.suptitle(f"Position: {pos_str}", fontsize=9)

    for ax, ch in zip(axes, present):
        row = ch_map[ch]
        try:
            sdt_obj   = SdtFile(str(row["filepath"]))
            intensity = sdt_obj.data[0].astype(float).sum(axis=2)
            pos_pix   = intensity[intensity > 0]
            vmax      = float(np.percentile(pos_pix, 99)) if pos_pix.size > 0 else 1.0
            ax.imshow(intensity, cmap="inferno", vmin=0, vmax=vmax)
        except Exception as exc:
            ax.text(0.5, 0.5, f"load error:\n{exc}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, color="red")
        fn_short = str(row.get("filename", ""))[:40]
        ax.set_title(f"{ch}\n{fn_short}", fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    fn_list = [ch_map[c]["filename"] for c in present]
    print(f"  {pos_str}  ->  {fn_list}")
    _n_shown += 1

print(f"\nShown: {_n_shown}  |  Skipped (already annotated): {_n_skipped_annot}")

# %% [markdown]
# ### Step 12c: Propagate laser-damage flag from annotation notes
#
# If you noted "laser damage" (or similar) in the notes column while annotating,
# this cell adds a boolean `laser_damage` column to position_annotation.csv so
# downstream notebooks can easily exclude those files.

# %%
if annot_path.exists():
    _ann = pd.read_csv(annot_path)
    if "notes" in _ann.columns:
        _damage_kw = re.compile(
            r"laser.?damage|damaged|overexposed|bleach|burn",
            re.IGNORECASE,
        )
        _ann["laser_damage"] = (
            _ann["notes"].fillna("").apply(lambda n: bool(_damage_kw.search(n)))
        )
        _ann.to_csv(annot_path, index=False)
        _n_damage = int(_ann["laser_damage"].sum())
        print(f"laser_damage column written to {annot_path}: "
              f"{_n_damage} file(s) flagged")
    else:
        print("No 'notes' column found in annotation CSV; skipping laser_damage flag.")
else:
    print(f"Annotation file not found: {annot_path}")

# %% [markdown]
# ## Summary
#
# At this point you have confirmed:
# - Which fit parameters are available per fixation type (using fit_map.csv)
# - best_fit_key() selects preferred n_components per fixation type
# - Shift is zero (or flagged if not)
# - Orientation: SDT and SPCImage exports are already aligned (ORIENTATION_TRANSFORM = None)
# - Chi^2 distribution and spatial pattern paired with intensity image
# - tau and amplitude distributions per fixation type
# - Amplitude ratio maps: a2/(a2+a3) for glu, a1/(a1+a2) for form/live
#
# Inputs read:
#   results/sdt_metadata_cal.csv
#   results/filepath_map.csv
#   results/fit_map.csv  -- fit-set/SDT pairs with n_components (written by Phase A)
#
# Saved:
#   results/fit_qc_summary.csv  -- all Phase C per-file quality stats (merged)
#     columns: filename, fixation_type,
#              chi2_mean, chi2_median, frac_hi,
#              fit_set_key, has_shift_export, shift_nonzero_px,
#                shift_max_abs_ps, shift_mean_ps, shift_std_ps,
#              tau1_n_valid, tau1_frac_lt_thresh, tau1_median_ps, tau1_min_photons
#   results/position_annotation.csv  -- fill annotation column before Phase D
#
# Before Phase D:
#   1. Fill in position_annotation.csv (single_cell / colony_deep / colony_edge /
#      colony_island / no_cells / other)
#   2. Thresholding (intensity mask per image) is Step 9 of Phase D

# %%

# %%
import winsound as _ws, time as _t
for _ in range(3):
    _ws.Beep(1000, 400)
    _t.sleep(1)
