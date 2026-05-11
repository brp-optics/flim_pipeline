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

# -- Configuration ----------------------------------------------------------
WIN_DATA_DIRS = [
    Path(r"E:\18_RK_Circadian\data\raw\20260429_KPC_fixed_dishes_on_SLIM"),
    Path(r"E:\18_RK_Circadian\data\raw\20260501_KPC_fixed_dishes_on_SLIM"),
]
LIN_DATA_DIRS = [
    Path("/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260429_KPC_fixed_dishes_on_SLIM"),
    Path("/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260501_KPC_fixed_dishes_on_SLIM"),
]
CURRENT_OS = "Win"
data_dirs = WIN_DATA_DIRS if CURRENT_OS == "Win" else LIN_DATA_DIRS

CHI2_LO = 0.8    # flag pixels below this
CHI2_HI = 2.0    # flag pixels above this

# Orientation of SPCImage .asc arrays relative to the SDT pixel array.
# OpenScan flips vertically vs BH. Confirmed in Step 8; update if wrong.
# Options: None | "flipud" | "fliplr" | "transpose" | "flipud+transpose"
ORIENTATION_TRANSFORM = "flipud"

# Amplitude ratio definition per fixation type:
#   glu:       a2/(a2+a3)  -- skip ultra-short artifact component a1 (~16 ps)
#   form/live: a1/(a1+a2)  -- standard free/total NADH ratio
RATIO_CFG = {
    "glu":  ("a2", "a3"),
    "form": ("a1", "a2"),
    "live": ("a1", "a2"),
}

# %%
# -- Load Phase B outputs ---------------------------------------------------
results_dir = Path("../results")

sdt_df = pd.read_csv(results_dir / "sdt_metadata_cal.csv")
fp_map = pd.read_csv(results_dir / "filepath_map.csv")
sdt_df = sdt_df.merge(fp_map[["filename", "filepath"]], on="filename", how="left")
sdt_df["filepath"]         = sdt_df["filepath"].map(Path)
sdt_df["acquisition_time"] = pd.to_datetime(sdt_df["acquisition_time"])
sdt_df["has_fit_export"]   = sdt_df["has_fit_export"].fillna(False).astype(bool)

print(f"Loaded {len(sdt_df)} SDT files")
print(sdt_df["file_type"].value_counts().to_string())

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


asc_paths = [fp for d in data_dirs for fp in sorted(d.rglob("*.asc"))]
print(f"\nFound {len(asc_paths)} .asc files -- loading...")

fit_data: dict[str, dict] = {}
n_failures = 0
for fp in asc_paths:
    result = _load_asc(fp)
    if result is None:
        n_failures += 1
        continue
    base = _strip_param_suffix(fp.stem)
    fit_data.setdefault(base, {})[result["param"]] = result["data"]
    fit_data[base][f"_path_{result['param']}"] = fp

print(f"Grouped into {len(fit_data)} image sets  ({n_failures} parse failures)")

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
    base = row.get("fit_base_stem")
    if not isinstance(base, str) or base not in fit_data:
        continue
    fd = fit_data[base]
    inv_rows.append({
        "filename":      row["filename"],
        "fixation_type": row.get("fixation_type"),
        **{p: (p in fd) for p in ALL_PARAMS},
    })

inv_df = pd.DataFrame(inv_rows)

print("Parameter availability by fixation type:")
for fix, grp in inv_df.groupby("fixation_type", dropna=False):
    n = len(grp)
    print(f"\n  {fix} ({n} files):")
    for p in ALL_PARAMS:
        if p in grp.columns and grp[p].any():
            print(f"    {p:12s}: {int(grp[p].sum())}/{n}")

# %%
# Shift check: shift was fixed to 0 in SPCImage -- verify
shift_issues = []
for _, row in sample_fits.iterrows():
    base = row.get("fit_base_stem")
    if not isinstance(base, str) or base not in fit_data:
        continue
    fd = fit_data[base]
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

base = orient_row.get("fit_base_stem", "")
fd   = fit_data.get(base, {})
ref_map = fd.get("photons") or fd.get("a1") or fd.get("a2")

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
    base = row.get("fit_base_stem")
    if not isinstance(base, str) or base not in fit_data:
        continue
    fd = fit_data[base]
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
# Spatial chi^2 map -- one representative file per fixation type
if fix_types:
    fig, axes = plt.subplots(1, len(fix_types),
                             figsize=(5 * len(fix_types), 4), squeeze=False)
    for ax, ft in zip(axes[0], fix_types):
        recs = [r for r in chi2_records if r["fixation_type"] == ft]
        if not recs:
            continue
        im = ax.imshow(recs[0]["_arr"], cmap="RdYlGn_r", vmin=0.5, vmax=3.0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="chi^2")
        ax.set_title(f"{ft}: {recs[0]['filename'][:30]}")
        ax.axis("off")
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
for fix_type in ["glu", "form", "live"]:
    rows = sample_fits[sample_fits["fixation_type"] == fix_type]
    if rows.empty:
        continue

    params_show = (["tau1", "tau2", "tau3", "a1", "a2", "a3", "chi2"]
                   if fix_type == "glu"
                   else ["tau1", "tau2", "a1", "a2", "chi2"])

    fig, axes = plt.subplots(1, len(params_show),
                             figsize=(3 * len(params_show), 3), squeeze=False)

    for ax, param in zip(axes[0], params_show):
        arrays = []
        for _, row in rows.iterrows():
            base = row.get("fit_base_stem")
            if not isinstance(base, str) or base not in fit_data:
                continue
            fd = fit_data[base]
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

        ax.hist(vals, bins=80, range=(lo, hi), density=True,
                color="steelblue", alpha=0.8)
        ax.set_xlabel(param)
        ax.set_title(f"{param}\nmedian={np.nanmedian(vals):.4g}")

    fig.suptitle(f"Fit parameters -- {fix_type}  ({len(rows)} files)", fontsize=10)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Step 11: Amplitude ratios
#
# Per-pixel ratio map and distribution across all files of each fixation type.
# Ratio convention set by RATIO_CFG at the top:
#   glu:       a2/(a2+a3)  -- fraction of non-artifact signal in shorter component
#   form/live: a1/(a1+a2)  -- fraction of total NADH signal in shorter component

# %%
for fix_type, (num_p, den_p) in RATIO_CFG.items():
    rows = sample_fits[sample_fits["fixation_type"] == fix_type]
    if rows.empty:
        continue

    ratio_maps = []
    for _, row in rows.iterrows():
        base = row.get("fit_base_stem")
        if not isinstance(base, str) or base not in fit_data:
            continue
        fd = fit_data[base]
        if num_p not in fd or den_p not in fd:
            continue
        a_n = apply_orientation(fd[num_p], ORIENTATION_TRANSFORM)
        a_d = apply_orientation(fd[den_p], ORIENTATION_TRANSFORM)
        denom = a_n + a_d
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(denom > 0, a_n / denom, np.nan)
        ratio_maps.append(ratio)

    if not ratio_maps:
        print(f"  {fix_type}: no data for {num_p}/({num_p}+{den_p})")
        continue

    label = f"{num_p}/({num_p}+{den_p})"
    flat  = np.concatenate([m.ravel() for m in ratio_maps])
    flat  = flat[np.isfinite(flat)]

    fig, (ax_map, ax_hist) = plt.subplots(1, 2, figsize=(10, 4))

    im = ax_map.imshow(ratio_maps[0], cmap="RdBu_r", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_map, label=label)
    ax_map.set_title(f"{fix_type}: ratio map (first file)")
    ax_map.axis("off")

    ax_hist.hist(flat, bins=80, range=(0, 1), density=True,
                 color="steelblue", alpha=0.8)
    ax_hist.set_xlabel(label)
    ax_hist.set_ylabel("density")
    ax_hist.set_title(f"{fix_type}: all files (n={len(ratio_maps)})\n"
                      f"mean={flat.mean():.3f}  median={np.median(flat):.3f}  "
                      f"std={flat.std():.3f}")
    plt.tight_layout()
    plt.show()

# %%
# Save chi^2 summary to results
chi2_save = chi2_df.copy()
chi2_save.to_csv(results_dir / "chi2_summary.csv", index=False)
print(f"Saved chi^2 summary to {results_dir / 'chi2_summary.csv'}")

# %% [markdown]
# ## Summary
#
# At this point you have confirmed:
# - Which fit parameters are available per fixation type
# - Shift is zero (or flagged if not)
# - Orientation transform needed to align .asc arrays with SDT pixels
# - Chi^2 distribution and spatial pattern (used for quality masking in Phase E)
# - tau and amplitude distributions per fixation type
# - Amplitude ratio maps: a2/(a2+a3) for glu, a1/(a1+a2) for form/live
#
# Saved: results/chi2_summary.csv
#
# Next: Phase D -- calibrated phasor analysis per sample file
