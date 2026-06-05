# %% [markdown]
#
# # Full analysis pipeline - Summary
#
# FLIM analysis pipeline
#
# # Phase A -- Inventory and metadata
#
# 1. **Discover all files.**
#    Recursively scan the data directory for `.sdt`, `.spc`, and ascii fit-export files.
#    -> *raw file list*
#
# 2. **Build metadata DataFrame.**
#    Parse sdt headers to extract `acquisition_time`, scan dimensions, bin width, rep rate. Tag each file with `file_type` (IRF-before, IRF-after, chroma-before, chroma-after, sample), `cell_appearance`, and `colony_size`. Match each sdt with its corresponding ascii fit export.
#    -> *df with one row per acquisition, sorted by acquisition\_time*
#
# 3. **Load ascii fit exports.**
#    Parse a1, a2, a3, tau1, tau2, tau3, photon count, chi^2 per pixel from ascii files. Store as numpy arrays keyed by filepath stem.
#    -> *dict of fit arrays per file*
#
# ## Phase B -- Instrument calibration
#
# 4. **Compare bracketing IRFs.**
#    Load before and after IRF sdt files. Overlay decay curves. Quantify peak channel position, FWHM, and integrated counts for each. Compute drift (delta_peak, delta_FWHM) between the two.
#    -> *IRF drift metrics; decision on interpolation vs averaging*
#
#    > WARNING: If peak shift > 1 time bin: flag session, consider per-file IRF interpolation by acquisition\_time.
#
# 5. **Compute phasor calibration from chroma slides.**
#    Load before/after chroma sdt files. Compute raw phasor (G, S) for each. Compare against the theoretical position for the known fluorophore lifetime at omega. Derive phase rotation and modulation correction factors. Check whether calibration drifted between before/after.
#    -> *phasor\_cal\_phase, phasor\_cal\_mod (per calibration timepoint)*
#
# 6. **Assign calibration to each sample.**
#    For each sample file, use its `acquisition_time` to determine which IRF and phasor calibration to apply (nearest, averaged, or interpolated). Record `irf_source` and `phasor_cal_source` columns in df.
#    -> *df updated with calibration provenance columns*
#
# ## Phase C -- Data integrity checks
#
# 7. **Verify orientation match between sdt and fit exports.**
#    For a few representative files, generate an intensity image from the sdt (sum over time bins) and compare against the photon count map from the ascii export. Check for flips (LR, UD) and transposition. Determine the transform needed, if any.
#    -> *orientation\_transform (None, "flipud", "fliplr", "transpose", etc.)*
#
# 8. **Spot-check exported fits against raw data.**
#    Select 3-5 bright pixels from a representative file. Fit mono-exponential from the raw sdt decay using your own fitter. Compare recovered tau and amplitude against the corresponding ascii export values. Verify agreement within tolerance.
#    -> *confidence that ascii exports are trustworthy (or a list of discrepancies)*
#
# ## Phase D -- Phasor analysis
#
# 9. **Compute intensity mask.**
#    For each sdt file, sum photon counts over time bins per pixel. Threshold to create a boolean mask excluding background pixels. Record `fraction_above_threshold` in df.
#    -> *mask array per file*
#
# 10. **Compute calibrated phasor per pixel.**
#     For each sdt file, compute raw (G, S) at the laser rep frequency using phasorpy. Apply the calibration correction from step 5. Store calibrated (G, S) arrays. Apply intensity mask before any summary statistics.
#     -> *G, S arrays per file; G\_mean, S\_mean columns in df*
#
# 11. **Phasor sanity check.**
#     Overlay phasor cloud on the universal semicircle. Verify the chroma reference point lands at its expected position after calibration. Check whether sample points fall on, inside, or outside the semicircle (indicating single-exp, multi-exp, or calibration error respectively).
#     -> *phasor semicircle plot; flag files with points outside the semicircle*
#
# 12. **Plot phasors grouped by file type.**
#     Generate one phasor plot per group (`file_type` / `cell_appearance` / `colony_size`). Color-code by group for quick visual feedback to collaborator.
#     -> *phasor summary figures for sharing*
#
# ## Phase E -- Exported fit analysis
#
# 13. **Apply orientation correction to fit arrays.**
#     Using the transform determined in step 7, flip/transpose the ascii fit arrays so they align pixel-for-pixel with the sdt data and phasor arrays.
#     -> *aligned fit arrays*
#
# 14. **Threshold on fit quality.**
#     Create a fit-quality mask from reduced chi^2 (e.g. 0.8 < chi^2 < 1.5). Optionally intersect with the intensity mask from step 9. Inspect the chi^2 map spatially for systematic patterns.
#     -> *fit\_quality\_mask per file; chi2\_mean column in df*
#
# 15. **Compute derived lifetime metrics.**
#     From the masked fit parameters, compute amplitude-weighted mean lifetime per pixel: tau\_mean = sum(a\_i * tau\_i) / sum(a\_i). Compute fractional contributions f\_i = a\_i * tau\_i / sum(a\_j * tau\_j). Add per-image summary statistics to df.
#     -> *tau\_mean array per file; tau\_mean\_mean, f1\_mean, f2\_mean columns in df*
#
# ## Phase F -- Group comparisons
#
# 16. **Check for photobleaching.**
#     If sdt files contain frame data, compare early vs late frame intensity. Flag files with significant bleaching (>10% decline) in df.
#     -> *bleaching\_flag, bleaching\_pct columns in df*
#
# 17. **Intra-group distributions.**
#     For each group, plot per-pixel lifetime distributions (histograms or KDEs) from each image overlaid. Assess within-group consistency: do images from the same condition look similar?
#     -> *intra-group overlay plots*
#
# 18. **Inter-group comparisons.**
#     Compare groups using per-image summary statistics (one value per image, not per pixel). Violin or box plots of tau\_mean, fractional contributions, phasor position. Statistical tests with the image (or colony) as the unit of analysis, not the pixel.
#     -> *inter-group comparison figures; statistical test results*
#
# 19. **Cross-validate phasor vs fit results.**
#     Compare phasor-derived lifetimes (tau\_phi, tau\_mod from step 10) against fit-derived mean lifetimes (step 15) on the same pixels. Agreement strengthens confidence; disagreement highlights model mismatch or calibration issues.
#     -> *phasor-vs-fit correlation plot; Bland-Altman plot*
#
# 20. **Export results.**
#     Save the final df as CSV. Export lifetime maps and phasor arrays as TIFF or HDF5. Package summary figures for collaborator.
#     -> *results/ directory with CSV, TIFFs, figures*
#
# %%

## Phase A -- Inventory and metadata
# 
# 1. **Discover all files.**
#    Recursively scan the data directory for `.sdt`, `.spc`, and ascii fit-export files.
#    -> *raw file list*
# 

from pathlib import Path
import pandas as pd
import numpy as np
import re
import warnings
from datetime import datetime
from sdtfile import SdtFile

# -- Configuration ----------------------------------------------------------
# Phase A is the AUTHORITATIVE source of session directories for the whole
# pipeline.  Edit the two lists below when adding a session, then re-run this
# notebook.  Phase A writes config/data_dirs.yaml and downstream phases read
# it via src.config.get_data_dirs().
#
# Listing both OS variants makes the notebook portable: get_data_dirs() picks
# whichever set exists on the current machine.

WIN_DATA_DIRS = [
    r"E:\18_RK_Circadian\data\raw\20260429_KPC_fixed_dishes_on_SLIM",
    r"E:\18_RK_Circadian\data\raw\20260501_KPC_fixed_dishes_on_SLIM",
    r"E:\18_RK_Circadian\data\raw\20260509_KPC_fixed_dishes_on_SLIM",
    r"E:\18_RK_Circadian\data\raw\20260508_KPC_live_on_SLIM",
    r"E:\18_RK_Circadian\data\raw\20260517_KPC_live_on_SLIM",
    r"E:\18_RK_Circadian\data\raw\20260520_KPC_fixed_dishes_SLIM",
    r"E:\18_RK_Circadian\data\raw\20260521_KPC_live_SLIM",
    r"E:\18_RK_Circadian\data\raw\20260522_KPC_fixed_dishes_SLIM",
]
LIN_DATA_DIRS = [
    "/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260429_KPC_fixed_dishes_on_SLIM",
    "/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260501_KPC_fixed_dishes_on_SLIM",
    "/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260509_KPC_fixed_dishes_on_SLIM",
    "/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260508_KPC_live_on_SLIM",
    "/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260517_KPC_live_on_SLIM",
    "/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260520_KPC_fixed_dishes_SLIM",
    "/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260521_KPC_live_SLIM",
    "/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260522_KPC_fixed_dishes_SLIM",
]

import sys as _sys_cfg_a
_sys_cfg_a.path.insert(0, str(Path("..").resolve()))
from src.config import save_data_dirs, get_data_dirs

# Persist the lists so downstream phases see what Phase A saw.
save_data_dirs(WIN_DATA_DIRS, LIN_DATA_DIRS)

# Then use the standard read path -- guarantees Phase A sees exactly what
# get_data_dirs() will return downstream (existence-filtered, OS-correct).
data_dirs = get_data_dirs()

for d in data_dirs:
    print(f"  Found: {d}  ({sum(1 for _ in d.rglob('*'))} files/dirs)")


# %% [markdown]
# # Phase A: Data inventory and loading
#
# Load all `.sdt` and `.asc` fit-export files from the acquisition sessions.
# Build a metadata DataFrame with one row per acquisition, sorted by time.
#
# **Data directories:**
# - `20260429_KPC_fixed_dishes_on_SLIM`
# - `20260501_KPC_fixed_dishes_on_SLIM`
# - `20260509_KPC_fixed_dishes_on_SLIM`
# - `20260508_KPC_live_on_SLIM`
# - `20260517_KPC_live_on_SLIM`
# - `20260520_KPC_fixed_dishes_SLIM`
# - `20260521_KPC_live_SLIM`
# - `2060522_KPC_fixed_dishes_SLIM`
#
# **Filename conventions:**
# ```
# {index}_{sample}_{ex_wavelength}nm_{ex_power}mW_u{usb}_poc{pockels}_{objective}_{em_filter}_g{gain}_z{zoom}_{pixels}pix_{mode}_{frames}f_{frame_idx}.sdt
# {index}_{sample}_{ex_wavelength}nm_{ex_power}mW_u{usb}_poc{pockels}_{objective}_{em_filter}_g{gain}_z{zoom}_{pixels}pix_{mode}_{frames}f_{frame_idx}_{variable}.asc
# ```

# %% [markdown]
# ## Step 1: Discover all files
#
# Find `.sdt` files (raw TCSPC data) and `.asc` files (SPCImage fit exports).

# %%
def find_data_files(roots: list[Path]) -> pd.DataFrame:
    """Recursively find all .sdt, .asc, and .irf files under the given roots."""
    _cat = {".sdt": "sdt", ".asc": "asc", ".irf": "irf"}
    records = []
    for root in roots:
        for fp in sorted(root.rglob("*")):
            if not fp.is_file():
                continue
            ext = fp.suffix.lower()
            if ext not in _cat:
                continue
            records.append({
                "filepath": fp,
                "filename": fp.name,
                "stem": fp.stem,
                "suffix": ext,
                "category": _cat[ext],
                "rel_dir": str(fp.relative_to(root).parent),
                "session_root": root.name,
                "size_mb": fp.stat().st_size / 1e6,
                "mtime": datetime.fromtimestamp(fp.stat().st_mtime),
            })
    return pd.DataFrame(records)


files_df = find_data_files(data_dirs)
print(f"Found {len(files_df)} files total:")
print(files_df["category"].value_counts().to_string())

sdt_files = files_df[files_df["category"] == "sdt"].copy()
asc_files = files_df[files_df["category"] == "asc"].copy()
irf_files = files_df[files_df["category"] == "irf"].copy()
print(f"\nSDT: {len(sdt_files)}  |  ASC: {len(asc_files)}  |  IRF exports: {len(irf_files)}")

# %% [markdown]
# ## Step 2: Parse filename metadata
#
# The filenames encode a rich set of acquisition parameters.
# We extract them with regex, falling back to `None` for unrecognised fields
# rather than failing.

# %%
# -- Filename metadata extractors -------------------------------------------
#
# Each extractor is a (column_name, regex_pattern, postprocessor) tuple.
# The regex is searched against the full filename (case-insensitive).
# The postprocessor converts the regex match group(1) to the desired type.
# If no match, the column gets None.

_FILENAME_EXTRACTORS: list[tuple[str, str, callable]] = [
    # Excitation wavelength: "740nm" -> 740
    ("wavelength_nm",   r"_(\d{3,4})nm",                lambda m: int(m.group(1))),

    # Laser power: "3037mW" -> 3037.0  (keep as mW for now)
    ("power_mW",        r"_(\d{3,5})mW",                lambda m: float(m.group(1))),

    # USB attenuator: "u1" -> 1, "u80" -> 80
    ("usb_atten",       r"_u(\d{1,3})_",                lambda m: int(m.group(1))),

    # Pockels cell: "poc0" -> 0.0, "poc0p2" -> 0.2, "poc0p15" -> 0.15
    ("pockels",         r"_poc(\d+(?:p\d+)?)_",         lambda m: float(m.group(1).replace("p", "."))),

    # Objective: "20x0p75NA" -> "20x0.75NA", "10x0p5NA" -> "10x0.5NA"
    ("objective",       r"_(\d+x\d+p?\d*NA)",           lambda m: m.group(1).replace("p", ".")),

    # EM filter center wavelength: "457s50" -> 457, "370s10" -> 370
    # Allow 4 digits to handle typos like "4570s50" (treat as "457s50")
    ("em_filter_nm",    r"_(\d{3,4})s\d+_",             lambda m: int(m.group(1)[:3])),

    # EM filter bandwidth: "457s50" -> 50, "370s10" -> 10
    ("em_bandwidth_nm", r"_\d{3,4}s(\d+)_",             lambda m: int(m.group(1))),

    # Gain: "g70" -> 70
    ("gain",            r"_g(\d+)_",                     lambda m: int(m.group(1))),

    # Z position: "z1" -> 1.0, "z0p5" -> 0.5
    ("z_position",      r"_z(\d+(?:p\d+)?)_",           lambda m: float(m.group(1).replace("p", "."))),

    # Pixel count: "256pix" -> 256
    ("n_pixels_cfg",    r"_(\d+)pix_",                   lambda m: int(m.group(1))),

    # Number of frames: "30f" -> 30, "120f" -> 120
    ("n_frames_cfg",    r"_(\d+)f_",                     lambda m: int(m.group(1))),

    # Frame index: last "_0000" before .sdt/.asc -> 0
    ("frame_index",     r"_(\d{4})\.\w+$",              lambda m: int(m.group(1))),

    # Sample index (leading number): "3_KPCWT..." -> 3
    ("sample_index",    r"^(\d+)_",                      lambda m: int(m.group(1))),

    # Position (if present): "pos5" or "posn2" -> 5 or 2
    ("position",        r"_posn?(\d+)",                  lambda m: int(m.group(1))),
]


def _extract_fixation_type(filename: str) -> str | None:
    """
    Extract fixation type from the sample description part of the filename.

    Known types: glu (glutaraldehyde), form (formaldehyde), live, chromablue, urea, pollen.
    """
    name = filename.lower()
    # Order matters: check multi-char patterns before substrings
    if "chromablue" in name or "chroma" in name:
        return "chromablue"
    if "pollen" in name:
        return "pollen"
    if "urea" in name:
        return "urea"
    if "glu" in name:
        return "glu"
    if "form" in name:
        return "form"
    if "live" in name:
        return "live"
    return None


def _extract_cell_type(filename: str) -> str | None:
    """
    Extract cell type from the sample description.

    Known types: KPCWT, BKO, DKO, KPC (sometimes used loosely).
    We match these case-insensitively at a word boundary or
    right after the sample index.
    """
    # Match KPCWT, BKO, DKO first (more specific), then KPC
    for ct in ["KPCWT", "BKO", "DKO"]:
        if re.search(rf"(?:^|\d_){ct}", filename, re.IGNORECASE):
            return ct
    # Bare "KPC" (but not if KPCWT already matched above)
    if re.search(r"(?:^|\d_)KPC(?!WT)", filename, re.IGNORECASE):
        return "KPC"
    return None


def _extract_treatment_duration(filename: str) -> str | None:
    """Extract treatment duration like '10min', '30min' if present."""
    m = re.search(r"(\d+min)", filename, re.IGNORECASE)
    return m.group(1) if m else None


def _extract_passage_info(filename: str) -> str | None:
    """Extract passage info like 'P4num4' -> 'P4num4'."""
    m = re.search(r"(P\d+num\d+)", filename, re.IGNORECASE)
    return m.group(1) if m else None


def _extract_session_date(session_root: str) -> str | None:
    """Extract YYYYMMDD from the session directory name."""
    m = re.match(r"(\d{8})_", session_root)
    return m.group(1) if m else None


def parse_filename_metadata(filename: str, session_root: str = "") -> dict:
    """
    Extract all metadata fields from a single filename.

    Returns a dict with one key per extracted field (None if not found).
    """
    meta = {}

    # Regex-based extractors
    for col, pattern, postproc in _FILENAME_EXTRACTORS:
        m = re.search(pattern, filename, re.IGNORECASE)
        meta[col] = postproc(m) if m else None

    # Custom extractors
    meta["fixation_type"] = _extract_fixation_type(filename)
    meta["cell_type"] = _extract_cell_type(filename)
    meta["treatment_duration"] = _extract_treatment_duration(filename)
    meta["passage_info"] = _extract_passage_info(filename)
    meta["session_date"] = _extract_session_date(session_root)

    return meta


# %%
# Parse metadata from all SDT filenames
fname_meta = sdt_files.apply(
    lambda row: parse_filename_metadata(row["filename"], row["session_root"]),
    axis=1,
).apply(pd.Series)

sdt_df = pd.concat([sdt_files, fname_meta], axis=1).reset_index(drop=True)

# %%
# Review: which columns have data vs all-None?
print("=== Filename metadata coverage ===")
meta_cols = fname_meta.columns.tolist()
for col in meta_cols:
    n_present = sdt_df[col].notna().sum()
    n_unique = sdt_df[col].nunique()
    examples = sdt_df[col].dropna().unique()[:5]
    print(f"  {col:25s}  {n_present:4d}/{len(sdt_df)} files  ({n_unique} unique)  ex: {list(examples)}")

# %%
# Show any filenames where key fields failed to parse
key_cols = ["wavelength_nm", "power_mW", "em_filter_nm", "fixation_type"]
missing_any = sdt_df[sdt_df[key_cols].isna().any(axis=1)]
if len(missing_any):
    print(f"\n=== {len(missing_any)} files missing key metadata ===")
    print(missing_any[["filename"] + key_cols].to_string())
else:
    print("\nAll files have key metadata fields populated.")

# %% [markdown]
# ## Step 2b: Classify file types
#
# Use the parsed metadata to classify: IRF, chroma calibration, or sample.

# %%
def classify_file_type(row: pd.Series) -> str:
    """
    Classify based on fixation_type and other parsed fields.

    chromablue -> chroma
    urea       -> irf  (urea slides are the IRF source for this instrument)
    pollen     -> pollen_reference
    glu/form/live with a cell_type -> sample
    """
    fix = row.get("fixation_type")
    if fix == "chromablue":
        return "chroma"
    if fix == "urea":
        return "irf"
    if fix == "pollen":
        return "pollen_reference"
    if row.get("cell_type") is not None:
        return "sample"
    return "other"


sdt_df["file_type"] = sdt_df.apply(classify_file_type, axis=1)

print("=== File type classification ===")
print(sdt_df["file_type"].value_counts().to_string())
print()

# Show a few from each type
for ft in sdt_df["file_type"].unique():
    subset = sdt_df[sdt_df["file_type"] == ft]
    print(f"\n--- {ft} ({len(subset)} files) ---")
    print(subset[["filename", "cell_type", "fixation_type", "em_filter_nm"]].head(5).to_string())

# %% [markdown]
# ## Step 2c: Metadata corrections
#
# Known naming inconsistencies corrected here.
# Add new corrections as they are discovered; document the reason.

# %%
# (1) Before session 20260509, files labelled "KPC" + live are actually KPCWT.
_mask_kpc = (
    (sdt_df["cell_type"] == "KPC") &
    (sdt_df["fixation_type"] == "live") &
    (sdt_df["session_date"] < "20260509")
)
sdt_df.loc[_mask_kpc, "cell_type"] = "KPCWT"
print(f"(1) KPC+live -> KPCWT (pre-20260509): {_mask_kpc.sum()} files corrected")

# (2) em_filter_nm 475 is a parsing artefact; correct to 457.
_mask_475 = sdt_df["em_filter_nm"] == 475
sdt_df.loc[_mask_475, "em_filter_nm"] = 457
print(f"(2) em_filter_nm 475 -> 457: {_mask_475.sum()} files corrected")

# (3) Position fallback: use frame_index when position is absent from filename.
sdt_df["position"] = sdt_df["position"].fillna(sdt_df["frame_index"])
print(f"(3) Position coverage after fallback: "
      f"{sdt_df['position'].notna().sum()}/{len(sdt_df)} files")

# %% [markdown]
# **IRF source:** Urea slides serve as the IRF measurement for this instrument.
# Files classified as `irf` above are the urea acquisitions.
# SPCImage also exports fitted IRFs as `.irf` files (discovered in `irf_files`);
# format is 256 whitespace-separated counts followed by 3 version numbers.
# IRF-before vs IRF-after assignment happens in Phase B based on `acquisition_time`.

# %%
def load_irf(filepath: Path) -> np.ndarray:
    """Load a BH SPCImage .irf export. Returns the 256-bin IRF histogram (unnormalized)."""
    tokens = filepath.read_text().split()
    return np.array(tokens[:256], dtype=float)


# %% [markdown]
# ## Step 3: Load SDT metadata from file headers
#
# Parse each `.sdt` file to extract acquisition timestamps,
# data shape, photon statistics, and time-axis parameters.

# %%


def extract_sdt_header_metadata(filepath: Path, channel: int = 0) -> dict:
    """
    Extract metadata from the SDT file header and data arrays.

    Returns dict of scalar values. Does NOT store the full decay array.
    """
    meta = {}
    try:
        sdt = SdtFile(str(filepath))
    except Exception as e:
        return {"load_error": str(e)}

    # -- Timestamp from info text block --
    try:
        info_text = sdt.info.id if hasattr(sdt.info, "id") else str(sdt.info)
        meta["info_text_snippet"] = info_text[:300]
    except Exception:
        info_text = ""

    meta["acquisition_time"] = _parse_bh_timestamp(
        info_text, fallback=datetime.fromtimestamp(filepath.stat().st_mtime)
    )

    # -- Header-level fields --
    try:
        meta["header_revision"] = int(sdt.header.revision)
    except (AttributeError, TypeError):
        pass

    # -- Measure info --
    if sdt.measure_info:
        mi = sdt.measure_info[0]
        for field in ["scan_x", "scan_y", "tac_range", "tac_gain", "collect_time", "scan_rxy"]:
            val = getattr(mi, field, None)
            if val is not None:
                meta[f"mi_{field}"] = float(val) if field in ("tac_range", "tac_gain", "collect_time") else int(val)

    # -- Data shape and photon stats --
    if sdt.data:
        meta["n_data_blocks"] = len(sdt.data)
        data = sdt.data[channel] if channel < len(sdt.data) else sdt.data[0]
        meta["data_shape"] = str(data.shape)
        meta["data_dtype"] = str(data.dtype)

        if data.ndim == 3:
            meta["data_nx"] = data.shape[0]
            meta["data_ny"] = data.shape[1]
            meta["data_nt"] = data.shape[2]
            intensity = data.sum(axis=2)
            meta["total_photons"] = int(data.sum())
            meta["max_photons_per_pixel"] = int(intensity.max())
            meta["mean_photons_per_pixel"] = float(intensity.mean())
            meta["median_photons_per_pixel"] = float(np.median(intensity))
        elif data.ndim == 2:
            meta["data_nt"] = data.shape[-1]
            meta["total_photons"] = int(data.sum())

    # -- Time axis --
    if sdt.times:
        t = sdt.times[channel] if channel < len(sdt.times) else sdt.times[0]
        meta["time_axis_len"] = len(t)
        if len(t) > 1:
            meta["bin_width_ns"] = float(t[1] - t[0]) * 1e9
            meta["time_range_ns"] = float(t[-1] - t[0]) * 1e9

    return meta


def _parse_bh_timestamp(info_text: str, fallback: datetime | None = None) -> datetime | None:
    """
    Parse acquisition timestamp from the BH SPCM info text block.

    Tries several common formats, falls back to mtime.
    """
    if not info_text:
        return fallback

    # Pattern: *DATE : 29 Apr 2026  *TIME : 14:32:05
    m = re.search(
        r"\*?DATE\s*:?\s*(\d{1,2}[\s\-]?\w{3}[\s\-]?\d{4})"
        r".*?"
        r"\*?TIME\s*:?\s*(\d{1,2}:\d{2}:\d{2})",
        info_text, re.DOTALL | re.IGNORECASE,
    )
    if m:
        date_str = m.group(1).replace("-", " ").strip()
        time_str = m.group(2).strip()
        for fmt in ["%d %b %Y %H:%M:%S", "%d %B %Y %H:%M:%S"]:
            try:
                return datetime.strptime(f"{date_str} {time_str}", fmt)
            except ValueError:
                continue

    # Pattern: #DATE DD-Mon-YYYY  #TIME HH:MM:SS
    m = re.search(
        r"#DATE\s+(\S+).*?#TIME\s+(\S+)",
        info_text, re.DOTALL | re.IGNORECASE,
    )
    if m:
        for fmt in ["%d-%b-%Y %H:%M:%S", "%d-%B-%Y %H:%M:%S"]:
            try:
                return datetime.strptime(f"{m.group(1)} {m.group(2)}", fmt)
            except ValueError:
                continue

    return fallback


# %%
print(f"Loading SDT header metadata from {len(sdt_df)} files...")

header_records = []
errors = []

for i, (_, row) in enumerate(sdt_df.iterrows()):
    if (i + 1) % 50 == 0 or i == 0:
        print(f"  [{i+1}/{len(sdt_df)}] {row['filename']}")

    meta = extract_sdt_header_metadata(row["filepath"])
    meta["_idx"] = i

    if "load_error" in meta:
        errors.append((row["filename"], meta["load_error"]))
    header_records.append(meta)

header_df = pd.DataFrame(header_records).set_index("_idx")

# Merge header metadata into the main dataframe
for col in header_df.columns:
    if col not in sdt_df.columns:
        sdt_df[col] = header_df[col].values

print(f"\nLoaded: {len(sdt_df)} files, {len(errors)} errors")
if errors:
    print("\n=== LOAD ERRORS ===")
    for fn, err in errors:
        print(f"  {fn}: {err}")

# %%
# Sort by acquisition time
sdt_df = sdt_df.sort_values("acquisition_time").reset_index(drop=True)

print("=== SDT metadata summary ===")
print(f"Time span: {sdt_df['acquisition_time'].min()} → {sdt_df['acquisition_time'].max()}")
print(f"Scan dimensions found: {sdt_df['data_shape'].value_counts().to_string()}")
if "total_photons" in sdt_df.columns:
    print(f"Total photons range: {sdt_df['total_photons'].min():.0f} - {sdt_df['total_photons'].max():.0f}")
    print(f"Mean photons/pixel: {sdt_df['mean_photons_per_pixel'].min():.1f} - {sdt_df['mean_photons_per_pixel'].max():.1f}")

# %% [markdown]
# ### Inspect: does the BH timestamp match filesystem mtime?
#
# The BH header timestamp is more reliable, but it's worth checking
# that they're consistent.

# %%
if "acquisition_time" in sdt_df.columns:
    sdt_df["time_delta_s"] = (
        sdt_df["acquisition_time"] - sdt_df["mtime"]
    ).dt.total_seconds().abs()
    large_delta = sdt_df[sdt_df["time_delta_s"] > 60]
    if len(large_delta):
        print(f"WARNING: {len(large_delta)} files have BH timestamp > 60s from mtime")
        print(large_delta[["filename", "acquisition_time", "mtime", "time_delta_s"]].head(10).to_string())
    else:
        print("All BH timestamps within 60s of filesystem mtime -- good.")

# %% [markdown]
# ## Step 3b: Inspect the parsed metadata
#
# Review the full table and check for anomalies.

# %%
# Summary by experimental group
print("=== Experimental groups ===\n")
sample_df = sdt_df[sdt_df["file_type"] == "sample"]

group_cols = ["cell_type", "fixation_type", "em_filter_nm"]
existing = [c for c in group_cols if c in sample_df.columns]
if existing:
    print(sample_df.groupby(existing).size().reset_index(name="n_files").to_string())

# %%
# Check for unexpected combinations
print("\n=== Unique values per parsed field ===")
for col in ["fixation_type", "cell_type", "wavelength_nm", "em_filter_nm",
            "pockels", "usb_atten", "gain", "objective", "n_frames_cfg"]:
    if col in sdt_df.columns:
        vals = sdt_df[col].dropna().unique()
        print(f"  {col:25s}: {sorted(vals)}")

# %% [markdown]
# ## Step 4: Load and match ascii fit exports (.asc)
#
# SPCImage exports one parameter per `.asc` file as a space-delimited
# 2D grid. The filename encodes which parameter it is.

# %%
def load_spcimage_asc(filepath: Path) -> dict | None:
    """
    Load a single SPCImage .asc export file.

    Returns dict with 'param_name', 'data' (2D ndarray), 'filepath'.
    Returns None if the file can't be parsed as a numeric grid.
    """
    param_name = _infer_param_name(filepath.stem)

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

    return {"param_name": param_name, "data": data, "filepath": filepath}


def _infer_param_name(stem: str) -> str:
    """Map filename stem suffix to a canonical parameter name."""
    stem_lower = stem.lower()
    patterns = [
        (r"[-_]a1[_%]?$",          "a1"),
        (r"[-_]a2[_%]?$",          "a2"),
        (r"[-_]a3[_%]?$",          "a3"),
        (r"[-_]t1$|[-_]tau1$",     "tau1"),
        (r"[-_]t2$|[-_]tau2$",     "tau2"),
        (r"[-_]t3$|[-_]tau3$",     "tau3"),
        (r"[-_]tm$|[-_]mean$",     "tau_mean"),
        (r"[-_]chi$|[-_]chisq$",   "chi2"),
        (r"[-_]photons?$|[-_]int(ensity)?$|[-_]cnt$", "photons"),
    ]
    for pattern, name in patterns:
        if re.search(pattern, stem_lower):
            return name
    # Fallback: use the last underscore-delimited token
    parts = stem_lower.rsplit("_", 1)
    return parts[-1] if len(parts) > 1 else stem_lower


def _strip_param_suffix(stem: str) -> str:
    """Remove the parameter suffix to get the base image name."""
    return re.sub(
        r"[-_]?(a[123]|t[123]|tau[123]|chi|chisq|photons?|intensity|cnt|tm|mean)[_%]?$",
        "",
        stem,
        flags=re.IGNORECASE,
    ).strip("_- ")


# %%
print(f"Indexing {len(asc_files)} .asc fit export files (metadata only, no array loading)...")

# fit_data key: "{session_root}::{rel_dir}::{base_stem}"
# Values: param_name -> Path  (arrays are loaded on-demand by get_fit_arrays / diagnostic)
# Loading full arrays here is unnecessary for building fit_map.csv and would consume
# tens of GB of RAM, causing heavy swapping during match_fits_to_sdt.
fit_data: dict[str, dict] = {}
n_stat_skipped = 0

for _, row in asc_files.iterrows():
    if re.search(r"_statistic", row["stem"], re.IGNORECASE):
        n_stat_skipped += 1
        continue

    param_name  = _infer_param_name(row["stem"])
    base        = _strip_param_suffix(row["stem"])
    fit_set_key = f"{row['session_root']}::{row['rel_dir']}::{base}"

    if fit_set_key not in fit_data:
        fit_data[fit_set_key] = {
            "_session_root": row["session_root"],
            "_folder":       row["rel_dir"],
            "_base_stem":    base,
        }
    # Store path as value; arrays loaded on-demand only when needed.
    fit_data[fit_set_key][param_name] = row["filepath"]

print(f"Indexed {len(fit_data)} fit sets  ({n_stat_skipped} statistics files skipped)")

print("\nFirst few fit sets:")
for key, params in list(fit_data.items())[:8]:
    param_names = sorted(k for k in params if not k.startswith("_"))
    print(f"  [{params['_folder']}] {params['_base_stem']}: {', '.join(param_names)}")


# %% [markdown]
# ## Step 4b: Match fit exports to SDT files
#
# Link each ascii export group to its corresponding SDT file.
# Acquisition timestamps for matched `.asc` files are inherited from
# the SDT (we don't care about mtime on the `.asc` files themselves).

# %%
def match_fits_to_sdt(sdt_df: pd.DataFrame,
                      fit_data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match ALL available fit sets to each SDT file by session + stem similarity.

    Returns (sdt_df_updated, fit_map_df).

    sdt_df_updated gains:
      has_fit_export      -- True if any fit set matched
      fit_base_stem       -- first matching fit_set_key (backward-compat)
      fit_params_available-- params of the first match
      n_fit_sets          -- total number of matched fit sets
      fit_set_keys        -- comma-separated list of all matching fit_set_keys

    fit_map_df columns: sdt_filename, fit_set_key, fit_folder, n_components,
                        params_available
    """
    # Per-session lookup: base_stem_lower -> [fit_set_keys]
    sess_lookup: dict[str, dict[str, list]] = {}
    for key, fd in fit_data.items():
        sess  = fd.get("_session_root", "")
        base  = fd.get("_base_stem", "").lower()
        sess_lookup.setdefault(sess, {}).setdefault(base, []).append(key)

    df = sdt_df.copy()
    df["has_fit_export"]      = False
    df["fit_base_stem"]       = None
    df["fit_params_available"]= None
    df["n_fit_sets"]          = 0
    df["fit_set_keys"]        = None

    fit_map_rows = []

    for idx, row in df.iterrows():
        sess      = row.get("session_root", "")
        sdt_lower = row["stem"].lower()

        matches = []
        for base_lower, keys in sess_lookup.get(sess, {}).items():
            if base_lower in sdt_lower or sdt_lower.startswith(base_lower):
                matches.extend(keys)

        # Deduplicate, preserve order
        seen, matches = set(), [k for k in matches
                                 if not (k in seen or seen.add(k))]
        if not matches:
            continue

        first_fd     = fit_data[matches[0]]
        first_params = sorted(k for k in first_fd if not k.startswith("_"))

        df.at[idx, "has_fit_export"]       = True
        df.at[idx, "fit_base_stem"]        = matches[0]
        df.at[idx, "fit_params_available"] = ", ".join(first_params)
        df.at[idx, "n_fit_sets"]           = len(matches)
        df.at[idx, "fit_set_keys"]         = ",".join(matches)

        for fit_key in matches:
            fd     = fit_data[fit_key]
            params = sorted(k for k in fd if not k.startswith("_"))
            n_comp = sum(1 for p in ("tau1", "tau2", "tau3") if p in fd)
            fit_map_rows.append({
                "sdt_filename":    row["filename"],
                "fit_set_key":     fit_key,
                "fit_folder":      fd.get("_folder", ""),
                "n_components":    n_comp,
                "params_available": ", ".join(params),
            })

    return df, pd.DataFrame(fit_map_rows)


sdt_df, fit_map_df = match_fits_to_sdt(sdt_df, fit_data)

n_with_fits = sdt_df["has_fit_export"].sum()
n_samples   = (sdt_df["file_type"] == "sample").sum()
print(f"=== Fit export matching ===")
print(f"  {n_with_fits} / {len(sdt_df)} total SDT files have at least one fit set")
print(f"  {sdt_df.loc[sdt_df['file_type'] == 'sample', 'has_fit_export'].sum()} "
      f"/ {n_samples} sample files")
if n_with_fits:
    ns = sdt_df.loc[sdt_df["has_fit_export"], "n_fit_sets"]
    print(f"  Fit sets per matched file: min={ns.min()}  max={ns.max()}  "
          f"mean={ns.mean():.1f}")
print(f"\n  n_components distribution across all fit sets:")
if not fit_map_df.empty:
    print(fit_map_df["n_components"].value_counts().sort_index().to_string())

unmatched = sdt_df[(sdt_df["file_type"] == "sample") & (~sdt_df["has_fit_export"])]
if len(unmatched):
    print(f"\n  {len(unmatched)} sample files WITHOUT fit exports:")
    print(unmatched[["filename"]].head(20).to_string())

# %% [markdown]
# ## Step 5: Quick sanity check -- load one file end-to-end

# %%
# Pick the first sample file with a fit export
_test_mask = (sdt_df["file_type"] == "sample") & (sdt_df["has_fit_export"])
if _test_mask.any():
    test_row = sdt_df[_test_mask].iloc[0]
else:
    test_row = sdt_df.iloc[0]

print(f"Test file: {test_row['filename']}")
print(f"  Type:       {test_row['file_type']}")
print(f"  Cell type:  {test_row.get('cell_type')}")
print(f"  Fixation:   {test_row.get('fixation_type')}")
print(f"  EM filter:  {test_row.get('em_filter_nm')} nm")
print(f"  Acquired:   {test_row.get('acquisition_time')}")
print(f"  Pockels:    {test_row.get('pockels')}")
print(f"  Wavelength: {test_row.get('wavelength_nm')} nm")
print(f"  Power:      {test_row.get('power_mW')} mW")

# Load raw decay
sdt_obj = SdtFile(str(test_row["filepath"]))
decay = sdt_obj.data[0]
time_axis = sdt_obj.times[0]
print(f"\n  Decay shape:  {decay.shape}")
print(f"  Time axis:    {time_axis.shape}, {time_axis[0]*1e9:.2f} - {time_axis[-1]*1e9:.2f} ns")
print(f"  Bin width:    {(time_axis[1] - time_axis[0])*1e9:.4f} ns")
print(f"  Total photons: {decay.sum():,.0f}")

# Intensity image
intensity = decay.sum(axis=2)
print(f"  Intensity:    shape={intensity.shape}, "
      f"min={intensity.min()}, max={intensity.max()}, mean={intensity.mean():.1f}")

# Check fit export shape match (loads arrays on-demand for the single test file only)
if test_row["has_fit_export"]:
    base = test_row["fit_base_stem"]
    param_names = sorted(k for k in fit_data[base] if not k.startswith("_"))
    print(f"\n  Fit params: {param_names}")

    for name in param_names:
        _fp = fit_data[base][name]   # stored as Path
        try:
            arr = np.loadtxt(str(_fp), dtype=float)
        except Exception:
            print(f"    {name:10s}: (could not load {_fp.name})")
            continue
        print(f"    {name:10s}: shape={arr.shape}, range=[{np.nanmin(arr):.3g}, {np.nanmax(arr):.3g}]")

        # Shape match check (once, using photons or first available param)
        if name == "photons" or (name == param_names[0] and "photons" not in param_names):
            sdt_shape = intensity.shape
            if arr.shape == sdt_shape:
                print(f"\n  Shape match: SDT {sdt_shape} == export {arr.shape}")
            else:
                print(f"\n  *** SHAPE MISMATCH: SDT {sdt_shape} vs export {arr.shape} ***")
                print(f"      Check for flip/transpose -- pipeline step 7")

# %% [markdown]
# ## Convenience functions for downstream notebooks

# %%
_sdt_cache: dict[str, SdtFile] = {}


def load_decay(filepath: Path, channel: int = 0, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Load TCSPC decay array and time axis from an SDT file.

    Returns (decay, time_axis) where:
      - decay: (scan_x, scan_y, n_timebins)
      - time_axis: (n_timebins,) in seconds
    """
    key = str(filepath)
    if use_cache and key in _sdt_cache:
        sdt = _sdt_cache[key]
    else:
        sdt = SdtFile(str(filepath))
        if use_cache:
            _sdt_cache[key] = sdt
    return sdt.data[channel], sdt.times[channel]


def get_fit_arrays(row: pd.Series,
                   fit_set_key: str = None) -> dict[str, np.ndarray] | None:
    """Get fit parameter arrays for a given SDT file row, or None.

    fit_set_key: specific key from fit_map_df; defaults to fit_base_stem (first match).
    Use fit_map_df to enumerate all available fit sets for a file.
    Arrays are loaded from disk on-demand (fit_data stores paths, not arrays).
    """
    if not row.get("has_fit_export"):
        return None
    key = fit_set_key or row.get("fit_base_stem")
    if not isinstance(key, str) or key not in fit_data:
        return None
    out = {}
    for k, v in fit_data[key].items():
        if k.startswith("_"):
            continue
        try:
            out[k] = np.loadtxt(str(v), dtype=float)
        except Exception:
            pass
    return out if out else None


def get_row(df: pd.DataFrame, filename: str = None, idx: int = None) -> pd.Series:
    """Look up a single row by filename or integer index."""
    if filename is not None:
        matches = df[df["filename"] == filename]
        if len(matches) == 0:
            raise KeyError(f"No file named '{filename}'")
        return matches.iloc[0]
    if idx is not None:
        return df.iloc[idx]
    raise ValueError("Provide either filename or idx")


# %% [markdown]
# ## Summary
#
# At this point you have:
# - **`sdt_df`**: one row per SDT file with parsed filename metadata,
#   BH header metadata, acquisition timestamps, and file type tags
# - **`fit_data`**: dict of `{base_stem: {param_name: 2D array}}`
# - **`load_decay(filepath)`** → `(decay, time_axis)`
# - **`get_fit_arrays(row)`** → `{param: array}` or `None`
#
# **Next**: `02_calibration.py` -- IRF comparison, chroma phasor
# calibration, temporal drift assessment.

# %%
# Save the metadata table for downstream notebooks
output_dir = Path("../results")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "sdt_metadata.csv"

# Drop non-portable columns
save_cols = [c for c in sdt_df.columns
             if c not in ("filepath", "info_text_snippet") and not c.startswith("_")]
sdt_df[save_cols].to_csv(output_path, index=False)
print(f"Saved metadata to {output_path} ({len(sdt_df)} rows, {len(save_cols)} columns)")

# Also save the filepath mapping separately (for re-loading on this machine)
fp_map = sdt_df[["filename", "session_root"]].copy()
fp_map["filepath"] = sdt_df["filepath"].astype(str)
fp_map.to_csv(output_dir / "filepath_map.csv", index=False)
print(f"Saved filepath map to {output_dir / 'filepath_map.csv'}")

# Save fit map: one row per (SDT file, fit set) -- used in Phase E to select fit model
fit_map_df.to_csv(output_dir / "fit_map.csv", index=False)
print(f"Saved fit map to {output_dir / 'fit_map.csv'} "
      f"({len(fit_map_df)} fit-set/SDT pairs)")

# %% [markdown]
# ## Verification: raw SDT headers vs parsed DataFrame
#
# Spot-check that filename parsing and BH header timestamps agree with
# the raw file content for a sample of files spread across the dataset.

# %%
verify_indices = [0, len(sdt_df) // 4, len(sdt_df) // 2, 3 * len(sdt_df) // 4, -1]
check_cols = [
    "acquisition_time", "wavelength_nm", "power_mW",
    "em_filter_nm", "pockels", "usb_atten", "gain",
    "cell_type", "fixation_type", "file_type",
    "data_shape", "bin_width_ns", "total_photons",
]

for i in verify_indices:
    row = sdt_df.iloc[i]
    print(f"\n{'='*60}")
    print(f"File: {row['filename']}")
    print("--- Raw BH header snippet ---")
    print(row.get("info_text_snippet", "(not stored)"))
    print("--- Parsed fields ---")
    for col in check_cols:
        if col in row.index:
            print(f"  {col:30s}: {row[col]}")

# %% [markdown]
# ## Export metadata to Excel for manual review

# %%
try:
    excel_path = output_dir / "sdt_metadata.xlsx"
    sdt_df[save_cols].to_excel(excel_path, index=False, freeze_panes=(1, 0))
    print(f"Saved Excel export to {excel_path}")
except ImportError:
    print("openpyxl not installed -- using CSV only (run: uv add --dev openpyxl)")

# %%

# %%
import winsound as _ws, time as _t
for _ in range(3):
    _ws.Beep(1000, 400)
    _t.sleep(1)
