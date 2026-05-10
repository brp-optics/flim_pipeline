# %% [markdown]

## Full analysis pipeline - Summary

# FLIM analysis pipeline

## Phase A — Inventory and metadata
# 
# 1. **Discover all files.**
#    Recursively scan the data directory for `.sdt`, `.spc`, and ascii fit-export files.
#    �넂 *raw file list*
# 
# 2. **Build metadata DataFrame.**
#    Parse sdt headers to extract `acquisition_time`, scan dimensions, bin width, rep rate. Tag each file with `file_type` (IRF-before, IRF-after, chroma-before, chroma-after, sample), `cell_appearance`, and `colony_size`. Match each sdt with its corresponding ascii fit export.
#    �넂 *df with one row per acquisition, sorted by acquisition\_time*
# 
# 3. **Load ascii fit exports.**
#    Parse a1, a2, a3, tau1, tau2, tau3, photon count, �눫� per pixel from ascii files. Store as numpy arrays keyed by filepath stem.
#    �넂 *dict of fit arrays per file*
# 
# ## Phase B — Instrument calibration
# 
# 4. **Compare bracketing IRFs.**
#    Load before and after IRF sdt files. Overlay decay curves. Quantify peak channel position, FWHM, and integrated counts for each. Compute drift (�봯eak, �봃WHM) between the two.
#    �넂 *IRF drift metrics; decision on interpolation vs averaging*
# 
#    > �슑 If peak shift > 1 time bin: flag session, consider per-file IRF interpolation by acquisition\_time.
# 
# 5. **Compute phasor calibration from chroma slides.**
#    Load before/after chroma sdt files. Compute raw phasor (G, S) for each. Compare against the theoretical position for the known fluorophore lifetime at ω. Derive phase rotation and modulation correction factors. Check whether calibration drifted between before/after.
#    �넂 *phasor\_cal\_phase, phasor\_cal\_mod (per calibration timepoint)*
# 
# 6. **Assign calibration to each sample.**
#    For each sample file, use its `acquisition_time` to determine which IRF and phasor calibration to apply (nearest, averaged, or interpolated). Record `irf_source` and `phasor_cal_source` columns in df.
#    �넂 *df updated with calibration provenance columns*
# 
# ## Phase C — Data integrity checks
# 
# 7. **Verify orientation match between sdt and fit exports.**
#    For a few representative files, generate an intensity image from the sdt (sum over time bins) and compare against the photon count map from the ascii export. Check for flips (LR, UD) and transposition. Determine the transform needed, if any.
#    �넂 *orientation\_transform (None, "flipud", "fliplr", "transpose", etc.)*
# 
# 8. **Spot-check exported fits against raw data.**
#    Select 3–5 bright pixels from a representative file. Fit mono-exponential from the raw sdt decay using your own fitter. Compare recovered τ and amplitude against the corresponding ascii export values. Verify agreement within tolerance.
#    �넂 *confidence that ascii exports are trustworthy (or a list of discrepancies)*
# 
# ## Phase D — Phasor analysis
# 
# 9. **Compute intensity mask.**
#    For each sdt file, sum photon counts over time bins per pixel. Threshold to create a boolean mask excluding background pixels. Record `fraction_above_threshold` in df.
#    �넂 *mask array per file*
# 
# 10. **Compute calibrated phasor per pixel.**
#     For each sdt file, compute raw (G, S) at the laser rep frequency using phasorpy. Apply the calibration correction from step 5. Store calibrated (G, S) arrays. Apply intensity mask before any summary statistics.
#     �넂 *G, S arrays per file; G\_mean, S\_mean columns in df*
# 
# 11. **Phasor sanity check.**
#     Overlay phasor cloud on the universal semicircle. Verify the chroma reference point lands at its expected position after calibration. Check whether sample points fall on, inside, or outside the semicircle (indicating single-exp, multi-exp, or calibration error respectively).
#     �넂 *phasor semicircle plot; flag files with points outside the semicircle*
# 
# 12. **Plot phasors grouped by file type.**
#     Generate one phasor plot per group (`file_type` / `cell_appearance` / `colony_size`). Color-code by group for quick visual feedback to collaborator.
#     �넂 *phasor summary figures for sharing*
# 
# ## Phase E — Exported fit analysis
# 
# 13. **Apply orientation correction to fit arrays.**
#     Using the transform determined in step 7, flip/transpose the ascii fit arrays so they align pixel-for-pixel with the sdt data and phasor arrays.
#     �넂 *aligned fit arrays*
# 
# 14. **Threshold on fit quality.**
#     Create a fit-quality mask from reduced �눫� (e.g. 0.8 < �눫� < 1.5). Optionally intersect with the intensity mask from step 9. Inspect the �눫� map spatially for systematic patterns.
#     �넂 *fit\_quality\_mask per file; chi2\_mean column in df*
# 
# 15. **Compute derived lifetime metrics.**
#     From the masked fit parameters, compute amplitude-weighted mean lifetime per pixel: τ\_mean = 誇(a\_i 쨌 τ\_i) / 誇(a\_i). Compute fractional contributions f\_i = a\_i 쨌 τ\_i / 誇(a\_j 쨌 τ\_j). Add per-image summary statistics to df.
#     �넂 *tau\_mean array per file; tau\_mean\_mean, f1\_mean, f2\_mean columns in df*
# 
# ## Phase F — Group comparisons
# 
# 16. **Check for photobleaching.**
#     If sdt files contain frame data, compare early vs late frame intensity. Flag files with significant bleaching (>10% decline) in df.
#     �넂 *bleaching\_flag, bleaching\_pct columns in df*
# 
# 17. **Intra-group distributions.**
#     For each group, plot per-pixel lifetime distributions (histograms or KDEs) from each image overlaid. Assess within-group consistency: do images from the same condition look similar?
#     �넂 *intra-group overlay plots*
# 
# 18. **Inter-group comparisons.**
#     Compare groups using per-image summary statistics (one value per image, not per pixel). Violin or box plots of τ\_mean, fractional contributions, phasor position. Statistical tests with the image (or colony) as the unit of analysis, not the pixel.
#     �넂 *inter-group comparison figures; statistical test results*
# 
# 19. **Cross-validate phasor vs fit results.**
#     Compare phasor-derived lifetimes (τ\_phi, τ\_mod from step 10) against fit-derived mean lifetimes (step 15) on the same pixels. Agreement strengthens confidence; disagreement highlights model mismatch or calibration issues.
#     �넂 *phasor-vs-fit correlation plot; Bland-Altman plot*
# 
# 20. **Export results.**
#     Save the final df as CSV. Export lifetime maps and phasor arrays as TIFF or HDF5. Package summary figures for collaborator.
#     �넂 *results/ directory with CSV, TIFFs, figures*
# 
# %%

## Phase A — Inventory and metadata
# 
# 1. **Discover all files.**
#    Recursively scan the data directory for `.sdt`, `.spc`, and ascii fit-export files.
#    �넂 *raw file list*
# 

# -- Configuration ----------------------------------------------------------
data_dirs = [ '/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260429_KPC_fixed_dishes_on_SLIM', '/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260501_KPC_fixed_dishes_on_SLIM']

win_data_dirs = [
    Path("E:\\18_RK_Circadian\\data\\raw\\20260429_KPC_fixed_dishes_on_SLIM"),
    Path("E:\\18_RK_Circadian\\data\\raw\\20260501_KPC_fixed_dishes_on_SLIM"),
    ]

current_os = "Win" ## Could we automate discovery of this?

from pathlib import Path
import pandas as pd
import numpy as np
import re
import warnings
from datetime import datetime


data_dirs = [ Path(d) for d in data_dirs ]

if current_os == "Win":
    data_dirs = win_data_dirs

# Verify directories exist
for d in data_dirs:
    if not d.exists():
        raise FileNotFoundError(f"Data directory not found: {d}")
    print(f"  Found: {d}  ({sum(1 for _ in d.rglob('*')) } files/dirs)")

    
# %% [markdown]
# ## Step 1: Discover all files
#
# Recursively find `.sdt` files and ascii fit exports.
# We'll handle `.spc` files too if they exist.
 
# %%
def find_data_files(roots: list[Path]) -> pd.DataFrame:
    """Recursively find all .sdt, .spc, and potential ascii fit-export files."""
    records = []
    # Common extensions for BH ascii exports: .asc, .txt, .dat, .csv, .tif
    # We'll be broad and filter later
    sdt_exts = {".sdt", ".spc"}
    ascii_exts = {".asc"} #, ".txt", ".dat", ".csv"}
    img_exts = {".tif"}
 
    for root in roots:
        for fp in sorted(root.rglob("*")):
            if not fp.is_file():
                continue
            ext = fp.suffix.lower()
            if ext in sdt_exts:
                category = "sdt" if ext == ".sdt" else "spc"
            elif ext in ascii_exts:
                category = "ascii_candidate"
            else:
                continue
 
            records.append({
                "filepath": fp,
                "filename": fp.name,
                "stem": fp.stem,
                "suffix": ext,
                "category": category,
                "session_dir": fp.relative_to(root).parts[0]
                    if len(fp.relative_to(root).parts) > 1 else "",
                "session_root": root.name,
                "size_mb": fp.stat().st_size / 1e6,
                "mtime": datetime.fromtimestamp(fp.stat().st_mtime),
            })
 
    return pd.DataFrame(records)
 
files_df = find_data_files(data_dirs)
print(f"Found {len(files_df)} files total:")
print(files_df["category"].value_counts().to_string())
print()
print(f"SDT files: {(files_df['category'] == 'sdt').sum()}")
print(f"SPC files: {(files_df['category'] == 'spc').sum()}")
print(f"ASCII candidates: {(files_df['category'] == 'ascii_candidate').sum()}")
 
# %%
# Quick look at the directory structure
print("=== SDT files by session ===")
sdt_files = files_df[files_df["category"] == "sdt"]
print(sdt_files.groupby("session_root")["filename"].count())
print()
print("=== First few SDT filenames ===")
print(sdt_files["filename"].head(20).to_string())
 
# %%
# Quick look at ascii files to understand the naming pattern
ascii_files = files_df[files_df["category"] == "ascii_candidate"]
print(f"=== ASCII candidate files ({len(ascii_files)}) ===")
print(ascii_files["filename"].head(20).to_string())


# %% [markdown]
# # Phase A: Data inventory and loading
#
# Load all `.sdt` and `.asc` fit-export files from the two acquisition sessions.
# Build a metadata DataFrame with one row per acquisition, sorted by time.
#
# **Data directories:**
# - `20260429_KPC_fixed_dishes_on_SLIM`
# - `20260501_KPC_fixed_dishes_on_SLIM`
#
# **Filename convention (decoded from first run):**
# ```
# {index}_{sample}_{wavelength}nm_{power}mW_{usb}_{pockels}_{objective}_{em_filter}_{gain}_{z}_{pixels}_{mode}_{frames}_{frame_idx}.sdt
# ```

# %%


# %% [markdown]
# ## Step 1: Discover all files
#
# Find `.sdt` files (raw TCSPC data) and `.asc` files (SPCImage fit exports).

# %%
def find_data_files(roots: list[Path]) -> pd.DataFrame:
    """Recursively find all .sdt and .asc files under the given roots."""
    records = []
    for root in roots:
        for fp in sorted(root.rglob("*")):
            if not fp.is_file():
                continue
            ext = fp.suffix.lower()
            if ext not in (".sdt", ".asc"):
                continue
            records.append({
                "filepath": fp,
                "filename": fp.name,
                "stem": fp.stem,
                "suffix": ext,
                "category": "sdt" if ext == ".sdt" else "asc",
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
print(f"\nSDT: {len(sdt_files)}  |  ASC: {len(asc_files)}")

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
    ("em_filter_nm",    r"_(\d{3})s\d+_",               lambda m: int(m.group(1))),

    # EM filter bandwidth: "457s50" -> 50, "370s10" -> 10
    ("em_bandwidth_nm", r"_\d{3}s(\d+)_",               lambda m: int(m.group(1))),

    # Gain: "g70" -> 70
    ("gain",            r"_g(\d+)_",                     lambda m: int(m.group(1))),

    # Z position: "z1" -> 1
    ("z_position",      r"_z(\d+)_",                     lambda m: int(m.group(1))),

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
    if "chromablue" in name:
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

sdt_df = pd.concat([sdt_files.reset_index(drop=True), fname_meta], axis=1)

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
    urea       -> urea_reference
    pollen     -> pollen_reference
    glu/form/live with a cell_type -> sample
    """
    fix = row.get("fixation_type")
    if fix == "chromablue":
        return "chroma"
    if fix == "urea":
        return "urea_reference"
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
# **Note:** No files matched as IRF — check if your IRF acquisitions
# are in a separate directory or have a different naming convention.
# You may need to add a keyword to `_extract_fixation_type()` or
# manually tag them:
# ```python
# sdt_df.loc[sdt_df["filename"].str.contains("some_irf_pattern"), "file_type"] = "irf_before"
# ```

# %% [markdown]
# ## Step 3: Load SDT metadata from file headers
#
# Parse each `.sdt` file to extract acquisition timestamps,
# data shape, photon statistics, and time-axis parameters.

# %%
from sdtfile import SdtFile


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
    print(f"Total photons range: {sdt_df['total_photons'].min():.0f} – {sdt_df['total_photons'].max():.0f}")
    print(f"Mean photons/pixel: {sdt_df['mean_photons_per_pixel'].min():.1f} – {sdt_df['mean_photons_per_pixel'].max():.1f}")

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
        print("All BH timestamps within 60s of filesystem mtime — good.")

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
print(f"Loading {len(asc_files)} .asc fit export files...")

fit_data: dict[str, dict[str, np.ndarray | Path]] = {}
parse_failures = []

for _, row in asc_files.iterrows():
    result = load_spcimage_asc(row["filepath"])
    if result is None:
        parse_failures.append(row["filename"])
        continue

    base = _strip_param_suffix(row["stem"])
    if base not in fit_data:
        fit_data[base] = {}

    fit_data[base][result["param_name"]] = result["data"]
    fit_data[base][f"_path_{result['param_name']}"] = result["filepath"]

print(f"Grouped into {len(fit_data)} image sets")
if parse_failures:
    print(f"  {len(parse_failures)} files failed to parse: {parse_failures[:10]}")

print("\nFirst few image sets:")
for base, params in list(fit_data.items())[:8]:
    param_names = sorted(k for k in params if not k.startswith("_"))
    shapes = [str(params[k].shape) for k in param_names]
    print(f"  {base}: {', '.join(f'{n} {s}' for n, s in zip(param_names, shapes))}")


# %% [markdown]
# ## Step 4b: Match fit exports to SDT files
#
# Link each ascii export group to its corresponding SDT file.
# Acquisition timestamps for matched `.asc` files are inherited from
# the SDT (we don't care about mtime on the `.asc` files themselves).

# %%
def match_fits_to_sdt(sdt_df: pd.DataFrame, fit_data: dict) -> pd.DataFrame:
    """
    Match ascii fit export groups to SDT files by filename stem similarity.

    Adds columns: has_fit_export, fit_params_available, fit_base_stem.
    """
    df = sdt_df.copy()
    df["has_fit_export"] = False
    df["fit_params_available"] = None
    df["fit_base_stem"] = None

    # Build lookup: lowered base stems -> original key
    fit_lookup = {k.lower(): k for k in fit_data}

    for idx, row in df.iterrows():
        sdt_stem = row["stem"].lower()

        # Try exact match
        if sdt_stem in fit_lookup:
            match_key = fit_lookup[sdt_stem]
        else:
            # Try containment: sdt stem starts with or contains a fit base
            match_key = None
            for base_lower, base_orig in fit_lookup.items():
                if base_lower in sdt_stem or sdt_stem in base_lower:
                    match_key = base_orig
                    break

        if match_key:
            param_names = sorted(k for k in fit_data[match_key] if not k.startswith("_"))
            df.at[idx, "has_fit_export"] = True
            df.at[idx, "fit_params_available"] = ", ".join(param_names)
            df.at[idx, "fit_base_stem"] = match_key

    return df


sdt_df = match_fits_to_sdt(sdt_df, fit_data)

n_with_fits = sdt_df["has_fit_export"].sum()
n_samples = (sdt_df["file_type"] == "sample").sum()
print(f"=== Fit export matching ===")
print(f"  {n_with_fits} / {len(sdt_df)} total SDT files have matching fit exports")
print(f"  {sdt_df.loc[sdt_df['file_type'] == 'sample', 'has_fit_export'].sum()} / {n_samples} sample files")

unmatched = sdt_df[(sdt_df["file_type"] == "sample") & (~sdt_df["has_fit_export"])]
if len(unmatched):
    print(f"\n  {len(unmatched)} sample files WITHOUT fit exports:")
    print(unmatched[["filename"]].head(20).to_string())

# %% [markdown]
# ## Step 5: Quick sanity check — load one file end-to-end

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
print(f"  Time axis:    {time_axis.shape}, {time_axis[0]*1e9:.2f} – {time_axis[-1]*1e9:.2f} ns")
print(f"  Bin width:    {(time_axis[1] - time_axis[0])*1e9:.4f} ns")
print(f"  Total photons: {decay.sum():,.0f}")

# Intensity image
intensity = decay.sum(axis=2)
print(f"  Intensity:    shape={intensity.shape}, "
      f"min={intensity.min()}, max={intensity.max()}, mean={intensity.mean():.1f}")

# Check fit export shape match
if test_row["has_fit_export"]:
    base = test_row["fit_base_stem"]
    param_names = sorted(k for k in fit_data[base] if not k.startswith("_"))
    print(f"\n  Fit params: {param_names}")

    for name in param_names:
        arr = fit_data[base][name]
        print(f"    {name:10s}: shape={arr.shape}, range=[{np.nanmin(arr):.3g}, {np.nanmax(arr):.3g}]")

    # Shape match check
    if "photons" in fit_data[base]:
        fit_shape = fit_data[base]["photons"].shape
        sdt_shape = intensity.shape
        if fit_shape == sdt_shape:
            print(f"\n  Shape match: SDT {sdt_shape} == export {fit_shape}")
        else:
            print(f"\n  *** SHAPE MISMATCH: SDT {sdt_shape} vs export {fit_shape} ***")
            print(f"      Check for flip/transpose — pipeline step 7")

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


def get_fit_arrays(row: pd.Series) -> dict[str, np.ndarray] | None:
    """Get fit parameter arrays for a given SDT file row, or None."""
    if not row.get("has_fit_export"):
        return None
    base = row["fit_base_stem"]
    if base not in fit_data:
        return None
    return {k: v for k, v in fit_data[base].items() if not k.startswith("_")}


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
# **Next**: `02_calibration.py` — IRF comparison, chroma phasor
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


# %% [markdown]
# ## Step 2: Load SDT metadata
#
# Parse each `.sdt` file header to extract acquisition timestamp,
# scan dimensions, time-bin parameters, and other instrument settings.
#
# **Note on `sdtfile` >= 2024.9.14**: `measure_info` fields are now
# returned as scalars, and some struct field names were updated to
# match the BH C reference. If you get attribute errors, check your
# `sdtfile` version.
 
# %%
from sdtfile import SdtFile
 
 
def extract_sdt_metadata(filepath: Path, channel: int = 0) -> dict:
    """
    Extract scalar metadata from a single SDT file.
 
    Returns a dict with instrument parameters. Handles both old and new
    sdtfile API field names gracefully.
    """
    meta = {"filepath": filepath, "filename": filepath.name}
 
    try:
        sdt = SdtFile(str(filepath))
    except Exception as e:
        meta["load_error"] = str(e)
        return meta
 
    # -- Timestamp extraction --
    # Strategy 1: parse from the info text block (most reliable)
    # The info block typically contains lines like:
    #   *DATE : 29 Apr 2026
    #   *TIME : 14:32:05
    try:
        info_text = sdt.info.id if hasattr(sdt.info, "id") else str(sdt.info)
        meta["info_text"] = info_text[:500]  # store first 500 chars for inspection
    except Exception:
        info_text = ""
 
    # Strategy 2: header creation_time / creation_date (C struct fields)
    try:
        meta["header_creation_time"] = str(sdt.header.creation_time)
        meta["header_creation_date"] = str(sdt.header.creation_date)
    except AttributeError:
        pass
 
    # Strategy 3: filesystem mtime as fallback
    meta["mtime"] = datetime.fromtimestamp(filepath.stat().st_mtime)
 
    # -- Parse acquisition timestamp from info text --
    # BH SPCM typically writes date/time in the info block
    meta["acquisition_time"] = _parse_bh_timestamp(info_text, fallback=meta["mtime"])
 
    # -- Measurement parameters from measure_info --
    if sdt.measure_info:
        mi = sdt.measure_info[0]
 
        # Scan dimensions
        meta["scan_x"] = int(getattr(mi, "scan_x", 0))
        meta["scan_y"] = int(getattr(mi, "scan_y", 0))
 
        # TAC parameters
        meta["tac_range"] = float(getattr(mi, "tac_range", 0))
        meta["tac_gain"] = float(getattr(mi, "tac_gain", 0))
 
        # Collection time
        meta["collect_time"] = float(getattr(mi, "collect_time", 0))
 
        # Scan parameters
        meta["scan_rxy"] = int(getattr(mi, "scan_rxy", 0))
 
    # -- Data shape info --
    if sdt.data:
        meta["n_data_blocks"] = len(sdt.data)
        data = sdt.data[channel] if channel < len(sdt.data) else sdt.data[0]
        meta["data_shape"] = str(data.shape)
        meta["data_dtype"] = str(data.dtype)
 
        if data.ndim == 3:
            meta["n_pixels_x"] = data.shape[0]
            meta["n_pixels_y"] = data.shape[1]
            meta["n_timebins"] = data.shape[2]
            meta["total_photons"] = int(data.sum())
            meta["max_photons_per_pixel"] = int(data.sum(axis=2).max())
            meta["mean_photons_per_pixel"] = float(data.sum(axis=2).mean())
        elif data.ndim == 2:
            # Might be a single-point decay or 1D scan
            meta["n_timebins"] = data.shape[-1]
            meta["total_photons"] = int(data.sum())
 
    # -- Time axis info --
    if sdt.times:
        t = sdt.times[channel] if channel < len(sdt.times) else sdt.times[0]
        meta["time_axis_len"] = len(t)
        meta["bin_width_ns"] = float(t[1] - t[0]) * 1e9 if len(t) > 1 else np.nan
        meta["time_range_ns"] = float(t[-1] - t[0]) * 1e9 if len(t) > 1 else np.nan
 
    return meta
 
 
def _parse_bh_timestamp(info_text: str, fallback: datetime | None = None) -> datetime | None:
    """
    Try to parse an acquisition timestamp from the BH info text block.
 
    The info block from SPCM typically looks like:
        *IDENTIFICATION
        *DATE : 29 Apr 2026  *TIME : 14:32:05
        ...
    or sometimes:
        #DATE    29-Apr-2026
        #TIME    14:32:05
 
    Falls back to mtime if parsing fails.
    """
    if not info_text:
        return fallback
 
    # Try pattern: *DATE : DD Mon YYYY  *TIME : HH:MM:SS
    m = re.search(
        r"\*?DATE\s*:?\s*(\d{1,2}[\s\-]?\w{3}[\s\-]?\d{4})"
        r".*?"
        r"\*?TIME\s*:?\s*(\d{1,2}:\d{2}:\d{2})",
        info_text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        date_str = m.group(1).replace("-", " ").strip()
        time_str = m.group(2).strip()
        for fmt in ["%d %b %Y %H:%M:%S", "%d %B %Y %H:%M:%S"]:
            try:
                return datetime.strptime(f"{date_str} {time_str}", fmt)
            except ValueError:
                continue
 
    # Try pattern: #DATE DD-Mon-YYYY  #TIME HH:MM:SS
    m = re.search(
        r"#DATE\s+(\S+).*?#TIME\s+(\S+)",
        info_text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        for fmt in ["%d-%b-%Y %H:%M:%S", "%d-%B-%Y %H:%M:%S"]:
            try:
                return datetime.strptime(f"{m.group(1)} {m.group(2)}", fmt)
            except ValueError:
                continue
 
    return fallback
 
# %%
# Build metadata for all SDT files
print(f"Loading metadata from {len(sdt_files)} SDT files...")
 
sdt_meta_records = []
errors = []
 
for i, row in sdt_files.iterrows():
    if (i + 1) % 10 == 0 or i == 0:
        print(f"  [{i+1}/{len(sdt_files)}] {row['filename']}")
    meta = extract_sdt_metadata(row["filepath"])
    meta["session_root"] = row["session_root"]
    meta["session_dir"] = row["session_dir"]
    if "load_error" in meta:
        errors.append(meta)
    sdt_meta_records.append(meta)
 
sdt_df = pd.DataFrame(sdt_meta_records)
print(f"\nLoaded: {len(sdt_df)} files, {len(errors)} errors")
 
if errors:
    print("\n=== LOAD ERRORS ===")
    for e in errors:
        print(f"  {e['filename']}: {e['load_error']}")
 
# %%
# Sort by acquisition time
sdt_df = sdt_df.sort_values("acquisition_time").reset_index(drop=True)
 
# Quick summary
print("=== SDT metadata summary ===")
print(f"Time span: {sdt_df['acquisition_time'].min()} �넂 {sdt_df['acquisition_time'].max()}")
print(f"Scan dimensions: {sdt_df['data_shape'].value_counts().to_string()}")
print(f"Total photons range: {sdt_df['total_photons'].min():.0f} – {sdt_df['total_photons'].max():.0f}")
print(f"Mean photons/pixel range: {sdt_df['mean_photons_per_pixel'].min():.1f} – {sdt_df['mean_photons_per_pixel'].max():.1f}")
 
# %% [markdown]
# ## Step 2b: Tag file types
#
# Classify each file as IRF, chroma calibration, or sample data.
# This needs to be adapted to your naming conventions.
# Below are common patterns — **edit the rules to match your filenames**.
 
# %%
def classify_file_type(filename: str, filepath: Path) -> str:
    """
    Classify a file as irf, chroma, or sample based on naming conventions.
 
    >>> classify_file_type("IRF_gold_before.sdt", Path("."))
    'irf'
    >>> classify_file_type("chroma_slide_after.sdt", Path("."))
    'chroma'
    >>> classify_file_type("KPC_dish3_colony1.sdt", Path("."))
    'sample'
 
    EDIT THESE RULES to match your actual naming scheme.
    """
    name_lower = filename.lower()
 
    # IRF patterns
    if any(kw in name_lower for kw in ["irf", "instrument_response", "ludox", "gold_nanorod", "scatter"]):
        if "after" in name_lower or "post" in name_lower:
            return "irf_after"
        elif "before" in name_lower or "pre" in name_lower:
            return "irf_before"
        return "irf"
 
    # Chroma calibration patterns
    if any(kw in name_lower for kw in ["chroma", "calibration", "reference", "rhodamine", "fluorescein"]):
        if "after" in name_lower or "post" in name_lower:
            return "chroma_after"
        elif "before" in name_lower or "pre" in name_lower:
            return "chroma_before"
        return "chroma"
 
    # Everything else is sample data
    return "sample"
 
 
sdt_df["file_type"] = sdt_df.apply(
    lambda row: classify_file_type(row["filename"], row["filepath"]), axis=1
)
 
print("=== File type classification ===")
print(sdt_df["file_type"].value_counts().to_string())
print()
print("--- Review a few classifications ---")
print(sdt_df[["filename", "file_type"]].head(20).to_string())
 
# %% [markdown]
# **ACTION REQUIRED**: Review the classifications above. If the auto-tagging
# is wrong, either:
# 1. Edit `classify_file_type()` to match your naming scheme, or
# 2. Manually override specific rows:
# ```python
# sdt_df.loc[sdt_df["filename"] == "some_file.sdt", "file_type"] = "irf_before"
# ```
 
# %% [markdown]
# ## Step 2c: Tag sample metadata
#
# For sample files, add columns for cell appearance and colony size.
# Again, adapt to your naming conventions or add manually.
 
# %%
def parse_sample_tags(filename: str) -> dict:
    """
    Extract cell_appearance and colony_size from the filename.
 
    EDIT THIS to match your naming convention. Common patterns:
    - KPC_dish3_colony1_large_epithelial.sdt
    - d3_c1_mesenchymal_small.sdt
 
    Returns dict with 'cell_appearance' and 'colony_size' (or None).
    """
    name_lower = filename.lower()
    tags = {"cell_appearance": None, "colony_size": None}
 
    # Cell appearance keywords -- adapt to your categories
    appearance_keywords = ["epithelial", "mesenchymal", "mixed", "round", "elongated", "spindle"]
    for kw in appearance_keywords:
        if kw in name_lower:
            tags["cell_appearance"] = kw
            break
 
    # Colony size keywords
    size_keywords = ["small", "medium", "large", "single", "cluster"]
    for kw in size_keywords:
        if kw in name_lower:
            tags["colony_size"] = kw
            break
 
    return tags
 
 
# Apply to sample files only
sample_mask = sdt_df["file_type"] == "sample"
sample_tags = sdt_df.loc[sample_mask, "filename"].apply(parse_sample_tags).apply(pd.Series)
sdt_df.loc[sample_mask, "cell_appearance"] = sample_tags["cell_appearance"].values
sdt_df.loc[sample_mask, "colony_size"] = sample_tags["colony_size"].values
 
print("=== Sample tag summary ===")
print(f"cell_appearance: {sdt_df['cell_appearance'].value_counts(dropna=False).to_string()}")
print()
print(f"colony_size: {sdt_df['colony_size'].value_counts(dropna=False).to_string()}")
 
# %% [markdown]
# If the auto-parsing didn't find your tags (lots of `None` above),
# you can load tags from a separate spreadsheet:
# ```python
# tags = pd.read_csv("sample_tags.csv")  # columns: filename, cell_appearance, colony_size
# sdt_df = sdt_df.merge(tags, on="filename", how="left", suffixes=("", "_manual"))
# ```
 
# %% [markdown]
# ## Step 3: Load and match ascii fit exports
#
# Parse the BH SPCImage ascii exports. These typically contain
# per-pixel fit parameters: a1, a2, a3, tau1, tau2, tau3, photon count,
# chi짼, etc.
#
# The exact format depends on your SPCImage export settings.
# Below we handle two common layouts.
 
# %%
def identify_ascii_format(filepath: Path, n_peek_lines: int = 30) -> dict | None:
    """
    Peek at an ascii file and determine if it's a BH fit export.
 
    Returns a dict describing the format, or None if not a fit export.
    """
    try:
        with open(filepath, "r", errors="replace") as f:
            lines = [f.readline() for _ in range(n_peek_lines)]
    except Exception:
        return None
 
    text = "".join(lines)
 
    # SPCImage ascii export usually has a header with parameter name
    # and then a 2D grid of values. Common indicators:
    if any(kw in text.lower() for kw in ["a1[%]", "t1[ps]", "chi", "photons", "a1 [%]", "tau1"]):
        return {"format": "spcimage_export", "filepath": filepath}
 
    # Could also be a single-line-per-pixel CSV format
    # with columns like: x, y, a1, tau1, a2, tau2, ...
    header = lines[0].strip().lower() if lines else ""
    if any(kw in header for kw in ["a1", "tau1", "t1", "chi"]):
        return {"format": "csv_per_pixel", "filepath": filepath}
 
    return None
 
 
def load_spcimage_ascii(filepath: Path) -> dict | None:
    """
    Load a BH SPCImage ascii export file.
 
    SPCImage exports one parameter per file as a space/tab-delimited 2D grid.
    The filename usually encodes the parameter, e.g.:
        image_a1.asc, image_t1.asc, image_chi.asc, image_photons.asc
 
    Returns dict with:
        - 'param_name': inferred parameter name
        - 'data': 2D numpy array
        - 'filepath': source path
    Or None if parsing fails.
    """
    # Infer parameter name from filename
    stem = filepath.stem.lower()
    param_name = _infer_param_name(stem)
 
    try:
        # SPCImage exports are typically space-delimited, no header row
        # Try loading as a numeric grid
        data = np.loadtxt(str(filepath), dtype=float)
    except ValueError:
        # Might have a header line — skip it
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
    """Infer the fit parameter name from a filename stem."""
    # Map common BH SPCImage export suffixes to canonical names
    patterns = {
        r"a1[_%]?$|_a1$": "a1",
        r"a2[_%]?$|_a2$": "a2",
        r"a3[_%]?$|_a3$": "a3",
        r"t1|tau1": "tau1",
        r"t2|tau2": "tau2",
        r"t3|tau3": "tau3",
        r"chi|chisq": "chi2",
        r"photon|intensity|cnt": "photons",
        r"tm|mean": "tau_mean",
    }
    for pattern, name in patterns.items():
        if re.search(pattern, stem):
            return name
    return stem  # fallback: use the stem itself
 
# %%
# Scan ascii candidates and identify fit exports
print("Scanning ascii files for fit exports...")
 
fit_export_info = []
for _, row in ascii_files.iterrows():
    fmt = identify_ascii_format(row["filepath"])
    if fmt:
        fmt["filename"] = row["filename"]
        fmt["stem"] = row["stem"]
        fmt["session_root"] = row["session_root"]
        fit_export_info.append(fmt)
 
print(f"Found {len(fit_export_info)} ascii fit export files")
if fit_export_info:
    fit_exports_df = pd.DataFrame(fit_export_info)
    print(fit_exports_df["format"].value_counts().to_string())
    print()
    print("First few:")
    print(fit_exports_df[["filename", "format"]].head(20).to_string())
 
# %%
# Load the actual fit data from ascii exports
# Group by the base image name (strip the parameter suffix)
print("Loading ascii fit data...")
 
fit_data = {}  # key: base_stem -> dict of param_name -> 2D array
 
for info in fit_export_info:
    result = load_spcimage_ascii(info["filepath"])
    if result is None:
        print(f"  WARN: could not parse {info['filename']}")
        continue
 
    # Determine the base image name by stripping the parameter suffix
    # e.g. "KPC_dish3_colony1_a1.asc" -> base = "KPC_dish3_colony1"
    base = re.sub(
        r"[_\-]?(a[123]|t[123]|tau[123]|chi|chisq|photon[s]?|intensity|cnt|tm|mean)[_%]?$",
        "",
        info["stem"],
        flags=re.IGNORECASE,
    ).strip("_- ")
 
    if base not in fit_data:
        fit_data[base] = {}
 
    fit_data[base][result["param_name"]] = result["data"]
    fit_data[base][f"{result['param_name']}_filepath"] = result["filepath"]
 
print(f"\nGrouped into {len(fit_data)} image sets")
for base, params in list(fit_data.items())[:5]:
    param_names = [k for k in params if not k.endswith("_filepath")]
    shapes = [str(params[k].shape) for k in param_names]
    print(f"  {base}: {', '.join(f'{n} {s}' for n, s in zip(param_names, shapes))}")
 
# %% [markdown]
# ## Step 3b: Match fit exports to SDT files
#
# Link each ascii export group to its corresponding SDT file.
 
# %%
def match_fits_to_sdt(sdt_df: pd.DataFrame, fit_data: dict) -> pd.DataFrame:
    """
    Match ascii fit export groups to SDT files by stem similarity.
 
    Adds columns: has_fit_export, fit_params_available, fit_base_stem
    """
    sdt_df = sdt_df.copy()
    sdt_df["has_fit_export"] = False
    sdt_df["fit_params_available"] = None
    sdt_df["fit_base_stem"] = None
 
    for idx, row in sdt_df.iterrows():
        sdt_stem = row["stem"].lower()
 
        # Try exact match first
        if sdt_stem in fit_data:
            match_key = sdt_stem
        else:
            # Try fuzzy: check if the sdt stem starts with or contains
            # any fit_data key, or vice versa
            match_key = None
            for base in fit_data:
                if base in sdt_stem or sdt_stem in base:
                    match_key = base
                    break
 
        if match_key:
            param_names = [k for k in fit_data[match_key] if not k.endswith("_filepath")]
            sdt_df.at[idx, "has_fit_export"] = True
            sdt_df.at[idx, "fit_params_available"] = ", ".join(sorted(param_names))
            sdt_df.at[idx, "fit_base_stem"] = match_key
 
    return sdt_df
 
 
sdt_df = match_fits_to_sdt(sdt_df, fit_data)
 
n_matched = sdt_df["has_fit_export"].sum()
n_total = len(sdt_df[sdt_df["file_type"] == "sample"])
print(f"=== Fit export matching ===")
print(f"  {n_matched}/{n_total} sample SDT files have matching fit exports")
print()
print("Files WITHOUT fit exports:")
no_fit = sdt_df[(sdt_df["file_type"] == "sample") & (~sdt_df["has_fit_export"])]
if len(no_fit):
    print(no_fit[["filename", "stem"]].to_string())
else:
    print("  (none — all matched!)")
 
# %% [markdown]
# ## Quick sanity check: display the assembled DataFrame
 
# %%
# Show key columns
display_cols = [
    "filename", "file_type", "acquisition_time", "session_root",
    "n_pixels_x", "n_pixels_y", "n_timebins",
    "total_photons", "mean_photons_per_pixel", "bin_width_ns",
    "cell_appearance", "colony_size",
    "has_fit_export", "fit_params_available",
]
# Only show columns that exist
display_cols = [c for c in display_cols if c in sdt_df.columns]
 
print(f"=== Full metadata table ({len(sdt_df)} rows) ===")
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
    print(sdt_df[display_cols].to_string())
 
# %% [markdown]
# ## Step 3c: Convenience loader for downstream analysis
#
# Functions to retrieve decay arrays and fit arrays by index or filename.
 
# %%
# Cache for loaded SDT data arrays (avoids re-reading from disk)
_sdt_cache: dict[str, SdtFile] = {}
 
 
def load_sdt_data(filepath: Path, channel: int = 0, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the TCSPC decay array and time axis from an SDT file.
 
    Returns:
        decay: array of shape (scan_x, scan_y, n_timebins)
        time_axis: array of shape (n_timebins,) in seconds
    """
    key = str(filepath)
    if use_cache and key in _sdt_cache:
        sdt = _sdt_cache[key]
    else:
        sdt = SdtFile(str(filepath))
        if use_cache:
            _sdt_cache[key] = sdt
 
    decay = sdt.data[channel]
    time_axis = sdt.times[channel]
    return decay, time_axis
 
 
def get_fit_arrays(sdt_row: pd.Series) -> dict[str, np.ndarray] | None:
    """
    Get the fit parameter arrays for a given SDT file row.
 
    Returns dict mapping param names to 2D arrays, or None if no fit export.
    """
    if not sdt_row.get("has_fit_export"):
        return None
 
    base = sdt_row["fit_base_stem"]
    if base not in fit_data:
        return None
 
    return {k: v for k, v in fit_data[base].items() if not k.endswith("_filepath")}
 
 
def get_row(sdt_df: pd.DataFrame, filename: str | None = None, idx: int | None = None) -> pd.Series:
    """Look up a single row by filename or integer index."""
    if filename is not None:
        matches = sdt_df[sdt_df["filename"] == filename]
        if len(matches) == 0:
            raise KeyError(f"No file named '{filename}'")
        return matches.iloc[0]
    elif idx is not None:
        return sdt_df.iloc[idx]
    else:
        raise ValueError("Provide either filename or idx")
 
# %% [markdown]
# ## Verify: load one file and inspect
 
# %%
# Pick the first sample file with a fit export
test_row = sdt_df[sdt_df["has_fit_export"]].iloc[0] if sdt_df["has_fit_export"].any() else sdt_df.iloc[0]
 
print(f"Test file: {test_row['filename']}")
print(f"  Type: {test_row['file_type']}")
print(f"  Acquired: {test_row['acquisition_time']}")
 
decay, time_axis = load_sdt_data(test_row["filepath"])
print(f"  Decay shape: {decay.shape}")
print(f"  Time axis: {time_axis.shape}, range = {time_axis[0]*1e9:.2f} – {time_axis[-1]*1e9:.2f} ns")
print(f"  Bin width: {(time_axis[1] - time_axis[0])*1e9:.4f} ns")
print(f"  Total photons: {decay.sum():,.0f}")
 
# Intensity image (sum over time bins)
intensity = decay.sum(axis=2)
print(f"  Intensity image: shape={intensity.shape}, min={intensity.min()}, max={intensity.max()}, mean={intensity.mean():.1f}")
 
# Check fit export if available
fits = get_fit_arrays(test_row)
if fits:
    print(f"\n  Fit parameters available: {list(fits.keys())}")
    for name, arr in fits.items():
        print(f"    {name}: shape={arr.shape}, range=[{arr.min():.3f}, {arr.max():.3f}]")
 
    # Verify shape match
    if "photons" in fits:
        shape_match = fits["photons"].shape == intensity.shape
        print(f"\n  Shape match (intensity vs photons export): {shape_match}")
        if not shape_match:
            print(f"    !!! SDT intensity: {intensity.shape} vs export photons: {fits['photons'].shape}")
            print(f"    You may need a flip/transpose — see pipeline step 7")
 
# %% [markdown]
# ## Summary
#
# At this point you have:
# - `sdt_df`: DataFrame with one row per SDT file, containing metadata,
#   acquisition timestamps, file type classification, and sample tags
# - `fit_data`: dict of per-image fit parameter arrays from ascii exports
# - `load_sdt_data(filepath)`: returns `(decay, time_axis)` for any file
# - `get_fit_arrays(row)`: returns fit parameter dict for any matched file
#
# **Next notebook**: `02_calibration.py` — IRF comparison, chroma
# calibration, and temporal drift assessment.
 
# %%
# Save the metadata table for use in subsequent notebooks
output_path = Path("../results/sdt_metadata.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
 
# Save without filepath (not portable) — keep filename and session_root for joining
save_cols = [c for c in sdt_df.columns if c not in ("filepath", "info_text")]
sdt_df[save_cols].to_csv(output_path, index=False)
print(f"Saved metadata to {output_path}")
print(f"Shape: {sdt_df.shape}")






## %% old code follows

# 
# ### IRF inspection
# Load and inspect the instrument response function before fitting.

# %%
import sdtfile
import matplotlib.pyplot as plt
import numpy as np
import os


# %%
# Inline images
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")

sdt = sdtfile.SdtFile("data/raw/irf_reference.sdt")
irf = sdt.data[0].sum(axis=(0, 1))  # collapse spatial dims

# %%
plt.plot(sdt.times[0] * 1e9, irf)
plt.xlabel("time (ns)")
plt.ylabel("photon counts")
plt.title("IRF")
plt.show()
# %%

%matplotlib inline
# %%
print(os.getcwd())


# %%

from src.preprocess import estimate_irf_width
fwhm = estimate_irf_width(irf, sdt.times[0])
print(f"IRF FWHM: {fwhm*1e12:.1f} ps")

# %%

import matplotlib.pyplot as plt


# %%
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.plot([1,2,3],[1,4,9])
plt.savefig('/tmp/test_plot.png')
print("saved")

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.plot([1,2,3],[1,4,9])
plt.show()

# %%
import matplotlib
print(matplotlib.get_backend())

# %%
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 4, 9])
plt.show()
