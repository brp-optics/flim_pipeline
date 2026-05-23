"""validate_flim.py -- throwaway validation script.

Shows calibrated phasor, tau_mean image, tau_mean histogram, and photon image
(with and without quality masking) for one center-of-acquisition-period file
per (session x fixation_type x channel).

Run from the project root:
    uv run python validate_flim.py

Output: results/figures/validation/validate_{session}_{fixation}_{channel}nm.png
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from sdtfile import SdtFile

# ============================================================================
# Configuration -- update CURRENT_OS and paths to match your machine
# ============================================================================

WIN_DATA_DIRS = [
    Path(r"E:\18_RK_Circadian\data\raw\20260429_KPC_fixed_dishes_on_SLIM"),
    Path(r"E:\18_RK_Circadian\data\raw\20260501_KPC_fixed_dishes_on_SLIM"),
    Path(r"E:\18_RK_Circadian\data\raw\20260509_KPC_fixed_dishes_on_SLIM"),
    Path(r"E:\18_RK_Circadian\data\raw\20260508_KPC_live_on_SLIM"),
    Path(r"E:\18_RK_Circadian\data\raw\20260517_KPC_live_on_SLIM"),
]
LIN_DATA_DIRS = [
    Path("/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260429_KPC_fixed_dishes_on_SLIM"),
    Path("/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260501_KPC_fixed_dishes_on_SLIM"),
    Path("/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260509_KPC_fixed_dishes_on_SLIM"),
    Path("/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260508_KPC_live_on_SLIM"),
    Path("/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260517_KPC_live_on_SLIM"),
]
CURRENT_OS = "Win"
data_dirs = WIN_DATA_DIRS if CURRENT_OS == "Win" else LIN_DATA_DIRS

RESULTS_DIR = Path("results")
OUT_DIR = Path("results/figures/validation")

REP_RATE_HZ = 80e6
OMEGA = 2 * np.pi * REP_RATE_HZ

# tau_mean colormap limits (ps) for spatial images and histogram x-axis
TAU_CLIM_PS = (400, 1400)

# Lifetime reference marks on the phasor universal semicircle (ns)
PHASOR_TAU_LABELS_NS = [1, 2, 3, 4, 5, 6, 7, 8]

# Preferred fit components per fixation type (mirrors Phase E)
PREFERRED_N_COMP = {"glu": 2, "form": 2, "live": 2}

# Amplitude-weighted tau_mean component pairs (a_key, tau_key)
TAUMEAN_CFG = {
    "glu":  (("a1", "tau1"), ("a2", "tau2")),
    "form": (("a1", "tau1"), ("a2", "tau2")),
    "live": (("a1", "tau1"), ("a2", "tau2")),
}

# Quality mask config (must match Phase E exactly to show the same mask)
CHI2_LO = 0.8
CHI2_HI = 2.0
MIN_PHOTONS_BY_NCOMP = {1: 500, 2: 3_000, 3: 8_000, None: 3_000}
TAU_BOUNDS = {
    "glu":  {"tau1": (50.0, 1500.0), "tau2": (800.0, 6000.0)},
    "form": {"tau1": (50.0, 1500.0), "tau2": (800.0, 6000.0)},
    "live": {"tau1": (50.0, 1500.0), "tau2": (800.0, 6000.0)},
}
TAU_MEAN_BOUNDS = {"glu": None, "form": None, "live": (250.0, None)}

# ============================================================================
# Self-contained helpers (minimal copies from Phase B/E notebooks)
# ============================================================================

def phasor_from_decay(decay, t_s, omega):
    """decay: (..., nt), t_s: (nt,) seconds. Returns G, S arrays."""
    cos_t = np.cos(omega * t_s)
    sin_t = np.sin(omega * t_s)
    total = decay.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.where(total > 0, (decay * cos_t).sum(axis=-1) / total, np.nan)
        S = np.where(total > 0, (decay * sin_t).sum(axis=-1) / total, np.nan)
    return G, S


def phasor_apply_cal(G, S, phase_corr, mod_corr):
    G_cal = mod_corr * (G * np.cos(phase_corr) - S * np.sin(phase_corr))
    S_cal = mod_corr * (G * np.sin(phase_corr) + S * np.cos(phase_corr))
    return G_cal, S_cal


def compute_tau_mean(fd, fixation_type):
    components = TAUMEAN_CFG.get(fixation_type)
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


def _infer_param_name(stem):
    s = stem.lower()
    for pattern, name in [
        (r"[-_]a1[_%]?$", "a1"),    (r"[-_]a2[_%]?$", "a2"),
        (r"[-_]a3[_%]?$", "a3"),    (r"[-_]t1$|[-_]tau1$", "tau1"),
        (r"[-_]t2$|[-_]tau2$", "tau2"), (r"[-_]t3$|[-_]tau3$", "tau3"),
        (r"[-_]chi$|[-_]chisq?$", "chi2"),
        (r"[-_]photons?$|[-_]int(ensity)?$|[-_]cnt$", "photons"),
    ]:
        if re.search(pattern, s):
            return name
    return stem.rsplit("_", 1)[-1] if "_" in stem else stem


def _strip_param_suffix(stem):
    return re.sub(
        r"[-_]?(a[123]|t[123]|tau[123]|chi|chisq?|photons?|intensity|cnt|"
        r"tm|tau_?mean|scatter|sc|shift|[gs])[_%]?$",
        "", stem, flags=re.IGNORECASE,
    ).strip("_- ")


def load_asc_fit_set(fit_dir, base_stem):
    fd = {}
    for fp in sorted(fit_dir.glob("*.asc")):
        if re.search(r"_statistic", fp.stem, re.IGNORECASE):
            continue
        if _strip_param_suffix(fp.stem).lower() != base_stem.lower():
            continue
        param = _infer_param_name(fp.stem)
        try:
            data = np.loadtxt(str(fp), dtype=float)
        except ValueError:
            try:
                data = np.loadtxt(str(fp), dtype=float, skiprows=1)
            except Exception:
                continue
        except Exception:
            continue
        if data.ndim == 2 and data.size > 0:
            fd[param] = data

    # BH phasor export: "{base_stem}_phasor.asc" is a two-column (G S) flat file,
    # one row per pixel in raster order.
    phasor_path = fit_dir / f"{base_stem}_phasor.asc"
    if phasor_path.exists():
        try:
            pdata = np.loadtxt(str(phasor_path), dtype=float)
            if pdata.ndim == 2 and pdata.shape[1] == 2:
                n = pdata.shape[0]
                # infer shape from other loaded arrays, or assume square
                shape2d = next(
                    (v.shape for v in fd.values() if isinstance(v, np.ndarray) and v.ndim == 2),
                    None,
                )
                if shape2d is None:
                    side = int(round(n ** 0.5))
                    shape2d = (side, side)
                if shape2d[0] * shape2d[1] == n:
                    fd["G"] = pdata[:, 0].reshape(shape2d)
                    fd["S"] = pdata[:, 1].reshape(shape2d)
        except Exception:
            pass

    return fd


def fit_dir_from_key(fit_set_key):
    session_root, rel_dir, base_stem = fit_set_key.split("::", 2)
    session_dir = next((d for d in data_dirs if d.name == session_root), None)
    fit_dir = (session_dir / Path(rel_dir)) if session_dir else None
    return fit_dir, session_root, rel_dir, base_stem


def best_fit_key_for(filename, fixation_type, fit_map_df):
    rows = fit_map_df[fit_map_df["sdt_filename"] == filename]
    if rows.empty:
        return None
    preferred = PREFERRED_N_COMP.get(str(fixation_type))
    if preferred is not None and "n_components" in rows.columns:
        exact = rows[rows["n_components"] == preferred]
        if not exact.empty:
            return str(exact.iloc[0]["fit_set_key"])
    if "n_components" in rows.columns:
        return str(rows.sort_values("n_components", ascending=False).iloc[0]["fit_set_key"])
    return str(rows.iloc[0]["fit_set_key"])


def compute_quality_mask(fd, fixation_type, b_val, n_components=None):
    """Return (mask, breakdown) where breakdown is per-criterion pass rates (0-100)."""
    shape = next(
        (v.shape for v in fd.values() if isinstance(v, np.ndarray) and v.ndim == 2),
        None,
    )
    if shape is None:
        return None, {}

    min_ph = MIN_PHOTONS_BY_NCOMP.get(n_components, MIN_PHOTONS_BY_NCOMP[None])
    photon_src = fd.get("photons")
    if photon_src is not None and photon_src.shape == shape:
        kernel_area = (2 * b_val + 1) ** 2
        smoothed    = uniform_filter(photon_src.astype(float), size=2 * b_val + 1)
        mask_photon = smoothed * kernel_area >= min_ph
    else:
        mask_photon = np.ones(shape, dtype=bool)

    chi2 = fd.get("chi2")
    if chi2 is not None and chi2.shape == shape:
        mask_chi2 = np.isfinite(chi2) & (chi2 >= CHI2_LO) & (chi2 <= CHI2_HI)
    else:
        mask_chi2 = np.ones(shape, dtype=bool)

    amp_names = [p for p in ("a1", "a2", "a3") if p in fd and fd[p].shape == shape]
    if amp_names:
        mask_amp = np.ones(shape, dtype=bool)
        for p in amp_names:
            mask_amp &= fd[p] >= 0.0
        mask_amp &= sum(fd[p] for p in amp_names) > 0.0
    else:
        mask_amp = np.ones(shape, dtype=bool)

    tau_cfg  = TAU_BOUNDS.get(fixation_type, {})
    mask_tau = np.ones(shape, dtype=bool)
    for param, (lo, hi) in tau_cfg.items():
        if param in fd and fd[param].shape == shape:
            arr = fd[param]
            mask_tau &= np.isfinite(arr) & (arr >= lo) & (arr <= hi)

    taumean_cfg = TAU_MEAN_BOUNDS.get(fixation_type)
    if taumean_cfg is not None:
        tau_m = compute_tau_mean(fd, fixation_type)
        if tau_m is not None and tau_m.shape == shape:
            lo_tm, hi_tm = taumean_cfg
            mask_taumean = np.isfinite(tau_m)
            if lo_tm is not None:
                mask_taumean &= tau_m >= lo_tm
            if hi_tm is not None:
                mask_taumean &= tau_m <= hi_tm
        else:
            mask_taumean = np.ones(shape, dtype=bool)
    else:
        mask_taumean = np.ones(shape, dtype=bool)

    mask_final = mask_photon & mask_chi2 & mask_amp & mask_tau & mask_taumean

    breakdown = {
        "photon": 100.0 * mask_photon.mean(),
        "chi2":   100.0 * mask_chi2.mean(),
        "amp":    100.0 * mask_amp.mean(),
        "tau":    100.0 * mask_tau.mean(),
        "taum":   100.0 * mask_taumean.mean(),
        "final":  100.0 * mask_final.mean(),
    }
    return mask_final, breakdown


def _draw_semicircle_with_labels(ax):
    """Draw universal semicircle with lifetime reference marks."""
    theta = np.linspace(0, np.pi, 300)
    ax.plot(0.5 + 0.5 * np.cos(theta), 0.5 * np.sin(theta),
            "k-", lw=0.8, alpha=0.5)
    for tau_ns in PHASOR_TAU_LABELS_NS:
        x = OMEGA * tau_ns * 1e-9
        G = 1.0 / (1.0 + x ** 2)
        S = x / (1.0 + x ** 2)
        ax.plot(G, S, "k.", ms=4, zorder=3)
        # offset label radially outward from semicircle center (0.5, 0)
        nx, ny = G - 0.5, S
        r = np.hypot(nx, ny)
        if r > 0:
            nx /= r; ny /= r
        ax.text(G + nx * 0.06, S + ny * 0.06, f"{tau_ns}",
                fontsize=5.5, ha="center", va="center", color="0.35")


# ============================================================================
# Load metadata
# ============================================================================

print("Loading metadata...")
sdt_df = pd.read_csv(RESULTS_DIR / "sdt_metadata_cal.csv")
fp_map = pd.read_csv(RESULTS_DIR / "filepath_map.csv")
sdt_df = sdt_df.merge(fp_map[["filename", "filepath"]], on="filename", how="left")
sdt_df["filepath"]         = sdt_df["filepath"].map(Path)
sdt_df["acquisition_time"] = pd.to_datetime(sdt_df["acquisition_time"])

fit_map_df = pd.read_csv(RESULTS_DIR / "fit_map.csv")
if "n_components" in fit_map_df.columns:
    fit_map_df["n_components"] = pd.to_numeric(
        fit_map_df["n_components"], errors="coerce"
    ).astype("Int64")

# Filter to calibrated sample files
samples = sdt_df[sdt_df["file_type"] == "sample"].copy()
for col in ("phasor_cal_phase_rad", "phasor_cal_mod"):
    samples[col] = pd.to_numeric(samples[col], errors="coerce")
samples = samples.dropna(subset=["phasor_cal_phase_rad", "phasor_cal_mod"])
samples["em_filter_nm"] = samples["em_filter_nm"].fillna("unknown").astype(str)

print(f"  {len(samples)} calibrated sample files")

# Keep only files that have a BH phasor export on disk
print("  Checking for _phasor.asc exports...", flush=True)

def _has_phasor_export(filename, fixation_type, verbose=False):
    fkey = best_fit_key_for(filename, fixation_type, fit_map_df)
    if fkey is None:
        if verbose:
            print(f"    [no fit key] {filename}")
        return False
    fit_dir, _, _, base_stem = fit_dir_from_key(fkey)
    if fit_dir is None:
        if verbose:
            print(f"    [fit_dir None] fkey={fkey[:60]}")
        return False
    phasor_path = fit_dir / f"{base_stem}_phasor.asc"
    if verbose:
        print(f"    checking: {phasor_path}  exists={phasor_path.exists()}")
    return phasor_path.exists()

# Diagnose first file to catch path issues early
_sample_row = samples.iloc[0]
print(f"  Diagnostic on first file: {_sample_row['filename']}")
_has_phasor_export(_sample_row["filename"], _sample_row["fixation_type"], verbose=True)

has_phasor = samples.apply(
    lambda r: _has_phasor_export(r["filename"], r["fixation_type"]), axis=1
)
samples = samples[has_phasor]
print(f"  {len(samples)} files with BH phasor export")

# Report any groups that were entirely filtered out
all_groups  = set(sdt_df[sdt_df["file_type"] == "sample"].groupby(
    ["session_root", "fixation_type", "em_filter_nm"]).groups.keys())
kept_groups = set(samples.groupby(
    ["session_root", "fixation_type", "em_filter_nm"]).groups.keys())
for g in sorted(all_groups - kept_groups):
    print(f"  [NO PHASOR EXPORT] skipping group {g}")

# Select one file per (session_root, fixation_type, em_filter_nm):
# sort by acquisition_time, pick the middle index
groups = samples.groupby(["session_root", "fixation_type", "em_filter_nm"], sort=True)
print(f"  {len(groups)} groups (session x fixation x channel)")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Per-group figure generation
# ============================================================================

for (session, fix_type, channel), grp in groups:
    grp_sorted = grp.sort_values("acquisition_time").reset_index(drop=True)
    row = grp_sorted.iloc[len(grp_sorted) // 2]

    label = f"{session}  {fix_type}  {channel}nm  (n={len(grp_sorted)} files)"
    print(f"\n{label}")
    print(f"  File: {row['filename']}")

    # ------------------------------------------------------------------
    # 1. Load SDT and compute calibrated phasor
    # ------------------------------------------------------------------
    try:
        sdt = SdtFile(str(row["filepath"]))
    except Exception as e:
        print(f"  [SKIP] Cannot open SDT: {e}")
        continue

    data = sdt.data[0].astype(float)
    t_s  = sdt.times[0]

    if data.ndim != 3:
        print(f"  [SKIP] Unexpected data shape {data.shape}")
        continue

    photons_sdt = data.sum(axis=-1)
    G_raw, S_raw = phasor_from_decay(data, t_s, OMEGA)

    # phasor_cal_phase_rad and phasor_cal_mod already include the urea correction
    phase_corr = float(row["phasor_cal_phase_rad"])
    mod_corr   = float(row["phasor_cal_mod"])
    G_cal, S_cal = phasor_apply_cal(G_raw, S_raw, phase_corr, mod_corr)

    # ------------------------------------------------------------------
    # 2. Load .asc fit set for tau_mean and photon images
    # ------------------------------------------------------------------
    fkey             = best_fit_key_for(row["filename"], fix_type, fit_map_df)
    fd               = {}
    b_val            = 2
    n_comp           = None
    fit_folder_disp  = "no fit"

    if fkey is not None:
        fit_dir, _, rel_dir, base_stem = fit_dir_from_key(fkey)
        if fit_dir is not None and fit_dir.exists():
            fd = load_asc_fit_set(fit_dir, base_stem)
            folder_name = Path(rel_dir).name
            m_b = re.search(r"(?:^|[-_])b(\d+)", folder_name, re.IGNORECASE)
            if m_b:
                b_val = int(m_b.group(1))
            m_c = re.search(r"[-_](\d+)component", folder_name, re.IGNORECASE)
            if not m_c:
                m_c = re.search(r"[-_]c(\d+)(?=[-_]|$)", folder_name, re.IGNORECASE)
            if m_c:
                n_comp = int(m_c.group(1))
            fit_folder_disp = folder_name
        else:
            fit_folder_disp = "fit dir not found"
    print(f"  Fit folder: {fit_folder_disp}  b={b_val}  nc={n_comp}  params={list(fd.keys())}")

    tau_m       = compute_tau_mean(fd, fix_type)
    photons_asc = fd.get("photons")

    # Prefer SPCImage phasor exports (_g.asc / _s.asc) over our computed phasor.
    # BH values are already calibrated by SPCImage's own IRF.
    bh_G = fd.get("G")
    bh_S = fd.get("S")
    if bh_G is not None and bh_S is not None:
        G_phasor   = bh_G
        S_phasor   = bh_S
        phasor_src = "BH export"
    else:
        G_phasor   = G_cal
        S_phasor   = S_cal
        phasor_src = "computed+cal"
    print(f"  Phasor source: {phasor_src}")

    # ------------------------------------------------------------------
    # 3. Quality mask + per-criterion breakdown
    # ------------------------------------------------------------------
    # Always compute breakdown from fd (shows current-config thresholds).
    # Override final mask with pre-saved .npy if available.
    quality_mask = None
    breakdown    = {}
    mask_source  = "unavailable"

    if fd:
        quality_mask, breakdown = compute_quality_mask(fd, fix_type, b_val, n_comp)
        mask_source = "computed"

    if fkey is not None:
        fit_dir, _, _, base_stem = fit_dir_from_key(fkey)
        if fit_dir is not None:
            npy_path = fit_dir / f"{base_stem}_fit_mask.npy"
            if npy_path.exists():
                quality_mask = np.load(str(npy_path))
                mask_source  = "saved .npy"

    if quality_mask is not None:
        pct = 100.0 * quality_mask.mean()
        print(f"  Mask ({mask_source}): {quality_mask.sum()}/{quality_mask.size} px  ({pct:.1f}%)")
        if breakdown:
            print(f"  Breakdown: photon {breakdown['photon']:.0f}%  chi2 {breakdown['chi2']:.0f}%  "
                  f"amp {breakdown['amp']:.0f}%  tau {breakdown['tau']:.0f}%  "
                  f"taum {breakdown['taum']:.0f}%  final {breakdown['final']:.0f}%")
    else:
        print("  Mask: unavailable (no fit data)")

    # Use .asc photons for display if available; fall back to SDT sum
    photons_disp = photons_asc if photons_asc is not None else photons_sdt
    ph_source    = "asc" if photons_asc is not None else "SDT sum"

    # Phasor validity (pixels with actual signal)
    phasor_valid = np.isfinite(G_phasor) & np.isfinite(S_phasor)

    # ------------------------------------------------------------------
    # 4. Build 2 x 4 figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        f"{label}\nfit: {fit_folder_disp}    file: {row['filename']}",
        fontsize=7.5, y=1.01,
    )

    col_titles = [
        f"Photon image ({ph_source})",
        "tau_mean map (ps)",
        "tau_mean histogram",
        "Calibrated phasor",
    ]
    row_titles = ["All pixels", "Quality-masked"]

    for ci, title in enumerate(col_titles):
        axes[0, ci].set_title(title, fontsize=9, fontweight="bold")

    for ri, title in enumerate(row_titles):
        axes[ri, 0].set_ylabel(title, fontsize=9, fontweight="bold", labelpad=8)

    for ri in range(2):
        # Effective mask for this row
        if ri == 0:
            eff_mask = phasor_valid
        else:
            if quality_mask is not None and quality_mask.shape == photons_disp.shape:
                eff_mask = quality_mask
            else:
                eff_mask = phasor_valid

        # ------ Col 0: photon image ------
        ax = axes[ri, 0]
        ph_show = photons_disp.astype(float).copy()
        if ri == 1:
            ph_show[~eff_mask] = np.nan
        vlo = float(np.nanpercentile(ph_show, 1))
        vhi = float(np.nanpercentile(ph_show, 99))
        im = ax.imshow(ph_show, cmap="gray", vmin=vlo, vmax=vhi)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Per-criterion breakdown overlay on the masked photon panel
        if ri == 1 and breakdown:
            # Highlight criteria that removed more than 5% of pixels
            lines = []
            for key, label_str in [("photon", "photon"), ("chi2", "chi2"),
                                    ("amp", "amp"), ("tau", "tau"),
                                    ("taum", "taum"), ("final", "FINAL")]:
                pct_val = breakdown[key]
                marker = " <" if (key != "final" and pct_val < 95.0) else "  "
                lines.append(f"{label_str:6s} {pct_val:5.1f}%{marker}")
            ax.text(0.03, 0.03, "\n".join(lines),
                    transform=ax.transAxes, fontsize=6,
                    va="bottom", ha="left", family="monospace",
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.55, pad=2, linewidth=0))
        ax.axis("off")

        # ------ Col 1: tau_mean spatial map ------
        ax = axes[ri, 1]
        if tau_m is not None:
            tm_show = tau_m.copy()
            if ri == 1:
                tm_show[~eff_mask] = np.nan
            im = ax.imshow(tm_show, cmap="turbo",
                           vmin=TAU_CLIM_PS[0], vmax=TAU_CLIM_PS[1])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ps")
        else:
            ax.text(0.5, 0.5, "no fit data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
        ax.axis("off")

        # ------ Col 2: tau_mean histogram ------
        ax = axes[ri, 2]
        if tau_m is not None:
            valid_tm = eff_mask & np.isfinite(tau_m) & (tau_m > 0)
            tm_vals  = tau_m[valid_tm]
            if len(tm_vals) > 10:
                ax.hist(tm_vals, bins=80, range=TAU_CLIM_PS,
                        color="steelblue", alpha=0.8, density=True)
                # photon-weighted mean to match SPCImage statistics panel
                ph_weights = photons_disp[valid_tm].astype(float) \
                    if photons_disp is not None else None
                if ph_weights is not None and ph_weights.sum() > 0:
                    mn = float(np.average(tm_vals, weights=ph_weights))
                    wt_label = "wtd mean"
                else:
                    mn = float(np.mean(tm_vals))
                    wt_label = "mean"
                ax.axvline(mn, color="crimson", lw=1.5,
                           label=f"{wt_label} {mn:.0f} ps  n={len(tm_vals)}")
                ax.legend(fontsize=7)
            else:
                ax.text(0.5, 0.5, f"n={len(tm_vals)} px", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
        else:
            ax.text(0.5, 0.5, "no fit data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_xlabel("tau_mean (ps)", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7)

        # ------ Col 3: calibrated phasor hexbin ------
        ax = axes[ri, 3]
        mask_flat = eff_mask.ravel()
        g_flat = G_phasor.ravel()[mask_flat]
        s_flat = S_phasor.ravel()[mask_flat]
        if len(g_flat) > 20:
            hb = ax.hexbin(g_flat, s_flat, gridsize=80, bins="log",
                           cmap="inferno", extent=(-0.1, 1.1, -0.05, 0.65))
            fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="log(n)")
        _draw_semicircle_with_labels(ax)
        ax.plot(1.0, 0.0, "b+", ms=10, mew=2, label="(1,0) scatter")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.05, 0.65)
        ax.set_xlabel("G", fontsize=8)
        ax.set_ylabel("S", fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        n_px = int(mask_flat.sum())
        ax.set_title(f"n={n_px} px  [{phasor_src}]", fontsize=7)

    plt.tight_layout()

    safe = lambda s: re.sub(r"[^A-Za-z0-9_-]", "_", str(s))
    out_name = f"validate_{safe(session)}_{safe(fix_type)}_{safe(channel)}nm.png"
    out_path = OUT_DIR / out_name
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

print("\nAll validation figures saved.")
