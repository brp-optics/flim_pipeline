"""compare_pairs.py -- side-by-side BKO vs KPCWT fluorescence lifetime comparison.

For each (fixation_type x em_filter_nm) group, matches BKO and KPCWT files by
median photon count (intensity similarity), then generates a 2x4 figure per pair
showing photons, tau_mean image, tau_mean histogram, and phasor overlay.

Run from the project root:
    uv run python compare_pairs.py

Output: results/figures/pairs/{fixation}_{channel}nm_pair{N:02d}.png
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.optimize import linear_sum_assignment
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

RESULTS_DIR = Path("results")
OUT_DIR = Path("results/figures/pairs")

N_PAIRS_PER_GROUP = 6        # max matched pairs shown per (fixation x channel) group
TAU_CLIM_PS = (400, 1400)    # shared tau_mean colorscale
PHASOR_TAU_LABELS_NS = [1, 2, 3, 4, 5, 6, 7, 8]
REP_RATE_HZ = 80e6
OMEGA = 2 * np.pi * REP_RATE_HZ

PREFERRED_N_COMP = {"glu": 2, "form": 2, "live": 2}
TAUMEAN_CFG = {
    "glu":  (("a1", "tau1"), ("a2", "tau2")),
    "form": (("a1", "tau1"), ("a2", "tau2")),
    "live": (("a1", "tau1"), ("a2", "tau2")),
}

# ============================================================================
# Self-contained helpers (copied from validate_flim.py)
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
# File data loading
# ============================================================================

def load_file_data(row, fit_map_df):
    """Load SDT + asc fit set for a single file row.

    Returns a dict with keys:
        photons   -- 2D array of photon counts (asc preferred, SDT sum fallback)
        tau_m     -- 2D tau_mean array, or None if unavailable
        G_cal     -- 2D calibrated phasor G (always from SDT + phasor_apply_cal)
        S_cal     -- 2D calibrated phasor S
        mask      -- 2D bool quality mask, or None
        median_ph -- float, median photon count over the full photon image
    Returns None on any critical failure.
    """
    filename   = row["filename"]
    fix_type   = row["fixation_type"]
    filepath   = Path(row["filepath"])

    # --- Open SDT ---
    try:
        sdt = SdtFile(str(filepath))
    except Exception as e:
        print(f"    [WARN] Cannot open SDT {filename}: {e}")
        return None

    data = sdt.data[0].astype(float)
    t_s  = sdt.times[0]

    if data.ndim != 3:
        print(f"    [WARN] Unexpected data shape {data.shape} in {filename}")
        return None

    photons_sdt = data.sum(axis=-1)

    # --- Calibrated phasor (always computed from raw decay) ---
    G_raw, S_raw = phasor_from_decay(data, t_s, OMEGA)
    phase_corr = float(row["phasor_cal_phase_rad"])
    mod_corr   = float(row["phasor_cal_mod"])
    G_cal, S_cal = phasor_apply_cal(G_raw, S_raw, phase_corr, mod_corr)

    # --- Load .asc fit set ---
    fkey = best_fit_key_for(filename, fix_type, fit_map_df)
    fd   = {}
    tau_m     = None
    mask      = None
    photons   = photons_sdt  # default fallback

    if fkey is not None:
        fit_dir, _, rel_dir, base_stem = fit_dir_from_key(fkey)
        if fit_dir is not None and fit_dir.exists():
            fd = load_asc_fit_set(fit_dir, base_stem)

            # Prefer asc photons over SDT sum
            if "photons" in fd and fd["photons"].shape == photons_sdt.shape:
                photons = fd["photons"]
            elif "photons" in fd:
                photons = fd["photons"]

            # tau_mean
            tau_m = compute_tau_mean(fd, fix_type)

            # Quality mask -- try saved .npy first
            npy_path = fit_dir / f"{base_stem}_fit_mask.npy"
            if npy_path.exists():
                try:
                    mask = np.load(str(npy_path)).astype(bool)
                except Exception:
                    mask = None

    median_ph = float(np.nanmedian(photons))

    return {
        "photons":   photons,
        "tau_m":     tau_m,
        "G_cal":     G_cal,
        "S_cal":     S_cal,
        "mask":      mask,
        "median_ph": median_ph,
    }


# ============================================================================
# Matching helpers
# ============================================================================

def match_pairs_by_intensity(bko_rows, kpcwt_rows, bko_medians, kpcwt_medians):
    """Use linear_sum_assignment to optimally pair BKO with KPCWT by median photons.

    Returns list of (bko_idx, kpcwt_idx) index pairs into the input lists.
    """
    n_bko   = len(bko_rows)
    n_kpcwt = len(kpcwt_rows)
    # Cost matrix: |log(median_bko) - log(median_kpcwt)| to match on relative scale.
    # Use log so a 2x difference looks the same regardless of absolute counts.
    log_bko   = np.log1p(np.array(bko_medians, dtype=float))
    log_kpcwt = np.log1p(np.array(kpcwt_medians, dtype=float))
    cost = np.abs(log_bko[:, None] - log_kpcwt[None, :])  # (n_bko, n_kpcwt)
    row_ind, col_ind = linear_sum_assignment(cost)
    return list(zip(row_ind.tolist(), col_ind.tolist()))


# ============================================================================
# Figure generation
# ============================================================================

def make_pair_figure(kpcwt_row, bko_row, kpcwt_data, bko_data,
                     fix_type, channel, pair_n, total_pairs):
    """Build 2x4 comparison figure for one matched pair.

    Rows: KPCWT (row 0), BKO (row 1)
    Cols: photons, tau_mean image, tau_mean histogram, phasor
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # ---- Shared colorscale for photons (joint 1st-99th percentile) ----
    ph_kpcwt = kpcwt_data["photons"].astype(float)
    ph_bko   = bko_data["photons"].astype(float)

    # Apply quality masks to NaN outside mask before computing joint range
    mask_kpcwt = kpcwt_data["mask"]
    mask_bko   = bko_data["mask"]

    def _masked_ph(ph, mask):
        arr = ph.copy()
        if mask is not None and mask.shape == ph.shape:
            arr[~mask] = np.nan
        return arr

    ph_k_m = _masked_ph(ph_kpcwt, mask_kpcwt)
    ph_b_m = _masked_ph(ph_bko,   mask_bko)

    all_ph = np.concatenate([ph_k_m[np.isfinite(ph_k_m)],
                              ph_b_m[np.isfinite(ph_b_m)]])
    if all_ph.size > 0:
        ph_vmin = float(np.percentile(all_ph, 1))
        ph_vmax = float(np.percentile(all_ph, 99))
    else:
        ph_vmin, ph_vmax = 0.0, 1.0

    # ---- Suptitle ----
    ratio = max(kpcwt_data["median_ph"], bko_data["median_ph"]) / max(
        min(kpcwt_data["median_ph"], bko_data["median_ph"]), 1.0
    )
    fig.suptitle(
        (f"{fix_type}  {channel}nm  |  intensity ratio: {ratio:.2f}x"
         f"  |  pair {pair_n}/{total_pairs}\n"
         f"KPCWT: {kpcwt_row['filename']} | median {kpcwt_data['median_ph']:.0f} ph\n"
         f"BKO:   {bko_row['filename']}   | median {bko_data['median_ph']:.0f} ph"),
        fontsize=8, y=1.01,
    )

    cell_types = ["KPCWT", "BKO"]
    rows_data  = [kpcwt_data, bko_data]
    rows_meta  = [kpcwt_row,  bko_row]

    for ri, (ctype, fdata, frow) in enumerate(zip(cell_types, rows_data, rows_meta)):
        mask = fdata["mask"]
        ph   = fdata["photons"].astype(float)
        tau_m = fdata["tau_m"]
        G_cal = fdata["G_cal"]
        S_cal = fdata["S_cal"]

        # Apply quality mask: NaN outside mask
        ph_show = ph.copy()
        if mask is not None and mask.shape == ph.shape:
            ph_show[~mask] = np.nan

        tau_show = None
        if tau_m is not None:
            tau_show = tau_m.copy()
            if mask is not None and mask.shape == tau_m.shape:
                tau_show[~mask] = np.nan

        # Row ylabel
        axes[ri, 0].set_ylabel(ctype, fontsize=10, fontweight="bold", labelpad=8)

        # ---- Col 0: Photons ----
        ax = axes[ri, 0]
        im = ax.imshow(ph_show, cmap="gray", vmin=ph_vmin, vmax=ph_vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")
        if ri == 0:
            ax.set_title("Photons", fontsize=9, fontweight="bold")

        # ---- Col 1: tau_mean image ----
        ax = axes[ri, 1]
        if ri == 0:
            ax.set_title("tau_mean (ps)", fontsize=9, fontweight="bold")
        if tau_show is not None:
            im = ax.imshow(tau_show, cmap="turbo",
                           vmin=TAU_CLIM_PS[0], vmax=TAU_CLIM_PS[1])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ps")
        else:
            ax.text(0.5, 0.5, "no fit data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
        ax.axis("off")

        # ---- Col 2: tau_mean histogram (both overlaid, drawn on row 0 only) ----
        ax = axes[ri, 2]
        if ri == 0:
            ax.set_title("tau_mean histogram", fontsize=9, fontweight="bold")
            # Draw both distributions on the same axes (row 0 panel)
            for hist_data, hist_mask, hist_ph, color, label_prefix in [
                (kpcwt_data["tau_m"], kpcwt_data["mask"],
                 kpcwt_data["photons"], "steelblue", "KPCWT"),
                (bko_data["tau_m"],   bko_data["mask"],
                 bko_data["photons"],  "darkorange", "BKO"),
            ]:
                if hist_data is None:
                    continue
                if hist_mask is not None and hist_mask.shape == hist_data.shape:
                    valid = hist_mask & np.isfinite(hist_data) & (hist_data > 0)
                else:
                    valid = np.isfinite(hist_data) & (hist_data > 0)
                vals = hist_data[valid]
                if len(vals) < 10:
                    continue
                ax.hist(vals, bins=80, range=TAU_CLIM_PS,
                        color=color, alpha=0.6, density=True, label=label_prefix)
                # photon-weighted mean
                ph_w = hist_ph[valid].astype(float) if hist_ph is not None else None
                if ph_w is not None and ph_w.sum() > 0:
                    mn = float(np.average(vals, weights=ph_w))
                else:
                    mn = float(np.mean(vals))
                ax.axvline(mn, color=color, lw=1.5, linestyle="--",
                           label=f"{label_prefix} wtd mean {mn:.0f} ps")
            ax.legend(fontsize=6.5)
            ax.set_xlabel("tau_mean (ps)", fontsize=8)
            ax.set_ylabel("density", fontsize=8)
            ax.tick_params(labelsize=7)
        else:
            # Row 1 histogram panel: leave empty with a note
            ax.axis("off")
            ax.text(0.5, 0.5, "(histogram above)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")

        # ---- Col 3: Phasor ----
        ax = axes[ri, 3]
        if ri == 0:
            ax.set_title("Phasor", fontsize=9, fontweight="bold")

        # Overlay both phasors on the same axes in row 0; just BKO in row 1.
        # For the shared phasor panel (row 0), plot KPCWT then BKO.
        if ri == 0:
            # Draw KPCWT (blues)
            g_k = kpcwt_data["G_cal"].ravel()
            s_k = kpcwt_data["S_cal"].ravel()
            fin_k = np.isfinite(g_k) & np.isfinite(s_k)
            if fin_k.sum() > 20:
                ax.hexbin(g_k[fin_k], s_k[fin_k], gridsize=60, bins="log",
                          cmap="Blues", alpha=0.7,
                          extent=(-0.1, 1.1, -0.05, 0.65))
            # Draw BKO (Oranges)
            g_b = bko_data["G_cal"].ravel()
            s_b = bko_data["S_cal"].ravel()
            fin_b = np.isfinite(g_b) & np.isfinite(s_b)
            if fin_b.sum() > 20:
                ax.hexbin(g_b[fin_b], s_b[fin_b], gridsize=60, bins="log",
                          cmap="Oranges", alpha=0.7,
                          extent=(-0.1, 1.1, -0.05, 0.65))
            # Legend proxies
            from matplotlib.patches import Patch
            ax.legend(
                handles=[
                    Patch(facecolor="steelblue",  alpha=0.7, label="KPCWT"),
                    Patch(facecolor="darkorange", alpha=0.7, label="BKO"),
                ],
                fontsize=7, loc="upper right",
            )
        else:
            # Row 1: just this cell's phasor for detail
            g_flat = G_cal.ravel()
            s_flat = S_cal.ravel()
            fin = np.isfinite(g_flat) & np.isfinite(s_flat)
            if fin.sum() > 20:
                cmap = "Blues" if ctype == "KPCWT" else "Oranges"
                ax.hexbin(g_flat[fin], s_flat[fin], gridsize=60, bins="log",
                          cmap=cmap, extent=(-0.1, 1.1, -0.05, 0.65))

        _draw_semicircle_with_labels(ax)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.05, 0.65)
        ax.set_aspect("equal")
        ax.set_xlabel("G", fontsize=8)
        ax.set_ylabel("S", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    return fig


# ============================================================================
# Main
# ============================================================================

def main():
    print("Loading metadata...")
    fit_sum = pd.read_csv(RESULTS_DIR / "fit_analysis_summary.csv")
    sdt_df  = pd.read_csv(RESULTS_DIR / "sdt_metadata_cal.csv")
    fp_map  = pd.read_csv(RESULTS_DIR / "filepath_map.csv")
    fit_map_df = pd.read_csv(RESULTS_DIR / "fit_map.csv")

    if "n_components" in fit_map_df.columns:
        fit_map_df["n_components"] = pd.to_numeric(
            fit_map_df["n_components"], errors="coerce"
        ).astype("Int64")

    # Merge filepath into sdt_df
    sdt_df = sdt_df.merge(fp_map[["filename", "filepath"]], on="filename", how="left")

    # Keep calibrated sample files
    samples = sdt_df[sdt_df["file_type"] == "sample"].copy()
    for col in ("phasor_cal_phase_rad", "phasor_cal_mod"):
        samples[col] = pd.to_numeric(samples[col], errors="coerce")
    samples = samples.dropna(subset=["phasor_cal_phase_rad", "phasor_cal_mod",
                                     "filepath"])
    samples = samples[samples["cell_type"].isin(["KPCWT", "BKO"])].copy()
    samples["em_filter_nm"] = samples["em_filter_nm"].fillna("unknown").astype(str)

    # Merge pct_final from fit_analysis_summary to filter low-quality files
    if "pct_final" in fit_sum.columns:
        pct_col = fit_sum[["filename", "pct_final"]].drop_duplicates("filename")
        samples = samples.merge(pct_col, on="filename", how="left")
        samples["pct_final"] = pd.to_numeric(samples["pct_final"], errors="coerce").fillna(0.0)
        n_before = len(samples)
        samples = samples[samples["pct_final"] > 1.0].copy()
        print(f"  Filtered by pct_final > 1.0: {n_before} -> {len(samples)} files")
    else:
        print("  [WARN] pct_final not in fit_analysis_summary, skipping quality filter")

    print(f"  {len(samples)} calibrated sample files with KPCWT/BKO cell_type")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    groups = samples.groupby(["fixation_type", "em_filter_nm"], sort=True)
    print(f"  {len(groups)} groups (fixation x channel)")

    total_figs = 0

    for (fix_type, channel), grp in groups:
        grp = grp.reset_index(drop=True)
        kpcwt_grp = grp[grp["cell_type"] == "KPCWT"].reset_index(drop=True)
        bko_grp   = grp[grp["cell_type"] == "BKO"].reset_index(drop=True)

        print(f"\n=== {fix_type}  {channel}nm  "
              f"(KPCWT n={len(kpcwt_grp)}, BKO n={len(bko_grp)}) ===")

        if len(kpcwt_grp) < 2 or len(bko_grp) < 2:
            print("  [SKIP] Need >= 2 files per cell type")
            continue

        # ---- Load photon data for all files in this group ----
        print("  Loading file data for intensity matching...")

        def _load_row(row):
            d = load_file_data(row, fit_map_df)
            return d

        kpcwt_data_list = []
        kpcwt_valid_rows = []
        for _, row in kpcwt_grp.iterrows():
            d = _load_row(row)
            if d is not None:
                kpcwt_data_list.append(d)
                kpcwt_valid_rows.append(row)
            else:
                print(f"    [SKIP KPCWT] {row['filename']}")

        bko_data_list = []
        bko_valid_rows = []
        for _, row in bko_grp.iterrows():
            d = _load_row(row)
            if d is not None:
                bko_data_list.append(d)
                bko_valid_rows.append(row)
            else:
                print(f"    [SKIP BKO] {row['filename']}")

        if len(kpcwt_data_list) < 2 or len(bko_data_list) < 2:
            print("  [SKIP] After loading, need >= 2 valid files per cell type")
            continue

        kpcwt_medians = [d["median_ph"] for d in kpcwt_data_list]
        bko_medians   = [d["median_ph"] for d in bko_data_list]

        # ---- Optimal intensity matching ----
        pairs = match_pairs_by_intensity(
            bko_valid_rows, kpcwt_valid_rows,
            bko_medians,    kpcwt_medians,
        )

        # Compute intensity ratio for each pair and sort (best match first)
        def _pair_ratio(bi, ki):
            mb = bko_medians[bi]
            mk = kpcwt_medians[ki]
            return max(mb, mk) / max(min(mb, mk), 1.0)

        pairs_sorted = sorted(pairs, key=lambda p: _pair_ratio(p[0], p[1]))
        pairs_sorted = pairs_sorted[:N_PAIRS_PER_GROUP]

        n_pairs = len(pairs_sorted)
        print(f"  Generating {n_pairs} pair figures...")

        safe = lambda s: re.sub(r"[^A-Za-z0-9_-]", "_", str(s))

        for pair_n, (bi, ki) in enumerate(pairs_sorted, start=1):
            bko_row   = bko_valid_rows[bi]
            kpcwt_row = kpcwt_valid_rows[ki]
            bko_d     = bko_data_list[bi]
            kpcwt_d   = kpcwt_data_list[ki]
            ratio     = _pair_ratio(bi, ki)

            print(f"  Pair {pair_n}/{n_pairs}:  ratio={ratio:.2f}x")
            print(f"    KPCWT: {kpcwt_row['filename']}  median={kpcwt_d['median_ph']:.0f} ph")
            print(f"    BKO:   {bko_row['filename']}   median={bko_d['median_ph']:.0f} ph")

            try:
                fig = make_pair_figure(
                    kpcwt_row, bko_row,
                    kpcwt_d,   bko_d,
                    fix_type, channel,
                    pair_n, n_pairs,
                )
            except Exception as e:
                print(f"    [WARN] Figure failed: {e}")
                plt.close("all")
                continue

            out_name = (f"{safe(fix_type)}_{safe(channel)}nm"
                        f"_pair{pair_n:02d}.png")
            out_path = OUT_DIR / out_name
            fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved: {out_path}")
            total_figs += 1

    print(f"\nDone. {total_figs} pair figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
