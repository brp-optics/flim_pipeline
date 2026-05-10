# %% [markdown]
# # Phase B: Instrument calibration
#
# Steps:
# 4. IRF characterization -- load urea decays, check drift, compare SPCImage exports
# 5. Chroma phasor calibration -- derive phase/modulation correction per timepoint
# 6. Assign calibration to each sample file

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from datetime import datetime
from sdtfile import SdtFile

# -- Configuration ----------------------------------------------------------
REP_RATE_HZ        = 80e6           # laser repetition rate
OMEGA              = 2 * np.pi * REP_RATE_HZ
CHROMA_TAU_REF_NS  = 1.000          # chromablue reference lifetime (ns)
                                    # range observed: 0.80-1.10 ns with urea IRF;
                                    # ~1.000 ns when using adjacent urea
IRF_CLUSTER_WIN_MIN = 10            # consecutive urea measurements within this
                                    # many minutes -> keep only the later one
IRF_PEAK_WARN_BINS  = 1             # warn if IRF peak shifts more than this many bins

# %%
# -- Load Phase A outputs ---------------------------------------------------
results_dir = Path("../results")

sdt_df = pd.read_csv(results_dir / "sdt_metadata.csv")
fp_map = pd.read_csv(results_dir / "filepath_map.csv")

# Re-attach filepaths (not portable, stored separately)
sdt_df = sdt_df.merge(fp_map[["filename", "filepath"]], on="filename", how="left")
sdt_df["filepath"] = sdt_df["filepath"].map(Path)
sdt_df["acquisition_time"] = pd.to_datetime(sdt_df["acquisition_time"])

print(f"Loaded {len(sdt_df)} SDT files")
print(sdt_df["file_type"].value_counts().to_string())

# %% [markdown]
# ## Step 4: IRF (urea) characterization
#
# Urea slides serve as the IRF source. We:
# 1. Load each urea decay (sum over spatial pixels -> 1D histogram)
# 2. Filter clustered measurements (two within IRF_CLUSTER_WIN_MIN -> keep later)
# 3. Compute peak channel and FWHM
# 4. Plot normalized overlaid decays
# 5. Check peak drift between before/after -- warn if > IRF_PEAK_WARN_BINS

# %%
def load_irf_decay_1d(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load urea SDT; return (decay_1d, time_axis_ns). Sums over spatial dims."""
    sdt = SdtFile(str(filepath))
    data = sdt.data[0].astype(float)          # sdtfile returns sum of all frames
    t_ns = sdt.times[0] * 1e9
    decay_1d = data.sum(axis=(0, 1)) if data.ndim == 3 else data.astype(float)
    return decay_1d, t_ns


def normalize_decay(decay: np.ndarray) -> np.ndarray:
    total = decay.sum()
    return decay / total if total > 0 else decay.copy()


def irf_peak_fwhm(decay: np.ndarray, t_ns: np.ndarray) -> dict:
    """Return peak_idx, peak_time_ns, fwhm_ns, total_counts."""
    peak_idx = int(np.argmax(decay))
    half_max = decay[peak_idx] / 2.0
    above = np.where(decay >= half_max)[0]
    bin_width_ns = float(t_ns[1] - t_ns[0]) if len(t_ns) > 1 else 1.0
    fwhm_ns = (above[-1] - above[0] + 1) * bin_width_ns
    return {
        "peak_idx":    peak_idx,
        "peak_time_ns": float(t_ns[peak_idx]),
        "fwhm_ns":     fwhm_ns,
        "total_counts": int(decay.sum()),
    }


def cluster_filter_irf(df: pd.DataFrame, window_min: float) -> pd.DataFrame:
    """Keep only the later of consecutive IRF pairs within window_min minutes."""
    df = df.sort_values("acquisition_time").reset_index(drop=True)
    dt_min = df["acquisition_time"].diff().dt.total_seconds().div(60)
    # Drop row i if row i+1 is within the window (i.e. i is the earlier of a pair)
    next_dt = dt_min.shift(-1)
    drop = (next_dt <= window_min) & next_dt.notna()
    dropped = df[drop]["filename"].tolist()
    if dropped:
        print(f"  Cluster filter: dropped {len(dropped)} earlier IRF(s): {dropped}")
    return df[~drop].reset_index(drop=True)


# %%
irf_all = sdt_df[sdt_df["file_type"] == "irf"].copy()
print(f"Found {len(irf_all)} urea (IRF) files across all sessions\n")

# Per-session analysis
irf_records = []

for session, grp in irf_all.groupby("session_root", sort=True):
    print(f"Session: {session}  ({len(grp)} urea files)")
    grp_filt = cluster_filter_irf(grp, IRF_CLUSTER_WIN_MIN)
    print(f"  After cluster filter: {len(grp_filt)} IRF(s)")

    for _, row in grp_filt.iterrows():
        decay, t_ns = load_irf_decay_1d(row["filepath"])
        metrics = irf_peak_fwhm(decay, t_ns)
        irf_records.append({
            "session_root":    session,
            "filename":        row["filename"],
            "acquisition_time": row["acquisition_time"],
            "decay":           decay,
            "t_ns":            t_ns,
            **metrics,
        })
        print(f"  {row['filename']}")
        print(f"    peak={metrics['peak_time_ns']:.3f} ns  FWHM={metrics['fwhm_ns']:.3f} ns  "
              f"counts={metrics['total_counts']:,}")

irf_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("decay", "t_ns")}
                        for r in irf_records])

# %%
# Plot normalized IRF decays
fig, axes = plt.subplots(1, len(irf_all["session_root"].unique()),
                         figsize=(5 * len(irf_all["session_root"].unique()), 4),
                         sharey=False, squeeze=False)

for ax, (session, sess_records) in zip(axes[0], _group_by_session(irf_records)):
    for rec in sess_records:
        norm = normalize_decay(rec["decay"])
        ax.plot(rec["t_ns"], norm, label=rec["filename"][:30])
    ax.set_title(f"IRF -- {session}")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Normalized counts")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.show()


def _group_by_session(records: list) -> list:
    from itertools import groupby
    keyed = sorted(records, key=lambda r: r["session_root"])
    return [(s, list(g)) for s, g in groupby(keyed, key=lambda r: r["session_root"])]


# %%
# Peak drift check: compare first vs last IRF per session
print("=== IRF peak drift per session ===")
for session, grp in irf_df.groupby("session_root"):
    if len(grp) < 2:
        print(f"  {session}: only 1 IRF -- no drift to measure")
        continue
    first = grp.iloc[0]
    last  = grp.iloc[-1]
    peak_shift_ns = abs(last["peak_time_ns"] - first["peak_time_ns"])
    t_ns_per_bin  = irf_records[0]["t_ns"][1] - irf_records[0]["t_ns"][0]
    peak_shift_bins = peak_shift_ns / t_ns_per_bin
    fwhm_shift_ns   = abs(last["fwhm_ns"] - first["fwhm_ns"])
    flag = "WARNING: large shift -- interpolation may be unreliable" \
           if peak_shift_bins > IRF_PEAK_WARN_BINS else "OK"
    print(f"  {session}:")
    print(f"    peak shift:  {peak_shift_ns:.3f} ns  ({peak_shift_bins:.1f} bins)  {flag}")
    print(f"    FWHM shift:  {fwhm_shift_ns:.3f} ns")
    print(f"    count ratio: {last['total_counts'] / first['total_counts']:.2f}  "
          f"(before={first['total_counts']:,}  after={last['total_counts']:,})")

# %% [markdown]
# ## Step 4b: Compare raw urea IRF vs SPCImage .irf exports
#
# SPCImage exports fitted IRFs as `.irf` files.
# Filename: `date_seq-et.irf` (single) or `date_seq1-seq2-et.irf` (pre-averaged sum).
# We compare the normalized raw urea decay against the normalized .irf export.

# %%
def load_irf_export(filepath: Path) -> np.ndarray:
    """Load BH SPCImage .irf export. Returns 256-bin IRF histogram (unnormalized)."""
    tokens = filepath.read_text().split()
    return np.array(tokens[:256], dtype=float)


def parse_irf_export_sequences(stem: str) -> list[int]:
    """Extract SDT sequence indices from .irf filename stem.

    '20260429_11-et'    -> [11]
    '20260429_11-12-et' -> [11, 12]
    """
    m = re.match(r"\d{8}_(.+)", stem)
    body = m.group(1) if m else stem
    body = re.sub(r"-et$", "", body, flags=re.IGNORECASE)
    return [int(x) for x in re.findall(r"\d+", body)]


# %%
irf_export_files = sdt_df[sdt_df["file_type"] == "irf"][[]].iloc[0:0]  # placeholder
# Re-load irf_files from Phase A context if available; otherwise rebuild from filepath_map
# irf_files was built in Phase A as: files_df[files_df["category"] == "irf"]
# Here we discover them from the same data directories
from sdtfile import SdtFile as _SdtFile  # already imported

_irf_export_paths = list(Path(fp_map["filepath"].iloc[0]).parents[2].rglob("*.irf"))
print(f"Found {len(_irf_export_paths)} .irf export files")

for irf_path in sorted(_irf_export_paths):
    seqs = parse_irf_export_sequences(irf_path.stem)
    export_decay = load_irf_export(irf_path)
    export_norm = normalize_decay(export_decay)

    # Find matching urea SDT rows by sample_index
    matching = irf_all[irf_all["sample_index"].isin(seqs)]
    if matching.empty:
        print(f"  {irf_path.name}: sequences {seqs} -- no matching urea SDT found")
        continue

    # Sum the matching raw urea decays
    raw_sum = np.zeros(256)
    for _, mrow in matching.iterrows():
        d, _ = load_irf_decay_1d(mrow["filepath"])
        raw_sum += d[:256]
    raw_norm = normalize_decay(raw_sum)

    corr = float(np.corrcoef(export_norm, raw_norm)[0, 1])
    max_diff = float(np.abs(export_norm - raw_norm).max())
    print(f"  {irf_path.name}: seqs={seqs}  r={corr:.6f}  max_abs_diff={max_diff:.2e}")

    if corr < 0.999:
        print(f"    WARNING: low correlation -- possible SPCImage bug or normalization issue")
        fig, ax = plt.subplots(figsize=(6, 3))
        t_ax = np.arange(256)
        ax.plot(t_ax, raw_norm, label="raw urea sum")
        ax.plot(t_ax, export_norm, label="SPCImage .irf", linestyle="--")
        ax.set_title(irf_path.name)
        ax.legend()
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## Step 4c: IRF interpolation function
#
# For use in Phase E (fit deconvolution). Not needed for phasor calibration.
#
# Strategy: normalize both IRFs to unit area, then linearly interpolate
# by fractional position of the sample between before/after acquisition times.
# Only valid when peak shift between before/after is small (verified above).

# %%
def interpolate_irf(decay_before: np.ndarray, t_before: datetime,
                    decay_after: np.ndarray,  t_after: datetime,
                    t_sample: datetime) -> tuple[np.ndarray, float]:
    """
    Linearly interpolate between two normalized IRFs.

    Returns (irf_interpolated, weight) where weight=0 -> before, weight=1 -> after.
    Both decays are normalized to unit area before interpolation.
    """
    t0 = t_before.timestamp()
    t1 = t_after.timestamp()
    ts = t_sample.timestamp()
    weight = np.clip((ts - t0) / (t1 - t0), 0.0, 1.0) if t1 > t0 else 0.0
    norm_b = normalize_decay(decay_before)
    norm_a = normalize_decay(decay_after)
    return (1.0 - weight) * norm_b + weight * norm_a, float(weight)


def get_bracketing_irfs(irf_df_sess: pd.DataFrame, t_sample: datetime,
                        irf_decays: dict) -> tuple:
    """
    Return (decay_before, t_before, decay_after, t_after) for a sample time.

    irf_decays: {filename: 1d decay array}
    If sample is before all IRFs or after all IRFs, clamps to nearest edge.
    """
    times = irf_df_sess["acquisition_time"].values
    fnames = irf_df_sess["filename"].values
    t_s = pd.Timestamp(t_sample)

    before_mask = times <= t_s.to_numpy()
    after_mask  = times >  t_s.to_numpy()

    if before_mask.any():
        idx_b = np.where(before_mask)[0][-1]
    else:
        idx_b = 0  # clamp to earliest

    if after_mask.any():
        idx_a = np.where(after_mask)[0][0]
    else:
        idx_a = len(times) - 1  # clamp to latest

    fn_b = fnames[idx_b]
    fn_a = fnames[idx_a]
    return (irf_decays[fn_b], pd.Timestamp(times[idx_b]),
            irf_decays[fn_a], pd.Timestamp(times[idx_a]))


# Build decay lookup for later use
irf_decay_lookup = {rec["filename"]: rec["decay"] for rec in irf_records}

# %% [markdown]
# ## Step 5: Chroma phasor calibration
#
# For each chroma slide acquisition:
# 1. Load the SDT, compute per-pixel phasor (G, S)
# 2. Check spatial uniformity (chroma slide should be spatially flat)
# 3. Mean G, S of the slide -> compare to theoretical for CHROMA_TAU_REF_NS
# 4. Derive phase correction (rad) and modulation correction (scale)

# %%
def phasor_from_decay(decay: np.ndarray, t_s: np.ndarray, omega: float):
    """
    Compute phasor (G, S) from TCSPC decay array.

    decay: (..., nt)  -- last axis is time
    t_s:   (nt,)      -- time axis in seconds
    Returns G, S arrays of shape (...), NaN where total counts == 0.
    """
    cos_t = np.cos(omega * t_s)
    sin_t = np.sin(omega * t_s)
    total = decay.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.where(total > 0, (decay * cos_t).sum(axis=-1) / total, np.nan)
        S = np.where(total > 0, (decay * sin_t).sum(axis=-1) / total, np.nan)
    return G, S


def phasor_theory(tau_ns: float, omega: float) -> tuple[float, float]:
    """Theoretical phasor for single-exponential lifetime tau_ns (in ns)."""
    x = omega * tau_ns * 1e-9
    return 1.0 / (1.0 + x**2), x / (1.0 + x**2)


def phasor_cal_factors(G_meas: float, S_meas: float,
                       G_ref: float,  S_ref: float) -> tuple[float, float]:
    """
    Compute phase offset (rad) and modulation scale from measured vs theoretical phasor.
    Apply with phasor_apply_cal().
    """
    phase_corr = np.arctan2(S_ref, G_ref) - np.arctan2(S_meas, G_meas)
    mod_ref  = np.hypot(G_ref,  S_ref)
    mod_meas = np.hypot(G_meas, S_meas)
    mod_corr = mod_ref / mod_meas if mod_meas > 0 else 1.0
    return float(phase_corr), float(mod_corr)


def phasor_apply_cal(G: np.ndarray, S: np.ndarray,
                     phase_corr: float, mod_corr: float):
    """Apply phase/modulation calibration to phasor arrays."""
    G_cal = mod_corr * (G * np.cos(phase_corr) - S * np.sin(phase_corr))
    S_cal = mod_corr * (G * np.sin(phase_corr) + S * np.cos(phase_corr))
    return G_cal, S_cal


# %%
chroma_all = sdt_df[sdt_df["file_type"] == "chroma"].sort_values("acquisition_time")
print(f"Found {len(chroma_all)} chroma files\n")

G_ref_theory, S_ref_theory = phasor_theory(CHROMA_TAU_REF_NS, OMEGA)
print(f"Theoretical phasor for tau={CHROMA_TAU_REF_NS} ns at {REP_RATE_HZ/1e6:.0f} MHz:")
print(f"  G_ref={G_ref_theory:.4f}  S_ref={S_ref_theory:.4f}  "
      f"|phasor|={np.hypot(G_ref_theory, S_ref_theory):.4f}")

chroma_cal_records = []

for _, row in chroma_all.iterrows():
    sdt = SdtFile(str(row["filepath"]))
    data = sdt.data[0].astype(float)    # (nx, ny, nt)
    t_s  = sdt.times[0]                 # seconds

    G, S = phasor_from_decay(data, t_s, OMEGA)

    # Spatial uniformity
    valid = np.isfinite(G) & np.isfinite(S)
    G_mean, S_mean = float(np.nanmean(G)), float(np.nanmean(S))
    G_std,  S_std  = float(np.nanstd(G)),  float(np.nanstd(S))

    phase_corr, mod_corr = phasor_cal_factors(G_mean, S_mean, G_ref_theory, S_ref_theory)

    print(f"\n{row['filename']}")
    print(f"  G_meas={G_mean:.4f} +/- {G_std:.4f}   S_meas={S_mean:.4f} +/- {S_std:.4f}")
    print(f"  phase_corr={np.degrees(phase_corr):.3f} deg   mod_corr={mod_corr:.4f}")
    print(f"  valid pixels: {valid.sum()} / {valid.size}")

    uniformity_ok = G_std < 0.05 and S_std < 0.05
    if not uniformity_ok:
        print(f"  WARNING: high spatial variance -- chroma slide may be non-uniform")

    chroma_cal_records.append({
        "filename":         row["filename"],
        "session_root":     row["session_root"],
        "acquisition_time": row["acquisition_time"],
        "G_meas":           G_mean,
        "S_meas":           S_mean,
        "G_std":            G_std,
        "S_std":            S_std,
        "phase_corr_rad":   phase_corr,
        "mod_corr":         mod_corr,
    })

chroma_cal_df = pd.DataFrame(chroma_cal_records)

# %%
# Plot: measured chroma phasors vs theoretical on universal semicircle
theta = np.linspace(0, np.pi, 300)
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(0.5 + 0.5 * np.cos(theta), 0.5 * np.sin(theta), "k-", lw=0.8, label="semicircle")
ax.scatter([G_ref_theory], [S_ref_theory], marker="*", s=120, color="black",
           zorder=5, label=f"theory {CHROMA_TAU_REF_NS} ns")
for _, rec in chroma_cal_df.iterrows():
    ax.scatter(rec["G_meas"], rec["S_meas"], label=rec["filename"][:25])
ax.set_xlabel("G")
ax.set_ylabel("S")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.02, 0.55)
ax.set_aspect("equal")
ax.legend(fontsize=7)
ax.set_title("Chroma phasor: measured vs theoretical")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 6: Assign phasor calibration to each sample
#
# For each sample file, find the bracketing chroma calibration measurements
# by acquisition_time and linearly interpolate phase_corr and mod_corr.
# Same cluster/normalization logic: if two chroma measurements within
# IRF_CLUSTER_WIN_MIN minutes, keep the later one.

# %%
def assign_phasor_cal(sdt_df: pd.DataFrame,
                      cal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add phasor calibration columns to sdt_df.

    For samples outside the calibration time range, clamps to nearest edge.
    """
    df = sdt_df.copy()
    df["phasor_cal_phase_rad"] = np.nan
    df["phasor_cal_mod"]       = np.nan
    df["phasor_cal_source"]    = None

    cal_sorted = cal_df.sort_values("acquisition_time").reset_index(drop=True)
    times = cal_sorted["acquisition_time"].values
    phases = cal_sorted["phase_corr_rad"].values
    mods   = cal_sorted["mod_corr"].values
    fnames = cal_sorted["filename"].values

    for idx, row in df[df["file_type"] == "sample"].iterrows():
        t_s = pd.Timestamp(row["acquisition_time"]).to_numpy()

        before_mask = times <= t_s
        after_mask  = times >  t_s

        if before_mask.any() and after_mask.any():
            ib = np.where(before_mask)[0][-1]
            ia = np.where(after_mask)[0][0]
        elif before_mask.any():
            ib = ia = np.where(before_mask)[0][-1]
        else:
            ib = ia = 0

        if ib == ia:
            phase = phases[ib]
            mod   = mods[ib]
            source = fnames[ib]
        else:
            t0 = pd.Timestamp(times[ib]).timestamp()
            t1 = pd.Timestamp(times[ia]).timestamp()
            ts = pd.Timestamp(t_s).timestamp()
            w = np.clip((ts - t0) / (t1 - t0), 0.0, 1.0)
            phase  = (1 - w) * phases[ib] + w * phases[ia]
            mod    = (1 - w) * mods[ib]   + w * mods[ia]
            source = f"interp({fnames[ib]}, {fnames[ia]}, w={w:.2f})"

        df.at[idx, "phasor_cal_phase_rad"] = phase
        df.at[idx, "phasor_cal_mod"]       = mod
        df.at[idx, "phasor_cal_source"]    = source

    return df


# %%
# Cluster-filter chroma calibrations before assigning
chroma_cal_df_filt = cluster_filter_irf(chroma_cal_df, IRF_CLUSTER_WIN_MIN)
print(f"Chroma calibration timepoints after cluster filter: {len(chroma_cal_df_filt)}")

sdt_df = assign_phasor_cal(sdt_df, chroma_cal_df_filt)

n_assigned = sdt_df["phasor_cal_phase_rad"].notna().sum()
n_samples  = (sdt_df["file_type"] == "sample").sum()
print(f"Assigned calibration to {n_assigned} / {n_samples} sample files")
print(f"\nCalibration summary:")
print(sdt_df[sdt_df["file_type"] == "sample"][
    ["filename", "acquisition_time", "phasor_cal_phase_rad", "phasor_cal_mod", "phasor_cal_source"]
].head(10).to_string())

# %%
# Save updated metadata
out_path = results_dir / "sdt_metadata_cal.csv"
save_cols = [c for c in sdt_df.columns if c not in ("filepath", "info_text_snippet")
             and not c.startswith("_")]
sdt_df[save_cols].to_csv(out_path, index=False)
print(f"Saved calibrated metadata to {out_path} ({len(sdt_df)} rows)")

# Also save chroma calibration table for inspection
chroma_cal_df_filt.to_csv(results_dir / "chroma_calibration.csv", index=False)
print(f"Saved chroma calibration table to {results_dir / 'chroma_calibration.csv'}")

# %% [markdown]
# ## Summary
#
# At this point you have:
# - `irf_df`: per-IRF peak/FWHM/counts, cluster-filtered
# - `irf_decay_lookup`: {filename -> 1D decay array} for Phase E fitting
# - `interpolate_irf()` / `get_bracketing_irfs()`: for Phase E deconvolution
# - `chroma_cal_df_filt`: phase_corr_rad, mod_corr per calibration timepoint
# - `sdt_df` updated with phasor_cal_phase_rad, phasor_cal_mod, phasor_cal_source
# - `phasor_apply_cal(G, S, phase_corr, mod_corr)`: apply to any phasor array
#
# **Next:** Phase C (data integrity checks) or Phase D (phasor analysis per sample)
