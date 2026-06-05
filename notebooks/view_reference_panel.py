# %% [markdown]
# # Reference Panel Viewer
#
# Loads the top-ranked candidate from each of the 4 groups produced by
# find_reference_images.py and shows a 4x7 grid:
#   rows : live KPCWT | live BKO | form KPCWT | form BKO
#   cols : photons | tau_mean | a1 | a2 | a1/a2 | chi2 | quality mask
#
# Each panel title shows mean-inside-mask vs mean-outside-mask,
# matching the format used in the coworker reference function.

# %%
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ---------------------------------------------------------------------------
# Config -- match OS paths to Phase E
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("../results")

# data_dirs are set by Phase A (notebooks/20260507_KPC_explore.py) and
# persisted to config/data_dirs.yaml.
import sys as _sys_cfg_v
_sys_cfg_v.path.insert(0, str(Path("..").resolve()))
from src.config import get_data_dirs
DATA_DIRS = get_data_dirs()

GROUPS = [("live", "KPCWT"), ("live", "BKO"), ("form", "KPCWT"), ("form", "BKO")]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session_dir(session_root):
    return next((d for d in DATA_DIRS if d.name == session_root), None)


def _infer_param(stem):
    s = stem.lower()
    for pat, name in [
        (r"[-_]a1[_%]?$",                              "a1"),
        (r"[-_]a2[_%]?$",                              "a2"),
        (r"[-_]a3[_%]?$",                              "a3"),
        (r"[-_]t1$|[-_]tau1$",                         "tau1"),
        (r"[-_]t2$|[-_]tau2$",                         "tau2"),
        (r"[-_]t3$|[-_]tau3$",                         "tau3"),
        (r"[-_]tm$|[-_]tau_?mean$|[-_]mean$",          "tau_mean"),
        (r"[-_]chi$|[-_]chisq?$",                      "chi2"),
        (r"[-_]photons?$|[-_]int(ensity)?$|[-_]cnt$",  "photons"),
        (r"[-_]shift$",                                 "shift"),
    ]:
        if re.search(pat, s):
            return name
    return stem.rsplit("_", 1)[-1]


def _strip_suffix(stem):
    return re.sub(
        r"[-_]?(a[123]|t[123]|tau[123]|chi|chisq?|photons?|intensity|cnt|"
        r"tm|tau_?mean|shift)[_%]?$",
        "", stem, flags=re.IGNORECASE,
    ).strip("_- ")


def load_asc(fit_dir, base_stem):
    fd = {}
    for fp in sorted(fit_dir.glob("*.asc")):
        if re.search(r"_statistic", fp.stem, re.IGNORECASE):
            continue
        if _strip_suffix(fp.stem).lower() != base_stem.lower():
            continue
        param = _infer_param(fp.stem)
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
    return fd


def compute_tau_mean(fd, fixation_type):
    if fixation_type in ("form", "live"):
        keys = (("a1", "tau1"), ("a2", "tau2"))
    elif fixation_type == "glu":
        keys = (("a2", "tau2"), ("a3", "tau3"))
    else:
        return None
    if any(a not in fd or t not in fd for a, t in keys):
        return None
    a_sum = sum(fd[a] for a, _ in keys)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            a_sum > 0,
            sum(fd[a] * fd[t] for a, t in keys) / a_sum,
            np.nan,
        )


def title_str(label, arr, mask):
    """mean-inside-mask vs mean-outside-mask, matching coworker format."""
    if arr is None:
        return label
    inside  = arr[mask  & np.isfinite(arr)]
    outside = arr[~mask & np.isfinite(arr)]
    i_mean  = inside.mean()  if inside.size  > 0 else float("nan")
    o_mean  = outside.mean() if outside.size > 0 else float("nan")
    return f"{label}\n{i_mean:.2f} vs {o_mean:.2f}"


def _imshow_panel(ax, arr, mask, label, cmap="viridis", plo=1, phi=99):
    if arr is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_title(label, fontsize=8)
        ax.axis("off")
        return
    finite = arr[np.isfinite(arr)]
    vlo = float(np.percentile(finite, plo)) if finite.size > 0 else 0.0
    vhi = float(np.percentile(finite, phi)) if finite.size > 0 else 1.0
    im = ax.imshow(arr, cmap=cmap, vmin=vlo, vmax=vhi, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title_str(label, arr, mask), fontsize=7.5)
    ax.axis("off")

# ---------------------------------------------------------------------------
# Load candidates and select top-1 per group
# ---------------------------------------------------------------------------

candidates = pd.read_csv(RESULTS_DIR / "reference_panel_candidates.csv")

TOP_N = 3

top_rows = []  # list of (ft, ct, row) tuples, TOP_N per group
for ft, ct in GROUPS:
    sub = candidates[
        (candidates["fixation_type"] == ft) & (candidates["cell_type"] == ct)
    ]
    if sub.empty:
        print(f"WARNING: no candidates for {ft} {ct}")
        for _ in range(TOP_N):
            top_rows.append((ft, ct, None))
    else:
        for _, row in sub.head(TOP_N).iterrows():
            top_rows.append((ft, ct, row))

# ---------------------------------------------------------------------------
# Load data for each top candidate
# ---------------------------------------------------------------------------

panel_data = []
for ft, ct, row in top_rows:
    if row is None:
        panel_data.append(None)
        continue

    key = row["fit_set_key"]
    session_root, rel_dir, base_stem = key.split("::", 2)
    sdir = _session_dir(session_root)
    if sdir is None:
        print(f"WARNING: session dir not found for {session_root}")
        panel_data.append(None)
        continue

    fit_dir = sdir / Path(rel_dir)
    if not fit_dir.exists():
        print(f"WARNING: fit dir not found: {fit_dir}")
        panel_data.append(None)
        continue

    fd = load_asc(fit_dir, base_stem)
    print(f"Loaded {row['fixation_type']} {row['cell_type']}:"
          f"  params={list(fd.keys())}  file=...{base_stem[-40:]}")

    mask_path = fit_dir / f"{base_stem}_fit_mask.npy"
    if mask_path.exists():
        mask = np.load(str(mask_path))
    else:
        print(f"  WARNING: no mask .npy found, using all-True mask")
        shape = next((v.shape for v in fd.values()), (1, 1))
        mask = np.ones(shape, dtype=bool)

    tau_m = fd.get("tau_mean") or compute_tau_mean(fd, row["fixation_type"])

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (
            np.where(fd["a2"] > 0, fd["a1"] / fd["a2"], np.nan)
            if "a1" in fd and "a2" in fd else None
        )

    panel_data.append({
        "row":     row,
        "fd":      fd,
        "mask":    mask,
        "tau_m":   tau_m,
        "ratio":   ratio,
    })

# ---------------------------------------------------------------------------
# 4 x 7 figure
# ---------------------------------------------------------------------------

COL_LABELS = ["photons", "tau_mean (ps)", "a1", "a2", "a1/a2", "chi2", "mask"]
CMAPS      = ["inferno", "RdBu_r",       None, None, "RdBu_r", None,   "gray_r"]

n_rows = len(top_rows)  # 4 groups x TOP_N = 12
fig, axes = plt.subplots(n_rows, 7, figsize=(25, 3.5 * n_rows))
fig.subplots_adjust(hspace=0.55, wspace=0.35)

for ri, ((ft, ct, _), pd_entry) in enumerate(zip(top_rows, panel_data)):
    row_axes = axes[ri]

    rank = ri % TOP_N + 1
    if pd_entry is None:
        for ax in row_axes:
            ax.axis("off")
        row_axes[0].text(0.5, 0.5, f"no data\n{ft} {ct}",
                         ha="center", va="center", transform=row_axes[0].transAxes)
        continue

    row  = pd_entry["row"]
    fd   = pd_entry["fd"]
    mask = pd_entry["mask"]

    arrays = [
        fd.get("photons"),
        pd_entry["tau_m"],
        fd.get("a1"),
        fd.get("a2"),
        pd_entry["ratio"],
        fd.get("chi2"),
        mask.astype(float),
    ]

    short_fn = str(row["filename"])[-50:]
    row_label = (f"#{rank} {ft} {ct}  |  {row['session_date']}\n"
                 f"poc={row['pockels']:.2f}  b={row['_bin']}"
                 f"  fr={int(row['frame_index'])}  pct={row['pct_final']:.0f}%\n"
                 f"...{short_fn}")

    for ci, (arr, label, cmap) in enumerate(zip(arrays, COL_LABELS, CMAPS)):
        ax = row_axes[ci]
        if ci == 6:
            # Mask panel: show boolean, no percentile clipping
            im = ax.imshow(mask, cmap="gray_r", vmin=0, vmax=1,
                           interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"mask\n{int(mask.sum())} px kept", fontsize=7.5)
            ax.axis("off")
        else:
            _imshow_panel(ax, arr, mask, label,
                          cmap=cmap if cmap else "viridis")

    # Row label on left side of photons panel
    row_axes[0].set_ylabel(row_label, fontsize=7, rotation=0,
                           labelpad=120, va="center")

plt.suptitle(f"Reference panel candidates -- top-{TOP_N} per group (closest to group median)",
             fontsize=11)
plt.savefig(RESULTS_DIR / "reference_panel_view.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: reference_panel_view.png")
