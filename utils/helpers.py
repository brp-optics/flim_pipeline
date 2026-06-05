"""Visualization helpers shared across the pipeline notebooks.

Migrated from quickpeek3 and view_reference_panel.  All functions are
parameterized so they do not depend on any notebook's module-level globals.

Functions:
  set2_palette         - dict mapping ordered cell types -> Set2 RGB colors
  violin_panel         - one violin + jittered scatter panel
  show_image_row       - 6-or-7 panel image row (photons + derived + mask)
  show_photons_grid    - grid of photons-only thumbnails for a DataFrame
  default_image_panels - canonical (key, cmap) pairs for 6-image display
"""

from __future__ import annotations

import textwrap
from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Palette factory
# ---------------------------------------------------------------------------

def set2_palette(cell_types: Sequence[str]) -> dict:
    """Map an ordered list of cell types -> Set2 colors (matches Phase F)."""
    n = max(len(cell_types), 1)
    set2 = plt.cm.Set2(np.linspace(0, 0.8, n))
    return {ct: set2[i] for i, ct in enumerate(cell_types)}


# ---------------------------------------------------------------------------
# Violin + scatter panel
# ---------------------------------------------------------------------------

def violin_panel(
    ax,
    sub,
    metric: str,
    ct_order: Sequence[str],
    ct_colors: dict,
    *,
    group_col: str = "cell_type",
    min_n: int = 2,
    jitter: float = 0.07,
    point_size: float = 22.0,
    line_width: float = 1.2,
    seed: int = 0,
) -> tuple:
    """Violin + jittered scatter with median/extrema lines (matches Phase F style).

    Args:
        ax:         matplotlib Axes.
        sub:        DataFrame with `group_col` and `metric`.
        metric:     numeric column.
        ct_order:   ordered list of categories to show (e.g. ['KPCWT', 'BKO']).
        ct_colors:  {category: color}.
        group_col:  column to group by.
        min_n:      groups with fewer rows are dropped.
        jitter:     +/- jitter range on x.
        point_size: scatter marker size.
        line_width: linewidth for median/extrema bars.
        seed:       RNG seed for jitter (reproducible).

    Returns (cts_here, groups).  groups is a list of value arrays parallel to cts_here.
    """
    cts_here = [ct for ct in ct_order if (sub[group_col] == ct).sum() >= min_n]
    groups   = [sub.loc[sub[group_col] == ct, metric].dropna().values for ct in cts_here]
    colors   = [ct_colors[ct] for ct in cts_here]

    if not groups:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        return cts_here, groups

    parts = ax.violinplot(groups, positions=range(len(groups)),
                          showmedians=True, showextrema=True)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.65)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("k")
            parts[key].set_linewidth(line_width)

    rng = np.random.default_rng(seed)
    for j, (grp, c) in enumerate(zip(groups, colors)):
        jit = rng.uniform(-jitter, jitter, len(grp))
        ax.scatter(j + jit, grp, s=point_size, color=c, alpha=0.7,
                   zorder=3, edgecolors="none")

    ax.set_xticks(range(len(cts_here)))
    ax.set_xticklabels(cts_here, fontsize=10)
    return cts_here, groups


# ---------------------------------------------------------------------------
# Image row (photons + tau_mean + a1 + a2 + a1/a2 + chi2 + mask)
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_PANELS = [
    ("photons",  "inferno"),
    ("tau_mean", "RdBu_r"),
    ("a1",       "viridis"),
    ("a2",       "viridis"),
    ("a1/a2",    "RdBu_r"),
    ("chi2",     "viridis"),
]


def show_image_row(
    ax_row,
    arrs: dict,
    panels: list | None = None,
    include_mask: bool  = True,
    raw_keys: Iterable[str] = ("photons",),
    fontsize: int = 8,
) -> None:
    """Render one row of image panels with rejected pixels grayed out.

    Args:
        ax_row:       1-D array of matplotlib Axes; needs len = len(panels) + 1
                      if include_mask else len(panels).
        arrs:         dict {param: 2-D ndarray} including 'mask' (bool).
        panels:       list of (key, cmap_name) tuples.  Defaults to the
                      photons/tau_mean/a1/a2/a1-over-a2/chi2 canonical set.
        include_mask: append a final mask panel.
        raw_keys:     keys NOT to mask out (typically just 'photons').
        fontsize:     panel title font size.
    """
    panels = panels if panels is not None else DEFAULT_IMAGE_PANELS
    mask   = arrs["mask"]
    for ax, (key, cmap_name) in zip(ax_row[:len(panels)], panels):
        arr     = arrs[key]
        display = arr if key in raw_keys else np.where(mask, arr, np.nan)
        cmap    = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("lightgray")
        finite  = display[np.isfinite(display)]
        vlo = float(np.percentile(finite, 1))  if finite.size else 0.0
        vhi = float(np.percentile(finite, 99)) if finite.size else 1.0
        im  = ax.imshow(display, cmap=cmap, vmin=vlo, vmax=vhi)
        inside = arr[mask & np.isfinite(arr)]
        i_m    = float(inside.mean()) if inside.size else float("nan")
        ax.set_title(f"{key}\n{i_m:.2f}", fontsize=fontsize)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")

    if include_mask:
        ax = ax_row[len(panels)]
        im = ax.imshow(mask.astype(float), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"mask\n{int(mask.sum())} px ({100*mask.mean():.0f}%)",
                     fontsize=fontsize)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")


# ---------------------------------------------------------------------------
# Photons grid
# ---------------------------------------------------------------------------

def violin_grid(
    df,
    metric: str,
    *,
    row_by: str | None = None,
    col_by: str | None = None,
    group_by: str = "cell_type",
    row_order: Sequence | None = None,
    col_order: Sequence | None = None,
    group_order: Sequence,
    group_colors: dict | None = None,
    figsize_per_panel: tuple = (4, 4),
    sharey: bool = True,
    min_n: int = 2,
    title_fmt: str = "{row_v} / {col_v}\nn={n_str}",
    row_label_fmt: str | None = None,
    col_label_fmt: str | None = None,
    suptitle: str | None = None,
    seed: int = 0,
):
    """Grid of violin+scatter panels — rows by `row_by`, cols by `col_by`.

    Each panel is one (row_value, col_value) slice of `df`, with `violin_panel`
    drawing the inner violin+jitter for groups along the x-axis.

    Args:
        df:           DataFrame.
        metric:       numeric column to compare.
        row_by:       column name for row faceting; None => 1 row.
        col_by:       column name for col faceting; None => 1 col.
        group_by:     column distinguishing the two groups within each panel.
        row_order:    sequence of row_by values to plot, in order; None => sorted unique.
        col_order:    sequence of col_by values; None => sorted unique.
        group_order:  ordered list of group labels (required).
        group_colors: {label: color}; defaults to set2_palette(group_order).
        figsize_per_panel: (w, h) inches.
        sharey:       share y-axis across panels.
        min_n:        a group with < min_n rows is dropped from a panel.
        title_fmt:    f-string with placeholders {row_v}, {col_v}, {n_str}.
                      For 1-d grids, the empty axis still substitutes ''.
        row_label_fmt / col_label_fmt: optional pre-formatters applied to the
                      row/col value before substitution (e.g. "{:.0f} nm").
        suptitle:     figure title.
        seed:         RNG seed for jitter (forwarded to violin_panel).

    Returns (fig, axes).  axes is always 2-D so callers can index axes[ri][ci].
    """
    if group_colors is None:
        group_colors = set2_palette(group_order)

    # Determine row / col axes
    def _vals(col):
        return sorted(df[col].dropna().unique()) if col else [None]

    rows = list(row_order) if row_order is not None else _vals(row_by)
    cols = list(col_order) if col_order is not None else _vals(col_by)

    n_rows = max(len(rows), 1)
    n_cols = max(len(cols), 1)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False, sharey=sharey,
    )

    def _fmt(value, fmt):
        if value is None:
            return ""
        if fmt is None:
            return str(value)
        try:
            return fmt.format(value)
        except Exception:
            return str(value)

    for ri, row_v in enumerate(rows):
        for ci, col_v in enumerate(cols):
            ax  = axes[ri][ci]
            sub = df
            if row_by and row_v is not None:
                sub = sub[sub[row_by] == row_v]
            if col_by and col_v is not None:
                sub = sub[sub[col_by] == col_v]
            sub = sub.dropna(subset=[metric])

            cts_here, _groups = violin_panel(
                ax, sub, metric,
                ct_order=group_order, ct_colors=group_colors,
                group_col=group_by, min_n=min_n,
                seed=seed,
            )

            n_str = "  ".join(
                f"{ct}:{(sub[group_by] == ct).sum()}"
                for ct in cts_here
            )
            ax.set_title(
                title_fmt.format(
                    row_v=_fmt(row_v, row_label_fmt),
                    col_v=_fmt(col_v, col_label_fmt),
                    n_str=n_str,
                ),
                fontsize=8,
            )

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)

    return fig, axes


def show_photons_grid(
    subset,
    mask_path_col: str = "fit_mask_path",
    filename_col:  str = "filename",
    cell_type_col: str = "cell_type",
    date_col:      str = "date",
    ncols: int = 4,
    title: str = "",
    wrap_filename: int = 55,
) -> None:
    """Plot a grid of photon-intensity thumbnails for an image-level subset.

    Expects each row to have a column with the path to the *_fit_mask.npy
    file; the photons map is loaded from the same folder by replacing the
    suffix with _photons.asc.
    """
    n = len(subset)
    if n == 0:
        print(f"(empty: {title})")
        return
    nrows = -(-n // ncols)
    plt.figure(figsize=(5.5 * ncols, 4.5 * nrows))
    for i, (_, row) in enumerate(subset.iterrows()):
        ph_path = str(row[mask_path_col]).replace("_fit_mask.npy", "_photons.asc")
        ax = plt.subplot(nrows, ncols, i + 1)
        try:
            ph = np.loadtxt(ph_path)
            finite = ph[ph > 0]
            vlo = float(np.percentile(finite, 1))  if finite.size else 0.0
            vhi = float(np.percentile(finite, 99)) if finite.size else 1.0
            plt.imshow(ph, cmap="inferno", vmin=vlo, vmax=vhi)
            plt.colorbar(fraction=0.046, pad=0.04)
        except Exception:
            plt.text(0.5, 0.5, "no .asc", ha="center",
                     transform=ax.transAxes)
        date_str = row[date_col].strftime("%m-%d") if hasattr(row[date_col], "strftime") else str(row[date_col])
        wrapped  = "\n".join(textwrap.wrap(str(row[filename_col]), width=wrap_filename))
        plt.title(f"{row[cell_type_col]} {date_str}\n{wrapped}", fontsize=7)
        plt.axis("off")
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()
