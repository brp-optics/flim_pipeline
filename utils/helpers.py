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
    """Map an ordered list of cell types to Set2 colors (matches Phase F palette).

    Parameters
    ----------
    cell_types : sequence of str
        Ordered list of group labels, e.g. ['KPCWT', 'BKO'].

    Returns
    -------
    dict
        {cell_type: rgba_array} where each value is a length-4 float array
        from matplotlib's Set2 colormap.

    Examples
    --------
    >>> ct_color = set2_palette(['KPCWT', 'BKO'])
    >>> violin_panel(ax, sub, 'tau_mean_median_ps', ['KPCWT', 'BKO'], ct_color)

    Dependencies
    ------------
    numpy, matplotlib.pyplot
    """
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

    Draws a violinplot for each group in ct_order that has >= min_n rows,
    overlaid with jittered scatter points.  Median and extrema bars are
    drawn in black.  Groups with < min_n rows are silently omitted.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    sub : pd.DataFrame
        Data slice; must contain `group_col` and `metric` columns.
    metric : str
        Numeric column to plot on the y-axis (e.g. 'tau_mean_median_ps').
    ct_order : sequence of str
        Ordered list of group labels; sets x-axis order.
    ct_colors : dict
        {group_label: color} for violin fill and scatter points.
    group_col : str, optional
        Column that identifies each group.  Default 'cell_type'.
    min_n : int, optional
        Minimum number of rows required to draw a violin.  Default 2.
    jitter : float, optional
        Half-width of uniform x-jitter for scatter points.  Default 0.07.
    point_size : float, optional
        Scatter marker size (matplotlib s units).  Default 22.0.
    line_width : float, optional
        Linewidth for median/cbars/cmins/cmaxes.  Default 1.2.
    seed : int, optional
        RNG seed for reproducible jitter.  Default 0.

    Returns
    -------
    tuple
        (cts_here, groups) where cts_here is the list of groups actually
        drawn (subset of ct_order with >= min_n rows) and groups is a
        parallel list of 1-D value arrays.

    Side Effects
    ------------
    Modifies ax in place (adds violin, scatter, and line artists; sets
    x-tick labels).

    Assumptions
    -----------
    NaN values in the metric column are dropped before plotting.

    Examples
    --------
    >>> ct_color = set2_palette(['KPCWT', 'BKO'])
    >>> violin_panel(ax, df_sub, 'tau_mean_median_ps',
    ...              ['KPCWT', 'BKO'], ct_color)
    >>> ax.set_ylabel('tau_mean (ps)')

    Dependencies
    ------------
    numpy, matplotlib.pyplot
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
    """Render one row of image panels with quality-rejected pixels grayed out.

    Displays each parameter map at [1st, 99th] percentile contrast.  Pixels
    outside the quality mask are shown as light gray (using cmap.set_bad).
    Optionally appends a mask panel showing accepted pixel fraction.

    Parameters
    ----------
    ax_row : array-like of matplotlib.axes.Axes
        1-D array of Axes objects.  Must have length = len(panels) + 1 when
        include_mask=True, or len(panels) otherwise.
    arrs : dict
        {param_name: 2-D ndarray}.  Must include 'mask' (bool array).
        Typical keys: 'photons', 'tau_mean', 'a1', 'a2', 'a1/a2', 'chi2'.
        Obtain from load_image_bundle().
    panels : list of (str, str), optional
        List of (param_key, colormap_name) pairs to render in order.
        Defaults to DEFAULT_IMAGE_PANELS (photons/tau_mean/a1/a2/a1_a2/chi2).
    include_mask : bool, optional
        If True (default), append a final grayscale panel showing the mask.
    raw_keys : iterable of str, optional
        Keys to render without masking (masked pixels shown as-is).
        Default ('photons',) so photon count is always shown unmasked.
    fontsize : int, optional
        Font size for panel titles.  Default 8.

    Side Effects
    ------------
    Modifies all Axes in ax_row in place (imshow, colorbar, title, axis off).

    Examples
    --------
    >>> arrs, stem = load_image_bundle(row['fit_mask_path'])
    >>> fig, axes = plt.subplots(1, 7, figsize=(21, 3))
    >>> show_image_row(axes, arrs)

    Dependencies
    ------------
    numpy, matplotlib.pyplot
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
    """Grid of violin+scatter panels faceted by row_by and col_by columns.

    Creates a figure with n_rows x n_cols subplots.  Each panel corresponds
    to one (row_value, col_value) slice of df and calls violin_panel() to
    draw the within-panel group comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Full analysis DataFrame (will be filtered per panel).
    metric : str
        Numeric column to compare across groups (e.g. 'tau_mean_median_ps').
    row_by : str, optional
        Column for row faceting (e.g. 'fixation_type').  None => 1 row.
    col_by : str, optional
        Column for column faceting (e.g. 'em_filter_nm').  None => 1 col.
    group_by : str, optional
        Column distinguishing the x-axis groups within each panel.
        Default 'cell_type'.
    row_order : sequence, optional
        Ordered row_by values; None => sorted unique.
    col_order : sequence, optional
        Ordered col_by values; None => sorted unique.
    group_order : sequence
        Ordered group labels (required).  Sets x-axis order in every panel.
    group_colors : dict, optional
        {group_label: color}; defaults to set2_palette(group_order).
    figsize_per_panel : tuple of float, optional
        (width, height) in inches per panel.  Default (4, 4).
    sharey : bool, optional
        Share the y-axis scale across all panels.  Default True.
    min_n : int, optional
        Groups with fewer rows are omitted from that panel.  Default 2.
    title_fmt : str, optional
        f-string for panel titles.  Placeholders: {row_v}, {col_v}, {n_str}
        (n_str is 'KPCWT:N  BKO:N').  For 1-D grids the unused value is ''.
    row_label_fmt : str, optional
        Format string applied to each row_by value before title substitution,
        e.g. '{:.0f}' to render a float as an integer.
    col_label_fmt : str, optional
        Format string applied to each col_by value.
    suptitle : str, optional
        Figure super-title.
    seed : int, optional
        RNG seed forwarded to violin_panel for reproducible jitter.

    Returns
    -------
    tuple
        (fig, axes) where axes is always 2-D (shape n_rows x n_cols) so
        callers can safely index axes[ri][ci].

    Side Effects
    ------------
    Creates and shows a matplotlib Figure.

    Examples
    --------
    >>> fig, axes = violin_grid(
    ...     df_analysis, 'tau_mean_median_ps',
    ...     row_by='fixation_type', col_by='em_filter_nm',
    ...     row_order=['form', 'live'], col_order=[457, 535],
    ...     group_order=['KPCWT', 'BKO'],
    ...     col_label_fmt='{:.0f} nm',
    ...     suptitle='tau_mean by fixation and channel',
    ... )

    Dependencies
    ------------
    numpy, matplotlib.pyplot, violin_panel, set2_palette
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

    Each thumbnail is a raw photon-count image displayed at [1st, 99th]
    percentile contrast.  The photon .asc is located by replacing
    '_fit_mask.npy' with '_photons.asc' in the mask path column.

    Parameters
    ----------
    subset : pd.DataFrame
        Rows to display.  Must have columns for mask path, filename,
        cell_type, and date (see parameter names below).
    mask_path_col : str, optional
        Column containing the _fit_mask.npy path.  Default 'fit_mask_path'.
    filename_col : str, optional
        Column with the .sdt filename for the panel title.  Default 'filename'.
    cell_type_col : str, optional
        Column with the cell type label.  Default 'cell_type'.
    date_col : str, optional
        Column with a datetime (or string) for the title.  Default 'date'.
    ncols : int, optional
        Number of thumbnail columns in the grid.  Default 4.
    title : str, optional
        Super-title for the figure.  Default ''.
    wrap_filename : int, optional
        Max characters per line when wrapping long filenames.  Default 55.

    Side Effects
    ------------
    Creates and shows a matplotlib Figure via plt.show().

    Assumptions
    -----------
    _photons.asc must exist alongside each _fit_mask.npy file.  Rows where
    the .asc is missing are shown as a 'no .asc' placeholder.

    Examples
    --------
    >>> show_photons_grid(filter_subset(df, fixation_type='form', em_filter_nm=457),
    ...                   title='form 457 nm -- all images')

    Dependencies
    ------------
    numpy, matplotlib.pyplot, textwrap
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
