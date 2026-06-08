# FLIM Pipeline API Reference

Quick reference for all public functions in `src/` and `utils/helpers.py`.
For full parameter descriptions see the docstrings (use `help(fn)` in a notebook).

---

## src.config

Load and manage YAML configuration and data-directory paths.

### `load_config(path=None) -> dict`

Reads `config/default.yaml` and normalizes types: YAML lists become tuples,
the string key `'default'` in `min_photons_by_ncomp` becomes `None`.
Call once at the top of a notebook and pass the result to downstream functions.

```python
from src.config import load_config
cfg = load_config()
# cfg['min_photons_by_ncomp']  -> {1: 500, 2: 1000, 3: 8000, None: 1000}
# cfg['tau_bounds']['form']    -> {'tau1': (50.0, 1500.0), 'tau2': (800.0, 6000.0)}
# cfg['taumean_cfg']['form']   -> (('a1', 'tau1'), ('a2', 'tau2'))
```

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| path | Path or str | `config/default.yaml` | YAML file to load |

Returns `dict` with keys `min_photons_by_ncomp`, `chi2_lo/hi`,
`tau_bounds`, `tau_mean_bounds`, `taumean_cfg`, `ratio_cfg`, `data_dirs`.

---

### `save_data_dirs(win_dirs, lin_dirs, *, path=None) -> Path`

Writes the authoritative list of .sdt session root directories to
`config/data_dirs.yaml`.  Called by Phase A; overwrites the file completely
on each call of this function.

```python
save_data_dirs(
    [r'E:\18_RK_Circadian\data\raw\20260429_KPC_fixed_dishes_on_SLIM'],
    ['/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260429_KPC_fixed_dishes_on_SLIM'],
)
```

| Parameter | Type | Meaning |
|-----------|------|---------|
| win_dirs | list of str/Path | Windows session-root paths |
| lin_dirs | list of str/Path | Linux session-root paths |
| path | Path or str | Destination file; default `config/data_dirs.yaml` |

Returns the `Path` written.  Side effect: creates/overwrites `data_dirs.yaml`.

---

### `get_data_dirs(config=None, *, require_exist=True) -> list[Path]`

Returns the .sdt session root directories as Path objects.  Reads
`config/data_dirs.yaml` first (Phase A output); falls back to
`config/default.yaml`.  Deduplicates win+lin lists and optionally filters
to paths that exist on disk.

```python
data_dirs = get_data_dirs()              # existing paths only
all_dirs  = get_data_dirs(require_exist=False)  # include offline paths
```

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| config | dict | None | Pre-loaded config dict; re-reads from disk if None |
| require_exist | bool | True | Raise FileNotFoundError if no path exists |

Returns `list[Path]`.

---

## src.io

Load SPCImage `.asc` exports, navigate fit folders, load Phase E masks.

### `parse_fit_folder(folder_name) -> dict`

Parses a BH SPCImage fit or export folder name into its components.

```python
parse_fit_folder('fitet-sz-b2-c2')
# -> {'b_val': 2, 'shift_free': False, 'kernel_size': 5,
#     'n_components': 2, 'is_export': False, ...}
```

| Key | Type | Meaning |
|-----|------|---------|
| b_val | int | Binning radius (kernel edge = 2*b+1) |
| shift_free | bool | True if IRF shift was fitted freely (-sf-) |
| kernel_size | int | Total kernel edge length = 2*b_val+1 |
| n_components | int or None | Number of exponential components |
| is_export | bool | True for exet* folders |
| export_format | str or None | 'asc' or 'tif' |
| color_lo/hi | float or None | tif colormap range (ps) |
| intensity_lo/hi | float or None | tif intensity LUT range |

---

### `fit_dir_from_key(fit_set_key, data_dirs) -> tuple`

Resolves a fit_set_key string to an absolute directory path.

```python
fit_dir, session_root, rel_dir, base_stem = fit_dir_from_key(
    row['fit_set_key'], data_dirs)
if fit_dir is not None:
    fd = load_asc_fit_set(fit_dir, base_stem)
```

| Parameter | Meaning |
|-----------|---------|
| fit_set_key | `'<session_root>::<rel_dir>::<base_stem>'` |
| data_dirs | session root Paths from `get_data_dirs()` |

Returns `(fit_dir, session_root, rel_dir, base_stem)`.
`fit_dir` is `None` if session_root is not found in data_dirs.

---

### `load_asc_fit_set(fit_dir, base_stem) -> dict`

Loads all `.asc` parameter maps for one .sdt base stem from a fit folder.

```python
fd = load_asc_fit_set(fit_dir, '3_KPCWT0430form20min_..._0000')
# fd.keys() -> {'a1', 'a2', 'tau1', 'tau2', 'chi2', 'photons'}
# fd['a1']  -> 2-D ndarray, shape (ny, nx)
```

Returns `{param_name: 2-D ndarray}`.  Files that fail to load or are not
2-D are silently skipped.

---

### `load_saved_mask(fit_set_key, data_dirs) -> np.ndarray or None`

Loads the Phase E quality mask (`{base_stem}_fit_mask.npy`) for a given
fit_set_key, or returns None if the file is absent.

```python
mask = load_saved_mask(row['fit_set_key'], data_dirs)
```

Returns a bool 2-D ndarray (True = pixel passed quality criteria) or None.

---

### `load_image_bundle(mask_path) -> tuple[dict, str]`

Convenience loader for the canonical set of images alongside a mask file.
Raises if any required `.asc` is missing; use `load_asc_fit_set()` for
partial or flexible loading.

```python
arrs, stem = load_image_bundle(row['fit_mask_path'])
# arrs.keys() -> {'photons', 'tau_mean', 'a1', 'a2', 'chi2', 'mask', 'a1/a2'}
plt.imshow(np.where(arrs['mask'], arrs['tau_mean'], np.nan), cmap='RdBu_r')
```

Returns `(arrs_dict, base_stem_str)`.

---

## src.preprocess

Condition labelling, Pockels classification, DataFrame assembly and filtering.

### `label_of(stem) -> str`

Parses a filename stem into a human-readable `'CELLTYPE condition'` label.

```python
label_of('3_KPCWT0430form20min_740nm...')  # -> 'KPCWT form 20min'
label_of('5_BKO0501live_740nm...')          # -> 'BKO live'
```

Condition priority: LIVE > FORM20MIN > FORM10MIN > FORM > GLU.
Returns `'KPCWT ?'` or `'BKO ?'` if condition is unrecognized.

---

### `pockels_class(values, threshold=0.3)`

Bins Pockels cell voltage into `'Low'` (<= threshold) or `'High'`
(> threshold).  Returns the same container type as the input.

```python
pockels_class(0.25)           # -> 'Low'
pockels_class(df['pockels'])  # -> pd.Series of 'Low'/'High'
```

NaN inputs pass through as `float('nan')`.

---

### `build_analysis_df(fit_analysis_csv, sdt_metadata_csv, *, ...) -> pd.DataFrame`

The central data-loading function.  Reads Phase E summary CSVs, merges
metadata and optional annotation files, and derives `date` and `PC` columns.

```python
df_analysis = build_analysis_df(
    'results/fit_analysis_summary.csv',
    'results/sdt_metadata_cal.csv',
    annotation_csv='results/position_annotation.csv',
)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| fit_analysis_csv | required | Phase E per-image summary |
| sdt_metadata_csv | required | Phase A/C acquisition metadata |
| annotation_csv | None | position_annotation.csv (adds 'annotation' column) |
| filepath_map_csv | None | filepath_map.csv (adds 'filepath' column) |
| pockels_threshold | 0.3 | 'Low'/'High' cutoff |
| derive_date | True | parse 'date' from session_root prefix |
| derive_pc | True | add 'PC' = 'Low'/'High' from pockels |
| extra_sdt_cols | (pockels, treatment_duration, ...) | columns to pull from sdt_metadata |

Returns `pd.DataFrame`, one row per .sdt image that survived Phase E.

---

### `filter_subset(df, **conditions) -> pd.DataFrame`

Filters a DataFrame to rows matching all conditions simultaneously (AND logic).
Returns a copy.

```python
sub = filter_subset(
    df_analysis,
    fixation_type='live',
    em_filter_nm=457,
    annotation='colony_edge',
    PC='Low',
)
```

Condition value types:
- Scalar: `df[col] == value`
- list/tuple/set: `df[col].isin(value)`
- callable: `df[col].map(callable)`

Raises `KeyError` if a column is missing from the DataFrame.

---

## src.phasor

Per-pixel phasor calculation, calibration, smoothing, and visualization.

Module-level defaults (override by passing kwargs):
`DEFAULT_REP_RATE_HZ = 80e6`, `DEFAULT_PHASOR_BLUR_SIGMA = 2.0`,
`DEFAULT_MASK_BLUR_SIGMA = 2.0`, `DEFAULT_MASK_FALLBACK_FRAC = 0.5`.

### `otsu_threshold(img) -> float`

Otsu threshold of a 2-D photon-count image (maximize between-class variance
on a 256-bin histogram).  Returns 0.0 on degenerate input.

```python
thresh = otsu_threshold(photons)
mask = photons >= thresh
```

---

### `intensity_mask(photon_img, fallback_frac=0.5, blur_sigma=2.0) -> (mask, threshold)`

Gaussian-blurred Otsu mask with a top-percentile fallback when Otsu keeps
<5% or >95% of pixels.

```python
mask, thresh = intensity_mask(photons)
print(f'{mask.mean()*100:.1f}% accepted')
```

---

### `smooth_phasor(G_cal, S_cal, photons, sigma=2.0) -> (G_sm, S_sm)`

Photon-weighted Gaussian smoothing of the phasor.  Equivalent to spatially
averaging the underlying TCSPC decays rather than averaging the per-pixel
phasors.

```python
G_sm, S_sm = smooth_phasor(G_cal, S_cal, photons)
```

---

### `get_time_ns(sdt_obj, n_bins) -> np.ndarray`

Returns bin-centre times in nanoseconds from a loaded `SdtFile`.  Falls back
to a uniform 12.5 ns grid if the times array has an unexpected length.

```python
t_ns = get_time_ns(sdt, decay.shape[2])
```

---

### `compute_phasor_raw(decay, time_ns, rep_rate_hz=80e6) -> (G, S, photons)`

Computes the first Fourier component of each pixel's TCSPC decay.

```python
G, S, photons = compute_phasor_raw(decay, time_ns)
# G, S in [0,1]; NaN where photons == 0
```

| Parameter | Shape | Meaning |
|-----------|-------|---------|
| decay | (ny, nx, n_timebins) | raw TCSPC histogram |
| time_ns | (n_timebins,) | bin-centre times in ns |

Returns `(G, S, photons)` each of shape `(ny, nx)`.

---

### `apply_phasor_cal(G_raw, S_raw, phase_corr, mod_corr) -> (G_cal, S_cal)`

Applies IRF calibration: rotates by `phase_corr` (radians) and scales by
`mod_corr`.  Parameters come from `phasor_cal_phase_rad` and `phasor_cal_mod`
columns in `sdt_metadata_cal.csv`.

```python
G_cal, S_cal = apply_phasor_cal(
    G, S, row['phasor_cal_phase_rad'], row['phasor_cal_mod'])
```

---

### `draw_semicircle(ax, *, rep_rate_hz=80e6, lifetime_ticks_ns=(...), set_axes=True)`

Draws the universal FLIM semicircle with lifetime tick marks on a phasor plot.
Single-exponential decays fall on the arc from (0,0) to (1,0).

```python
fig, ax = plt.subplots()
ax.scatter(G_cal[mask], S_cal[mask], s=1, alpha=0.3)
draw_semicircle(ax)
```

Side effect: modifies `ax` in place.

---

## src.metrics

Per-image statistics, effect sizes, and multiple-testing correction.

### `compute_tau_mean(fd, fixation_type, taumean_cfg=None) -> np.ndarray or None`

Amplitude-weighted tau_mean = sum(a_i * tau_i) / sum(a_i) over the component
pairs for the given fixation type.  Returns `None` if required keys are absent.

```python
tau_m = compute_tau_mean(fd, 'form')
median_ps = float(np.nanmedian(tau_m[mask]))
```

Default component pairs: glu: (a2,tau2)+(a3,tau3); form/live: (a1,tau1)+(a2,tau2).

---

### `rank_biserial_r(x, y) -> float`

Effect size for Mann-Whitney U: r = 2U/(n1*n2) - 1.  Positive r means x
tends to be larger than y.

```python
r = rank_biserial_r(kpcwt_vals, bko_vals)
```

---

### `benjamini_hochberg(pvals) -> np.ndarray`

BH FDR correction.  Returns q-values in the same order as input.
Test significance with `q <= alpha`.

```python
q = benjamini_hochberg(p_list)
sig = q <= 0.05
```

---

### `effect_stats(sub, metric, group_col='cell_type', group_a='KPCWT', group_b='BKO', n_boot=2000, seed=0) -> dict or None`

Median difference + 95% bootstrap CI + Mann-Whitney U + rank-biserial r.

```python
stats = effect_stats(df, 'tau_mean_median_ps')
# keys: diff, ci_lo, ci_hi, p, r, n_a, n_b
```

Returns None if either group has < 2 finite observations.

---

### `perm_test(sub, metric, ..., n_perm=2000, seed=0) -> tuple or None`

Permutation test on the median difference.  Returns
`(obs_diff, null_distribution, p_empirical)`.

```python
obs, null, p = perm_test(df, 'amp_ratio_median', n_perm=5000)
```

---

## src.fitting

Fit-set selection and per-pixel quality masking.

Module-level defaults (all overridable):
`DEFAULT_MIN_PHOTONS_BY_NCOMP = {1: 500, 2: 1000, 3: 8000, None: 1000}`,
`DEFAULT_CHI2_LO = 0.8`, `DEFAULT_CHI2_HI = 2.0`,
`DEFAULT_PREFERRED_N_COMP = {'glu': 3, 'form': 2, 'live': 2}`.

### `best_fit_key(filename, fit_map_df, fixation_type=None, preferred_n_comp=None) -> str or None`

Selects the highest-bin, shift-zero fit_set_key for an image, matching
the preferred number of components for the fixation type.

```python
key = best_fit_key(row['filename'], fit_map_df, fixation_type=row['fixation_type'])
if key:
    fit_dir, _, _, stem = fit_dir_from_key(key, data_dirs)
    fd = load_asc_fit_set(fit_dir, stem)
```

Returns `None` if no shift-zero fit with the preferred n_components exists.

---

### `compute_fit_mask(fd, fixation_type, b_val, ...) -> (mask, breakdown)`

Applies 5 quality criteria and returns a boolean pixel mask and a dict
summarizing how many pixels pass each criterion.

```python
mask, brk = compute_fit_mask(fd, 'form', b_val=2, n_components=2)
print(f'{brk["pct_final"]:.1f}% of pixels passed')
np.save('myfile_fit_mask.npy', mask)
```

Criteria: (1) photon count, (2) chi-squared, (3) amplitude positivity,
(4) tau bounds, (5) tau_mean bounds (skipped when bound is None).

`breakdown` keys: `n_total`, `n_photon_ok`, `n_chi2_ok`, `n_amp_ok`,
`n_tau_ok`, `n_taumean_ok`, `n_final`, `pct_final`.

---

## utils.helpers

Visualization helpers shared across notebooks.

### `set2_palette(cell_types) -> dict`

Maps an ordered list of group labels to Set2 colors.  Matches the Phase F
color scheme.

```python
ct_color = set2_palette(['KPCWT', 'BKO'])
```

---

### `violin_panel(ax, sub, metric, ct_order, ct_colors, *, ...) -> (cts_here, groups)`

Single violin + jittered scatter panel.  The core building block for all
violin figures in the pipeline.

```python
ct_color = set2_palette(['KPCWT', 'BKO'])
violin_panel(ax, df_sub, 'tau_mean_median_ps', ['KPCWT', 'BKO'], ct_color)
ax.set_ylabel('tau_mean (ps)')
```

Key kwargs: `group_col='cell_type'`, `min_n=2`, `jitter=0.07`, `seed=0`.
Returns `(cts_here, groups)` for downstream annotation.

---

### `violin_grid(df, metric, *, row_by, col_by, group_order, ...) -> (fig, axes)`

Creates a grid of violin panels faceted by up to two columns.

```python
fig, axes = violin_grid(
    df_analysis, 'tau_mean_median_ps',
    row_by='fixation_type', col_by='em_filter_nm',
    row_order=['form', 'live'], col_order=[457, 535],
    group_order=['KPCWT', 'BKO'],
    col_label_fmt='{:.0f} nm',
    suptitle='tau_mean',
)
```

`axes` is always 2-D; index as `axes[ri][ci]`.

---

### `show_image_row(ax_row, arrs, panels=None, include_mask=True, ...)`

Renders a row of parameter-map panels with quality-rejected pixels grayed out.

```python
arrs, stem = load_image_bundle(row['fit_mask_path'])
fig, axes = plt.subplots(1, 7, figsize=(21, 3))
show_image_row(axes, arrs)
```

---

### `show_photons_grid(subset, *, ncols=4, title='', ...)`

Plots a grid of photon-intensity thumbnails for a DataFrame subset.
Useful for a quick visual inventory of a filtered image set.

```python
show_photons_grid(
    filter_subset(df_analysis, fixation_type='form', em_filter_nm=457),
    title='form 457 nm',
)
```
